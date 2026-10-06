"""Pipeline de preprocesamiento del modelo de probabilidad de default (PD).

Reproduce, para datos nuevos, las transformaciones que se aplicaron a las variables
durante el desarrollo del modelo (notebook ``modelo_riesgo``):

1. Validación del esquema (columnas requeridas, IDs únicos, datos no vacíos).
2. Estandarización de ``application_date`` (formatos ``D/MM/AA`` y ``AAAA-MM-DD``).
3. Tipificación: enteras, continuas, booleanas, categóricas y ordinales.
   Un ingreso no numérico pasa a NaN y deja la bandera ``ingreso_corrupto``.
4. Reporte de valores fuera de rango (solo se reportan, no se corrigen).
5. Winsorización de la cola alta con límites FIJOS calculados en train (nunca se
   recalculan con datos nuevos).
6. Variables construidas (``pago_prom`` y otras candidatas).
7. Formato final para el modelo (``has_bureau_record`` como 0/1).

Principios: la función no modifica el DataFrame de entrada, no aprende nada de los
datos que recibe (todo parámetro viene de la configuración o del artefacto del
modelo) y no imputa: los nulos de las variables de pago los maneja el bin ``NULO``
del modelo. Los logs registran conteos, no valores de clientes.

Uso::

    from preprocesamiento import preprocesar, cargar_limites_winsor
    limites = cargar_limites_winsor("artefactos/modelo_pd.json")
    resultado = preprocesar(df_crudo, limites_winsor=limites)
    features = resultado.datos
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #


class ErrorValidacionDatos(ValueError):
    """Los datos de entrada no cumplen el contrato esperado por el pipeline."""


@dataclass(frozen=True)
class ConfigPreprocesamiento:
    """Parámetros del pipeline. Es inmutable: para cambiar algo se usa ``replace``."""

    # Columnas mínimas para poder puntuar con el modelo actual
    requeridas: tuple[str, ...] = (
        "application_id", "application_date", "avg_monthly_topups", "has_bureau_record",
        "utility_payment_ontime_rate", "telco_payment_ontime_rate",
    )
    id_col: str = "application_id"
    fecha_col: str = "application_date"
    # Formatos de fecha aceptados, en orden de prueba
    formatos_fecha: tuple[str, ...] = ("%Y-%m-%d", "%d/%m/%y")

    enteras: tuple[str, ...] = ("age", "app_tenure_months", "avg_monthly_topups",
                                "num_transactions_last_90d", "monto_bono_referidos", "requested_amount")
    continuas: tuple[str, ...] = ("monthly_income_declared", "pct_p2p_transfers", "bureau_score",
                                  "utility_payment_ontime_rate", "telco_payment_ontime_rate",
                                  "digital_engagement_score")
    booleanas: tuple[str, ...] = ("has_bureau_record", "disbursed_flag")
    categoricas: tuple[str, ...] = ("sexo", "product_type")
    ordinales: Mapping[str, tuple[int, ...]] = field(default_factory=lambda: {"device_price_tier": (1, 2, 3, 4, 5)})
    # Rangos esperados: solo se reportan, no se corrigen
    rangos: Mapping[str, tuple[float, float]] = field(default_factory=lambda: {
        "age": (18, 65), "pct_p2p_transfers": (0, 1), "utility_payment_ontime_rate": (0, 1),
        "telco_payment_ontime_rate": (0, 1), "digital_engagement_score": (0, 100), "bureau_score": (300, 900),
    })
    # True: ante datos inválidos se lanza error. False: se marcan como NaN/NaT y se registra un warning
    modo_estricto: bool = True


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #


@dataclass
class ResultadoPreprocesamiento:
    datos: pd.DataFrame
    reporte: dict


# --------------------------------------------------------------------------- #
# Utilidades de bajo nivel
# --------------------------------------------------------------------------- #


def _ids_muestra(ids: pd.Series, mascara: pd.Series, n: int = 5) -> list[str]:
    """Primeros ``n`` IDs de las filas con problema (para mensajes de error)."""
    return ids[mascara].astype(str).head(n).tolist()


def validar_esquema(df: pd.DataFrame, config: ConfigPreprocesamiento) -> None:
    """Verifica que el DataFrame se pueda procesar. Lanza ``ErrorValidacionDatos``."""
    if not isinstance(df, pd.DataFrame):
        raise ErrorValidacionDatos(f"Se esperaba un DataFrame y llegó {type(df).__name__}")
    if df.empty:
        raise ErrorValidacionDatos("El DataFrame de entrada está vacío")
    faltantes = [c for c in config.requeridas if c not in df.columns]
    if faltantes:
        raise ErrorValidacionDatos(f"Faltan columnas requeridas: {faltantes}")
    if df[config.id_col].isna().any():
        raise ErrorValidacionDatos(f"Hay valores nulos en {config.id_col}")
    duplicados = df[config.id_col].duplicated(keep=False)
    if duplicados.any():
        raise ErrorValidacionDatos(
            f"IDs duplicados en {config.id_col}: {_ids_muestra(df[config.id_col], duplicados)}")


def estandarizar_fecha(serie: pd.Series, formatos: Sequence[str], ids: pd.Series | None = None,
                       estricto: bool = True) -> pd.Series:
    """Convierte una columna de texto a ``datetime64`` probando cada formato en orden.

    Las filas que ningún formato reconoce quedan como NaT. En modo estricto se lanza
    ``ErrorValidacionDatos`` indicando los IDs afectados.
    """
    if pd.api.types.is_datetime64_any_dtype(serie):
        return pd.to_datetime(serie).dt.normalize()

    texto = serie.astype("string").str.strip()
    resultado = pd.to_datetime(pd.Series(pd.NaT, index=serie.index))
    for formato in formatos:
        pendiente = resultado.isna() & texto.notna()
        if not pendiente.any():
            break
        parcial = pd.to_datetime(texto.where(pendiente), format=formato, errors="coerce")
        resultado = resultado.combine_first(parcial)

    invalidas = resultado.isna()
    if invalidas.any():
        mensaje = f"{int(invalidas.sum())} fechas no reconocidas en '{serie.name}' (formatos {list(formatos)})"
        if ids is not None:
            mensaje += f"; ejemplos de ID: {_ids_muestra(ids, invalidas)}"
        if estricto:
            raise ErrorValidacionDatos(mensaje)
        logger.warning(mensaje)
    return resultado


def convertir_numericas(df: pd.DataFrame, columnas: Sequence[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convierte a ``float64``. Lo no numérico pasa a NaN. Devuelve cuántos valores se corrompieron."""
    out = df.copy()
    corruptos: dict[str, int] = {}
    for col in columnas:
        if col not in out.columns:
            continue
        original = out[col]
        convertido = pd.to_numeric(original, errors="coerce").astype("float64")
        n_corruptos = int((convertido.isna() & original.notna()).sum())
        if n_corruptos:
            corruptos[col] = n_corruptos
            logger.warning("%s: %d valores no numéricos pasaron a NaN", col, n_corruptos)
        out[col] = convertido
    return out, corruptos


_VERDADERO = {"true", "1", "t", "si", "sí", "yes", "y"}
_FALSO = {"false", "0", "f", "no", "n"}


def a_booleano(serie: pd.Series) -> pd.Series:
    """Convierte True/False, 0/1 o texto ('True', 'false', 'si', 'no') a ``bool``.

    Nulos o valores desconocidos lanzan ``ErrorValidacionDatos``: una bandera de buró
    ausente no se puede asumir.
    """
    if serie.isna().any():
        raise ErrorValidacionDatos(f"'{serie.name}' tiene {int(serie.isna().sum())} valores nulos")
    if serie.dtype == bool:
        return serie.copy()
    normal = serie.astype("string").str.strip().str.lower()
    resultado = pd.Series(np.nan, index=serie.index, dtype="object")
    resultado[normal.isin(_VERDADERO)] = True
    resultado[normal.isin(_FALSO)] = False
    if resultado.isna().any():
        raros = sorted(set(serie[resultado.isna()].astype(str)))[:5]
        raise ErrorValidacionDatos(f"'{serie.name}' tiene valores no booleanos: {raros}")
    return resultado.astype(bool)


def tipificar(df: pd.DataFrame, config: ConfigPreprocesamiento) -> tuple[pd.DataFrame, dict]:
    """Asigna el tipo correcto a cada variable presente. No imputa, no agrupa y no escala."""
    out = df.copy()
    ids = out[config.id_col]
    novedades: dict = {}

    out[config.fecha_col] = estandarizar_fecha(out[config.fecha_col], config.formatos_fecha, ids, config.modo_estricto)

    out, corruptos = convertir_numericas(out, [*config.enteras, *config.continuas])
    novedades["valores_no_numericos"] = corruptos
    # Bandera del ingreso corrupto (igual que en el desarrollo del modelo)
    if "monthly_income_declared" in df.columns:
        out["ingreso_corrupto"] = df["monthly_income_declared"].notna() & out["monthly_income_declared"].isna()
    if corruptos and config.modo_estricto:
        solo_ingreso = set(corruptos) <= {"monthly_income_declared"}
        if not solo_ingreso:       # el ingreso corrupto es un caso conocido y tolerado; los demás no
            raise ErrorValidacionDatos(f"Valores no numéricos en columnas: {corruptos}")

    for col in config.booleanas:
        if col in out.columns:
            out[col] = a_booleano(out[col])

    for col in config.categoricas:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip().astype("category")

    for col, niveles in config.ordinales.items():
        if col in out.columns:
            valor = pd.to_numeric(out[col], errors="coerce")
            invalido = valor.notna() & ~valor.isin(niveles)
            if invalido.any():
                logger.warning("%s: %d valores fuera de %s pasaron a NaN", col, int(invalido.sum()), list(niveles))
            out[col] = pd.Categorical(valor.where(~invalido).astype("Int64"), dtype=pd.CategoricalDtype(list(niveles), ordered=True))
    return out, novedades


def reportar_rangos(df: pd.DataFrame, rangos: Mapping[str, tuple[float, float]]) -> dict[str, int]:
    """Cuenta valores fuera de rango por variable. Solo reporta: no corrige nada."""
    fuera: dict[str, int] = {}
    for col, (minimo, maximo) in rangos.items():
        if col not in df.columns:
            continue
        n = int(((df[col] < minimo) | (df[col] > maximo)).sum())
        if n:
            fuera[col] = n
            logger.warning("%s: %d valores fuera del rango esperado [%s, %s]", col, n, minimo, maximo)
    return fuera


def winsorizar(df: pd.DataFrame, limites_superiores: Mapping[str, float] | None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recorta la cola alta con límites FIJOS (percentil 99 de train). La cola baja NO se toca.

    La cola baja de ``avg_monthly_topups`` es información de riesgo (25.9% de default
    en el 1% inferior), por eso solo se recorta por arriba.
    """
    out = df.copy()
    recortados: dict[str, int] = {}
    for col, limite in (limites_superiores or {}).items():
        if col not in out.columns:
            continue
        if not np.isfinite(limite):
            raise ErrorValidacionDatos(f"Límite de winsorización inválido para {col}: {limite}")
        n = int((out[col] > limite).sum())
        out[col] = out[col].clip(upper=limite)
        recortados[col] = n
        logger.info("%s: %d valores recortados al límite %.2f", col, n, limite)
    return out, recortados


def _dividir(num: pd.Series, den: pd.Series) -> pd.Series:
    """División que devuelve NaN (no infinito) cuando el denominador es 0."""
    return (num / den).replace([np.inf, -np.inf], np.nan)


def construir_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables derivadas evaluadas en el desarrollo. El modelo actual usa ``pago_prom``.

    - ``pago_prom``: promedio de pago a tiempo de servicios públicos y telco (ignora un nulo; NaN si faltan ambos).
    """
    out = df.copy()
    out["pago_prom"] = out[["utility_payment_ontime_rate", "telco_payment_ontime_rate"]].mean(axis=1)

    # Candidatas que no entraron al modelo; se conservan para monitoreo y reentrenamientos
    if {"avg_monthly_topups", "monthly_income_declared"} <= set(out.columns):
        out["topups_sobre_ingreso"] = _dividir(out["avg_monthly_topups"], out["monthly_income_declared"])
    if {"avg_monthly_topups", "requested_amount"} <= set(out.columns):
        out["topups_sobre_monto"] = _dividir(out["avg_monthly_topups"], out["requested_amount"])
    if {"requested_amount", "monthly_income_declared"} <= set(out.columns):
        out["monto_sobre_ingreso"] = _dividir(out["requested_amount"], out["monthly_income_declared"])
    if "num_transactions_last_90d" in out.columns:
        out["sin_transacciones"] = (out["num_transactions_last_90d"] == 0).astype(int)
        if "app_tenure_months" in out.columns:      # meses observables dentro de la ventana de 90 días (máximo 3)
            out["trx_por_mes"] = _dividir(out["num_transactions_last_90d"], np.minimum(out["app_tenure_months"], 3))
    if {"pct_p2p_transfers", "num_transactions_last_90d"} <= set(out.columns):
        out["p2p_con_trx"] = out["pct_p2p_transfers"].where(out["num_transactions_last_90d"] > 0)
    if "monto_bono_referidos" in out.columns:
        out["tiene_bono"] = (out["monto_bono_referidos"] > 0).astype(int)
    return out


def formato_para_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Deja las columnas en el formato que consume el scoring: ``has_bureau_record`` como 0/1."""
    out = df.copy()
    out["has_bureau_record"] = out["has_bureau_record"].astype(int)
    return out


def cargar_limites_winsor(ruta_artefacto: str | Path) -> dict[str, float]:
    """Lee los límites de winsorización guardados junto al modelo (calculados en train)."""
    with open(ruta_artefacto, encoding="utf-8") as f:
        artefacto = json.load(f)
    if "limite_winsor_p99" not in artefacto:
        raise ErrorValidacionDatos(f"El artefacto {ruta_artefacto} no tiene 'limite_winsor_p99'")
    return {k: float(v) for k, v in artefacto["limite_winsor_p99"].items()}


def configurar_logging(nivel: int = logging.INFO, archivo: str | Path | None = None) -> None:
    """Configura el logging de la aplicación (consola y, opcionalmente, un archivo).

    Es un ayudante para scripts y jobs; el módulo en sí solo usa ``logging.getLogger``.
    """
    manejadores: list[logging.Handler] = [logging.StreamHandler()]
    if archivo is not None:
        manejadores.append(logging.FileHandler(archivo, encoding="utf-8"))
    logging.basicConfig(level=nivel, handlers=manejadores, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# --------------------------------------------------------------------------- #
# Función principal
# --------------------------------------------------------------------------- #


def preprocesar(df: pd.DataFrame, config: ConfigPreprocesamiento | None = None,
                limites_winsor: Mapping[str, float] | None = None) -> ResultadoPreprocesamiento:
    """Aplica el pipeline completo y devuelve los datos listos para el scoring y un reporte.

    Parámetros
    ----------
    df: datos crudos con las columnas de ``dataset_candidato`` (sin la variable objetivo).
    config: parámetros del pipeline (por defecto, los del desarrollo del modelo).
    limites_winsor: ``{columna: límite superior}`` calculado en train
        (ver ``cargar_limites_winsor``). Si es None no se winsoriza y se registra un warning.

    El DataFrame de entrada no se modifica.
    """
    config = config or ConfigPreprocesamiento()
    inicio = time.perf_counter()
    validar_esquema(df, config)
    logger.info("Preprocesamiento v%s: %d filas, %d columnas", __version__, len(df), df.shape[1])
    if limites_winsor is None:
        logger.warning("No se entregaron límites de winsorización: se omite el recorte de la cola alta")

    datos, novedades = tipificar(df, config)
    fuera_de_rango = reportar_rangos(datos, config.rangos)
    datos, recortados = winsorizar(datos, limites_winsor)
    datos = construir_variables(datos)
    datos = formato_para_modelo(datos)

    reporte = {
        "version": __version__,
        "filas": len(datos),
        "nulos": {c: int(n) for c, n in datos.isna().sum().items() if n},
        "valores_no_numericos": novedades["valores_no_numericos"],
        "fuera_de_rango": fuera_de_rango,
        "winsorizados": recortados,
        "ingreso_corrupto": int(datos["ingreso_corrupto"].sum()) if "ingreso_corrupto" in datos else 0,
    }
    logger.info("Preprocesamiento terminado en %.3f s. Nulos por columna: %s", time.perf_counter() - inicio, reporte["nulos"])
    return ResultadoPreprocesamiento(datos=datos, reporte=reporte)
