"""Scoring: aplica el artefacto JSON del modelo (bins, WoE y coeficientes) a datos ya preprocesados.

    from scoring import puntuar
    pd_series = puntuar(df_crudo, "artefactos/modelo_pd.json")

PD = 1 / (1 + exp(-(intercepto + sum(coeficiente_v * WoE_v(bin del cliente))))).
No depende de scikit-learn: solo de pandas y numpy.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from preprocesamiento import ConfigPreprocesamiento, ErrorValidacionDatos, cargar_limites_winsor, preprocesar

logger = logging.getLogger(__name__)


class ErrorArtefacto(ValueError):
    """El JSON del modelo está incompleto o es inconsistente."""


def cargar_artefacto(ruta: str | Path) -> dict:
    """Lee y valida el artefacto del modelo."""
    with open(ruta, encoding="utf-8") as f:
        art = json.load(f)
    validar_artefacto(art)
    logger.info("Artefacto cargado desde %s: variables=%s", ruta, list(art["variables"]))
    return art


def validar_artefacto(art: Mapping) -> None:
    for clave in ("intercepto", "variables"):
        if clave not in art:
            raise ErrorArtefacto(f"Falta la clave '{clave}' en el artefacto")
    if not art["variables"]:
        raise ErrorArtefacto("El artefacto no tiene variables")
    for nombre, info in art["variables"].items():
        for clave in ("coeficiente", "bins", "woe_por_bin"):
            if clave not in info:
                raise ErrorArtefacto(f"Falta '{clave}' en la variable '{nombre}'")
        bins = info["bins"]
        n_bins = len(bins["bordes"]) - 1 if "bordes" in bins else len(bins.get("niveles", []))
        if n_bins <= 0:
            raise ErrorArtefacto(f"Bins vacíos en '{nombre}'")
        if len(info["woe_por_bin"]) != n_bins + 1:       # +1 por el bin NULO, siempre el último
            raise ErrorArtefacto(f"'{nombre}': {len(info['woe_por_bin'])} WoE para {n_bins} bins + NULO")


def posicion_bin(serie: pd.Series, bins: Mapping) -> np.ndarray:
    """Número de bin de cada valor (0..n-1). Nulos y niveles no vistos en train van al bin NULO (el último)."""
    if "bordes" in bins:
        codigos = pd.cut(pd.to_numeric(serie, errors="coerce"), bins["bordes"], include_lowest=True).cat.codes.to_numpy()
        n_bins = len(bins["bordes"]) - 1
    else:
        niveles = [str(n) for n in bins["niveles"]]
        texto = serie.astype("string")
        codigos = pd.Categorical(texto.where(texto.isin(niveles)), categories=niveles).codes
        n_bins = len(niveles)
        desconocidos = int((texto.notna() & (codigos == -1)).sum())
        if desconocidos:
            logger.warning("%s: %d niveles no vistos en entrenamiento; se asignan al bin NULO", serie.name, desconocidos)
    return np.where(codigos == -1, n_bins, codigos)


def calcular_pd(features: pd.DataFrame, art: Mapping) -> pd.Series:
    """PD de cada fila de ``features`` (salida de ``preprocesar``)."""
    faltantes = [v for v in art["variables"] if v not in features.columns]
    if faltantes:
        raise ErrorValidacionDatos(f"Faltan variables del modelo en los datos: {faltantes}")
    z = np.full(len(features), float(art["intercepto"]))
    for nombre, info in art["variables"].items():
        posicion = posicion_bin(features[nombre], info["bins"])
        z += info["coeficiente"] * np.asarray(info["woe_por_bin"], dtype=float)[posicion]
    return pd.Series(1.0 / (1.0 + np.exp(-z)), index=features.index, name="pd")


def puntuar(df_crudo: pd.DataFrame, ruta_artefacto: str | Path, config: ConfigPreprocesamiento | None = None) -> pd.Series:
    """Camino completo: datos crudos -> preprocesamiento -> PD (indexada por ``application_id``)."""
    art = cargar_artefacto(ruta_artefacto)
    limites = cargar_limites_winsor(ruta_artefacto)
    resultado = preprocesar(df_crudo, config=config, limites_winsor=limites)
    pd_ = calcular_pd(resultado.datos, art)
    pd_.index = resultado.datos[(config or ConfigPreprocesamiento()).id_col].to_numpy()
    logger.info("Scoring terminado: %d clientes, PD media %.4f", len(pd_), pd_.mean())
    return pd_
