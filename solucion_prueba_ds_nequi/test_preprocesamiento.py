"""Pruebas unitarias del preprocesamiento. Se ejecutan con ``python -m unittest discover -s tests``
o con ``pytest tests`` (usan solo la librería estándar)."""
import logging
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocesamiento import (ConfigPreprocesamiento, ErrorValidacionDatos, a_booleano, cargar_limites_winsor,  # noqa: E402
                              construir_variables, convertir_numericas, estandarizar_fecha, preprocesar,
                              reportar_rangos, validar_esquema, winsorizar)

CONFIG = ConfigPreprocesamiento()


def crudo(n: int = 4, **cambios) -> pd.DataFrame:
    """DataFrame crudo mínimo y válido (como llega de ``dataset_candidato.csv``)."""
    df = pd.DataFrame({
        "application_id": [f"CAND-{i:05d}" for i in range(n)],
        "application_date": ["1/10/24", "01/10/24", "15/11/24", "31/12/24"][:n] + ["01/01/24"] * max(0, n - 4),
        "avg_monthly_topups": [10000, 50000, 120000, 400000][:n] + [30000] * max(0, n - 4),
        "has_bureau_record": [True, False, "True", "False"][:n] + [True] * max(0, n - 4),
        "utility_payment_ontime_rate": [0.8, 0.6, np.nan, np.nan][:n] + [0.7] * max(0, n - 4),
        "telco_payment_ontime_rate": [0.9, np.nan, 0.5, np.nan][:n] + [0.7] * max(0, n - 4),
        "monthly_income_declared": ["1000000", "abc", "2000000", "500000"][:n] + ["900000"] * max(0, n - 4),
        "requested_amount": [500000, 1000000, 200000, 0][:n] + [300000] * max(0, n - 4),
        "age": [25, 40, 17, 66][:n] + [30] * max(0, n - 4),
        "device_price_tier": [1, 3, 5, 9][:n] + [2] * max(0, n - 4),
        "sexo": [" H", "M ", "NB", "H"][:n] + ["M"] * max(0, n - 4),
    })
    for col, val in cambios.items():
        df[col] = val
    return df


class TestFechas(unittest.TestCase):
    def test_formatos_mixtos_se_estandarizan(self):
        s = pd.Series(["1/10/24", "01/10/24", "2024-10-01", "31/12/24"], name="application_date")
        r = estandarizar_fecha(s, CONFIG.formatos_fecha)
        self.assertEqual(list(r.dt.strftime("%Y-%m-%d")), ["2024-10-01", "2024-10-01", "2024-10-01", "2024-12-31"])

    def test_fecha_invalida_modo_estricto_reporta_ids(self):
        s = pd.Series(["01/10/24", "no-es-fecha"], name="application_date")
        ids = pd.Series(["A1", "B2"])
        with self.assertRaises(ErrorValidacionDatos) as ctx:
            estandarizar_fecha(s, CONFIG.formatos_fecha, ids)
        self.assertIn("B2", str(ctx.exception))

    def test_fecha_invalida_modo_no_estricto_queda_nat_y_loguea(self):
        s = pd.Series(["01/10/24", "32/13/24"], name="application_date")
        with self.assertLogs("preprocesamiento", level=logging.WARNING):
            r = estandarizar_fecha(s, CONFIG.formatos_fecha, estricto=False)
        self.assertTrue(r.iloc[1] is pd.NaT or pd.isna(r.iloc[1]))
        self.assertFalse(pd.isna(r.iloc[0]))

    def test_columna_ya_datetime_se_respeta(self):
        s = pd.Series(pd.to_datetime(["2024-10-01 13:45"]))
        self.assertEqual(estandarizar_fecha(s, CONFIG.formatos_fecha).iloc[0], pd.Timestamp("2024-10-01"))


class TestTipos(unittest.TestCase):
    def test_numericas_no_numericas_pasan_a_nan_y_se_cuentan(self):
        df = pd.DataFrame({"x": ["1", "abc", None, "3.5"]})
        out, corruptos = convertir_numericas(df, ["x"])
        self.assertEqual(corruptos, {"x": 1})            # None no cuenta como corrupto
        self.assertEqual(out["x"].isna().sum(), 2)
        self.assertEqual(out["x"].dtype, "float64")

    def test_booleanos_aceptan_varias_representaciones(self):
        r = a_booleano(pd.Series(["True", "false", "1", "0", "Si", "no"], name="b"))
        self.assertEqual(list(r), [True, False, True, False, True, False])

    def test_booleano_nulo_o_desconocido_lanza_error(self):
        with self.assertRaises(ErrorValidacionDatos):
            a_booleano(pd.Series([True, None], name="has_bureau_record"))
        with self.assertRaises(ErrorValidacionDatos):
            a_booleano(pd.Series(["quizas"], name="has_bureau_record"))

    def test_ingreso_corrupto_genera_bandera(self):
        r = preprocesar(crudo()).datos
        self.assertEqual(list(r["ingreso_corrupto"]), [False, True, False, False])
        self.assertTrue(np.isnan(r.loc[1, "monthly_income_declared"]))

    def test_ordinal_fuera_de_niveles_pasa_a_nan(self):
        r = preprocesar(crudo()).datos
        self.assertTrue(pd.isna(r.loc[3, "device_price_tier"]))      # el 9 no existe
        self.assertTrue(r["device_price_tier"].cat.ordered)

    def test_categoricas_se_limpian_espacios(self):
        r = preprocesar(crudo()).datos
        self.assertEqual(list(r["sexo"].astype(str)), ["H", "M", "NB", "H"])


class TestRangosYWinsor(unittest.TestCase):
    def test_rangos_reporta_pero_no_corrige(self):
        df = pd.DataFrame({"age": [17.0, 30.0, 66.0]})
        fuera = reportar_rangos(df, {"age": (18, 65)})
        self.assertEqual(fuera, {"age": 2})
        self.assertEqual(list(df["age"]), [17.0, 30.0, 66.0])

    def test_winsor_recorta_solo_cola_alta(self):
        df = pd.DataFrame({"x": [1.0, 50.0, 1000.0]})
        out, n = winsorizar(df, {"x": 100.0})
        self.assertEqual(list(out["x"]), [1.0, 50.0, 100.0])        # el mínimo no se toca
        self.assertEqual(n, {"x": 1})

    def test_winsor_sin_limites_no_cambia_nada(self):
        df = pd.DataFrame({"x": [1.0, 1e9]})
        out, n = winsorizar(df, None)
        pd.testing.assert_frame_equal(out, df)
        self.assertEqual(n, {})

    def test_winsor_ignora_columnas_ausentes_y_rechaza_limite_invalido(self):
        df = pd.DataFrame({"x": [1.0]})
        out, n = winsorizar(df, {"no_existe": 5.0})
        self.assertEqual(n, {})
        with self.assertRaises(ErrorValidacionDatos):
            winsorizar(df, {"x": float("nan")})

    def test_winsor_no_modifica_la_entrada(self):
        df = pd.DataFrame({"x": [1.0, 1000.0]})
        winsorizar(df, {"x": 100.0})
        self.assertEqual(df["x"].max(), 1000.0)

    def test_cargar_limites_desde_artefacto(self):
        ruta = Path(__file__).parent / "data" / "modelo_pd.json"
        limites = cargar_limites_winsor(ruta)
        self.assertEqual(set(limites), {"avg_monthly_topups"})
        self.assertGreater(limites["avg_monthly_topups"], 0)


class TestVariablesConstruidas(unittest.TestCase):
    def test_pago_prom_promedia_y_tolera_un_nulo(self):
        df = pd.DataFrame({"utility_payment_ontime_rate": [0.8, 0.6, np.nan, np.nan],
                           "telco_payment_ontime_rate": [0.6, np.nan, 0.5, np.nan]})
        r = construir_variables(df)["pago_prom"]
        self.assertAlmostEqual(r[0], 0.7)
        self.assertAlmostEqual(r[1], 0.6)
        self.assertAlmostEqual(r[2], 0.5)
        self.assertTrue(np.isnan(r[3]))

    def test_division_por_cero_devuelve_nan_no_infinito(self):
        df = pd.DataFrame({"utility_payment_ontime_rate": [0.5], "telco_payment_ontime_rate": [0.5],
                           "avg_monthly_topups": [100.0], "requested_amount": [0.0],
                           "monthly_income_declared": [0.0]})
        r = construir_variables(df)
        self.assertTrue(np.isnan(r.loc[0, "topups_sobre_monto"]))
        self.assertTrue(np.isnan(r.loc[0, "topups_sobre_ingreso"]))

    def test_p2p_solo_con_transacciones(self):
        df = pd.DataFrame({"utility_payment_ontime_rate": [0.5, 0.5], "telco_payment_ontime_rate": [0.5, 0.5],
                           "pct_p2p_transfers": [0.4, 0.4], "num_transactions_last_90d": [0, 5]})
        r = construir_variables(df)
        self.assertTrue(np.isnan(r.loc[0, "p2p_con_trx"]))
        self.assertEqual(r.loc[1, "p2p_con_trx"], 0.4)
        self.assertEqual(list(r["sin_transacciones"]), [1, 0])


class TestEsquema(unittest.TestCase):
    def test_columnas_faltantes(self):
        with self.assertRaises(ErrorValidacionDatos) as ctx:
            validar_esquema(crudo().drop(columns=["has_bureau_record"]), CONFIG)
        self.assertIn("has_bureau_record", str(ctx.exception))

    def test_ids_duplicados(self):
        df = crudo()
        df.loc[1, "application_id"] = df.loc[0, "application_id"]
        with self.assertRaises(ErrorValidacionDatos):
            validar_esquema(df, CONFIG)

    def test_vacio_y_tipo_incorrecto(self):
        with self.assertRaises(ErrorValidacionDatos):
            validar_esquema(crudo().iloc[0:0], CONFIG)
        with self.assertRaises(ErrorValidacionDatos):
            validar_esquema([1, 2, 3], CONFIG)          # type: ignore[arg-type]


class TestPipeline(unittest.TestCase):
    def test_no_modifica_la_entrada(self):
        df = crudo()
        copia = df.copy(deep=True)
        preprocesar(df, limites_winsor={"avg_monthly_topups": 100000.0})
        pd.testing.assert_frame_equal(df, copia)

    def test_es_determinista(self):
        a = preprocesar(crudo(), limites_winsor={"avg_monthly_topups": 100000.0}).datos
        b = preprocesar(crudo(), limites_winsor={"avg_monthly_topups": 100000.0}).datos
        pd.testing.assert_frame_equal(a, b)

    def test_salida_lista_para_el_modelo(self):
        r = preprocesar(crudo(), limites_winsor={"avg_monthly_topups": 100000.0})
        d = r.datos
        self.assertEqual(list(d["has_bureau_record"]), [1, 0, 1, 0])
        self.assertEqual(d["has_bureau_record"].dtype.kind, "i")
        self.assertEqual(d["avg_monthly_topups"].max(), 100000.0)        # 400000 recortado
        self.assertEqual(d["avg_monthly_topups"].min(), 10000.0)         # cola baja intacta
        self.assertIn("pago_prom", d.columns)
        self.assertEqual(d["application_date"].dtype.kind, "M")
        self.assertEqual(r.reporte["winsorizados"], {"avg_monthly_topups": 2})
        self.assertEqual(r.reporte["ingreso_corrupto"], 1)

    def test_sin_limites_advierte(self):
        with self.assertLogs("preprocesamiento", level=logging.WARNING) as cm:
            preprocesar(crudo())
        self.assertTrue(any("winsorización" in m for m in cm.output))

    def test_reporte_cuenta_nulos_y_fuera_de_rango(self):
        r = preprocesar(crudo(), limites_winsor={}).reporte
        self.assertEqual(r["fuera_de_rango"], {"age": 2})                 # 17 y 66
        self.assertEqual(r["nulos"]["pago_prom"], 1)

    def test_modo_estricto_rechaza_numerica_corrupta_distinta_de_ingreso(self):
        df = crudo()
        df["avg_monthly_topups"] = df["avg_monthly_topups"].astype(object)
        df.loc[0, "avg_monthly_topups"] = "abc"
        with self.assertRaises(ErrorValidacionDatos):
            preprocesar(df)
        r = preprocesar(df, config=replace(CONFIG, modo_estricto=False))
        self.assertTrue(np.isnan(r.datos.loc[0, "avg_monthly_topups"]))

    def test_registra_logs_informativos_sin_valores_de_clientes(self):
        with self.assertLogs("preprocesamiento", level=logging.INFO) as cm:
            preprocesar(crudo(), limites_winsor={"avg_monthly_topups": 100000.0})
        texto = " ".join(cm.output)
        self.assertIn("Preprocesamiento", texto)
        self.assertNotIn("CAND-", texto)                                   # los IDs no se loguean en el camino feliz


if __name__ == "__main__":
    unittest.main()
