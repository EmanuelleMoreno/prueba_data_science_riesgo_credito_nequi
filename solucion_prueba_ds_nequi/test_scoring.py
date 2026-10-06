"""Pruebas unitarias del scoring con un artefacto sintético (sin depender del modelo real)."""
import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocesamiento import ErrorValidacionDatos  # noqa: E402
from scoring import ErrorArtefacto, calcular_pd, posicion_bin, validar_artefacto  # noqa: E402

ARTEFACTO = {
    "intercepto": -2.0,
    "variables": {
        "x": {"coeficiente": -1.0, "bins": {"bordes": [-1e18, 10.0, 20.0, 1e18]}, "woe_por_bin": [-0.5, 0.0, 0.5, 0.25]},
        "flag": {"coeficiente": -1.0, "bins": {"niveles": ["0", "1"]}, "woe_por_bin": [0.4, -0.4, 0.0]},
    },
}


class TestPosicionBin(unittest.TestCase):
    def test_bordes_incluyen_limite_superior(self):
        pos = posicion_bin(pd.Series([5.0, 10.0, 10.5, 20.0, 99.0]), ARTEFACTO["variables"]["x"]["bins"])
        self.assertEqual(list(pos), [0, 0, 1, 1, 2])

    def test_nulo_va_al_ultimo_bin(self):
        pos = posicion_bin(pd.Series([np.nan]), ARTEFACTO["variables"]["x"]["bins"])
        self.assertEqual(list(pos), [3])

    def test_niveles_conocidos_y_no_vistos(self):
        pos = posicion_bin(pd.Series([0, 1, 7], name="flag"), ARTEFACTO["variables"]["flag"]["bins"])
        self.assertEqual(list(pos), [0, 1, 2])             # el 7 no existía en train -> NULO


class TestCalcularPD(unittest.TestCase):
    def test_pd_coincide_con_calculo_manual(self):
        df = pd.DataFrame({"x": [5.0, 15.0, np.nan], "flag": [0, 1, 1]}, index=[10, 11, 12])
        pd_ = calcular_pd(df, ARTEFACTO)
        z = [-2.0 + (-1.0) * (-0.5) + (-1.0) * 0.4,
             -2.0 + (-1.0) * 0.0 + (-1.0) * (-0.4),
             -2.0 + (-1.0) * 0.25 + (-1.0) * (-0.4)]
        esperado = [1 / (1 + math.exp(-v)) for v in z]
        np.testing.assert_allclose(pd_.to_numpy(), esperado, rtol=1e-12)
        self.assertEqual(list(pd_.index), [10, 11, 12])

    def test_pd_siempre_entre_0_y_1(self):
        df = pd.DataFrame({"x": np.linspace(-100, 100, 50), "flag": [0, 1] * 25})
        pd_ = calcular_pd(df, ARTEFACTO)
        self.assertTrue(((pd_ > 0) & (pd_ < 1)).all())

    def test_falta_una_variable_del_modelo(self):
        with self.assertRaises(ErrorValidacionDatos):
            calcular_pd(pd.DataFrame({"x": [1.0]}), ARTEFACTO)

    def test_orden_de_filas_no_cambia_el_resultado(self):
        df = pd.DataFrame({"x": [5.0, 15.0, 25.0], "flag": [0, 1, 0]})
        a = calcular_pd(df, ARTEFACTO)
        b = calcular_pd(df.iloc[::-1], ARTEFACTO).iloc[::-1]
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


class TestValidarArtefacto(unittest.TestCase):
    def test_artefacto_valido(self):
        validar_artefacto(ARTEFACTO)

    def test_clave_faltante(self):
        art = copy.deepcopy(ARTEFACTO)
        del art["intercepto"]
        with self.assertRaises(ErrorArtefacto):
            validar_artefacto(art)

    def test_woe_con_longitud_incorrecta(self):
        art = copy.deepcopy(ARTEFACTO)
        art["variables"]["x"]["woe_por_bin"] = [0.1, 0.2]
        with self.assertRaises(ErrorArtefacto):
            validar_artefacto(art)

    def test_sin_variables(self):
        with self.assertRaises(ErrorArtefacto):
            validar_artefacto({"intercepto": 0, "variables": {}})


if __name__ == "__main__":
    unittest.main()
