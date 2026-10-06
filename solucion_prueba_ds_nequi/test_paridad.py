"""Prueba de paridad: el pipeline de producción reproduce las PD calculadas en el notebook.

Usa una muestra de 1.200 solicitudes CRUDAS (formato original de fechas) y las PD de referencia
generadas por el notebook con el modelo final."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring import puntuar  # noqa: E402

DATA = Path(__file__).parent / "data"


class TestParidadConNotebook(unittest.TestCase):
    def test_pd_identicas_al_notebook(self):
        crudo = pd.read_csv(DATA / "dataset_crudo_muestra.csv", dtype={"application_date": str})
        ref = pd.read_csv(DATA / "pd_referencia.csv").set_index("application_id")["pd_referencia"]
        obtenido = puntuar(crudo, DATA / "modelo_pd.json")
        self.assertEqual(len(obtenido), len(ref))
        diferencia = np.abs(obtenido.loc[ref.index].to_numpy() - ref.to_numpy()).max()
        self.assertLess(diferencia, 1e-9)

    def test_pd_en_rango_razonable(self):
        crudo = pd.read_csv(DATA / "dataset_crudo_muestra.csv", dtype={"application_date": str})
        pd_ = puntuar(crudo, DATA / "modelo_pd.json")
        self.assertTrue(((pd_ > 0) & (pd_ < 1)).all())
        self.assertTrue(0.05 < pd_.mean() < 0.20)


if __name__ == "__main__":
    unittest.main()
