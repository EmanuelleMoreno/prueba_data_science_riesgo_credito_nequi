# Pipeline de PD: preprocesamiento + scoring

```
datos crudos ──► preprocesamiento.preprocesar ──► scoring.calcular_pd (modelo_pd.json) ──► PD
```

| Archivo | Qué hace |
|---|---|
| `preprocesamiento.py` | Validación de esquema, fechas, tipos, rangos, winsorización (límites fijos de train), variables construidas, logs |
| `scoring.py` | Lee `modelo_pd.json` (bins, WoE, coeficientes) y calcula la PD. `puntuar(df_crudo, ruta_json)` hace todo el camino |
| `tests/` | Pruebas unitarias y una prueba de paridad contra las PD del notebook |

```python
from preprocesamiento import configurar_logging
from scoring import puntuar

configurar_logging(archivo="pipeline_pd.log")
pd_clientes = puntuar(df_crudo, "artefactos/modelo_pd.json")   # Serie indexada por application_id
```

Pruebas (solo librería estándar; también corren con `pytest tests`):

```
python -m unittest discover -s tests -v
```

Dependencias: `pandas>=2.0`, `numpy`. Los límites de winsorización, bins, WoE e intercepto viven en el JSON del
modelo: **nunca se recalculan con datos nuevos**. Si se reentrena, se genera un JSON nuevo y se versiona junto al código.
