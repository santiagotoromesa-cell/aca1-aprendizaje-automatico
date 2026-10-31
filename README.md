# ACA 1 · Aprendizaje Automático (CUN)

**Estimación del valor de la vivienda en California con Machine Learning clásico y despliegue en Streamlit**

Integrantes: Santiago Toro Mesa y Darwin Lozano Lozano · Docente: Dr. Hamilton Rivera Flor
Especialización en Inteligencia Artificial · Corporación Unificada Nacional de Educación Superior (CUN)

| Archivo | Contenido |
|---|---|
| `ACA_1_Aprendizaje_Automatico.ipynb` | Notebook completo: EDA, limpieza, modelos, métricas y conclusiones |
| `app.py` | Interfaz interactiva en Streamlit |
| `procesamiento.py` | Funciones auxiliares de limpieza (las mismas del notebook) |
| `requirements.txt` | Librerías que instala Streamlit Community Cloud |
| `housing.csv` | Datos: California Housing Prices (Pace & Barry, 1997), versión de Kaggle |
| `configuracion_modelo.json` | Hiperparámetros elegidos y métricas de prueba exportados por el notebook |
| `predicciones_prueba.csv` | Valores reales y estimados del conjunto de prueba |

**Modelo final:** Random Forest ajustado (100 árboles, profundidad máxima 25, `max_features="sqrt"`).
**Resultados en prueba:** MAE ≈ USD 27.000 · RMSE ≈ USD 41.900 · R² ≈ 0,82.

Ejecutar localmente: `pip install -r requirements.txt` y luego `streamlit run app.py`.
