"""
ACA 1 - Aprendizaje Automático (CUN) · Santiago Toro Mesa y Darwin Lozano Lozano
App de Streamlit: estimación del valor de la vivienda típica de un barrio de California
con el Random Forest ajustado en el notebook del proyecto.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from procesamiento import VARIABLES_NUMERICAS, VARIABLES_TEXTO, km_a_ciudad, limpiar

st.set_page_config(page_title="Valor de vivienda · California", page_icon="🏠", layout="wide")

CERCANIA = {  # etiqueta en español -> código del dataset
    "Interior (lejos del mar)": "INLAND",
    "A menos de 1 hora del mar": "<1H OCEAN",
    "Cerca del mar": "NEAR OCEAN",
    "Cerca de la bahía (San Francisco)": "NEAR BAY",
}
CIUDADES = {  # punto de partida para ubicar el barrio
    "Los Ángeles": (34.05, -118.24),
    "San Francisco": (37.77, -122.42),
    "San Diego": (32.72, -117.16),
    "San José": (37.34, -121.89),
    "Oakland": (37.80, -122.27),
    "Sacramento": (38.58, -121.49),
    "Fresno": (36.74, -119.79),
    "Bakersfield": (35.37, -119.02),
    "Santa Bárbara": (34.42, -119.70),
}


# ---------------------------------------------------------------- datos y modelos
@st.cache_data
def cargar_datos():
    datos = limpiar(pd.read_csv("housing.csv"))
    with open("configuracion_modelo.json", encoding="utf-8") as archivo:
        configuracion = json.load(archivo)
    predicciones = pd.read_csv("predicciones_prueba.csv")
    return datos, configuracion, predicciones


@st.cache_resource
def entrenar_modelos(datos, parametros, k):
    """Entrena el Random Forest con los hiperparámetros del notebook y la misma partición (random_state=42)."""
    X = datos[VARIABLES_NUMERICAS + VARIABLES_TEXTO]
    y = datos["median_house_value"]
    X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.20, random_state=42)

    preparacion = ColumnTransformer([
        ("numericas", SimpleImputer(strategy="median"), VARIABLES_NUMERICAS),
        ("texto", OneHotEncoder(handle_unknown="ignore"), VARIABLES_TEXTO),
    ])
    modelo = Pipeline([("preparacion", preparacion),
                       ("modelo", RandomForestRegressor(**parametros, random_state=42, n_jobs=-1))])
    modelo.fit(X_ent, y_ent)
    estimado = modelo.predict(X_pru)
    control = {"MAE": mean_absolute_error(y_pru, estimado), "R2": r2_score(y_pru, estimado)}

    # Zonas K-Means (ubicación + ingreso, escalados)
    escalador = StandardScaler()
    Z = escalador.fit_transform(datos[["latitude", "longitude", "median_income"]])
    zonas = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    return modelo, control, escalador, zonas


def nombrar_zonas(datos, etiquetas):
    resumen = datos.assign(zona=etiquetas).groupby("zona").agg(
        barrios=("median_house_value", "size"),
        ingreso=("median_income", "mean"),
        valor=("median_house_value", "mean"),
        latitud=("latitude", "mean"))
    resumen["nombre"] = [
        ("Norte (San Francisco)" if f.latitud > 36 else "Sur (Los Ángeles)")
        + (" – ingreso alto" if f.ingreso > 4 else " – ingreso medio-bajo")
        for f in resumen.itertuples()]
    return resumen


datos, configuracion, predicciones_prueba = cargar_datos()
with st.spinner("Preparando el modelo (solo la primera vez, unos segundos)..."):
    modelo, control, escalador_zonas, modelo_zonas = entrenar_modelos(
        datos, configuracion["parametros_random_forest"], configuracion["numero_zonas"])
resumen_zonas = nombrar_zonas(datos, modelo_zonas.labels_)
metricas = configuracion["metricas_prueba"]["Random Forest ajustado"]

# ---------------------------------------------------------------- barra lateral
st.sidebar.header("Datos del barrio")
ciudad = st.sidebar.selectbox("Ciudad de referencia", list(CIUDADES))
lat0, lon0 = CIUDADES[ciudad]
latitud = st.sidebar.slider("Latitud (ajuste fino)", 32.5, 42.0, lat0, 0.01)
longitud = st.sidebar.slider("Longitud (ajuste fino)", -124.4, -114.3, lon0, 0.01)
cercania = st.sidebar.selectbox("Cercanía al mar", list(CERCANIA), index=1)
ingreso_usd = st.sidebar.slider("Ingreso anual del hogar típico (USD)", 5000, 150000, 50000, 1000)
antiguedad = st.sidebar.slider("Antigüedad de la vivienda típica (años)", 1, 52, 30)
cuartos = st.sidebar.slider("Cuartos por vivienda (todos los espacios)", 1.0, 15.0, 5.0, 0.1)
dormitorios = st.sidebar.slider("Dormitorios por vivienda", 0.5, 6.0, 1.0, 0.1)
personas = st.sidebar.slider("Personas por vivienda", 1.0, 10.0, 2.8, 0.1)

if dormitorios > cuartos:
    st.sidebar.warning("Los dormitorios no pueden ser más que los cuartos; se igualan.")
    dormitorios = cuartos

barrio = pd.DataFrame([{
    "longitude": longitud, "latitude": latitud, "housing_median_age": antiguedad,
    "median_income": ingreso_usd / 10000, "cuartos_por_hogar": cuartos,
    "personas_por_hogar": personas, "dormitorios_por_cuarto": dormitorios / cuartos,
    "km_a_ciudad": float(km_a_ciudad(latitud, longitud)), "ocean_proximity": CERCANIA[cercania],
}])
valor = float(modelo.predict(barrio)[0])
zona_barrio = int(modelo_zonas.predict(escalador_zonas.transform(
    barrio[["latitude", "longitude", "median_income"]]))[0])

# ---------------------------------------------------------------- encabezado
st.title("🏠 ¿Cuánto vale la vivienda típica de un barrio en California?")
st.caption("ACA 1 · Aprendizaje Automático · CUN — Santiago Toro Mesa y Darwin Lozano Lozano · "
           "Pilar: Ventas (analítica comercial inmobiliaria)")

pestana_1, pestana_2, pestana_3, pestana_4 = st.tabs(
    ["Estimar valor", "Explorar los datos", "Desempeño del modelo", "Zonas (K-Means)"])

# ---------------------------------------------------------------- 1. estimación
with pestana_1:
    st.write("Ajuste los datos del barrio en el panel de la izquierda. El modelo responde al instante.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Valor estimado de la vivienda", f"USD {valor:,.0f}")
    c2.metric("Rango probable (± error promedio)",
              f"USD {max(valor - metricas['MAE'], 0):,.0f} – {valor + metricas['MAE']:,.0f}")
    c3.metric("Distancia a LA o SF", f"{barrio['km_a_ciudad'][0]:,.0f} km")
    st.info(f"**Zona K-Means del barrio:** {resumen_zonas.loc[zona_barrio, 'nombre']} "
            f"(valor promedio de la zona: USD {resumen_zonas.loc[zona_barrio, 'valor']:,.0f}).")
    if valor > 450000:
        st.warning("El estimado está cerca de USD 500.000, el límite del modelo: el valor real podría ser mayor.")

    columna_mapa, _ = st.columns([3, 2])
    fig, eje = plt.subplots(figsize=(6, 5.5))
    eje.scatter(datos["longitude"], datos["latitude"], c=datos["median_house_value"],
                cmap="viridis", s=2, alpha=0.4)
    eje.scatter([longitud], [latitud], color="red", s=180, marker="*", edgecolor="black", label="Barrio ingresado")
    eje.set_xlabel("Longitud")
    eje.set_ylabel("Latitud")
    eje.set_title("Ubicación del barrio (color = valor de la vivienda)")
    eje.legend(loc="upper right")
    columna_mapa.pyplot(fig)

    with st.expander("¿Cómo se calcula?"):
        st.write(
            "Un **Random Forest** de 100 árboles de decisión, entrenado con "
            f"{configuracion['barrios_entrenamiento']:,} barrios del censo de California de 1990. "
            "Cada árbol hace preguntas sobre los datos del barrio (ingreso, ubicación, cercanía al mar...) "
            "y el resultado es el promedio de las 100 respuestas. Valores en dólares de 1990; "
            "el modelo es válido para viviendas de hasta USD 500.000.")

# ---------------------------------------------------------------- 2. exploración
with pestana_2:
    st.subheader("Datos depurados usados por el modelo")
    filtro = st.multiselect("Filtrar por cercanía al mar", list(CERCANIA), default=list(CERCANIA))
    rango = st.slider("Ingreso anual del hogar (USD)", 0, 160000, (0, 160000), 5000)
    vista = datos[datos["ocean_proximity"].isin([CERCANIA[f] for f in filtro])
                  & datos["median_income"].between(rango[0] / 10000, rango[1] / 10000)]

    d1, d2, d3 = st.columns(3)
    d1.metric("Barrios seleccionados", f"{len(vista):,}")
    d2.metric("Valor promedio", f"USD {vista['median_house_value'].mean():,.0f}" if len(vista) else "—")
    d3.metric("Ingreso promedio", f"USD {vista['median_income'].mean() * 10000:,.0f}" if len(vista) else "—")

    if len(vista):
        g1, g2 = st.columns(2)
        with g1:
            fig, eje = plt.subplots(figsize=(6, 5))
            puntos = eje.scatter(vista["longitude"], vista["latitude"], c=vista["median_house_value"],
                                 cmap="viridis", s=3, alpha=0.5)
            fig.colorbar(puntos, label="Valor (USD)")
            eje.set_title("Mapa del valor de la vivienda")
            eje.set_xlabel("Longitud")
            eje.set_ylabel("Latitud")
            st.pyplot(fig)
        with g2:
            fig, eje = plt.subplots(figsize=(6, 5))
            eje.scatter(vista["median_income"] * 10000, vista["median_house_value"], s=3, alpha=0.3)
            eje.set_title("Ingreso del hogar contra valor de la vivienda")
            eje.set_xlabel("Ingreso anual (USD)")
            eje.set_ylabel("Valor de la vivienda (USD)")
            st.pyplot(fig)
        with st.expander("Ver tabla de datos"):
            st.dataframe(vista.drop(columns=["total_rooms", "total_bedrooms", "population", "households"]))

# ---------------------------------------------------------------- 3. desempeño
with pestana_3:
    st.subheader("Calificación con barrios que el modelo nunca vio")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE: error promedio", f"USD {metricas['MAE']:,.0f}")
    m2.metric("RMSE", f"USD {metricas['RMSE']:,.0f}")
    m3.metric("R²: variación explicada", f"{metricas['R2'] * 100:.1f} %")

    tabla = pd.DataFrame(configuracion["metricas_prueba"]).T
    tabla.index.name = "Modelo"
    st.dataframe(tabla.style.format({"MAE": "USD {:,.0f}", "RMSE": "USD {:,.0f}", "R2": "{:.4f}"}))
    st.caption(f"Control al iniciar la app (mismo modelo, misma partición): "
               f"MAE USD {control['MAE']:,.0f} · R² {control['R2']:.4f}")

    g1, g2 = st.columns(2)
    with g1:
        fig, eje = plt.subplots(figsize=(6, 5))
        eje.scatter(predicciones_prueba["real"], predicciones_prueba["estimado"], s=4, alpha=0.25)
        eje.plot([0, 500000], [0, 500000], "r--", label="Acierto perfecto")
        eje.set_xlabel("Valor real (USD)")
        eje.set_ylabel("Valor estimado (USD)")
        eje.set_title("Reales contra estimados")
        eje.legend()
        st.pyplot(fig)
    with g2:
        nombres = [n.split("__")[1] for n in modelo.named_steps["preparacion"].get_feature_names_out()]
        importancia = pd.Series(modelo.named_steps["modelo"].feature_importances_, index=nombres).sort_values()
        fig, eje = plt.subplots(figsize=(6, 5))
        eje.barh(importancia.index, importancia.values * 100)
        eje.set_xlabel("Importancia (%)")
        eje.set_title("Variables en las que se apoya el modelo")
        st.pyplot(fig)

    st.markdown(
        "**Cómo leerlo:** el modelo explica cerca del 82 % de las diferencias de precio entre barrios y se equivoca, "
        "en promedio, unos USD 27.000. Si se respondiera siempre el precio promedio, el error sería de USD 79.300. "
        "**Limitaciones:** datos de 1990, válido hasta USD 500.000 y con más error en los barrios muy baratos o muy caros.")

# ---------------------------------------------------------------- 4. zonas
with pestana_4:
    st.subheader("Zonas de barrios parecidos (aprendizaje no supervisado)")
    st.write("K-Means agrupó los barrios por **ubicación e ingreso**, sin conocer el precio. "
             "Después se calculó el valor promedio de cada zona.")
    fig, eje = plt.subplots(figsize=(7, 6))
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for zona, fila in resumen_zonas.iterrows():
        parte = datos[modelo_zonas.labels_ == zona]
        eje.scatter(parte["longitude"], parte["latitude"], s=3, alpha=0.5,
                    color=colores[zona % len(colores)], label=fila["nombre"])
    eje.scatter([longitud], [latitud], color="black", s=180, marker="*", label="Barrio ingresado")
    leyenda = eje.legend(loc="upper right", fontsize=8)
    for marcador in getattr(leyenda, "legend_handles", getattr(leyenda, "legendHandles", [])):
        marcador.set_sizes([40])
    eje.set_xlabel("Longitud")
    eje.set_ylabel("Latitud")
    columna_zonas, _ = st.columns([3, 2])
    columna_zonas.pyplot(fig)

    tabla_zonas = resumen_zonas.set_index("nombre")[["barrios", "ingreso", "valor"]].rename(columns={
        "barrios": "Barrios", "ingreso": "Ingreso promedio (USD)", "valor": "Valor promedio (USD)"})
    tabla_zonas["Ingreso promedio (USD)"] *= 10000
    st.dataframe(tabla_zonas.style.format("{:,.0f}"))

st.divider()
st.caption("Datos: Pace, R. K., & Barry, R. (1997). California housing prices [Conjunto de datos]. Kaggle. "
           "Modelo: scikit-learn (Pedregosa et al., 2011). Interfaz: Streamlit.")
