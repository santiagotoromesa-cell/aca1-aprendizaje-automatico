"""
Funciones auxiliares del proyecto ACA 1 (Aprendizaje Automático, CUN).
Aplican exactamente las mismas reglas de limpieza y las mismas variables nuevas del notebook,
para que la app de Streamlit trabaje sobre los mismos datos con los que se evaluó el modelo.
"""
import numpy as np
import pandas as pd

# Coordenadas de las dos ciudades grandes de California
LAT_LA, LON_LA = 34.05, -118.24   # Los Ángeles
LAT_SF, LON_SF = 37.77, -122.42   # San Francisco

VARIABLES_NUMERICAS = ["longitude", "latitude", "housing_median_age", "median_income",
                       "cuartos_por_hogar", "personas_por_hogar", "dormitorios_por_cuarto", "km_a_ciudad"]
VARIABLES_TEXTO = ["ocean_proximity"]


def distancia_km(lat, lon, lat_ciudad, lon_ciudad):
    """Distancia aproximada en km (1 grado de latitud ~ 111 km)."""
    lado_vertical = (lat - lat_ciudad) * 111
    lado_horizontal = (lon - lon_ciudad) * 111 * np.cos(np.radians(lat))
    return np.sqrt(lado_vertical ** 2 + lado_horizontal ** 2)


def km_a_ciudad(lat, lon):
    """Kilómetros a la ciudad grande más cercana (Los Ángeles o San Francisco)."""
    return np.minimum(distancia_km(lat, lon, LAT_LA, LON_LA),
                      distancia_km(lat, lon, LAT_SF, LON_SF))


def limpiar(datos):
    """Reglas de limpieza del Paso 2 del notebook. Los vacíos se rellenan dentro del modelo."""
    datos = datos.copy()
    datos = datos[datos["ocean_proximity"] != "ISLAND"]          # 5 barrios de isla
    datos = datos[datos["median_house_value"] < 500001]          # precio cortado en el tope
    datos["cuartos_por_hogar"] = datos["total_rooms"] / datos["households"]
    datos["personas_por_hogar"] = datos["population"] / datos["households"]
    datos["dormitorios_por_cuarto"] = datos["total_bedrooms"] / datos["total_rooms"]
    datos = datos[~((datos["cuartos_por_hogar"] > 20) | (datos["personas_por_hogar"] > 10))]  # imposibles
    datos["km_a_ciudad"] = km_a_ciudad(datos["latitude"], datos["longitude"])
    return datos
