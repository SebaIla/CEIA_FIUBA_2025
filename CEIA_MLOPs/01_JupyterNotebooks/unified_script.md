Eres un ingeniero de machine learning que tiene como tarea entregar en un repositorio de github en donde se implemente en LOCAL un ciclo de desarrollo desde el modelo establecido en la jupyter notebook compartida hasta la generación final del artefacto del modelo entrenado. 

Ten en cuenta:

Para la creación del ambiente utiliza la distribución Anaconda
Para el registro de los modelos y los experimentos utiliza MLFLOW
Para la orquestación de procesos utiliza METAFLOW

Entrega:

1. Estructura de carpetas para poder armar en el repositorio.
2. Paso a paso de los archivos creados y el código implementado.
3. Resumen final que desarrolle una guía de como ejecutar el entregable.

# Collision Model Jupyter Notebook

## Ingesta de información y transformación de datos inicial

import requests as rq
import json
import pandas as pd
import numpy as np
import time


#---------------------------------------------------------
# TRAFIC COLLISION DATA FROM 2010 TO PRESENT (Los Angeles)
#---------------------------------------------------------

# 1) Ingesta de datos desde via API Endpoint

api_ep_la_collisions = 'https://data.lacity.org/resource/d5tf-ez2w.json'

LIMIT = 50000 # Maximum allowed per request (SODA 2.0)
MAX_RETRIES = 3
RETRY_DELAY = 10 # Seconds to wait between attempts

# Storage for the data
la_collisions_data = []
offset = 0
total_retrieved = 0
EXPECTED_TOTAL_ROWS = 621677

print("Starting data retrieval...")

while True:
    retries = 0
    success = False
    last_error_message = ""

    while retries < MAX_RETRIES and not success:
        try:
            params = {'$limit': LIMIT, '$offset': offset}
            response = rq.get(api_ep_la_collisions, params=params)

            if response.status_code == 200:
                chunk = response.json()
                # The condition 'if not chunk:' is redundant as len(chunk) < LIMIT will cover it
                # if the chunk is empty.

                la_collisions_data.extend(chunk)
                retrieved = len(chunk)
                total_retrieved += retrieved
                print(f"✅ Partially ingested {retrieved} rows (Total: {total_retrieved})")
                offset += retrieved
                success = True

            else:
                last_error_message = f"HTTP Error {response.status_code}: {response.text}"
                print(f"⚠️ {last_error_message}. Retrying...")
                retries += 1
                if retries < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        except Exception as e:
            last_error_message = f"Exception: {str(e)}"
            print(f"⚠️ {last_error_message}. Retrying...")
            retries += 1
            if retries < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    if not success:
        print(f"❌ Failed after {MAX_RETRIES} attempts. Stopping. Last error: {last_error_message}")
        break

    # If the retrieved chunk is less than the LIMIT, it means we've reached the end of the data.
    if len(chunk) < LIMIT:
        break

print(f"✅ Final ingestion completed: {len(la_collisions_data)} rows retrieved.")

# Added: Optional: Verify the total number of rows
if len(la_collisions_data) == EXPECTED_TOTAL_ROWS:
    print(f"🎉 Successfully retrieved all {EXPECTED_TOTAL_ROWS} expected rows.")
else:
    print(f"❗ Warning: Expected {EXPECTED_TOTAL_ROWS} rows, but retrieved {len(la_collisions_data)} rows.")

# Transformamos los datos en dataframe utilizando la funcion json_normalize

df_la_collisions = pd.json_normalize(la_collisions_data)

df_la_collisions.info()

df = df_la_collisions

# Mostrar las primeras filas del dataset
df.head()

# Eliminar columnas irrelevantes para modelado temporal
columnas_a_eliminar = [
    'dr_no', 'rpt_dist_no', 'crm_cd', 'crm_cd_desc',
    'cross_street', 'location_1.human_address'
]
# Eliminar columnas con nombres tipo @computed
columnas_a_eliminar += [col for col in df.columns if col.startswith(':@computed')]

# Aplicar eliminación
df_limpio = df.drop(columns=columnas_a_eliminar)

# Convertir columnas de fecha a datetime
df_limpio['date_rptd'] = pd.to_datetime(df_limpio['date_rptd'], errors='coerce')
df_limpio['date_occ'] = pd.to_datetime(df_limpio['date_occ'], errors='coerce')

# Revisar valores nulos
nulos = df_limpio.isnull().sum()

# Mostrar resumen del dataframe limpio y los nulos encontrados
df_limpio.info(), nulos

# Asegurarse de que vict_age sea numérico
df_limpio['vict_age'] = pd.to_numeric(df_limpio['vict_age'], errors='coerce')

# Imputar vict_age con la mediana
df_limpio['vict_age'] = df_limpio['vict_age'].fillna(df_limpio['vict_age'].median())

# Imputar categóricas con 'Unknown'
df_limpio['vict_sex'] = df_limpio['vict_sex'].fillna('Unknown')
df_limpio['vict_descent'] = df_limpio['vict_descent'].fillna('Unknown')
df_limpio['premis_desc'] = df_limpio['premis_desc'].fillna('Unknown')

# Imputar premis_cd con la moda
df_limpio['premis_cd'] = df_limpio['premis_cd'].fillna(df_limpio['premis_cd'].mode()[0])

# Eliminar mocodes si no se necesita
df_limpio = df_limpio.drop(columns=['mocodes'])

# Verificar que no haya nulos
nulos_post = df_limpio.isnull().sum()
print(nulos_post)


# Seleccionar columnas numéricas a revisar
columnas_numericas = ['time_occ', 'vict_age', 'location_1.latitude', 'location_1.longitude']

for col in columnas_numericas:
    df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')

# Aplicar detección de outliers usando IQR
outliers_info = {}

for col in columnas_numericas:
    Q1 = df_limpio[col].dropna().quantile(0.25)
    Q3 = df_limpio[col].dropna().quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df_limpio[(df_limpio[col] < limite_inferior) | (df_limpio[col] > limite_superior)]
    outliers_info[col] = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Limite inferior': limite_inferior,
        'Limite superior': limite_superior,
        'Cantidad outliers': len(outliers)
    }

pd.DataFrame(outliers_info).T

# Crear una copia para filtrar los outliers
df_filtrado = df_limpio.copy()

# Aplicar filtros por cada columna con outliers
for col in ['vict_age', 'location_1.latitude', 'location_1.longitude']:
    Q1 = df_filtrado[col].quantile(0.25)
    Q3 = df_filtrado[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtrado = df_filtrado[(df_filtrado[col] >= lower_bound) & (df_filtrado[col] <= upper_bound)]

df_filtrado.shape

## Preparación del dataset para entrenamiento de modelo de Random Forest

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = df_filtrado

df['date_occ'] = pd.to_datetime(df['date_occ'])

# Filtrar por Hollywood y agrupar por fecha
df = df[df['area_name'] == 'Hollywood']
daily = df.groupby('date_occ').size().reset_index(name='collision_count')

# Crear features
daily['dayofyear'] = daily['date_occ'].dt.dayofyear
daily['weekday'] = daily['date_occ'].dt.weekday
daily['year'] = daily['date_occ'].dt.year
daily['lag_1'] = daily['collision_count'].shift(1)
daily['lag_2'] = daily['collision_count'].shift(2)
daily['rolling_mean_3'] = daily['collision_count'].rolling(3).mean()

# Limpiar NaNs
daily = daily.dropna()

# Variables
X = daily[['dayofyear', 'weekday', 'year', 'lag_1', 'lag_2', 'rolling_mean_3']]
y = daily['collision_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)