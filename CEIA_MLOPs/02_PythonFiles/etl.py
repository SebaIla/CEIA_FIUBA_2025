import requests as rq
import pandas as pd
import numpy as np
import time
import os
import json

def extract_data():
    api_ep_la_collisions = 'https://data.lacity.org/resource/d5tf-ez2w.json'
    LIMIT = 50000
    MAX_RETRIES = 3
    RETRY_DELAY = 10
    la_collisions_data = []
    offset = 0
    total_retrieved = 0

    print("Starting data retrieval...")
    while True:
        retries = 0
        success = False
        while retries < MAX_RETRIES and not success:
            try:
                params = {'$limit': LIMIT, '$offset': offset}
                response = rq.get(api_ep_la_collisions, params=params)
                if response.status_code == 200:
                    chunk = response.json()
                    la_collisions_data.extend(chunk)
                    retrieved = len(chunk)
                    total_retrieved += retrieved
                    print(f"✅ Partially ingested {retrieved} rows (Total: {total_retrieved})")
                    offset += retrieved
                    success = True
                else:
                    print(f"⚠️ HTTP Error {response.status_code}. Retrying...")
                    retries += 1
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                print(f"⚠️ Exception: {str(e)}. Retrying...")
                retries += 1
                time.sleep(RETRY_DELAY)
        
        if not success:
            print(f"❌ Failed after {MAX_RETRIES} attempts. Stopping.")
            break
        if len(chunk) < LIMIT:
            break
    
    print(f"✅ Final ingestion completed: {len(la_collisions_data)} rows retrieved.")
    return la_collisions_data

def transform_data(la_collisions_data):
    df = pd.json_normalize(la_collisions_data)
    
    # Clean data
    columnas_a_eliminar = [
        'dr_no', 'rpt_dist_no', 'crm_cd', 'crm_cd_desc',
        'cross_street', 'location_1.human_address'
    ]
    columnas_a_eliminar += [col for col in df.columns if col.startswith(':@computed')]
    df_limpio = df.drop(columns=columnas_a_eliminar)
    
    # Convert dates
    df_limpio['date_rptd'] = pd.to_datetime(df_limpio['date_rptd'], errors='coerce')
    df_limpio['date_occ'] = pd.to_datetime(df_limpio['date_occ'], errors='coerce')
    
    # Handle missing values
    df_limpio['vict_age'] = pd.to_numeric(df_limpio['vict_age'], errors='coerce')
    df_limpio['vict_age'] = df_limpio['vict_age'].fillna(df_limpio['vict_age'].median())
    df_limpio['vict_sex'] = df_limpio['vict_sex'].fillna('Unknown')
    df_limpio['vict_descent'] = df_limpio['vict_descent'].fillna('Unknown')
    df_limpio['premis_desc'] = df_limpio['premis_desc'].fillna('Unknown')
    df_limpio['premis_cd'] = df_limpio['premis_cd'].fillna(df_limpio['premis_cd'].mode()[0])
    df_limpio = df_limpio.drop(columns=['mocodes'])
    
    # Handle outliers
    columnas_numericas = ['time_occ', 'vict_age', 'location_1.latitude', 'location_1.longitude']
    for col in columnas_numericas:
        df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')
    
    df_filtrado = df_limpio.copy()
    for col in ['vict_age', 'location_1.latitude', 'location_1.longitude']:
        Q1 = df_filtrado[col].quantile(0.25)
        Q3 = df_filtrado[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_filtrado = df_filtrado[(df_filtrado[col] >= lower_bound) & 
                                 (df_filtrado[col] <= upper_bound)]
    
    # Prepare time-series data
    df_filtrado = df_filtrado[df_filtrado['area_name'] == 'Hollywood']
    daily = df_filtrado.groupby('date_occ').size().reset_index(name='collision_count')
    daily['dayofyear'] = daily['date_occ'].dt.dayofyear
    daily['weekday'] = daily['date_occ'].dt.weekday
    daily['year'] = daily['date_occ'].dt.year
    daily['lag_1'] = daily['collision_count'].shift(1)
    daily['lag_2'] = daily['collision_count'].shift(2)
    daily['rolling_mean_3'] = daily['collision_count'].rolling(3).mean()
    daily = daily.dropna()
    
    return daily

def save_data(daily, filename='data/processed_data.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    daily.to_csv(filename, index=False)
    print(f"✅ Processed data saved to {filename}")

if __name__ == "__main__":
    # Extract
    raw_data = extract_data()
    
    # Transform
    processed_data = transform_data(raw_data)
    
    # Load
    save_data(processed_data)