import requests
import pandas as pd
import numpy as np
import time
import os
import json


def extract_data():
    api_endpoint_la_collisions = 'https://data.lacity.org/resource/d5tf-ez2w.json'
    limit = 50000
    max_retries = 3
    retry_delay = 10
    la_collisions_data = []
    offset = 0
    total_retrieved = 0

    print("Starting data retrieval...")
    while True:
        retries = 0
        success = False
        while retries < max_retries and not success:
            try:
                params = {'$limit': limit, '$offset': offset}
                response = requests.get(api_endpoint_la_collisions, params=params)
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
                    time.sleep(retry_delay)
            except Exception as e:
                print(f"⚠️ Exception: {str(e)}. Retrying...")
                retries += 1
                time.sleep(retry_delay)
        
        if not success:
            print(f"❌ Failed after {max_retries} attempts. Stopping.")
            break
        if len(chunk) < limit:
            break
    
    print(f"✅ Final ingestion completed: {len(la_collisions_data)} rows retrieved.")
    return la_collisions_data


def transform_data(la_collisions_data):
    df = pd.json_normalize(la_collisions_data)
    
    # Clean data
    columns_to_remove = [
        'dr_no', 'rpt_dist_no', 'crm_cd', 'crm_cd_desc',
        'cross_street', 'location_1.human_address'
    ]
    columns_to_remove += [col for col in df.columns if col.startswith(':@computed')]
    df_clean = df.drop(columns=columns_to_remove)
    
    # Convert dates
    df_clean['date_rptd'] = pd.to_datetime(df_clean['date_rptd'], errors='coerce')
    df_clean['date_occ'] = pd.to_datetime(df_clean['date_occ'], errors='coerce')
    
    # Handle missing values
    df_clean['vict_age'] = pd.to_numeric(df_clean['vict_age'], errors='coerce')
    df_clean['vict_age'] = df_clean['vict_age'].fillna(df_clean['vict_age'].median())
    df_clean['vict_sex'] = df_clean['vict_sex'].fillna('Unknown')
    df_clean['vict_descent'] = df_clean['vict_descent'].fillna('Unknown')
    df_clean['premis_desc'] = df_clean['premis_desc'].fillna('Unknown')
    df_clean['premis_cd'] = df_clean['premis_cd'].fillna(df_clean['premis_cd'].mode()[0])
    df_clean = df_clean.drop(columns=['mocodes'])
    
    # Handle outliers
    numeric_columns = ['time_occ', 'vict_age', 'location_1.latitude', 'location_1.longitude']
    for col in numeric_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    df_filtered = df_clean.copy()
    for col in ['vict_age', 'location_1.latitude', 'location_1.longitude']:
        q1 = df_filtered[col].quantile(0.25)
        q3 = df_filtered[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & 
                                 (df_filtered[col] <= upper_bound)]
    
    # Prepare time-series data
    df_filtered = df_filtered[df_filtered['area_name'] == 'Hollywood']
    daily = df_filtered.groupby('date_occ').size().reset_index(name='collision_count')
    daily['dayofyear'] = daily['date_occ'].dt.dayofyear
    daily['weekday'] = daily['date_occ'].dt.weekday
    daily['year'] = daily['date_occ'].dt.year
    daily['lag_1'] = daily['collision_count'].shift(1)
    daily['lag_2'] = daily['collision_count'].shift(2)
    daily['rolling_mean_3'] = daily['collision_count'].rolling(3).mean()
    daily = daily.dropna()
    
    return daily


def save_data(daily, filename='data/processed/processed_data.csv'):
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