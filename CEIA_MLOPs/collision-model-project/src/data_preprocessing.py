import pandas as pd
import os

def preprocess_data(df):
    # Eliminar columnas irrelevantes
    columnas_a_eliminar = [
        'dr_no', 'rpt_dist_no', 'crm_cd', 'crm_cd_desc',
        'cross_street', 'location_1.human_address'
    ]
    columnas_a_eliminar += [col for col in df.columns if col.startswith(':@computed')]
    df_limpio = df.drop(columns=columnas_a_eliminar)
    
    # Convertir fechas
    df_limpio['date_rptd'] = pd.to_datetime(df_limpio['date_rptd'], errors='coerce')
    df_limpio['date_occ'] = pd.to_datetime(df_limpio['date_occ'], errors='coerce')
    
    # Imputar valores
    df_limpio['vict_age'] = pd.to_numeric(df_limpio['vict_age'], errors='coerce')
    df_limpio['vict_age'] = df_limpio['vict_age'].fillna(df_limpio['vict_age'].median())
    
    df_limpio['vict_sex'] = df_limpio['vict_sex'].fillna('Unknown')
    df_limpio['vict_descent'] = df_limpio['vict_descent'].fillna('Unknown')
    df_limpio['premis_desc'] = df_limpio['premis_desc'].fillna('Unknown')
    df_limpio['premis_cd'] = df_limpio['premis_cd'].fillna(df_limpio['premis_cd'].mode()[0])
    
    # Eliminar mocodes
    df_limpio = df_limpio.drop(columns=['mocodes'])
    
    # Filtrar outliers
    df_filtrado = df_limpio.copy()
    for col in ['vict_age', 'location_1.latitude', 'location_1.longitude']:
        Q1 = df_filtrado[col].quantile(0.25)
        Q3 = df_filtrado[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_filtrado = df_filtrado[(df_filtrado[col] >= lower_bound) & 
                                 (df_filtrado[col] <= upper_bound)]
    
    # Preparar datos para series temporales
    df_filtrado['date_occ'] = pd.to_datetime(df_filtrado['date_occ'])
    df_hollywood = df_filtrado[df_filtrado['area_name'] == 'Hollywood']
    daily = df_hollywood.groupby('date_occ').size().reset_index(name='collision_count')
    
    # Crear características temporales
    daily['dayofyear'] = daily['date_occ'].dt.dayofyear
    daily['weekday'] = daily['date_occ'].dt.weekday
    daily['year'] = daily['date_occ'].dt.year
    daily['lag_1'] = daily['collision_count'].shift(1)
    daily['lag_2'] = daily['collision_count'].shift(2)
    daily['rolling_mean_3'] = daily['collision_count'].rolling(3).mean()
    
    # Eliminar NaNs
    daily = daily.dropna()
    
    # Guardar datos procesados
    os.makedirs('data/processed', exist_ok=True)
    daily.to_csv('data/processed/processed_data.csv', index=False)
    
    return daily

if __name__ == "__main__":
    # Para prueba directa, cargar datos crudos
    df_raw = pd.read_csv('data/raw/la_collisions_raw.csv')
    preprocess_data(df_raw)