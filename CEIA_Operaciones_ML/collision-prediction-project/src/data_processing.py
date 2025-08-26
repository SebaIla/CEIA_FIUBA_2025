# Importación de librerías necesarias
import pandas as pd          # Para manipulación de datos tabulares
import numpy as np           # Para operaciones numéricas
import json                  # Para leer y escribir archivos JSON
from pathlib import Path     # Para manejar rutas de archivos de forma segura
import mlflow                # Para registrar experimentos de machine learning
import os                    # Para interactuar con el sistema operativo

# Configura MLflow para registrar experimentos localmente
def setup_mlflow():
    """Setup MLflow for basic tracking without model registry"""
    try:
        # Define la ruta local donde se guardarán los experimentos
        mlruns_path = os.path.join(os.getcwd(), "mlruns")
        
        # Crea el directorio si no existe
        os.makedirs(mlruns_path, exist_ok=True)
        
        # Establece la URI de seguimiento para MLflow
        mlflow.set_tracking_uri(mlruns_path)
        
        # Define el nombre del experimento
        mlflow.set_experiment("Collision Prediction")
        
        print(f"MLflow tracking setup complete. Using directory: {mlruns_path}")
        return True
    except Exception as e:
        # Si ocurre un error, muestra advertencia y continúa sin MLflow
        print(f"MLflow setup warning: {e}")
        print("MLflow tracking will be limited. Continuing without full MLflow functionality.")
        return False

# Carga los datos crudos desde un archivo JSON
def load_raw_data(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    return data

# Procesa los datos: limpieza, imputación y filtrado
def process_data(data):
    df = pd.json_normalize(data)  # Convierte JSON anidado en DataFrame plano
    
    # Lista de columnas irrelevantes a eliminar
    columnas_a_eliminar = [
        'dr_no', 'rpt_dist_no', 'crm_cd', 'crm_cd_desc',
        'cross_street', 'location_1.human_address'
    ]
    # También elimina columnas computadas automáticamente
    columnas_a_eliminar += [col for col in df.columns if col.startswith(':@computed')]
    df_limpio = df.drop(columns=columnas_a_eliminar)
    
    # Convierte columnas de fechas
    df_limpio['date_rptd'] = pd.to_datetime(df_limpio['date_rptd'], errors='coerce')
    df_limpio['date_occ'] = pd.to_datetime(df_limpio['date_occ'], errors='coerce')
    
    # Imputa valores nulos en columnas clave
    df_limpio['vict_age'] = pd.to_numeric(df_limpio['vict_age'], errors='coerce')
    df_limpio['vict_age'] = df_limpio['vict_age'].fillna(df_limpio['vict_age'].median())
    df_limpio['vict_sex'] = df_limpio['vict_sex'].fillna('Unknown')
    df_limpio['vict_descent'] = df_limpio['vict_descent'].fillna('Unknown')
    df_limpio['premis_desc'] = df_limpio['premis_desc'].fillna('Unknown')
    df_limpio['premis_cd'] = df_limpio['premis_cd'].fillna(df_limpio['premis_cd'].mode()[0])
    
    # Elimina columna 'mocodes' si existe
    if 'mocodes' in df_limpio.columns:
        df_limpio = df_limpio.drop(columns=['mocodes'])
    
    # Convierte columnas numéricas
    columnas_numericas = ['time_occ', 'vict_age', 'location_1.latitude', 'location_1.longitude']
    for col in columnas_numericas:
        if col in df_limpio.columns:
            df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')
    
    # Filtra outliers usando el método del rango intercuartílico (IQR)
    df_filtrado = df_limpio.copy()
    for col in ['vict_age', 'location_1.latitude', 'location_1.longitude']:
        if col in df_filtrado.columns and not df_filtrado[col].isnull().all():
            Q1 = df_filtrado[col].quantile(0.25)
            Q3 = df_filtrado[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_filtrado = df_filtrado[(df_filtrado[col] >= lower_bound) & (df_filtrado[col] <= upper_bound)]
    
    return df_filtrado

# Genera features específicas para el área de Hollywood
def prepare_hollywood_features(df):
    # Verifica que exista la columna 'area_name'
    if 'area_name' not in df.columns:
        print("Warning: 'area_name' column not found. Cannot filter for Hollywood.")
        return pd.DataFrame()
    
    # Filtra registros de Hollywood
    df = df[df['area_name'] == 'Hollywood']
    
    # Agrupa por fecha de ocurrencia y cuenta colisiones por día
    daily = df.groupby('date_occ').size().reset_index(name='collision_count')
    
    # Crea variables temporales
    daily['dayofyear'] = daily['date_occ'].dt.dayofyear
    daily['weekday'] = daily['date_occ'].dt.weekday
    daily['year'] = daily['date_occ'].dt.year
    daily['lag_1'] = daily['collision_count'].shift(1)
    daily['lag_2'] = daily['collision_count'].shift(2)
    daily['rolling_mean_3'] = daily['collision_count'].rolling(3).mean()
    
    # Elimina filas con valores nulos
    daily = daily.dropna()
    
    return daily

# Guarda los datos procesados en formato CSV
def save_processed_data(data, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

# Punto de entrada del script
if __name__ == "__main__":
    # Intenta configurar MLflow
    mlflow_available = setup_mlflow()
    
    try:
        # Si MLflow está disponible, inicia un run y registra parámetros
        if mlflow_available:
            with mlflow.start_run(run_name="data_processing"):
                mlflow.log_param("data_source", "LA API")
                mlflow.log_param("target_area", "Hollywood")
                
                # Carga y procesa los datos
                raw_data = load_raw_data("data/raw/la_collisions_raw.json")
                processed_data = process_data(raw_data)
                feature_data = prepare_hollywood_features(processed_data)
                
                # Verifica que haya datos válidos
                if feature_data.empty:
                    print("Error: No data available after processing. Check if 'area_name' column exists and contains 'Hollywood'.")
                    exit(1)
                
                # Guarda y registra los datos procesados
                save_processed_data(feature_data, "data/processed/hollywood_collisions.csv")
                mlflow.log_artifact("data/processed/hollywood_collisions.csv")
                print("Data processing completed and logged to MLflow")
        else:
            # Si MLflow no está disponible, procesa sin registrar
            raw_data = load_raw_data("data/raw/la_collisions_raw.json")
            processed_data = process_data(raw_data)
            feature_data = prepare_hollywood_features(processed_data)
            
            if feature_data.empty:
                print("Error: No data available after processing. Check if 'area_name' column exists and contains 'Hollywood'.")
                exit(1)
            
            save_processed_data(feature_data, "data/processed/hollywood_collisions.csv")
            print("Data processing completed without MLflow logging")
            
    except Exception as e:
        # Si ocurre un error, intenta procesar sin MLflow como fallback
        print(f"Error during data processing: {e}")
        print("Trying to process data without MLflow...")
        
        raw_data = load_raw_data("data/raw/la_collisions_raw.json")
        processed_data = process_data(raw_data)
        feature_data = prepare_hollywood_features(processed_data)
        
        if feature_data.empty:
            print("Error: No data available after processing. Check if 'area_name' column exists and contains 'Hollywood'.")
            exit(1)
        
        save_processed_data(feature_data, "data/processed/hollywood_collisions.csv")
        print("Data processing completed without MLflow logging after error")
