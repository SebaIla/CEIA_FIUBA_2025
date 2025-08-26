# Importación de librerías necesarias
import pandas as pd                      # Para manipulación de datos tabulares
import numpy as np                       # Para operaciones numéricas
from sklearn.model_selection import train_test_split  # Para dividir datos en entrenamiento y prueba
from sklearn.ensemble import RandomForestRegressor    # Algoritmo de regresión basado en árboles
from sklearn.metrics import mean_squared_error, r2_score  # Métricas de evaluación
import mlflow                           # Para registrar experimentos de ML
import mlflow.sklearn                   # Para registrar modelos de scikit-learn en MLflow
from pathlib import Path                # Para manejar rutas de archivos
import os                               # Para operaciones del sistema
import joblib                           # Para guardar el modelo entrenado

# Configura MLflow con backend SQLite para evitar problemas de compatibilidad en Windows
def setup_mlflow():
    """Configura MLflow con backend SQLite"""
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Usa base de datos local en lugar de archivos
        mlflow.set_experiment("Collision Prediction")   # Define el nombre del experimento
        print("MLflow setup with SQLite backend")
        return True
    except Exception as e:
        print(f"MLflow setup warning: {e}")
        print("Continuing without MLflow...")
        return False

# Carga los datos procesados desde un archivo CSV
def load_processed_data(input_path):
    return pd.read_csv(input_path)

# Entrena el modelo de predicción de colisiones
def train_model(data):
    # Define las variables independientes (features) y la dependiente (target)
    X = data[['dayofyear', 'weekday', 'year', 'lag_1', 'lag_2', 'rolling_mean_3']]
    y = data['collision_count']
    
    # Divide los datos en conjunto de entrenamiento y prueba (sin mezclar el orden temporal)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Crea y entrena el modelo Random Forest con 200 árboles y profundidad máxima de 10
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Realiza predicciones sobre el conjunto de prueba
    rf_preds = rf.predict(X_test)
    
    # Calcula métricas de evaluación
    mse = mean_squared_error(y_test, rf_preds)
    rmse = np.sqrt(mse)                 # Error cuadrático medio raíz
    r2 = r2_score(y_test, rf_preds)     # Coeficiente de determinación
    
    return rf, rmse, r2, X_test, y_test

# Guarda el modelo entrenado manualmente como archivo .joblib
def save_model_manually(model, model_path):
    """Guarda el modelo usando joblib"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved manually to: {model_path}")

# Punto de entrada del script
if __name__ == "__main__":
    # Configura MLflow
    mlflow_available = setup_mlflow()
    
    # Carga los datos procesados
    data = load_processed_data("data/processed/hollywood_collisions.csv")
    
    # Entrena el modelo y obtiene métricas
    model, rmse, r2, X_test, y_test = train_model(data)
    print(f"Model training completed. RMSE: {rmse}, R2: {r2}")
    
    try:
        # Si MLflow está disponible, registra el experimento
        if mlflow_available:
            with mlflow.start_run(run_name="random_forest_training"):
                # Registra parámetros del modelo
                mlflow.log_param("n_estimators", 200)
                mlflow.log_param("max_depth", 10)
                mlflow.log_param("random_state", 42)
                
                # Registra métricas de evaluación
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                
                # Registra el modelo entrenado
                mlflow.sklearn.log_model(model, "random_forest_model")
                print("Model logged to MLflow")
                
                # Registra las predicciones como artefacto
                results_df = pd.DataFrame({
                    'actual': y_test,
                    'predicted': model.predict(X_test)
                })
                results_df.to_csv("prediction_results.csv", index=False)
                mlflow.log_artifact("prediction_results.csv")
        else:
            print("MLflow not available for model logging")
            
    except Exception as e:
        print(f"MLflow logging failed: {e}")
        print("Continuing without MLflow...")
    
    # Guarda el modelo manualmente como respaldo
    save_model_manually(model, "models/random_forest_model.joblib")
    
    print("Model artifact is available for use!")
