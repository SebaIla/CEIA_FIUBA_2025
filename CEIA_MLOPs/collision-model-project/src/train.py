import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def train_model():
    # Cargar datos procesados
    data = pd.read_csv('data/processed/processed_data.csv')
    
    # Preparar características y objetivo
    X = data[['dayofyear', 'weekday', 'year', 'lag_1', 'lag_2', 'rolling_mean_3']]
    y = data['collision_count']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Configurar MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Collision_Prediction")
    
    with mlflow.start_run():
        # Entrenar modelo
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluar modelo
        predictions = rf.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Registrar parámetros y métricas
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        # Registrar modelo
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        print(f"Model trained with MSE: {mse}, R2: {r2}")
    
    return rf

if __name__ == "__main__":
    train_model()