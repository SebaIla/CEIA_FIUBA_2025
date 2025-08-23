import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os


def load_test_data(filename='data/processed/processed_data.csv'):
    """Cargar datos para testing"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"El archivo {filename} no existe. Ejecuta primero etl.py")
    
    return pd.read_csv(filename, parse_dates=['date_occ'])


def test_model():
    """Probar el modelo entrenado"""
    # Verificar que el modelo existe
    model_path = 'models/random_forest_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo {model_path} no existe. Ejecuta primero train.py")
    
    # Cargar modelo y datos
    model = joblib.load(model_path)
    data = load_test_data()
    
    # Preparar características
    X = data[['dayofyear', 'weekday', 'year', 'lag_1', 'lag_2', 'rolling_mean_3']]
    y = data['collision_count']
    
    # Hacer predicciones
    predictions = model.predict(X)
    
    # Calcular métricas
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Guardar resultados
    results = pd.DataFrame({
        'date_occ': data['date_occ'],
        'actual': y,
        'predicted': predictions
    })
    
    os.makedirs('results', exist_ok=True)
    results.to_csv('results/predictions.csv', index=False)
    print("✅ Resultados guardados en 'results/predictions.csv'")


if __name__ == "__main__":
    test_model()