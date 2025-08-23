import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os


def load_data(filename='data/processed/processed_data.csv'):
    """Cargar datos procesados"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"El archivo {filename} no existe. Ejecuta primero etl.py")
    
    return pd.read_csv(filename, parse_dates=['date_occ'])


def prepare_features(data):
    """Preparar características para el modelo"""
    X = data[['dayofyear', 'weekday', 'year', 'lag_1', 'lag_2', 'rolling_mean_3']]
    y = data['collision_count']
    return X, y


def train_model():
    """Entrenar el modelo de Random Forest"""
    # Cargar datos
    data = load_data()
    
    # Preparar características
    X, y = prepare_features(data)
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Train R²: {train_score:.4f}")
    print(f"Test R²: {test_score:.4f}")
    
    # Guardar modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.pkl')
    print("✅ Modelo guardado en 'models/random_forest_model.pkl'")
    
    return model, X_test, y_test


if __name__ == "__main__":
    train_model()