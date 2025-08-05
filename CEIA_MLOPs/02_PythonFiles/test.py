import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filename='data/processed_data.csv'):
    return pd.read_csv(filename, parse_dates=['date_occ'])

def load_model(filename='models/random_forest_model.pkl'):
    return joblib.load(filename)

def prepare_test_data(df):
    X, y = prepare_features(df)
    split_index = int(len(X) * 0.8)
    return X.iloc[split_index:], y.iloc[split_index:]

def prepare_features(df):
    X = df[['dayofyear', 'weekday', 'year', 'lag_1', 'lag_2', 'rolling_mean_3']]
    y = df['collision_count']
    return X, y

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"📊 Model Evaluation:")
    print(f"  - MSE: {mse:.2f}")
    print(f"  - R²: {r2:.4f}")

if __name__ == "__main__":
    # Load data and model
    df = load_data()
    model = load_model()
    
    # Prepare test set
    X_test, y_test = prepare_test_data(df)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)