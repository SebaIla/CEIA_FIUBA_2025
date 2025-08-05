import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(filename='data/processed_data.csv'):
    df = pd.read_csv(filename, parse_dates=['date_occ'])
    return df

def prepare_features(df):
    X = df[['dayofyear', 'weekday', 'year', 'lag_1', 'lag_2', 'rolling_mean_3']]
    y = df['collision_count']
    return X, y

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename='models/random_forest_model.pkl'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"✅ Model saved to {filename}")

if __name__ == "__main__":
    # Load processed data
    df = load_data()
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data (time-series split)
    split_index = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Save model
    save_model(model)