import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filename='/app/data/processed/processed_data.csv'):
    """Load processed data from CSV file"""
    print(f"{datetime.now()} - Loading data from {filename}")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist. Run etl.py first")
    
    df = pd.read_csv(filename, parse_dates=['date_occ'])
    print(f"{datetime.now()} - Loaded {len(df)} records")
    return df


def load_model(filename='/app/models/random_forest_model.pkl'):
    """Load trained model from file"""
    print(f"{datetime.now()} - Loading model from {filename}")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} does not exist. Run train.py first")
    
    model = joblib.load(filename)
    print(f"{datetime.now()} - Model loaded successfully")
    return model


def load_training_metrics(filename='/app/models/training_metrics.json'):
    """Load training metrics from file"""
    print(f"{datetime.now()} - Loading training metrics from {filename}")
    if not os.path.exists(filename):
        print(f"{datetime.now()} - Warning: Training metrics file not found")
        return None
    
    with open(filename, 'r') as f:
        metrics = json.load(f)
    return metrics


def prepare_test_data(df):
    """Prepare test data (last 20% of time series)"""
    print(f"{datetime.now()} - Preparing test data")
    
    # Check if all required columns are present
    required_columns = ['dayofyear', 'weekday', 'year', 'lag_1', 'lag_2', 'rolling_mean_3', 'collision_count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in test data: {missing_columns}")
    
    X, y = prepare_features(df)
    
    # Use the same split as in training (last 20% for testing)
    split_index = int(len(X) * 0.8)
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    
    print(f"{datetime.now()} - Test set prepared: {X_test.shape} features, {y_test.shape} target")
    return X_test, y_test


def prepare_features(df):
    """Prepare features and target variable"""
    X = df[['dayofyear', 'weekday', 'year', 'lag_1', 'lag_2', 'rolling_mean_3']]
    y = df['collision_count']
    return X, y


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data"""
    print(f"{datetime.now()} - Evaluating model performance")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100  # Avoid division by zero
    
    # Print results
    print(f"\n{datetime.now()} - 📊 Model Evaluation Results:")
    print(f"  - MSE (Mean Squared Error): {mse:.4f}")
    print(f"  - RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  - MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  - MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"  - R² Score: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'y_test': y_test.values,
        'y_pred': y_pred
    }


def compare_with_training_metrics(test_metrics, training_metrics):
    """Compare test metrics with training metrics"""
    if training_metrics is None:
        return
    
    print(f"\n{datetime.now()} - 📈 Comparison with Training Performance:")
    print(f"  Metric        | Training | Test     | Difference")
    print(f"  --------------|----------|----------|-----------")
    
    metrics_to_compare = ['rmse', 'r2']
    for metric in metrics_to_compare:
        train_val = training_metrics.get(metric, float('nan'))
        test_val = test_metrics.get(metric, float('nan'))
        diff = test_val - train_val
        print(f"  {metric.upper():12} | {train_val:8.4f} | {test_val:8.4f} | {diff:+.4f}")
    
    # Check for overfitting
    rmse_diff = test_metrics['rmse'] - training_metrics.get('test_rmse', training_metrics.get('rmse', float('nan')))
    if rmse_diff > 1.0:
        print(f"\n{datetime.now()} - ⚠️  Warning: Significant performance drop detected - possible overfitting")


def save_test_results(test_metrics, filename='/app/results/test_results.json'):
    """Save test results to JSON file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Prepare results for saving (remove numpy arrays)
    results_to_save = {k: v for k, v in test_metrics.items() if k not in ['y_test', 'y_pred']}
    results_to_save['test_date'] = datetime.now().isoformat()
    
    with open(filename, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"{datetime.now()} - Test results saved to {filename}")


def create_visualizations(test_metrics, df, X_test):
    """Create visualization plots for model performance"""
    print(f"{datetime.now()} - Creating visualizations")
    
    os.makedirs('/app/results/plots', exist_ok=True)
    
    try:
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(test_metrics['y_test'], test_metrics['y_pred'], alpha=0.5)
        plt.plot([test_metrics['y_test'].min(), test_metrics['y_test'].max()], 
                 [test_metrics['y_test'].min(), test_metrics['y_test'].max()], 'r--')
        plt.xlabel('Actual Collision Count')
        plt.ylabel('Predicted Collision Count')
        plt.title('Actual vs Predicted Collision Counts')
        plt.savefig('/app/results/plots/actual_vs_predicted.png')
        plt.close()
        
        # Time series plot
        test_dates = df['date_occ'].iloc[-len(test_metrics['y_test']):]
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, test_metrics['y_test'], label='Actual', marker='o')
        plt.plot(test_dates, test_metrics['y_pred'], label='Predicted', marker='x')
        plt.xlabel('Date')
        plt.ylabel('Collision Count')
        plt.title('Time Series: Actual vs Predicted Collisions')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('/app/results/plots/time_series_comparison.png')
        plt.close()
        
        # Residual plot
        residuals = test_metrics['y_test'] - test_metrics['y_pred']
        plt.figure(figsize=(10, 6))
        plt.scatter(test_metrics['y_pred'], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.savefig('/app/results/plots/residual_plot.png')
        plt.close()
        
        print(f"{datetime.now()} - Visualizations saved to /app/results/plots/")
        
    except Exception as e:
        print(f"{datetime.now()} - Warning: Could not create visualizations: {str(e)}")


def main():
    """Main testing function"""
    try:
        print(f"{datetime.now()} - Starting model testing process")
        
        # Load data and model
        df = load_data()
        model = load_model()
        training_metrics = load_training_metrics()
        
        # Prepare test data
        X_test, y_test = prepare_test_data(df)
        
        # Evaluate model
        test_metrics = evaluate_model(model, X_test, y_test)
        
        # Compare with training performance
        compare_with_training_metrics(test_metrics, training_metrics)
        
        # Save results
        save_test_results(test_metrics)
        
        # Create visualizations
        create_visualizations(test_metrics, df, X_test)
        
        print(f"\n{datetime.now()} - ✅ Model testing completed successfully!")
        
        return test_metrics
        
    except Exception as e:
        print(f"{datetime.now()} - ❌ Model testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()