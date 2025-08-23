import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime


def test_model():
    try:
        print(f"{datetime.now()} - Starting model testing")
        
        # Código de testing
        # ... (mantener el mismo código de testing)
        
        print(f"{datetime.now()} - Model testing completed successfully")
    except Exception as e:
        print(f"{datetime.now()} - Model testing failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_model()