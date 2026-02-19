import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_data_generation():
    """Test synthetic data generation"""
    from generate_data import generate_data
    
    # Generate small test dataset
    test_path = 'data/raw/test_data_gen.csv'
    os.makedirs('data/raw', exist_ok=True)
    
    df = generate_data(n_customers=100, output_path=test_path)
    
    assert len(df) == 100
    assert df['monthly_charges'].min() >= 20
    assert df['monthly_charges'].max() <= 150
    assert 'churned' in df.columns
    assert set(df['contract_type'].unique()).issubset({'Month-to-Month', 'One Year', 'Two Year'})
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)

def test_preprocessing():
    """Test data preprocessing"""
    from generate_data import generate_data
    import train_pipeline
    
    # Generate test data
    os.makedirs('data/raw', exist_ok=True)
    generate_data(n_customers=100, output_path='data/raw/customer_data.csv')
    
    # Test preprocessing
    try:
        X_train, X_test, y_train, y_test = train_pipeline.load_and_preprocess()
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert X_train.shape[1] == 8  # 8 features
    except Exception as e:
        pytest.fail(f"Preprocessing failed: {e}")

def test_model_performance():
    """Test model achieves minimum performance"""
    from generate_data import generate_data
    import train_pipeline
    
    # Disable MLflow tracking for tests
    os.environ['MLFLOW_TRACKING_URI'] = ''
    
    # Generate test data
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    generate_data(n_customers=1000, output_path='data/raw/customer_data.csv')
    
    X_train, X_test, y_train, y_test = train_pipeline.load_and_preprocess()
    
    # Train without MLflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Minimum acceptable performance
    assert f1 > 0.4, f"F1 score {f1:.4f} below threshold"
    assert accuracy > 0.5, f"Accuracy {accuracy:.4f} below threshold"
