import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import sys
sys.path.append('..')

def test_data_generation():
    """Test synthetic data generation"""
    from generate_data import generate_data
    
    df = pd.DataFrame({
        'customer_id': range(1, 101),
        'monthly_charges': np.random.uniform(20, 150, 100),
        'support_tickets': np.random.poisson(2, 100),
        'churned': np.random.binomial(1, 0.3, 100)
    })
    
    assert len(df) == 100
    assert df['monthly_charges'].min() >= 20
    assert df['monthly_charges'].max() <= 150
    assert 'churned' in df.columns

def test_preprocessing():
    """Test data preprocessing"""
    import train_pipeline
    
    # Generate small test dataset
    df = pd.DataFrame({
        'customer_id': range(1, 101),
        'account_age_days': np.random.randint(30, 1825, 100),
        'monthly_charges': np.random.uniform(20, 150, 100),
        'total_charges': np.random.uniform(100, 8000, 100),
        'support_tickets': np.random.poisson(2, 100),
        'contract_type': np.random.choice(['Month-to-Month', 'One Year'], 100),
        'payment_method': np.random.choice(['Credit Card', 'Bank Transfer'], 100),
        'monthly_usage_gb': np.random.uniform(5, 500, 100),
        'num_services': np.random.randint(1, 6, 100),
        'churned': np.random.binomial(1, 0.3, 100)
    })
    df.to_csv('data/raw/test_data.csv', index=False)
    
    # Should not raise errors
    try:
        X_train, X_test, y_train, y_test = train_pipeline.load_and_preprocess()
        assert len(X_train) > 0
        assert len(X_test) > 0
    except Exception as e:
        pytest.fail(f"Preprocessing failed: {e}")

def test_model_performance():
    """Test model achieves minimum performance"""
    import train_pipeline
    
    X_train, X_test, y_train, y_test = train_pipeline.load_and_preprocess()
    model, metrics = train_pipeline.train_model(X_train, y_train, X_test, y_test)
    
    # Minimum acceptable F1 score
    assert metrics['f1'] > 0.5, f"F1 score {metrics['f1']} below threshold"
    assert metrics['accuracy'] > 0.6, f"Accuracy {metrics['accuracy']} below threshold"