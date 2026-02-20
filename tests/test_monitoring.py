import pytest
import pandas as pd
import numpy as np
from monitoring.drift_detector import DriftDetector
from monitoring.collect_predictions import PredictionLogger
import os

def test_drift_detector():
    """Test drift detection"""
    # Create reference data
    np.random.seed(42)
    reference = pd.DataFrame({
        'account_age_days': np.random.randint(30, 1825, 1000),
        'monthly_charges': np.random.uniform(20, 150, 1000),
        'total_charges': np.random.uniform(100, 8000, 1000),
        'support_tickets': np.random.poisson(2, 1000),
        'monthly_usage_gb': np.random.uniform(5, 500, 1000),
        'num_services': np.random.randint(1, 6, 1000)
    })
    
    os.makedirs('data/raw', exist_ok=True)
    reference.to_csv('data/raw/test_reference.csv', index=False)
    
    # Create current data (no drift)
    current = pd.DataFrame({
        'account_age_days': np.random.randint(30, 1825, 500),
        'monthly_charges': np.random.uniform(20, 150, 500),
        'total_charges': np.random.uniform(100, 8000, 500),
        'support_tickets': np.random.poisson(2, 500),
        'monthly_usage_gb': np.random.uniform(5, 500, 500),
        'num_services': np.random.randint(1, 6, 500)
    })
    
    detector = DriftDetector('data/raw/test_reference.csv', threshold=0.05)
    result = detector.calculate_drift(current)
    
    assert 'overall_drift_score' in result
    assert 'drift_detected' in result
    assert result['overall_drift_score'] >= 0

def test_drift_detector_with_drift():
    """Test drift detection with actual drift"""
    np.random.seed(42)
    
    # Reference data
    reference = pd.DataFrame({
        'monthly_charges': np.random.uniform(20, 150, 1000)
    })
    reference.to_csv('data/raw/test_reference_drift.csv', index=False)
    
    # Drifted data (shifted distribution)
    current = pd.DataFrame({
        'monthly_charges': np.random.uniform(100, 300, 500)  # Different range
    })
    
    detector = DriftDetector('data/raw/test_reference_drift.csv', threshold=0.05)
    result = detector.calculate_drift(current)
    
    # Should detect drift
    assert result['drift_detected'] == True
    assert result['overall_drift_score'] > 0.1

def test_prediction_logger():
    """Test prediction logging"""
    logger = PredictionLogger(log_path='data/test_predictions/')
    
    customer_data = {
        'customer_id': 123,
        'account_age_days': 365,
        'monthly_charges': 50.0
    }
    
    prediction = {
        'churn_prediction': True,
        'churn_probability': 0.75,
        'risk_level': 'high'
    }
    
    logger.log_prediction(customer_data, prediction)
    logger.flush()
    
    # Verify log file exists
    assert os.path.exists('data/test_predictions/')