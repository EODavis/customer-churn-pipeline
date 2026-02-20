from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Setup test client
client = TestClient(app)

def setup_module():
    """Create dummy model for testing"""
    os.makedirs('models', exist_ok=True)
    
    # Create dummy model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_dummy = np.random.rand(100, 8)
    y_dummy = np.random.randint(0, 2, 100)
    model.fit(X_dummy, y_dummy)
    joblib.dump(model, 'models/churn_model_test.pkl')
    
    # Create dummy encoders
    le_contract = LabelEncoder()
    le_contract.fit(['Month-to-Month', 'One Year', 'Two Year'])
    joblib.dump(le_contract, 'models/contract_encoder.pkl')
    
    le_payment = LabelEncoder()
    le_payment.fit(['Credit Card', 'Bank Transfer', 'Electronic Check'])
    joblib.dump(le_payment, 'models/payment_encoder.pkl')

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_predict_endpoint():
    payload = {
        "customer_id": 12345,
        "account_age_days": 730,
        "monthly_charges": 89.99,
        "total_charges": 2159.76,
        "support_tickets": 3,
        "contract_type": "Month-to-Month",
        "payment_method": "Credit Card",
        "monthly_usage_gb": 150.5,
        "num_services": 4
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert 0 <= data["churn_probability"] <= 1

def test_predict_invalid_data():
    payload = {
        "customer_id": 12345,
        "account_age_days": -100,  # Invalid
        "monthly_charges": 89.99,
        "total_charges": 2159.76,
        "support_tickets": 3,
        "contract_type": "Invalid Type",  # Invalid
        "payment_method": "Credit Card",
        "monthly_usage_gb": 150.5,
        "num_services": 4
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_batch_prediction():
    payload = {
        "customers": [
            {
                "customer_id": 1,
                "account_age_days": 365,
                "monthly_charges": 50.0,
                "total_charges": 600.0,
                "support_tickets": 1,
                "contract_type": "One Year",
                "payment_method": "Credit Card",
                "monthly_usage_gb": 100.0,
                "num_services": 2
            },
            {
                "customer_id": 2,
                "account_age_days": 730,
                "monthly_charges": 100.0,
                "total_charges": 2400.0,
                "support_tickets": 5,
                "contract_type": "Month-to-Month",
                "payment_method": "Electronic Check",
                "monthly_usage_gb": 300.0,
                "num_services": 5
            }
        ]
    }
    
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "n_features" in data
```

### Update `requirements.txt`
```
mlflow==2.9.2
dvc==3.38.1
prefect==2.14.10
scikit-learn==1.3.2
pandas==2.1.4
great-expectations==0.18.8
pytest==7.4.3
pytest-cov==4.1.0
fastapi==0.104.1
uvicorn[standard]==0.24.0
httpx==0.25.2
locust==2.19.1