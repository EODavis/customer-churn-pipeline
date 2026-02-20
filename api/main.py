from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from datetime import datetime
import os
from typing import List

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability",
    version="1.0.0"
)

# Load model and encoders at startup
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/churn_model_latest.pkl')
model = None
contract_encoder = None
payment_encoder = None

@app.on_event("startup")
async def load_model():
    global model, contract_encoder, payment_encoder
    try:
        # Find latest model
        model_files = [f for f in os.listdir('models') if f.startswith('churn_model_') and f.endswith('.pkl')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            model = joblib.load(f'models/{latest_model}')
            contract_encoder = joblib.load('models/contract_encoder.pkl')
            payment_encoder = joblib.load('models/payment_encoder.pkl')
            print(f"✓ Loaded model: {latest_model}")
        else:
            print("⚠ No model found, using dummy model")
    except Exception as e:
        print(f"✗ Error loading model: {e}")

class CustomerFeatures(BaseModel):
    customer_id: int = Field(..., description="Customer ID")
    account_age_days: int = Field(..., ge=1, le=3650, description="Days since account creation")
    monthly_charges: float = Field(..., ge=0, le=500, description="Monthly charges in USD")
    total_charges: float = Field(..., ge=0, le=50000, description="Total charges to date")
    support_tickets: int = Field(..., ge=0, le=100, description="Number of support tickets")
    contract_type: str = Field(..., description="Contract type: Month-to-Month, One Year, Two Year")
    payment_method: str = Field(..., description="Payment method: Credit Card, Bank Transfer, Electronic Check")
    monthly_usage_gb: float = Field(..., ge=0, le=10000, description="Monthly data usage in GB")
    num_services: int = Field(..., ge=1, le=10, description="Number of subscribed services")

    class Config:
        schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]

@app.get("/")
async def root():
    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """Predict churn for a single customer"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode categorical features
        contract_encoded = contract_encoder.transform([customer.contract_type])[0]
        payment_encoded = payment_encoder.transform([customer.payment_method])[0]
        
        # Prepare features in correct order
        features = np.array([[
            customer.account_age_days,
            customer.monthly_charges,
            customer.total_charges,
            customer.support_tickets,
            customer.monthly_usage_gb,
            customer.num_services,
            contract_encoded,
            payment_encoded
        ]])
        
        # Predict
        churn_prob = model.predict_proba(features)[0][1]
        churn_pred = bool(churn_prob >= 0.5)
        
        # Risk level
        if churn_prob < 0.3:
            risk_level = "low"
        elif churn_prob < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(float(churn_prob), 4),
            churn_prediction=churn_pred,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for multiple customers"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for customer in request.customers:
        try:
            pred = await predict_churn(customer)
            predictions.append(pred)
        except Exception as e:
            predictions.append({
                "customer_id": customer.customer_id,
                "error": str(e)
            })
    
    return {
        "predictions": predictions,
        "total": len(predictions),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get model metadata"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "n_features": model.n_features_in_,
        "n_estimators": getattr(model, 'n_estimators', None),
        "feature_names": [
            "account_age_days", "monthly_charges", "total_charges",
            "support_tickets", "monthly_usage_gb", "num_services",
            "contract_type_encoded", "payment_method_encoded"
        ]
    }

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from api.metrics import (
    track_prediction_metrics,
    active_model_version,
    data_drift_score
)

# ... existing imports ...

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
@track_prediction_metrics  # Add decorator
async def predict_churn(customer: CustomerFeatures):
    # ... existing code ...
    pass

# Set model version on startup
@app.on_event("startup")
async def load_model():
    global model, contract_encoder, payment_encoder
    # ... existing code ...
    
    # Set model version metric
    model_files = [f for f in os.listdir('models') if f.startswith('churn_model_') and f.endswith('.pkl')]
    if model_files:
        version = len(model_files)
        active_model_version.set(version)