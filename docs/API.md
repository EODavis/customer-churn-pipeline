# Churn Prediction API Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-02-20T10:30:00"
}
```

### 2. Single Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "customer_id": 12345,
  "churn_probability": 0.7234,
  "churn_prediction": true,
  "risk_level": "high",
  "timestamp": "2025-02-20T10:30:00"
}
```

### 3. Batch Prediction
```http
POST /predict/batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "customers": [
    { /* customer 1 data */ },
    { /* customer 2 data */ }
  ]
}
```

### 4. Model Info
```http
GET /model/info
```

## Error Codes
- `200`: Success
- `422`: Validation Error
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

## Rate Limits
- 100 requests/minute per IP
- 1000 requests/hour per IP

## Python Client
```python
from api_client import ChurnPredictionClient

client = ChurnPredictionClient("http://localhost:8000")
result = client.predict(customer_data)
```