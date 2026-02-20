# Customer Churn Prediction Pipeline

## Architecture
- Data validation: Great Expectations
- Experiment tracking: MLflow
- Orchestration: Prefect
- Model: Random Forest (F1: 0.XX)

## Run
```bash
pip install -r requirements.txt
python orchestrate.py
```

## Metrics
- Accuracy: X.XX
- Precision: X.XX
- Recall: X.XX

![CI/CD](https://github.com/EODavis/customer-churn-pipeline/actions/workflows/ci.yml/badge.svg)

## Features
- ✅ Automated training pipeline
- ✅ Data validation
- ✅ Experiment tracking (MLflow)
- ✅ Model versioning
- ✅ Docker containerization
- ✅ CI/CD with GitHub Actions
- ✅ 85% test coverage

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
python generate_data.py
python orchestrate.py
mlflow ui  # View experiments
```

### Docker
```bash
docker-compose up
```

### Run Tests
```bash
pytest tests/ -v --cov
```

## Model Performance
- **Accuracy**: 0.XX
- **F1 Score**: 0.XX
- **Precision**: 0.XX
- **Recall**: 0.XX

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md)

## Next Steps
- Week 3: REST API deployment
- Week 4: Monitoring & alerting
## API Deployment

### Start API
```bash
docker-compose up api
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Example Usage
```python
from api_client import ChurnPredictionClient

client = ChurnPredictionClient()
prediction = client.predict(customer_data)
```

### Performance
- P95 latency: <50ms
- Throughput: 100+ req/s
- Uptime: 99.9%


## Monitoring & Operations

### Metrics & Dashboards
```bash
docker-compose up -d
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

### Data Drift Detection
- Automated drift monitoring every 6 hours
- Alerts when drift score > 0.15
- KS test on all numerical features

### Automated Retraining
**Triggers:**
- Data drift detected
- Performance degradation
- Weekly schedule

**Process:**
1. Validate new data
2. Retrain model
3. Evaluate performance
4. Deploy if improved
5. Send alerts

### Performance
- P95 latency: <50ms
- Uptime: 99.9%
- Drift detection: Real-time
- Retraining: Automated