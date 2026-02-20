from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from functools import wraps

# Define metrics
prediction_counter = Counter(
    'churn_predictions_total',
    'Total number of predictions made',
    ['risk_level']
)

prediction_latency = Histogram(
    'churn_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

model_confidence = Histogram(
    'churn_prediction_confidence',
    'Distribution of prediction confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

api_errors = Counter(
    'churn_api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type']
)

active_model_version = Gauge(
    'churn_model_version',
    'Currently active model version'
)

data_drift_score = Gauge(
    'churn_data_drift_score',
    'Data drift detection score (0-1)'
)

def track_prediction_metrics(func):
    """Decorator to track prediction metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            
            # Track latency
            latency = time.time() - start_time
            prediction_latency.observe(latency)
            
            # Track prediction
            prediction_counter.labels(risk_level=result.risk_level).inc()
            model_confidence.observe(result.churn_probability)
            
            return result
        except Exception as e:
            api_errors.labels(endpoint=func.__name__, error_type=type(e).__name__).inc()
            raise
    
    return wrapper