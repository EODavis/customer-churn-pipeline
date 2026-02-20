import pandas as pd
import json
from datetime import datetime, timedelta
import os

class PredictionLogger:
    """Log predictions for drift monitoring"""
    
    def __init__(self, log_path: str = 'data/predictions/'):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.current_date = datetime.now().date()
        self.current_batch = []
    
    def log_prediction(self, customer_data: dict, prediction: dict):
        """Log a single prediction"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'customer_id': customer_data['customer_id'],
            **{k: v for k, v in customer_data.items() if k != 'customer_id'},
            'prediction': prediction['churn_prediction'],
            'probability': prediction['churn_probability'],
            'risk_level': prediction['risk_level']
        }
        
        self.current_batch.append(log_entry)
        
        # Flush to disk every 100 predictions
        if len(self.current_batch) >= 100:
            self.flush()
    
    def flush(self):
        """Write batch to disk"""
        if not self.current_batch:
            return
        
        filename = f"{self.log_path}/predictions_{self.current_date}.jsonl"
        with open(filename, 'a') as f:
            for entry in self.current_batch:
                f.write(json.dumps(entry) + '\n')
        
        self.current_batch = []
    
    def get_daily_predictions(self, date=None) -> pd.DataFrame:
        """Load predictions for a specific date"""
        if date is None:
            date = datetime.now().date()
        
        filename = f"{self.log_path}/predictions_{date}.jsonl"
        if not os.path.exists(filename):
            return pd.DataFrame()
        
        data = []
        with open(filename, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        return pd.DataFrame(data)

# Usage in API - update api/main.py
from monitoring.collect_predictions import PredictionLogger

prediction_logger = PredictionLogger()

@app.post("/predict", response_model=PredictionResponse)
@track_prediction_metrics
async def predict_churn(customer: CustomerFeatures):
    # ... existing prediction code ...
    
    # Log prediction
    prediction_logger.log_prediction(
        customer_data=customer.dict(),
        prediction=result.dict()
    )
    
    return result