import requests
from typing import List, Dict, Optional

class ChurnPredictionClient:
    """Python client for Churn Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict(self, customer_data: Dict) -> Dict:
        """Predict churn for single customer"""
        response = self.session.post(
            f"{self.base_url}/predict",
            json=customer_data
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, customers: List[Dict]) -> Dict:
        """Predict churn for multiple customers"""
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json={"customers": customers}
        )
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get model metadata"""
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()

# Usage example
if __name__ == "__main__":
    client = ChurnPredictionClient()
    
    # Health check
    print("Health:", client.health_check())
    
    # Single prediction
    customer = {
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
    
    result = client.predict(customer)
    print(f"\nPrediction for customer {result['customer_id']}:")
    print(f"  Churn probability: {result['churn_probability']:.2%}")
    print(f"  Risk level: {result['risk_level']}")