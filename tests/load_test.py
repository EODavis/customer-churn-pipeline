from locust import HttpUser, task, between
import random

class ChurnPredictionUser(HttpUser):
    wait_time = between(1, 3)
    
    contract_types = ['Month-to-Month', 'One Year', 'Two Year']
    payment_methods = ['Credit Card', 'Bank Transfer', 'Electronic Check']
    
    @task(3)
    def predict_single(self):
        """Test single prediction endpoint"""
        payload = {
            "customer_id": random.randint(1, 100000),
            "account_age_days": random.randint(30, 1825),
            "monthly_charges": round(random.uniform(20, 150), 2),
            "total_charges": round(random.uniform(100, 8000), 2),
            "support_tickets": random.randint(0, 10),
            "contract_type": random.choice(self.contract_types),
            "payment_method": random.choice(self.payment_methods),
            "monthly_usage_gb": round(random.uniform(5, 500), 2),
            "num_services": random.randint(1, 6)
        }
        
        self.client.post("/predict", json=payload)
    
    @task(1)
    def predict_batch(self):
        """Test batch prediction endpoint"""
        customers = []
        for _ in range(random.randint(5, 20)):
            customers.append({
                "customer_id": random.randint(1, 100000),
                "account_age_days": random.randint(30, 1825),
                "monthly_charges": round(random.uniform(20, 150), 2),
                "total_charges": round(random.uniform(100, 8000), 2),
                "support_tickets": random.randint(0, 10),
                "contract_type": random.choice(self.contract_types),
                "payment_method": random.choice(self.payment_methods),
                "monthly_usage_gb": round(random.uniform(5, 500), 2),
                "num_services": random.randint(1, 6)
            })
        
        self.client.post("/predict/batch", json={"customers": customers})
    
    @task(1)
    def health_check(self):
        """Test health endpoint"""
        self.client.get("/health")
    
    @task(1)
    def model_info(self):
        """Test model info endpoint"""
        self.client.get("/model/info")