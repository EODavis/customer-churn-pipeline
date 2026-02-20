import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000"

def single_prediction_latency():
    """Measure single prediction latency"""
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
    
    start = time.time()
    response = requests.post(f"{API_URL}/predict", json=payload)
    latency = (time.time() - start) * 1000  # ms
    
    return latency if response.status_code == 200 else None

def run_benchmark(n_requests=100):
    """Run performance benchmark"""
    print(f"Running {n_requests} requests...")
    
    latencies = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(lambda _: single_prediction_latency(), range(n_requests))
        latencies = [r for r in results if r is not None]
    
    if latencies:
        print(f"\n=== Performance Metrics ===")
        print(f"Total requests: {len(latencies)}")
        print(f"Mean latency: {statistics.mean(latencies):.2f}ms")
        print(f"Median latency: {statistics.median(latencies):.2f}ms")
        print(f"P95 latency: {statistics.quantiles(latencies, n=20)[18]:.2f}ms")
        print(f"P99 latency: {statistics.quantiles(latencies, n=100)[98]:.2f}ms")
        print(f"Min latency: {min(latencies):.2f}ms")
        print(f"Max latency: {max(latencies):.2f}ms")

if __name__ == "__main__":
    # Check health
    health = requests.get(f"{API_URL}/health")
    if health.status_code != 200:
        print("API not healthy!")
        exit(1)
    
    run_benchmark(100)