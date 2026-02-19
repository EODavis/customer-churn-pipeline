import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data(n_customers=50000, output_path='data/raw/customer_data.csv'):
    """Generate synthetic customer churn data"""
    np.random.seed(42)

    data = {
        'customer_id': range(1, n_customers + 1),
        'account_age_days': np.random.randint(30, 1825, n_customers),
        'monthly_charges': np.random.uniform(20, 150, n_customers),
        'total_charges': np.random.uniform(100, 8000, n_customers),
        'support_tickets': np.random.poisson(2, n_customers),
        'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_customers, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check'], n_customers, p=[0.4, 0.3, 0.3]),
        'monthly_usage_gb': np.random.uniform(5, 500, n_customers),
        'num_services': np.random.randint(1, 6, n_customers),
    }

    # Create churn target
    churn_probability = (
        0.1 + 
        (data['monthly_charges'] / 150) * 0.2 +
        (data['support_tickets'] / 10) * 0.3 +
        (np.array(data['contract_type']) == 'Month-to-Month') * 0.25
    )
    churn_probability = np.clip(churn_probability, 0, 0.8)
    data['churned'] = np.random.binomial(1, churn_probability)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records. Churn rate: {df['churned'].mean():.2%}")
    return df

if __name__ == "__main__":
    generate_data()
