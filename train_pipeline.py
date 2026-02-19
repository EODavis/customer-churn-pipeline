import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime
import os

def load_and_preprocess():
    df = pd.read_csv('data/raw/customer_data.csv')
    
    # Encode categoricals
    le_contract = LabelEncoder()
    le_payment = LabelEncoder()
    df['contract_type_encoded'] = le_contract.fit_transform(df['contract_type'])
    df['payment_method_encoded'] = le_payment.fit_transform(df['payment_method'])
    
    # Save encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(le_contract, 'models/contract_encoder.pkl')
    joblib.dump(le_payment, 'models/payment_encoder.pkl')
    
    feature_cols = ['account_age_days', 'monthly_charges', 'total_charges', 
                    'support_tickets', 'monthly_usage_gb', 'num_services',
                    'contract_type_encoded', 'payment_method_encoded']
    
    X = df[feature_cols]
    y = df['churned']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, X_test, y_test):
    # Check if MLflow is configured
    use_mlflow = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db') != ''
    
    if use_mlflow:
        mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))
        mlflow.set_experiment("customer-churn")
        mlflow.start_run()
    
    # Log parameters
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    }
    
    if use_mlflow:
        mlflow.log_params(params)
    
    # Train
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    if use_mlflow:
        mlflow.log_metrics(metrics)
    
    # Save model
    model_path = f"models/churn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(model, model_path)
    
    if use_mlflow:
        mlflow.log_artifact(model_path)
        mlflow.end_run()
    
    print(f"Model trained. F1 Score: {metrics['f1']:.4f}")
    return model, metrics

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    model, metrics = train_model(X_train, y_train, X_test, y_test)
