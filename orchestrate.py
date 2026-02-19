from prefect import flow, task
import subprocess

@task
def validate_data():
    subprocess.run(['python', 'validate_data.py'], check=True)
    
@task
def train_model():
    subprocess.run(['python', 'train_pipeline.py'], check=True)
    
@flow
def ml_pipeline():
    validate_data()
    train_model()

if __name__ == "__main__":
    ml_pipeline()
