# Pipeline Architecture

## Components

### 1. Data Generation (`generate_data.py`)
- Synthetic customer data
- 50,000 records
- Realistic churn patterns

### 2. Data Validation (`validate_data.py`)
- Great Expectations checks
- Schema validation
- Data quality gates

### 3. Training Pipeline (`train_pipeline.py`)
- Preprocessing
- Model training (Random Forest)
- Metric logging to MLflow

### 4. Orchestration (`orchestrate.py`)
- Prefect workflow
- Task dependencies
- Error handling

### 5. Model Registry (`model_registry.py`)
- Version tracking
- Production promotion
- Rollback capability

## CI/CD Flow
```
Push to GitHub
    ↓
Run Tests (pytest)
    ↓
Validate Data
    ↓
Train Model
    ↓
Register Model Version
    ↓
Build Docker Image
    ↓
Upload Artifacts
```

## Deployment
```bash
# Local
docker-compose up

# Production (coming Week 3)
- FastAPI serving
- Kubernetes deployment
- Auto-scaling
```
```

### Create `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# MLflow
mlruns/
mlflow.db

# Models
models/*.pkl
!models/.gitkeep

# Data
data/raw/*.csv
data/processed/*.csv
!data/raw/.gitkeep

# Docker
*.tar.gz

# Tests
.pytest_cache/
.coverage
coverage.xml

# IDE
.vscode/
.idea/