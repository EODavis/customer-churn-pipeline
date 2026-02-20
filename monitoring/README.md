# Monitoring & Alerting

## Components

### 1. Metrics (Prometheus)
- Prediction latency
- Request rate
- Error rate
- Model confidence distribution
- Data drift score

**View**: http://localhost:9090

### 2. Dashboard (Grafana)
- Real-time metrics visualization
- Alerts configuration
- Historical trends

**Access**: http://localhost:3000 (admin/admin)

### 3. Drift Detection
```bash
python monitoring/drift_detector.py \
  --reference data/raw/customer_data.csv \
  --current data/predictions/predictions_2025-02-20.jsonl
```

### 4. Automated Retraining
```bash
python monitoring/retrain_trigger.py
```

**Triggers:**
- Data drift > 0.15
- Min 1000 new samples
- Min 7 days since last retrain

### 5. Alerts
- Email alerts for drift/performance issues
- Slack notifications
- Log file: `monitoring/alerts.log`

## Configuration

Set environment variables:
```bash
export SMTP_SERVER=smtp.gmail.com
export ALERT_EMAIL=your-email@example.com
export ALERT_PASSWORD=your-password
export RECIPIENT_EMAIL=team@example.com
```