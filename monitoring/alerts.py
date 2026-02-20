import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime

class AlertManager:
    """Manage alerts for model performance and drift"""
    
    def __init__(self):
        self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.environ.get('SMTP_PORT', '587'))
        self.sender_email = os.environ.get('ALERT_EMAIL', 'alerts@example.com')
        self.sender_password = os.environ.get('ALERT_PASSWORD', '')
        self.recipient_email = os.environ.get('RECIPIENT_EMAIL', 'mlops-team@example.com')
    
    def send_email_alert(self, subject: str, body: str):
        """Send email alert"""
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = self.recipient_email
        msg['Subject'] = f"[Churn Model Alert] {subject}"
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            print(f"✓ Alert sent: {subject}")
        except Exception as e:
            print(f"✗ Failed to send alert: {e}")
    
    def send_slack_alert(self, message: str):
        """Send Slack alert (placeholder)"""
        # Implement Slack webhook integration
        print(f"📢 Slack alert: {message}")
    
    def alert_drift_detected(self, drift_score: float, features: list):
        """Alert when data drift is detected"""
        subject = "Data Drift Detected"
        body = f"""
Data drift detected in churn prediction model!

Drift Score: {drift_score:.4f}
Drifted Features: {', '.join(features)}
Timestamp: {datetime.now()}

Action Required:
- Review data quality
- Consider model retraining
- Check feature engineering pipeline

View dashboard: http://localhost:3000
"""
        self.send_email_alert(subject, body)
    
    def alert_performance_degradation(self, current_f1: float, baseline_f1: float):
        """Alert when model performance degrades"""
        subject = "Model Performance Degradation"
        body = f"""
Model performance has degraded!

Current F1 Score: {current_f1:.4f}
Baseline F1 Score: {baseline_f1:.4f}
Degradation: {((baseline_f1 - current_f1) / baseline_f1 * 100):.2f}%

Recommended Actions:
- Trigger model retraining
- Investigate data quality issues
- Review recent feature changes
"""
        self.send_email_alert(subject, body)
    
    def alert_retrain_complete(self, old_f1: float, new_f1: float):
        """Alert when retraining completes"""
        improvement = ((new_f1 - old_f1) / old_f1 * 100)
        subject = "Model Retraining Complete"
        body = f"""
Model retraining completed successfully!

Previous F1 Score: {old_f1:.4f}
New F1 Score: {new_f1:.4f}
Improvement: {improvement:+.2f}%

Model has been deployed to production.
"""
        self.send_email_alert(subject, body)