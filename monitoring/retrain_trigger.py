import schedule
import time
from datetime import datetime, timedelta
import pandas as pd
import subprocess
import os
from monitoring.drift_detector import DriftDetector
from monitoring.collect_predictions import PredictionLogger

class RetrainTrigger:
    """Automated model retraining trigger"""
    
    def __init__(
        self,
        drift_threshold: float = 0.15,
        min_days_between_retrains: int = 7,
        min_new_samples: int = 1000
    ):
        self.drift_threshold = drift_threshold
        self.min_days_between_retrains = min_days_between_retrains
        self.min_new_samples = min_new_samples
        self.last_retrain_date = None
        self.prediction_logger = PredictionLogger()
    
    def should_retrain(self) -> tuple[bool, str]:
        """Determine if model should be retrained"""
        
        # Check 1: Minimum time between retrains
        if self.last_retrain_date:
            days_since_retrain = (datetime.now().date() - self.last_retrain_date).days
            if days_since_retrain < self.min_days_between_retrains:
                return False, f"Last retrain was {days_since_retrain} days ago (min: {self.min_days_between_retrains})"
        
        # Check 2: Sufficient new data
        recent_data = self.prediction_logger.get_daily_predictions()
        if len(recent_data) < self.min_new_samples:
            return False, f"Only {len(recent_data)} new samples (min: {self.min_new_samples})"
        
        # Check 3: Data drift
        detector = DriftDetector('data/raw/customer_data.csv', threshold=0.05)
        drift_result = detector.calculate_drift(recent_data)
        
        if drift_result['overall_drift_score'] > self.drift_threshold:
            return True, f"Data drift detected: {drift_result['overall_drift_score']:.4f} > {self.drift_threshold}"
        
        return False, f"No drift detected: {drift_result['overall_drift_score']:.4f}"
    
    def trigger_retrain(self):
        """Execute retraining pipeline"""
        print(f"\n{'='*60}")
        print(f"[{datetime.now()}] Checking retrain conditions...")
        
        should_retrain, reason = self.should_retrain()
        print(f"Decision: {'RETRAIN' if should_retrain else 'SKIP'}")
        print(f"Reason: {reason}")
        
        if should_retrain:
            print("\nStarting retraining pipeline...")
            try:
                # Run training pipeline
                subprocess.run(['python', 'train_pipeline.py'], check=True)
                
                # Update last retrain date
                self.last_retrain_date = datetime.now().date()
                
                print("✓ Retraining completed successfully")
                
                # Send alert (placeholder)
                self.send_alert("Model retrained successfully", reason)
            
            except Exception as e:
                print(f"✗ Retraining failed: {e}")
                self.send_alert("Model retraining FAILED", str(e))
        
        print(f"{'='*60}\n")
    
    def send_alert(self, subject: str, message: str):
        """Send alert (placeholder for email/slack integration)"""
        alert_log = 'monitoring/alerts.log'
        os.makedirs('monitoring', exist_ok=True)
        
        with open(alert_log, 'a') as f:
            f.write(f"[{datetime.now()}] {subject}: {message}\n")
        
        print(f"📧 Alert logged: {subject}")
    
    def start_monitoring(self):
        """Start scheduled monitoring"""
        # Check daily at 2 AM
        schedule.every().day.at("02:00").do(self.trigger_retrain)
        
        # Also check every 6 hours
        schedule.every(6).hours.do(self.trigger_retrain)
        
        print("🔄 Retrain monitoring started")
        print("  - Daily check: 02:00 AM")
        print("  - Periodic check: Every 6 hours")
        
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    trigger = RetrainTrigger(
        drift_threshold=0.15,
        min_days_between_retrains=7,
        min_new_samples=1000
    )
    
    # Run immediate check
    trigger.trigger_retrain()
    
    # Start scheduled monitoring
    # trigger.start_monitoring()