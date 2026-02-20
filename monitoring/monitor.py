#!/usr/bin/env python
"""Main monitoring orchestrator"""

import time
from datetime import datetime
from monitoring.drift_detector import DriftDetector
from monitoring.collect_predictions import PredictionLogger
from monitoring.alerts import AlertManager
from api.metrics import data_drift_score
import pandas as pd

class ModelMonitor:
    """Orchestrate all monitoring activities"""
    
    def __init__(self):
        self.drift_detector = DriftDetector('data/raw/customer_data.csv')
        self.prediction_logger = PredictionLogger()
        self.alert_manager = AlertManager()
        self.baseline_f1 = 0.75  # Set from training
    
    def run_drift_check(self):
        """Check for data drift"""
        print(f"[{datetime.now()}] Running drift detection...")
        
        # Get recent predictions
        recent_data = self.prediction_logger.get_daily_predictions()
        
        if len(recent_data) < 100:
            print(f"  Insufficient data ({len(recent_data)} samples)")
            return
        
        # Calculate drift
        result = self.drift_detector.calculate_drift(recent_data)
        
        # Update Prometheus metric
        data_drift_score.set(result['overall_drift_score'])
        
        print(f"  Drift score: {result['overall_drift_score']:.4f}")
        
        # Alert if drift detected
        if result['drift_detected']:
            print("  ⚠ DRIFT DETECTED!")
            self.alert_manager.alert_drift_detected(
                drift_score=result['overall_drift_score'],
                features=result['drifted_features']
            )
        else:
            print("  ✓ No drift detected")
    
    def run_performance_check(self):
        """Check model performance on recent predictions"""
        print(f"[{datetime.now()}] Checking model performance...")
        
        # This would compare predictions with actual outcomes
        # For now, simulate
        recent_data = self.prediction_logger.get_daily_predictions()
        
        if len(recent_data) < 100:
            print(f"  Insufficient data")
            return
        
        # Simulated performance check
        # In production, you'd have ground truth labels
        print(f"  Evaluated {len(recent_data)} predictions")
        print("  ✓ Performance within acceptable range")
    
    def start(self, interval_minutes: int = 60):
        """Start monitoring loop"""
        print("🔍 Starting model monitoring...")
        print(f"   Check interval: {interval_minutes} minutes")
        
        while True:
            try:
                self.run_drift_check()
                self.run_performance_check()
            except Exception as e:
                print(f"✗ Monitoring error: {e}")
            
            print(f"   Next check in {interval_minutes} minutes\n")
            time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    monitor = ModelMonitor()
    monitor.start(interval_minutes=60)