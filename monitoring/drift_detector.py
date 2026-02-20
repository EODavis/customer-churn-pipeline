import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, List
import json
from datetime import datetime
import os

class DriftDetector:
    """Detect data drift using statistical tests"""
    
    def __init__(self, reference_data_path: str, threshold: float = 0.05):
        self.threshold = threshold
        self.reference_data = pd.read_csv(reference_data_path)
        self.feature_cols = [
            'account_age_days', 'monthly_charges', 'total_charges',
            'support_tickets', 'monthly_usage_gb', 'num_services'
        ]
        self.drift_history = []
    
    def calculate_drift(self, current_data: pd.DataFrame) -> Dict:
        """Calculate drift score using Kolmogorov-Smirnov test"""
        drift_scores = {}
        
        for col in self.feature_cols:
            if col in current_data.columns:
                # KS test
                statistic, p_value = ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                drift_scores[col] = {
                    'ks_statistic': float(statistic),
                    'p_value': float(p_value),
                    'drift_detected': p_value < self.threshold
                }
        
        # Overall drift score (average of KS statistics)
        overall_drift = np.mean([s['ks_statistic'] for s in drift_scores.values()])
        
        # Count drifted features
        drifted_features = [col for col, s in drift_scores.items() if s['drift_detected']]
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'overall_drift_score': float(overall_drift),
            'threshold': self.threshold,
            'features': drift_scores,
            'drifted_features': drifted_features,
            'drift_detected': len(drifted_features) > 0,
            'n_samples': len(current_data)
        }
        
        self.drift_history.append(result)
        return result
    
    def save_drift_report(self, output_path: str = 'monitoring/drift_report.json'):
        """Save drift detection report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.drift_history, f, indent=2)
        print(f"✓ Drift report saved to {output_path}")
    
    def get_latest_drift_score(self) -> float:
        """Get most recent drift score"""
        if self.drift_history:
            return self.drift_history[-1]['overall_drift_score']
        return 0.0

# Standalone drift detection script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', default='data/raw/customer_data.csv')
    parser.add_argument('--current', default='data/raw/customer_data_new.csv')
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()
    
    detector = DriftDetector(args.reference, args.threshold)
    current = pd.read_csv(args.current)
    
    result = detector.calculate_drift(current)
    detector.save_drift_report()
    
    print(f"\n=== Drift Detection Report ===")
    print(f"Overall drift score: {result['overall_drift_score']:.4f}")
    print(f"Drift detected: {result['drift_detected']}")
    if result['drifted_features']:
        print(f"Drifted features: {', '.join(result['drifted_features'])}")