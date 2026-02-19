import mlflow
import joblib
from datetime import datetime
import json
import os

class ModelRegistry:
    def __init__(self, registry_path='models/registry.json'):
        self.registry_path = registry_path
        self.registry = self._load_registry()
    
    def _load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {'models': [], 'production': None}
    
    def _save_registry(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_path, metrics, metadata=None):
        """Register a new model version"""
        version = len(self.registry['models']) + 1
        
        model_info = {
            'version': version,
            'path': model_path,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'status': 'staging'
        }
        
        self.registry['models'].append(model_info)
        self._save_registry()
        
        print(f"✓ Registered model v{version}")
        print(f"  F1: {metrics['f1']:.4f}")
        return version
    
    def promote_to_production(self, version):
        """Promote a model version to production"""
        model = next((m for m in self.registry['models'] if m['version'] == version), None)
        
        if not model:
            raise ValueError(f"Model version {version} not found")
        
        # Demote current production model
        if self.registry['production']:
            old_prod = next(m for m in self.registry['models'] 
                          if m['version'] == self.registry['production'])
            old_prod['status'] = 'archived'
        
        # Promote new model
        model['status'] = 'production'
        self.registry['production'] = version
        self._save_registry()
        
        print(f"✓ Promoted v{version} to production")
    
    def get_production_model(self):
        """Get current production model"""
        if not self.registry['production']:
            return None
        
        return next(m for m in self.registry['models'] 
                   if m['version'] == self.registry['production'])
    
    def list_models(self):
        """List all registered models"""
        print("\n=== Model Registry ===")
        for model in self.registry['models']:
            prod_marker = " [PRODUCTION]" if model['version'] == self.registry['production'] else ""
            print(f"v{model['version']}{prod_marker} - F1: {model['metrics']['f1']:.4f} - {model['status']}")

# Usage example
if __name__ == "__main__":
    registry = ModelRegistry()
    registry.list_models()