"""
Model Management Utilities for the ML API.

This module provides utilities for:
1. Loading and validating models
2. Model versioning and deployment
3. Model performance monitoring
4. A/B testing support
"""

import os
import json
import logging
import shutil
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

class ModelManager:
    """Comprehensive model management system."""
    
    def __init__(self, models_base_dir: str = "models"):
        """Initialize the model manager.
        
        Args:
            models_base_dir: Base directory for storing models
        """
        self.models_base_dir = models_base_dir
        self.models_registry = {}
        self.deployment_history = []
        
        # Create directories if they don't exist
        os.makedirs(models_base_dir, exist_ok=True)
        os.makedirs(os.path.join(models_base_dir, "active"), exist_ok=True)
        os.makedirs(os.path.join(models_base_dir, "archive"), exist_ok=True)
        os.makedirs(os.path.join(models_base_dir, "staging"), exist_ok=True)
        
        self._load_registry()
    
    def _load_registry(self):
        """Load the models registry from disk."""
        registry_path = os.path.join(self.models_base_dir, "registry.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    self.models_registry = json.load(f)
                logger.info(f"Loaded {len(self.models_registry)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.models_registry = {}
    
    def _save_registry(self):
        """Save the models registry to disk."""
        registry_path = os.path.join(self.models_base_dir, "registry.json")
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.models_registry, f, indent=2, default=str)
            logger.info("Models registry saved")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_model(self, model_name: str, model_path: str, metadata: Dict, 
                      version: str = None) -> bool:
        """Register a new model in the system.
        
        Args:
            model_name: Unique name for the model
            model_path: Path to the model file
            metadata: Model metadata including performance metrics
            version: Version string (auto-generated if None)
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Validate model can be loaded
            model = joblib.load(model_path)
            
            # Create model entry
            model_entry = {
                "model_name": model_name,
                "version": version,
                "model_path": model_path,
                "metadata": metadata,
                "registered_at": datetime.now().isoformat(),
                "status": "registered",
                "performance_history": [],
                "deployment_count": 0
            }
            
            # Store in registry
            model_key = f"{model_name}:{version}"
            self.models_registry[model_key] = model_entry
            
            # Save registry
            self._save_registry()
            
            logger.info(f"Model registered: {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            return False
    
    def deploy_model(self, model_name: str, version: str = None, 
                    deployment_type: str = "active") -> bool:
        """Deploy a model to active, staging, or archive.
        
        Args:
            model_name: Name of the model to deploy
            version: Version to deploy (latest if None)
            deployment_type: "active", "staging", or "archive"
            
        Returns:
            True if deployment successful, False otherwise
        """
        try:
            # Find model in registry
            if version is None:
                # Find latest version
                versions = [key for key in self.models_registry.keys() 
                           if key.startswith(f"{model_name}:")]
                if not versions:
                    logger.error(f"No versions found for model {model_name}")
                    return False
                model_key = sorted(versions)[-1]  # Latest version
            else:
                model_key = f"{model_name}:{version}"
            
            if model_key not in self.models_registry:
                logger.error(f"Model not found in registry: {model_key}")
                return False
            
            model_entry = self.models_registry[model_key]
            
            # Copy model to deployment directory
            target_dir = os.path.join(self.models_base_dir, deployment_type)
            target_path = os.path.join(target_dir, f"{model_name}_model.joblib")
            
            shutil.copy2(model_entry["model_path"], target_path)
            
            # Create deployment metadata
            deployment_metadata = {
                **model_entry["metadata"],
                "deployed_at": datetime.now().isoformat(),
                "deployment_type": deployment_type,
                "version": model_entry["version"]
            }
            
            metadata_path = os.path.join(target_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(deployment_metadata, f, indent=2)
            
            # Update registry
            model_entry["status"] = f"deployed_{deployment_type}"
            model_entry["deployment_count"] += 1
            model_entry["last_deployed"] = datetime.now().isoformat()
            
            # Record deployment history
            deployment_record = {
                "model_key": model_key,
                "deployment_type": deployment_type,
                "deployed_at": datetime.now().isoformat(),
                "target_path": target_path
            }
            self.deployment_history.append(deployment_record)
            
            self._save_registry()
            
            logger.info(f"Model deployed: {model_key} to {deployment_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_name}: {e}")
            return False
    
    def validate_model(self, model_name: str, test_data_path: str = None) -> Dict:
        """Validate a model's performance.
        
        Args:
            model_name: Name of the model to validate
            test_data_path: Path to test data (uses sample data if None)
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Load model from active deployment
            model_path = os.path.join(self.models_base_dir, "active", f"{model_name}_model.joblib")
            metadata_path = os.path.join(self.models_base_dir, "active", f"{model_name}_metadata.json")
            
            if not os.path.exists(model_path):
                return {"error": f"Model {model_name} not found in active deployment"}
            
            model = joblib.load(model_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load test data
            if test_data_path and os.path.exists(test_data_path):
                test_data = pd.read_csv(test_data_path)
            else:
                # Create sample data for validation
                test_data = self._create_sample_data()
            
            # Prepare features similar to training
            X_test, y_test = self._prepare_test_features(test_data, metadata)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            
            validation_results = {
                "model_name": model_name,
                "validation_timestamp": datetime.now().isoformat(),
                "test_samples": len(y_test),
                "metrics": {
                    "r2_score": r2,
                    "rmse": rmse,
                    "mae": mae
                },
                "training_metrics": metadata.get("performance", {}),
                "performance_degradation": {
                    "r2_change": metadata.get("performance", {}).get("r2", 0) - r2,
                    "rmse_change": rmse - metadata.get("performance", {}).get("rmse", 0)
                },
                "status": "passed" if r2 > 0.1 else "failed"
            }
            
            # Update model registry with validation results
            for model_key, model_entry in self.models_registry.items():
                if model_entry["model_name"] == model_name and model_entry["status"].startswith("deployed"):
                    model_entry["performance_history"].append(validation_results)
                    break
            
            self._save_registry()
            
            logger.info(f"Model validation completed: {model_name}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Model validation failed for {model_name}: {e}")
            return {"error": str(e)}
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for model validation."""
        np.random.seed(42)
        n_samples = 50
        
        return pd.DataFrame({
            'City Name': np.random.choice(['BOSTON', 'NEW YORK', 'CHICAGO'], n_samples),
            'Package': np.random.choice(['bushel cartons', '1/2 bushel cartons'], n_samples),
            'Variety': np.random.choice(['PIE TYPE', 'CARVING'], n_samples),
            'Date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            'Low Price': np.random.uniform(10, 20, n_samples),
            'High Price': np.random.uniform(20, 30, n_samples)
        })
    
    def _prepare_test_features(self, test_data: pd.DataFrame, metadata: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare test features in the same format as training data."""
        # Basic processing
        test_data['Date'] = pd.to_datetime(test_data['Date'])
        test_data['Month'] = test_data['Date'].dt.month
        test_data['DayOfYear'] = test_data['Date'].dt.dayofyear
        test_data['Price'] = (test_data['Low Price'] + test_data['High Price']) / 2
        
        feature_names = metadata.get('feature_names', ['DayOfYear', 'Month'])
        
        # Create feature matrix
        X = pd.DataFrame()
        
        for feature_name in feature_names:
            if feature_name == 'DayOfYear':
                X[feature_name] = test_data['DayOfYear']
            elif feature_name == 'Month':
                X[feature_name] = test_data['Month']
            elif feature_name.startswith('Variety_'):
                variety = feature_name.replace('Variety_', '')
                X[feature_name] = (test_data['Variety'] == variety).astype(int)
            elif feature_name.startswith('City_'):
                city = feature_name.replace('City_', '')
                X[feature_name] = (test_data['City Name'] == city).astype(int)
            elif feature_name.startswith('Package_'):
                package = feature_name.replace('Package_', '')
                X[feature_name] = (test_data['Package'] == package).astype(int)
            else:
                X[feature_name] = 0  # Default for unknown features
        
        y = test_data['Price']
        
        return X, y
    
    def list_models(self, status_filter: str = None) -> List[Dict]:
        """List all models in the registry.
        
        Args:
            status_filter: Filter by status ("registered", "deployed_active", etc.)
            
        Returns:
            List of model information dictionaries
        """
        models_list = []
        
        for model_key, model_entry in self.models_registry.items():
            if status_filter is None or model_entry["status"] == status_filter:
                models_list.append({
                    "model_key": model_key,
                    "model_name": model_entry["model_name"],
                    "version": model_entry["version"],
                    "status": model_entry["status"],
                    "registered_at": model_entry["registered_at"],
                    "last_deployed": model_entry.get("last_deployed"),
                    "deployment_count": model_entry["deployment_count"],
                    "performance_metrics": model_entry["metadata"].get("performance", {})
                })
        
        return sorted(models_list, key=lambda x: x["registered_at"], reverse=True)
    
    def get_deployment_history(self) -> List[Dict]:
        """Get deployment history."""
        return self.deployment_history
    
    def rollback_model(self, model_name: str, target_version: str = None) -> bool:
        """Rollback a model to a previous version.
        
        Args:
            model_name: Name of the model to rollback
            target_version: Version to rollback to (previous if None)
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            # Find deployment history for this model
            model_deployments = [d for d in self.deployment_history 
                               if d["model_key"].startswith(f"{model_name}:")]
            
            if len(model_deployments) < 2:
                logger.error(f"Not enough deployment history for rollback: {model_name}")
                return False
            
            if target_version is None:
                # Rollback to previous version
                target_deployment = model_deployments[-2]
            else:
                # Find specific version
                target_deployment = None
                for deployment in model_deployments:
                    if deployment["model_key"].endswith(f":{target_version}"):
                        target_deployment = deployment
                        break
                
                if target_deployment is None:
                    logger.error(f"Target version not found: {target_version}")
                    return False
            
            # Deploy the target version
            model_key_parts = target_deployment["model_key"].split(":")
            return self.deploy_model(model_key_parts[0], model_key_parts[1], "active")
            
        except Exception as e:
            logger.error(f"Rollback failed for {model_name}: {e}")
            return False


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Management CLI")
    parser.add_argument("command", choices=["register", "deploy", "validate", "list", "rollback"],
                       help="Command to execute")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--model-path", help="Path to model file (for register)")
    parser.add_argument("--version", help="Model version")
    parser.add_argument("--deployment-type", default="active", 
                       choices=["active", "staging", "archive"], help="Deployment type")
    parser.add_argument("--test-data", help="Path to test data for validation")
    parser.add_argument("--models-dir", default="models", help="Models base directory")
    
    args = parser.parse_args()
    
    # Initialize model manager
    manager = ModelManager(args.models_dir)
    
    if args.command == "register":
        if not args.model_path:
            print("Error: --model-path required for register command")
            return
        
        # Load metadata (simplified for CLI)
        metadata = {"performance": {"r2": 0.5, "rmse": 2.0}}
        success = manager.register_model(args.model_name, args.model_path, metadata, args.version)
        print(f"Registration {'successful' if success else 'failed'}")
    
    elif args.command == "deploy":
        success = manager.deploy_model(args.model_name, args.version, args.deployment_type)
        print(f"Deployment {'successful' if success else 'failed'}")
    
    elif args.command == "validate":
        results = manager.validate_model(args.model_name, args.test_data)
        print(json.dumps(results, indent=2))
    
    elif args.command == "list":
        models = manager.list_models()
        for model in models:
            print(f"{model['model_key']}: {model['status']} (deployed {model['deployment_count']} times)")
    
    elif args.command == "rollback":
        success = manager.rollback_model(args.model_name, args.version)
        print(f"Rollback {'successful' if success else 'failed'}")


if __name__ == "__main__":
    main()