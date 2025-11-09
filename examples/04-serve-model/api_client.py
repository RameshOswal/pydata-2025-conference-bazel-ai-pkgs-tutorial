"""
API Client for testing the ML Model Serving API.

This client provides:
1. Easy testing of prediction endpoints
2. Batch prediction testing
3. Performance benchmarking
4. Health monitoring integration
"""

import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from pydantic import BaseModel


class PumpkinFeatures(BaseModel):
    """Input features for pumpkin price prediction."""
    day_of_year: int
    month: int
    variety: str
    city: str
    package: str


class APIClient:
    """Client for the ML Model Serving API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", 
                 monitor_url: str = "http://localhost:8001"):
        """Initialize the API client.
        
        Args:
            base_url: Base URL of the model serving API
            monitor_url: Base URL of the health monitoring API
        """
        self.base_url = base_url.rstrip('/')
        self.monitor_url = monitor_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        try:
            response = self.session.get(f"{self.base_url}/models/{model_name}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def activate_model(self, model_name: str) -> Dict[str, Any]:
        """Activate a specific model."""
        try:
            response = self.session.post(f"{self.base_url}/models/{model_name}/activate", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_single(self, features: PumpkinFeatures, model_name: str = None) -> Dict[str, Any]:
        """Make a single prediction."""
        try:
            start_time = time.time()
            
            params = {"model_name": model_name} if model_name else {}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=features.dict(),
                params=params,
                timeout=30
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            response.raise_for_status()
            result = response.json()
            result["client_response_time_ms"] = response_time_ms
            
            # Log to health monitor if available
            self._log_to_monitor(model_name or "default", response_time_ms, True)
            
            return result
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            self._log_to_monitor(model_name or "default", response_time_ms, False, str(e))
            return {"error": str(e), "response_time_ms": response_time_ms}
    
    def predict_batch(self, samples: List[PumpkinFeatures], model_name: str = None) -> Dict[str, Any]:
        """Make batch predictions."""
        try:
            start_time = time.time()
            
            request_data = {
                "samples": [sample.dict() for sample in samples],
                "model_name": model_name
            }
            
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=request_data,
                timeout=60
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            response.raise_for_status()
            result = response.json()
            result["client_response_time_ms"] = response_time_ms
            
            # Log to health monitor
            self._log_to_monitor(model_name or "default", response_time_ms, True)
            
            return result
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            self._log_to_monitor(model_name or "default", response_time_ms, False, str(e))
            return {"error": str(e), "response_time_ms": response_time_ms}
    
    def get_sample_prediction(self) -> Dict[str, Any]:
        """Get a sample prediction using default values."""
        try:
            response = self.session.get(f"{self.base_url}/predict/sample", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def _log_to_monitor(self, model_name: str, response_time_ms: float, 
                       success: bool, error_msg: str = None):
        """Log prediction to health monitor."""
        try:
            self.session.post(
                f"{self.monitor_url}/log-prediction",
                params={
                    "model_name": model_name,
                    "response_time_ms": response_time_ms,
                    "success": success,
                    "error_msg": error_msg
                },
                timeout=5
            )
        except:
            pass  # Ignore monitoring errors
    
    def benchmark_api(self, num_requests: int = 100, concurrent: bool = False) -> Dict[str, Any]:
        """Benchmark API performance."""
        print(f"Starting benchmark with {num_requests} requests...")
        
        # Create sample features
        sample_features = PumpkinFeatures(
            day_of_year=280,
            month=10,
            variety="PIE TYPE",
            city="BOSTON",
            package="bushel cartons"
        )
        
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        response_times = []
        errors = []
        
        if concurrent:
            # Concurrent requests (simplified - using threads would be better)
            results = []
            for i in range(num_requests):
                result = self.predict_single(sample_features)
                results.append(result)
                
                if "error" in result:
                    failed_requests += 1
                    errors.append(result["error"])
                else:
                    successful_requests += 1
                    response_times.append(result.get("client_response_time_ms", 0))
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{num_requests} requests...")
        else:
            # Sequential requests
            for i in range(num_requests):
                result = self.predict_single(sample_features)
                
                if "error" in result:
                    failed_requests += 1
                    errors.append(result["error"])
                else:
                    successful_requests += 1
                    response_times.append(result.get("client_response_time_ms", 0))
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{num_requests} requests...")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        if response_times:
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            max_response_time = np.max(response_times)
            min_response_time = np.min(response_times)
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
            max_response_time = min_response_time = 0
        
        return {
            "benchmark_config": {
                "num_requests": num_requests,
                "concurrent": concurrent,
                "total_time_seconds": total_time
            },
            "results": {
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": successful_requests / num_requests if num_requests > 0 else 0,
                "requests_per_second": num_requests / total_time if total_time > 0 else 0
            },
            "response_times": {
                "avg_ms": avg_response_time,
                "p95_ms": p95_response_time,
                "p99_ms": p99_response_time,
                "max_ms": max_response_time,
                "min_ms": min_response_time
            },
            "errors": errors[:10] if errors else []  # Show first 10 errors
        }
    
    def test_various_inputs(self) -> Dict[str, Any]:
        """Test API with various input combinations."""
        test_cases = [
            {
                "name": "typical_fall_pumpkin",
                "features": PumpkinFeatures(
                    day_of_year=280, month=10, variety="PIE TYPE", 
                    city="BOSTON", package="bushel cartons"
                )
            },
            {
                "name": "early_season_carving",
                "features": PumpkinFeatures(
                    day_of_year=250, month=9, variety="CARVING", 
                    city="NEW YORK", package="1/2 bushel cartons"
                )
            },
            {
                "name": "late_season_variety",
                "features": PumpkinFeatures(
                    day_of_year=320, month=11, variety="PIE TYPE", 
                    city="CHICAGO", package="bushel baskets"
                )
            },
            {
                "name": "edge_case_early_year",
                "features": PumpkinFeatures(
                    day_of_year=1, month=1, variety="PIE TYPE", 
                    city="BOSTON", package="bushel cartons"
                )
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            print(f"Testing: {test_case['name']}")
            result = self.predict_single(test_case['features'])
            results[test_case['name']] = {
                "input": test_case['features'].dict(),
                "output": result
            }
        
        return results
    
    def get_monitor_status(self) -> Dict[str, Any]:
        """Get health monitor status."""
        try:
            response = self.session.get(f"{self.monitor_url}/uptime", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_monitor_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts from health monitor."""
        try:
            response = self.session.get(f"{self.monitor_url}/alerts", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return [{"error": str(e)}]


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML API Client")
    parser.add_argument("command", choices=["health", "models", "predict", "benchmark", "test"],
                       help="Command to execute")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--monitor-url", default="http://localhost:8001", help="Monitor URL")
    parser.add_argument("--model-name", help="Specific model to use")
    parser.add_argument("--num-requests", type=int, default=100, 
                       help="Number of requests for benchmark")
    parser.add_argument("--concurrent", action="store_true", 
                       help="Use concurrent requests for benchmark")
    
    args = parser.parse_args()
    
    # Initialize client
    client = APIClient(args.api_url, args.monitor_url)
    
    if args.command == "health":
        result = client.health_check()
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == "models":
        models = client.list_models()
        print(json.dumps(models, indent=2, default=str))
    
    elif args.command == "predict":
        # Use sample prediction
        result = client.get_sample_prediction()
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == "benchmark":
        result = client.benchmark_api(args.num_requests, args.concurrent)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == "test":
        results = client.test_various_inputs()
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()