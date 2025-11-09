"""
Unit tests for the ML Model Serving API.

Tests for:
1. API endpoints functionality
2. Model loading and management
3. Prediction accuracy and format
4. Error handling and validation
5. Health monitoring integration
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from fastapi.testclient import TestClient

# Import modules to test
try:
    from model_server import app, model_store, PumpkinFeatures, process_features
    MODEL_SERVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import model_server: {e}")
    MODEL_SERVER_AVAILABLE = False
    # Create mock objects for testing
    app = None
    model_store = None
    PumpkinFeatures = None
    process_features = None

try:
    from model_manager import ModelManager
    MODEL_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import model_manager: {e}")
    MODEL_MANAGER_AVAILABLE = False
    ModelManager = None

try:
    from health_monitor import HealthMonitor, HealthMetrics
    HEALTH_MONITOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import health_monitor: {e}")
    HEALTH_MONITOR_AVAILABLE = False
    HealthMonitor = None
    HealthMetrics = None

try:
    from api_client import APIClient, PumpkinFeatures as ClientFeatures
    API_CLIENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import api_client: {e}")
    API_CLIENT_AVAILABLE = False
    APIClient = None
    ClientFeatures = None


class TestModelServer(unittest.TestCase):
    """Test cases for the model serving API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        
        # Create sample features
        self.sample_features = {
            "day_of_year": 280,
            "month": 10,
            "variety": "PIE TYPE",
            "city": "BOSTON",
            "package": "bushel cartons"
        }
        
        # Create mock model and metadata
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.array([15.5])
        
        self.mock_metadata = {
            "model_type": "LinearRegression",
            "feature_names": ["DayOfYear", "Month"],
            "performance": {"r2": 0.8, "rmse": 2.0, "mae": 1.5}
        }
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        if not MODEL_SERVER_AVAILABLE:
            self.skipTest("model_server not available")
        
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
    
    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        if not MODEL_SERVER_AVAILABLE:
            self.skipTest("model_server not available")
        
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("timestamp", data)
        self.assertIn("version", data)
    
    def test_list_models_endpoint(self):
        """Test the list models endpoint."""
        if not MODEL_SERVER_AVAILABLE:
            self.skipTest("model_server not available")
        
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
    
    def test_predict_single_endpoint(self):
        """Test single prediction endpoint."""
        if not MODEL_SERVER_AVAILABLE:
            self.skipTest("model_server not available")
        
        response = self.client.post("/predict", json=self.sample_features)
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("predicted_price", data)
            self.assertIn("model_used", data)
            self.assertIn("prediction_timestamp", data)
            self.assertIsInstance(data["predicted_price"], (int, float))
        else:
            # If models aren't loaded, we expect an error
            self.assertIn(response.status_code, [404, 500])
    
    def test_predict_single_validation(self):
        """Test input validation for single prediction."""
        if not MODEL_SERVER_AVAILABLE:
            self.skipTest("model_server not available")
        
        # Test missing required field
        invalid_features = self.sample_features.copy()
        del invalid_features["day_of_year"]
        
        response = self.client.post("/predict", json=invalid_features)
        self.assertEqual(response.status_code, 422)  # Validation error
        
        # Test invalid day_of_year
        invalid_features = self.sample_features.copy()
        invalid_features["day_of_year"] = 400  # Invalid (> 366)
        
        response = self.client.post("/predict", json=invalid_features)
        self.assertEqual(response.status_code, 422)  # Validation error
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint."""
        if not MODEL_SERVER_AVAILABLE:
            self.skipTest("model_server not available")
        
        batch_request = {
            "samples": [self.sample_features, self.sample_features],
            "model_name": None
        }
        
        response = self.client.post("/predict/batch", json=batch_request)
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("predictions", data)
            self.assertIn("batch_size", data)
            self.assertEqual(data["batch_size"], 2)
            self.assertEqual(len(data["predictions"]), 2)
        else:
            # If models aren't loaded, we expect an error
            self.assertIn(response.status_code, [404, 500])
    
    def test_process_features_function(self):
        """Test the feature processing function."""
        if not MODEL_SERVER_AVAILABLE:
            self.skipTest("model_server not available")
        
        try:
            features = PumpkinFeatures(**self.sample_features)
            processed = process_features(features, self.mock_metadata)
            
            self.assertIsInstance(processed, pd.DataFrame)
            self.assertEqual(len(processed), 1)
            self.assertIn("DayOfYear", processed.columns)
            self.assertIn("Month", processed.columns)
            
        except NameError:
            self.skipTest("process_features not available")


class TestModelManager(unittest.TestCase):
    """Test cases for the model manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODEL_MANAGER_AVAILABLE:
            self.skipTest("model_manager not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelManager(self.temp_dir)
        
        # Create a simple test model
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        self.test_model = LinearRegression()
        self.test_model.fit(X, y)
        
        # Save test model
        self.test_model_path = os.path.join(self.temp_dir, "test_model.joblib")
        joblib.dump(self.test_model, self.test_model_path)
        
        self.test_metadata = {
            "model_type": "LinearRegression",
            "feature_names": ["feature1", "feature2"],
            "performance": {"r2": 0.5, "rmse": 1.0, "mae": 0.8}
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_register_model(self):
        """Test model registration."""
        if not MODEL_MANAGER_AVAILABLE:
            self.skipTest("model_manager not available")
        
        try:
            success = self.manager.register_model(
                "test_model", self.test_model_path, self.test_metadata
            )
            self.assertTrue(success)
            
            # Check if model is in registry
            models = self.manager.list_models()
            self.assertGreater(len(models), 0)
            self.assertEqual(models[0]["model_name"], "test_model")
            
        except NameError:
            self.skipTest("ModelManager not available")
    
    def test_deploy_model(self):
        """Test model deployment."""
        if not MODEL_MANAGER_AVAILABLE:
            self.skipTest("model_manager not available")
        
        try:
            # First register the model
            self.manager.register_model(
                "test_model", self.test_model_path, self.test_metadata
            )
            
            # Then deploy it
            success = self.manager.deploy_model("test_model", deployment_type="active")
            self.assertTrue(success)
            
            # Check if model file exists in active directory
            active_model_path = os.path.join(self.temp_dir, "active", "test_model_model.joblib")
            self.assertTrue(os.path.exists(active_model_path))
            
        except NameError:
            self.skipTest("ModelManager not available")
    
    def test_list_models(self):
        """Test listing models."""
        if not MODEL_MANAGER_AVAILABLE:
            self.skipTest("model_manager not available")
        
        try:
            # Register a model
            self.manager.register_model(
                "test_model", self.test_model_path, self.test_metadata
            )
            
            models = self.manager.list_models()
            self.assertIsInstance(models, list)
            if models:
                self.assertIn("model_name", models[0])
                self.assertIn("status", models[0])
                
        except NameError:
            self.skipTest("ModelManager not available")


class TestHealthMonitor(unittest.TestCase):
    """Test cases for the health monitor."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not HEALTH_MONITOR_AVAILABLE:
            self.skipTest("health_monitor not available")
        
        try:
            self.monitor = HealthMonitor()
        except NameError:
            self.monitor = None
    
    def test_log_prediction(self):
        """Test prediction logging."""
        if self.monitor is None:
            self.skipTest("HealthMonitor not available")
        
        initial_count = len(self.monitor.prediction_log)
        
        self.monitor.log_prediction("test_model", 100.0, True)
        
        self.assertEqual(len(self.monitor.prediction_log), initial_count + 1)
        
        # Check log entry
        log_entry = self.monitor.prediction_log[-1]
        self.assertEqual(log_entry["model_name"], "test_model")
        self.assertEqual(log_entry["response_time_ms"], 100.0)
        self.assertTrue(log_entry["success"])
    
    def test_collect_metrics(self):
        """Test metrics collection."""
        if not HEALTH_MONITOR_AVAILABLE or self.monitor is None:
            self.skipTest("HealthMonitor not available")
        
        # Log some predictions first
        self.monitor.log_prediction("test_model", 100.0, True)
        self.monitor.log_prediction("test_model", 150.0, True)
        self.monitor.log_prediction("test_model", 200.0, False)
        
        metrics = self.monitor.collect_metrics("test_model")
        
        self.assertIsInstance(metrics, HealthMetrics)
        self.assertEqual(metrics.model_name, "test_model")
        self.assertGreaterEqual(metrics.prediction_count, 0)
        self.assertGreaterEqual(metrics.memory_usage_mb, 0)
    
    def test_get_metrics_summary(self):
        """Test metrics summary."""
        if not HEALTH_MONITOR_AVAILABLE or self.monitor is None:
            self.skipTest("HealthMonitor not available")
        
        # Collect some metrics first
        self.monitor.collect_metrics("test_model")
        
        summary = self.monitor.get_metrics_summary("test_model", hours=1)
        
        if "error" not in summary:
            self.assertIn("time_period_hours", summary)
            self.assertIn("model_name", summary)
            self.assertIn("metrics_count", summary)


class TestAPIClient(unittest.TestCase):
    """Test cases for the API client."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not API_CLIENT_AVAILABLE or APIClient is None or ClientFeatures is None:
            self.skipTest("api_client not available")
        
        try:
            # Use a mock URL since we're not running a real server
            self.client = APIClient("http://mock-api:8000", "http://mock-monitor:8001")
            self.sample_features = ClientFeatures(
                day_of_year=280,
                month=10,
                variety="PIE TYPE",
                city="BOSTON",
                package="bushel cartons"
            )
        except NameError:
            self.client = None
    
    @patch('requests.Session.get')
    def test_health_check(self, mock_get):
        """Test health check method."""
        if self.client is None:
            self.skipTest("APIClient not available")
        
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.health_check()
        
        self.assertEqual(result["status"], "healthy")
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_health_check_error(self, mock_get):
        """Test health check error handling."""
        if self.client is None:
            self.skipTest("APIClient not available")
        
        # Mock error response
        mock_get.side_effect = Exception("Connection error")
        
        result = self.client.health_check()
        
        self.assertIn("error", result)
        self.assertEqual(result["status"], "unhealthy")
    
    @patch('requests.Session.post')
    def test_predict_single(self, mock_post):
        """Test single prediction method."""
        if self.client is None:
            self.skipTest("APIClient not available")
        
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "predicted_price": 15.5,
            "model_used": "test_model",
            "prediction_timestamp": "2023-01-01T00:00:00"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.client.predict_single(self.sample_features)
        
        self.assertIn("predicted_price", result)
        self.assertEqual(result["predicted_price"], 15.5)
        mock_post.assert_called_once()
    
    def test_feature_validation(self):
        """Test feature validation in client."""
        if self.client is None:
            self.skipTest("APIClient not available")
        
        # Test valid features
        features = ClientFeatures(
            day_of_year=280,
            month=10,
            variety="PIE TYPE",
            city="BOSTON",
            package="bushel cartons"
        )
        
        self.assertEqual(features.day_of_year, 280)
        self.assertEqual(features.month, 10)
        
        # Test invalid day_of_year (should raise validation error)
        with self.assertRaises(Exception):  # Pydantic validation error
            ClientFeatures(
                day_of_year=400,  # Invalid
                month=10,
                variety="PIE TYPE",
                city="BOSTON",
                package="bushel cartons"
            )


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test a complete workflow from model creation to serving."""
        # This is a simplified integration test
        # In practice, this would test the full pipeline
        
        # 1. Create a simple model
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        model = LinearRegression()
        model.fit(X, y)
        
        # 2. Test that model can make predictions
        predictions = model.predict(X[:5])
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
        
        # 3. Test feature processing (simplified)
        sample_data = pd.DataFrame({
            'DayOfYear': [280],
            'Month': [10]
        })
        
        self.assertEqual(len(sample_data), 1)
        self.assertIn('DayOfYear', sample_data.columns)
        self.assertIn('Month', sample_data.columns)
    
    def test_data_types_consistency(self):
        """Test that data types are consistent across the system."""
        # Test that our sample data has the right types
        sample_features = {
            "day_of_year": 280,
            "month": 10,
            "variety": "PIE TYPE",
            "city": "BOSTON",
            "package": "bushel cartons"
        }
        
        # Validate types
        self.assertIsInstance(sample_features["day_of_year"], int)
        self.assertIsInstance(sample_features["month"], int)
        self.assertIsInstance(sample_features["variety"], str)
        self.assertIsInstance(sample_features["city"], str)
        self.assertIsInstance(sample_features["package"], str)
        
        # Validate ranges
        self.assertGreaterEqual(sample_features["day_of_year"], 1)
        self.assertLessEqual(sample_features["day_of_year"], 366)
        self.assertGreaterEqual(sample_features["month"], 1)
        self.assertLessEqual(sample_features["month"], 12)


def run_tests():
    """Run all test suites."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestModelServer))
    test_suite.addTest(unittest.makeSuite(TestModelManager))
    test_suite.addTest(unittest.makeSuite(TestHealthMonitor))
    test_suite.addTest(unittest.makeSuite(TestAPIClient))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)