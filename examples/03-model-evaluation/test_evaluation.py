"""
Unit tests for model evaluation pipeline.

Tests for:
1. Model loading and metadata validation
2. Evaluation metrics calculation
3. Cross-validation procedures
4. Drift detection algorithms
"""

import unittest
import os
import tempfile
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Import modules to test
try:
    from model_evaluator import (
        load_models_and_metadata, 
        evaluate_model_performance,
        perform_cross_validation
    )
    from cross_validation import (
        perform_kfold_cv,
        generate_learning_curves
    )
    from model_drift_detection import (
        detect_data_drift,
        simulate_model_performance_monitoring
    )
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")


class TestModelEvaluator(unittest.TestCase):
    """Test cases for model evaluation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'DayOfYear': np.random.randint(1, 365, n_samples),
            'Month': np.random.randint(1, 13, n_samples),
            'Variety': np.random.choice(['PIE TYPE', 'CARVING'], n_samples),
            'City': np.random.choice(['BOSTON', 'NEW YORK'], n_samples),
            'Package': np.random.choice(['24 inch bins', '36 inch bins'], n_samples),
            'Price': np.random.normal(10, 3, n_samples)
        })
        
        # Create temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.models_dir)
        
        # Create sample model and metadata
        X = self.sample_data[['DayOfYear', 'Month']]
        y = self.sample_data['Price']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Save model
        model_path = os.path.join(self.models_dir, 'test_model.joblib')
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'test_model': {
                'model_file': 'test_model.joblib',
                'model_type': 'LinearRegression',
                'feature_names': ['DayOfYear', 'Month'],
                'performance': {
                    'r2': 0.5,
                    'rmse': 2.0,
                    'mae': 1.5
                }
            }
        }
        
        metadata_path = os.path.join(self.models_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_models_and_metadata(self):
        """Test model and metadata loading."""
        try:
            models_info = load_models_and_metadata(self.models_dir)
            
            self.assertIn('test_model', models_info)
            self.assertIn('model', models_info['test_model'])
            self.assertIn('metadata', models_info['test_model'])
            
            # Check model type
            model = models_info['test_model']['model']
            self.assertIsInstance(model, LinearRegression)
            
            # Check metadata
            metadata = models_info['test_model']['metadata']
            self.assertEqual(metadata['model_type'], 'LinearRegression')
            self.assertEqual(len(metadata['feature_names']), 2)
            
        except NameError:
            self.skipTest("load_models_and_metadata not available")
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation."""
        try:
            # Create simple model
            X = self.sample_data[['DayOfYear', 'Month']]
            y = self.sample_data['Price']
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Evaluate performance
            results = evaluate_model_performance(model, X, y, 'test_model')
            
            # Check required fields
            required_fields = ['model_name', 'r2_score', 'rmse', 'mae', 'predictions', 'residuals']
            for field in required_fields:
                self.assertIn(field, results)
            
            # Check value ranges
            self.assertGreaterEqual(results['r2_score'], -1)
            self.assertLessEqual(results['r2_score'], 1)
            self.assertGreaterEqual(results['rmse'], 0)
            self.assertGreaterEqual(results['mae'], 0)
            
            # Check array lengths
            self.assertEqual(len(results['predictions']), len(y))
            self.assertEqual(len(results['residuals']), len(y))
            
        except NameError:
            self.skipTest("evaluate_model_performance not available")
    
    def test_cross_validation_basic(self):
        """Test basic cross-validation functionality."""
        try:
            X = self.sample_data[['DayOfYear', 'Month']]
            y = self.sample_data['Price']
            model = LinearRegression()
            
            # Test k-fold CV
            cv_results = perform_kfold_cv(X, y, model, k=3)
            
            # Check required fields
            required_fields = ['method', 'k', 'test_r2', 'train_r2', 'test_rmse', 'train_rmse']
            for field in required_fields:
                self.assertIn(field, cv_results)
            
            # Check array lengths
            self.assertEqual(len(cv_results['test_r2']), 3)
            self.assertEqual(len(cv_results['test_rmse']), 3)
            
        except NameError:
            self.skipTest("perform_kfold_cv not available")
    
    def test_learning_curves(self):
        """Test learning curve generation."""
        try:
            X = self.sample_data[['DayOfYear', 'Month']]
            y = self.sample_data['Price']
            model = LinearRegression()
            
            # Generate learning curves
            lc_results = generate_learning_curves(X, y, model, cv=3)
            
            # Check required fields
            required_fields = ['train_sizes', 'train_scores_mean', 'val_scores_mean']
            for field in required_fields:
                self.assertIn(field, lc_results)
            
            # Check array lengths match
            n_points = len(lc_results['train_sizes'])
            self.assertEqual(len(lc_results['train_scores_mean']), n_points)
            self.assertEqual(len(lc_results['val_scores_mean']), n_points)
            
        except NameError:
            self.skipTest("generate_learning_curves not available")


class TestDriftDetection(unittest.TestCase):
    """Test cases for drift detection functionality."""
    
    def setUp(self):
        """Set up drift detection test fixtures."""
        np.random.seed(42)
        
        # Create reference data
        n_ref = 100
        self.reference_data = pd.DataFrame({
            'DayOfYear': np.random.normal(200, 50, n_ref),
            'Month': np.random.normal(6, 2, n_ref),
            'Price': np.random.normal(10, 3, n_ref)
        })
        
        # Create current data with slight drift
        n_curr = 100
        self.current_data = pd.DataFrame({
            'DayOfYear': np.random.normal(220, 55, n_curr),  # Slight drift
            'Month': np.random.normal(6.5, 2.2, n_curr),    # Slight drift
            'Price': np.random.normal(12, 3.5, n_curr)      # Larger drift
        })
        
        # Create current data without drift (for negative test)
        self.current_data_no_drift = pd.DataFrame({
            'DayOfYear': np.random.normal(200, 50, n_curr),
            'Month': np.random.normal(6, 2, n_curr),
            'Price': np.random.normal(10, 3, n_curr)
        })
    
    def test_detect_data_drift_with_drift(self):
        """Test drift detection when drift is present."""
        try:
            features = ['DayOfYear', 'Month', 'Price']
            drift_results = detect_data_drift(
                self.reference_data, self.current_data, features, alpha=0.05
            )
            
            # Check all features are analyzed
            for feature in features:
                self.assertIn(feature, drift_results)
            
            # Check required fields
            required_fields = ['ks_statistic', 'ks_pvalue', 'drift_detected']
            for feature in features:
                for field in required_fields:
                    self.assertIn(field, drift_results[feature])
            
            # Price should show drift (largest difference)
            self.assertTrue(drift_results['Price']['drift_detected'])
            
        except NameError:
            self.skipTest("detect_data_drift not available")
    
    def test_detect_data_drift_without_drift(self):
        """Test drift detection when no drift is present."""
        try:
            features = ['DayOfYear', 'Month', 'Price']
            drift_results = detect_data_drift(
                self.reference_data, self.current_data_no_drift, features, alpha=0.05
            )
            
            # Most features should not show drift
            drift_detected_count = sum(
                result['drift_detected'] for result in drift_results.values()
            )
            
            # Allow for some false positives due to randomness
            self.assertLessEqual(drift_detected_count, 1)
            
        except NameError:
            self.skipTest("detect_data_drift not available")
    
    def test_model_performance_monitoring(self):
        """Test model performance monitoring simulation."""
        try:
            # Create a simple model for testing
            models_info = {
                'test_model': {
                    'model': LinearRegression(),
                    'metadata': {
                        'model_type': 'LinearRegression',
                        'feature_names': ['DayOfYear', 'Month']
                    }
                }
            }
            
            # Fit model on reference data
            X_ref = self.reference_data[['DayOfYear', 'Month']]
            y_ref = self.reference_data['Price']
            models_info['test_model']['model'].fit(X_ref, y_ref)
            
            # Monitor performance
            monitoring_results = simulate_model_performance_monitoring(
                models_info, self.reference_data, self.current_data
            )
            
            # Check structure
            self.assertIn('test_model', monitoring_results)
            
            result = monitoring_results['test_model']
            required_fields = ['reference_performance', 'current_performance', 'degradation']
            for field in required_fields:
                self.assertIn(field, result)
            
            # Check performance metrics
            ref_perf = result['reference_performance']
            curr_perf = result['current_performance']
            
            for perf in [ref_perf, curr_perf]:
                self.assertIn('r2', perf)
                self.assertIn('rmse', perf)
                self.assertGreaterEqual(perf['rmse'], 0)
            
        except NameError:
            self.skipTest("simulate_model_performance_monitoring not available")


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing utilities."""
    
    def test_sample_data_creation(self):
        """Test that sample data can be created and processed."""
        # Create sample data similar to pumpkins dataset
        n_samples = 50
        sample_data = pd.DataFrame({
            'City Name': np.random.choice(['BOSTON', 'NEW YORK', 'CHICAGO'], n_samples),
            'Package': np.random.choice(['24 inch bins', '36 inch bins', '1/2 bushel cartons'], n_samples),
            'Variety': np.random.choice(['PIE TYPE', 'CARVING'], n_samples),
            'Date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            'Low Price': np.random.uniform(5, 15, n_samples),
            'High Price': np.random.uniform(15, 25, n_samples)
        })
        
        # Basic processing
        sample_data['Month'] = sample_data['Date'].dt.month
        sample_data['DayOfYear'] = sample_data['Date'].dt.dayofyear
        sample_data['Price'] = (sample_data['Low Price'] + sample_data['High Price']) / 2
        
        # Validate processing
        self.assertEqual(len(sample_data), n_samples)
        self.assertTrue(all(sample_data['Month'].between(1, 12)))
        self.assertTrue(all(sample_data['DayOfYear'].between(1, 366)))
        self.assertTrue(all(sample_data['Price'] > 0))
    
    def test_feature_engineering(self):
        """Test feature engineering transformations."""
        # Create test data
        data = pd.DataFrame({
            'Variety': ['PIE TYPE', 'CARVING', 'PIE TYPE'],
            'City': ['BOSTON', 'NEW YORK', 'BOSTON'],
            'Month': [6, 7, 8],
            'DayOfYear': [150, 200, 250]
        })
        
        # One-hot encoding test
        variety_encoded = pd.get_dummies(data['Variety'], prefix='Variety')
        city_encoded = pd.get_dummies(data['City'], prefix='City')
        
        # Check dimensions
        self.assertEqual(variety_encoded.shape[0], len(data))
        self.assertEqual(city_encoded.shape[0], len(data))
        
        # Check column names
        self.assertIn('Variety_PIE TYPE', variety_encoded.columns)
        self.assertIn('Variety_CARVING', variety_encoded.columns)
        self.assertIn('City_BOSTON', city_encoded.columns)
        self.assertIn('City_NEW YORK', city_encoded.columns)


def run_tests():
    """Run all test suites."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestModelEvaluator))
    test_suite.addTest(unittest.makeSuite(TestDriftDetection))
    test_suite.addTest(unittest.makeSuite(TestDataProcessing))
    
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