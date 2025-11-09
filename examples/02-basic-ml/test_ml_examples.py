#!/usr/bin/env python3
"""
Tests for the ML examples.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import os


class TestMLExamples(unittest.TestCase):
    """Test cases for ML examples."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic test data similar to pumpkin data
        np.random.seed(42)
        n_samples = 100
        
        self.test_data = pd.DataFrame({
            'DayOfYear': np.random.randint(200, 350, n_samples),
            'Month': np.random.randint(8, 13, n_samples),
            'Variety': np.random.choice(['PIE TYPE', 'CARVING', 'ORANGE'], n_samples),
            'City': np.random.choice(['BALTIMORE', 'BOSTON', 'NEW YORK'], n_samples),
            'Package': np.random.choice(['1 1/9 bushel cartons', '1/2 bushel cartons'], n_samples),
            'Low Price': np.random.uniform(2, 8, n_samples),
            'High Price': np.random.uniform(8, 15, n_samples)
        })
        
        # Calculate price (similar to actual data processing)
        self.test_data['Price'] = (self.test_data['Low Price'] + self.test_data['High Price']) / 2
        
        # Add some noise based on day of year for realistic correlation
        day_factor = (self.test_data['DayOfYear'] - 275) / 100  # Peak around day 275
        self.test_data['Price'] += day_factor * 2 + np.random.normal(0, 0.5, n_samples)
        self.test_data['Price'] = np.clip(self.test_data['Price'], 1, 20)  # Keep reasonable bounds
    
    def test_data_loading(self):
        """Test that data loading works correctly."""
        # Check data shape
        self.assertEqual(len(self.test_data), 100)
        self.assertIn('Price', self.test_data.columns)
        self.assertIn('DayOfYear', self.test_data.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(self.test_data['Price']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.test_data['DayOfYear']))
    
    def test_linear_regression(self):
        """Test linear regression functionality."""
        X = self.test_data[['DayOfYear']]
        y = self.test_data['Price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        pred = model.predict(X_test)
        
        # Check predictions
        self.assertEqual(len(pred), len(X_test))
        self.assertTrue(np.all(pred > 0))  # Prices should be positive
        self.assertTrue(np.all(pred < 50))  # Prices should be reasonable
        
        # Check model has coefficients
        self.assertIsNotNone(model.coef_)
        self.assertIsNotNone(model.intercept_)
    
    def test_polynomial_regression(self):
        """Test polynomial regression functionality."""
        X = self.test_data[['DayOfYear']]
        y = self.test_data['Price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create polynomial pipeline
        pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        pred = pipeline.predict(X_test)
        
        # Check predictions
        self.assertEqual(len(pred), len(X_test))
        self.assertTrue(np.all(pred > 0))  # Prices should be positive
        
        # Check pipeline has expected components
        self.assertEqual(len(pipeline.steps), 2)
        self.assertIsInstance(pipeline.steps[0][1], PolynomialFeatures)
        self.assertIsInstance(pipeline.steps[1][1], LinearRegression)
    
    def test_one_hot_encoding(self):
        """Test one-hot encoding functionality."""
        # Test variety encoding
        variety_encoded = pd.get_dummies(self.test_data['Variety'])
        
        # Check encoding
        unique_varieties = self.test_data['Variety'].unique()
        self.assertEqual(len(variety_encoded.columns), len(unique_varieties))
        
        # Check that each row sums to 1 (one-hot property)
        row_sums = variety_encoded.sum(axis=1)
        self.assertTrue(np.all(row_sums == 1))
        
        # Check data types are correct (should be uint8 or int64)
        for col in variety_encoded.columns:
            self.assertTrue(pd.api.types.is_integer_dtype(variety_encoded[col]))
    
    def test_feature_engineering(self):
        """Test feature engineering functionality."""
        # Create complete feature set
        X_complete = pd.get_dummies(self.test_data['Variety'], prefix='Variety') \
            .join(self.test_data[['Month', 'DayOfYear']]) \
            .join(pd.get_dummies(self.test_data['City'], prefix='City')) \
            .join(pd.get_dummies(self.test_data['Package'], prefix='Package'))
        
        # Check feature set
        self.assertGreater(len(X_complete.columns), 5)  # Should have multiple features
        self.assertIn('Month', X_complete.columns)
        self.assertIn('DayOfYear', X_complete.columns)
        
        # Check that variety columns exist
        variety_cols = [col for col in X_complete.columns if 'Variety_' in col]
        self.assertGreater(len(variety_cols), 0)
        
        # Check no missing values
        self.assertEqual(X_complete.isnull().sum().sum(), 0)
    
    def test_model_comparison(self):
        """Test that different models can be compared."""
        X = self.test_data[['DayOfYear']]
        y = self.test_data['Price']
        
        models = {
            'Linear': LinearRegression(),
            'Polynomial_2': make_pipeline(PolynomialFeatures(2), LinearRegression())
        }
        
        results = []
        
        for model_name, model in models.items():
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            pred = model.predict(X_test)
            
            # Calculate score
            score = model.score(X_test, y_test)
            
            results.append({
                'model': model_name,
                'score': score,
                'predictions': pred
            })
        
        # Check we got results for both models
        self.assertEqual(len(results), 2)
        
        # Check all models produced valid predictions
        for result in results:
            self.assertTrue(np.all(result['predictions'] > 0))
            self.assertTrue(np.isfinite(result['score']))
    
    def test_data_file_exists(self):
        """Test that the data file exists in the expected location."""
        # This test assumes the data file is in the standard location
        data_path = 'examples/02-basic-ml/data/US-pumpkins.csv'
        
        # In a Bazel test environment, we might need to check different paths
        possible_paths = [
            data_path,
            'data/US-pumpkins.csv',
            '../data/US-pumpkins.csv'
        ]
        
        file_exists = False
        for path in possible_paths:
            if os.path.exists(path):
                file_exists = True
                # If file exists, try to load it
                try:
                    df = pd.read_csv(path)
                    self.assertGreater(len(df), 0)
                    self.assertIn('Date', df.columns)
                    break
                except Exception as e:
                    self.fail(f"Could not load data file {path}: {e}")
        
        # Skip this test if running in an environment where data file is not available
        if not file_exists:
            self.skipTest("Data file not found in test environment")


if __name__ == '__main__':
    unittest.main()