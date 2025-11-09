#!/usr/bin/env python3
"""
Polynomial Regression Example for Pumpkin Price Prediction

This example demonstrates polynomial regression using scikit-learn
to capture non-linear relationships in pumpkin pricing.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def load_and_process_data(data_path):
    """Load and process the pumpkin data."""
    print("Loading pumpkin data...")
    pumpkins = pd.read_csv(data_path)
    print(f"Loaded {len(pumpkins)} rows of data")
    
    # Filter to only include bushel measurements
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    print(f"After filtering for bushel measurements: {len(pumpkins)} rows")
    
    # Select relevant columns
    new_columns = ['Package', 'Variety', 'City Name', 'Month', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)
    
    # Calculate average price
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2
    
    # Extract month and day of year
    month = pd.DatetimeIndex(pumpkins['Date']).month
    day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
    
    # Create new dataframe
    new_pumpkins = pd.DataFrame({
        'Month': month,
        'DayOfYear': day_of_year, 
        'City': pumpkins['City Name'], 
        'Package': pumpkins['Package'], 
        'Low Price': pumpkins['Low Price'],
        'High Price': pumpkins['High Price'], 
        'Price': price
    })
    
    # Adjust prices based on package size
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/1.1
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price*2
    
    return new_pumpkins


def train_polynomial_regression(X, y, degree=2, test_size=0.2, random_state=0):
    """Train a polynomial regression model."""
    print(f"\nTraining polynomial regression model (degree={degree})...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create polynomial pipeline
    pipeline = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    train_score = pipeline.score(X_train, y_train)
    
    print(f"Root Mean Squared Error: {mse:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Training Score: {train_score:.3f}")
    print(f"Mean error percentage: {mse/np.mean(pred)*100:.1f}%")
    
    return pipeline, X_test, y_test, pred


def visualize_polynomial_results(model, X, y, X_test, y_test, pred, degree, output_dir):
    """Visualize the polynomial regression results."""
    print("Creating polynomial regression visualization...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot all data points
    plt.scatter(X, y, alpha=0.3, color='lightblue', label='All data points')
    
    # Plot test data points
    plt.scatter(X_test, y_test, alpha=0.6, color='blue', label='Test data')
    
    # Plot polynomial regression curve
    X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_poly = model.predict(X_range)
    plt.plot(X_range, y_poly, 'r-', linewidth=2, label=f'Polynomial regression (degree {degree})')
    
    plt.xlabel('Day of Year')
    plt.ylabel('Price ($)')
    plt.title(f'Polynomial Regression (Degree {degree}): Pumpkin Price vs Day of Year')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'polynomial_regression_degree_{degree}.png'))
    plt.close()
    
    print(f"Polynomial regression visualization saved to {output_dir}")


def compare_polynomial_degrees(X, y, degrees=[1, 2, 3, 4], output_dir=None):
    """Compare polynomial regression with different degrees."""
    print("\nComparing polynomial degrees...")
    
    results = []
    
    for degree in degrees:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Create and train model
        pipeline = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        train_score = pipeline.score(X_train, y_train)
        
        results.append({
            'degree': degree,
            'mse': mse,
            'r2': r2,
            'train_score': train_score,
            'error_pct': mse/np.mean(pred)*100
        })
        
        print(f"Degree {degree}: MSE={mse:.3f}, R²={r2:.3f}, Train Score={train_score:.3f}, Error={mse/np.mean(pred)*100:.1f}%")
    
    # Create comparison visualization if output directory is provided
    if output_dir:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot([r['degree'] for r in results], [r['mse'] for r in results], 'bo-')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Root Mean Squared Error')
        plt.title('MSE vs Polynomial Degree')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot([r['degree'] for r in results], [r['r2'] for r in results], 'ro-')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('R² Score')
        plt.title('R² Score vs Polynomial Degree')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot([r['degree'] for r in results], [r['error_pct'] for r in results], 'go-')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Error Percentage (%)')
        plt.title('Error % vs Polynomial Degree')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'polynomial_degree_comparison.png'))
        plt.close()
        print(f"Degree comparison visualization saved to {output_dir}")
    
    return results


def predict_price_for_day(model, day_of_year):
    """Predict price for a specific day."""
    prediction = model.predict([[day_of_year]])
    print(f"Predicted price for day {day_of_year}: ${prediction[0]:.2f}")
    return prediction[0]


def main():
    parser = argparse.ArgumentParser(description='Polynomial regression for pumpkin price prediction')
    parser.add_argument('--data_path', default='examples/02-basic-ml/data/US-pumpkins.csv',
                       help='Path to the pumpkin CSV data file')
    parser.add_argument('--output_dir', default='outputs/02-basic-ml',
                       help='Directory to save outputs')
    parser.add_argument('--degree', type=int, default=2,
                       help='Polynomial degree (default: 2)')
    parser.add_argument('--compare_degrees', action='store_true',
                       help='Compare different polynomial degrees')
    parser.add_argument('--predict_day', type=int, default=256,
                       help='Day of year to predict price for (default: 256, programmer\'s day)')
    
    args = parser.parse_args()
    
    # Handle relative paths from current working directory
    original_cwd = os.environ.get('BUILD_WORKING_DIRECTORY', os.getcwd())
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(original_cwd, args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process data
    pumpkins_df = load_and_process_data(args.data_path)
    
    # Use all pumpkin data
    print(f"All pumpkins: {len(pumpkins_df)} rows")
    
    # Prepare features and target
    X = pumpkins_df['DayOfYear'].to_numpy().reshape(-1, 1)
    y = pumpkins_df['Price']
    
    # Train polynomial regression model
    model, X_test, y_test, pred = train_polynomial_regression(X, y, degree=args.degree)
    
    # Visualize results
    visualize_polynomial_results(model, X, y, X_test, y_test, pred, args.degree, args.output_dir)
    
    # Compare different degrees if requested
    if args.compare_degrees:
        compare_polynomial_degrees(X, y, degrees=[1, 2, 3, 4, 5], output_dir=args.output_dir)
    
    # Make a prediction for programmer's day
    predict_price_for_day(model, args.predict_day)
    
    print("\nPolynomial regression analysis complete!")


if __name__ == "__main__":
    main()