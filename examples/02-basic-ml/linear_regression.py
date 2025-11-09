#!/usr/bin/env python3
"""
Linear Regression Example for Pumpkin Price Prediction

This example demonstrates linear regression using scikit-learn
to predict pumpkin prices based on day of year.

Based on Microsoft ML-For-Beginners tutorial:
https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/3-Linear/solution/notebook.ipynb
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


def analyze_correlation(pumpkins_df):
    """Analyze correlation between DayOfYear and Price."""
    # Check correlation for all pumpkins
    correlation = pumpkins_df['DayOfYear'].corr(pumpkins_df['Price'])
    print(f"Correlation between DayOfYear and Price: {correlation:.4f}")
    
    return pumpkins_df


def visualize_data(data, output_dir):
    """Create visualizations of the data."""
    print("Creating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Scatter plot of price vs day of year
    plt.figure(figsize=(12, 8))
    plt.scatter(data['DayOfYear'], data['Price'], alpha=0.7, color='blue')
    plt.xlabel('Day of Year')
    plt.ylabel('Price ($)')
    plt.title('Pumpkin Prices vs Day of Year')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'pumpkin_prices_scatter.png'))
    plt.close()
    
    # Bar chart of average prices by city
    plt.figure(figsize=(10, 6))
    data.groupby('City')['Price'].mean().plot(kind='bar')
    plt.xlabel('City')
    plt.ylabel('Average Price ($)')
    plt.title('Average Pumpkin Prices by City')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_prices_by_city.png'))
    plt.close()
    
    # Bar chart of average prices by package type
    plt.figure(figsize=(10, 6))
    data.groupby('Package')['Price'].mean().plot(kind='bar')
    plt.xlabel('Package Type')
    plt.ylabel('Average Price ($)')
    plt.title('Average Pumpkin Prices by Package Type')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_prices_by_package.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def train_linear_regression(X, y, test_size=0.2, random_state=0):
    """Train a linear regression model."""
    print("\nTraining linear regression model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create and train model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    
    # Make predictions
    pred = lin_reg.predict(X_test)
    
    # Calculate metrics
    mse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    
    print(f"Root Mean Squared Error: {mse:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Mean error percentage: {mse/np.mean(pred)*100:.1f}%")
    
    return lin_reg, X_test, y_test, pred


def visualize_regression_results(model, X_test, y_test, pred, output_dir):
    """Visualize the regression results."""
    print("Creating regression visualization...")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, alpha=0.6, label='Actual prices')
    plt.plot(sorted(X_test.flatten()), model.predict(sorted(X_test)), 
             'r-', label='Linear regression line')
    plt.xlabel('Day of Year')
    plt.ylabel('Price ($)')
    plt.title('Linear Regression: Pumpkin Price vs Day of Year')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'linear_regression_results.png'))
    plt.close()
    
    print(f"Regression visualization saved to {output_dir}")


def predict_price_for_day(model, day_of_year):
    """Predict price for a specific day."""
    prediction = model.predict([[day_of_year]])
    print(f"Predicted price for day {day_of_year}: ${prediction[0]:.2f}")
    return prediction[0]


def main():
    parser = argparse.ArgumentParser(description='Linear regression for pumpkin price prediction')
    parser.add_argument('--data_path', default='examples/02-basic-ml/data/US-pumpkins.csv',
                       help='Path to the pumpkin CSV data file')
    parser.add_argument('--output_dir', default='outputs/02-basic-ml',
                       help='Directory to save outputs')
    parser.add_argument('--predict_day', type=int, default=256,
                       help='Day of year to predict price for (default: 256, programmer\'s day)')
    
    args = parser.parse_args()
    
    # Handle relative paths from current working directory
    original_cwd = os.environ.get('BUILD_WORKING_DIRECTORY', os.getcwd())
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(original_cwd, args.output_dir)
    
    # Load and process data
    pumpkins_df = load_and_process_data(args.data_path)
    
    # Create visualizations of all data
    visualize_data(pumpkins_df, args.output_dir)
    
    # Analyze correlation
    analyzed_pumpkins = analyze_correlation(pumpkins_df)
    
    # Prepare features and target
    X = analyzed_pumpkins['DayOfYear'].to_numpy().reshape(-1, 1)
    y = analyzed_pumpkins['Price']
    
    # Train linear regression model
    model, X_test, y_test, pred = train_linear_regression(X, y)
    
    # Visualize results
    visualize_regression_results(model, X_test, y_test, pred, args.output_dir)
    
    # Make a prediction for programmer's day
    predict_price_for_day(model, args.predict_day)
    
    print("\nLinear regression analysis complete!")


if __name__ == "__main__":
    main()