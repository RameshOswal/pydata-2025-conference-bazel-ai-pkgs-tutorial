"""
Pumpkin Price Prediction Tutorial - Data Processing
This script demonstrates how to prepare and visualize pumpkin market data for ML.
Based on Microsoft's ML-For-Beginners tutorial adapted for Bazel.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
import os


def load_pumpkin_data(data_path: str) -> pd.DataFrame:
    """Load the pumpkin dataset from CSV file."""
    print(f"Loading pumpkin data from: {data_path}")
    pumpkins = pd.read_csv(data_path)
    print(f"Loaded {len(pumpkins)} rows of data")
    return pumpkins


def clean_and_filter_data(pumpkins: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter the pumpkin data."""
    print("Filtering data to only include bushel measurements...")
    
    # Filter to only include bushel measurements for consistency
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    print(f"After filtering: {len(pumpkins)} rows remaining")
    
    return pumpkins


def prepare_features(pumpkins: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for machine learning."""
    print("Preparing features...")
    
    # Select only the columns we need
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    
    # Calculate average price
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2
    
    # Extract month from date
    month = pd.DatetimeIndex(pumpkins['Date']).month
    
    # Create new dataframe with cleaned data
    new_pumpkins = pd.DataFrame({
        'Month': month,
        'Package': pumpkins['Package'],
        'Low Price': pumpkins['Low Price'],
        'High Price': pumpkins['High Price'],
        'Price': price
    })
    
    # Normalize pricing per bushel
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price / (1 + 1/9)
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price / (1/2)
    
    print("Feature preparation complete")
    return new_pumpkins


def visualize_data(data, output_dir):
    """Create visualizations of the pumpkin data."""
    print("Creating data visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    try:
        # Create a scatter plot of Low Price vs High Price
        plt.figure(figsize=(10, 6))
        plt.scatter(data['Low Price'], data['High Price'], alpha=0.6)
        plt.xlabel('Low Price ($)')
        plt.ylabel('High Price ($)')
        plt.title('Pumpkin Price Analysis: Low vs High Prices')
        plt.grid(True, alpha=0.3)
        
        scatter_path = os.path.join(output_dir, 'pumpkin_scatter.png')
        plt.savefig(scatter_path)
        plt.close()
        print(f"Scatter plot saved to: {scatter_path}")
        
        # Create a bar chart of average prices by month
        monthly_avg = data.groupby('Month')['Price'].mean()
        plt.figure(figsize=(10, 6))
        monthly_avg.plot(kind='bar')
        plt.xlabel('Month')
        plt.ylabel('Average Price ($)')
        plt.title('Average Pumpkin Prices by Month')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        
        bar_path = os.path.join(output_dir, 'pumpkin_monthly_avg.png')
        plt.savefig(bar_path)
        plt.close()
        print(f"Bar chart saved to: {bar_path}")
        
        print(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Process pumpkin price data')
    parser.add_argument('--data_path', required=True, help='Path to the pumpkin CSV data file')
    parser.add_argument('--output_dir', default='outputs', help='Directory to save outputs')
    parser.add_argument('--save_processed', action='store_true', help='Save processed data to CSV')
    
    args = parser.parse_args()
    
    # Store the original working directory to ensure outputs go to the right place
    original_cwd = os.environ.get('BUILD_WORKING_DIRECTORY', os.getcwd())
    if args.output_dir and not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(original_cwd, args.output_dir)
    
    # Load and process data
    raw_data = load_pumpkin_data(args.data_path)
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Missing values:\n{raw_data.isnull().sum()}")
    
    # Clean and filter data
    filtered_data = clean_and_filter_data(raw_data)
    
    # Prepare features
    processed_data = prepare_features(filtered_data)
    
    # Display basic statistics
    print("\nProcessed data summary:")
    print(processed_data.describe())
    
    # Create visualizations
    visualize_data(processed_data, args.output_dir)
    
    # Save processed data if requested
    if args.save_processed:
        output_file = os.path.join(args.output_dir, 'processed_pumpkins.csv')
        processed_data.to_csv(output_file, index=False)
        print(f"Processed data saved to: {output_file}")
    
    print("Data processing complete!")


if __name__ == "__main__":
    main()