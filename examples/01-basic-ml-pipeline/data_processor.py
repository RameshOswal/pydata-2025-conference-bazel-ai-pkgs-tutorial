"""
Pumpkin Price Prediction Tutorial - Data Processing
This script demonstrates how to prepare and visualize pumpkin market data for ML.
Based on Microsoft's ML-For-Beginners tutorial adapted for Bazel.
"""

import pandas as pd
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


def visualize_data(pumpkins: pd.DataFrame, output_dir: str = ".") -> None:
    """Create visualizations of the pumpkin data."""
    print("Creating data visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(pumpkins['Price'], pumpkins['Month'])
    plt.xlabel('Price ($)')
    plt.ylabel('Month')
    plt.title('Pumpkin Prices by Month')
    plt.savefig(os.path.join(output_dir, 'pumpkin_scatter.png'))
    plt.close()
    
    # Bar chart of average prices by month
    plt.figure(figsize=(10, 6))
    monthly_avg = pumpkins.groupby(['Month'])['Price'].mean()
    monthly_avg.plot(kind='bar')
    plt.xlabel('Month')
    plt.ylabel('Average Pumpkin Price ($)')
    plt.title('Average Pumpkin Prices by Month')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pumpkin_monthly_avg.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Process pumpkin market data')
    parser.add_argument('--data_path', required=True, help='Path to the US-pumpkins.csv file')
    parser.add_argument('--output_dir', default='.', help='Directory to save outputs')
    parser.add_argument('--save_processed', action='store_true', help='Save processed data to CSV')
    
    args = parser.parse_args()
    
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