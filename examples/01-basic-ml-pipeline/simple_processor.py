#!/usr/bin/env python3
"""
Simple CSV data processor that works without external dependencies.
This demonstrates basic Bazel usage for ML data pipelines.
"""

import csv
import sys
import os
import argparse


def load_and_analyze_data(csv_path):
    """Load CSV data and perform basic analysis."""
    print(f"Loading data from: {csv_path}")
    
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    print(f"Loaded {len(rows)} rows with columns: {headers}")
    
    # Basic analysis
    if rows:
        print("\nFirst few rows:")
        for i, row in enumerate(rows[:3]):
            print(f"Row {i+1}: {dict(row)}")
    
    # Analyze pumpkin prices (basic version without pandas)
    prices = []
    for row in rows:
        try:
            low = float(row.get('Low Price', 0))
            high = float(row.get('High Price', 0))
            if low > 0 and high > 0:
                avg_price = (low + high) / 2
                prices.append(avg_price)
        except (ValueError, TypeError):
            continue
    
    if prices:
        print(f"\nPrice analysis:")
        print(f"  Number of valid prices: {len(prices)}")
        print(f"  Average price: ${sum(prices)/len(prices):.2f}")
        print(f"  Min price: ${min(prices):.2f}")
        print(f"  Max price: ${max(prices):.2f}")
    
    return len(rows), len(prices)


def main():
    parser = argparse.ArgumentParser(description='Analyze pumpkin data')
    parser.add_argument('--data_path', required=True, help='Path to CSV file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: File not found: {args.data_path}")
        return 1
    
    try:
        rows, valid_prices = load_and_analyze_data(args.data_path)
        print(f"\n✓ Successfully processed {rows} rows, {valid_prices} valid prices")
        return 0
    except Exception as e:
        print(f"Error processing data: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())