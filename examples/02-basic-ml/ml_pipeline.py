#!/usr/bin/env python3
"""
Complete ML Pipeline Example for Pumpkin Price Prediction

This example demonstrates a complete machine learning pipeline including:
- Data preprocessing
- Feature engineering (one-hot encoding for categorical variables)
- Multiple regression models
- Model comparison and evaluation
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import joblib  # For model serialization
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
        'Variety': pumpkins['Variety'], 
        'City': pumpkins['City Name'], 
        'Package': pumpkins['Package'], 
        'Low Price': pumpkins['Low Price'],
        'High Price': pumpkins['High Price'], 
        'Price': price
    })
    
    # Adjust prices based on package size
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/1.1
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price*2
    
    print("\nData summary:")
    print(new_pumpkins.describe())
    print(f"\nVarieties: {list(new_pumpkins['Variety'].unique())}")
    print(f"Cities: {list(new_pumpkins['City'].unique())}")
    print(f"Package types: {list(new_pumpkins['Package'].unique())}")
    
    return new_pumpkins


def create_feature_sets(pumpkins_df):
    """Create different feature sets for model comparison."""
    feature_sets = {}
    
    # 1. Day of year only (simple)
    feature_sets['day_only'] = {
        'X': pumpkins_df[['DayOfYear']],
        'description': 'Day of Year only'
    }
    
    # 2. Day of year + Month
    feature_sets['day_month'] = {
        'X': pumpkins_df[['DayOfYear', 'Month']],
        'description': 'Day of Year + Month'
    }
    
    # 3. Variety only (one-hot encoded)
    variety_encoded = pd.get_dummies(pumpkins_df['Variety'], prefix='Variety')
    feature_sets['variety_only'] = {
        'X': variety_encoded,
        'description': 'Variety (one-hot encoded)'
    }
    
    # 4. Complete feature set (all categorical variables one-hot encoded)
    X_complete = pd.get_dummies(pumpkins_df['Variety'], prefix='Variety') \
        .join(pumpkins_df[['Month', 'DayOfYear']]) \
        .join(pd.get_dummies(pumpkins_df['City'], prefix='City')) \
        .join(pd.get_dummies(pumpkins_df['Package'], prefix='Package'))
    
    feature_sets['complete'] = {
        'X': X_complete,
        'description': 'All features (one-hot encoded categorical)'
    }
    
    return feature_sets


def train_and_evaluate_model(X, y, model_name, model, test_size=0.2, random_state=0):
    """Train and evaluate a model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    pred = model.predict(X_test)
    
    # Calculate metrics
    mse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    train_score = model.score(X_train, y_train)
    
    return {
        'model_name': model_name,
        'model': model,
        'mse': mse,
        'r2': r2,
        'mae': mae,
        'train_score': train_score,
        'error_pct': mse/np.mean(pred)*100 if np.mean(pred) != 0 else 0,
        'X_test': X_test,
        'y_test': y_test,
        'pred': pred,
        'data_shape': X.shape
    }


def compare_models_and_features(pumpkins_df, output_dir):
    """Compare different models and feature sets."""
    print("\nComparing models and feature sets...")
    
    feature_sets = create_feature_sets(pumpkins_df)
    y = pumpkins_df['Price']
    
    # Define models to test
    models = {
        'Linear': LinearRegression(),
        'Polynomial_2': make_pipeline(PolynomialFeatures(2), LinearRegression()),
        'Polynomial_3': make_pipeline(PolynomialFeatures(3), LinearRegression())
    }
    
    results = []
    
    for feature_name, feature_info in feature_sets.items():
        X = feature_info['X']
        print(f"\n--- Feature set: {feature_info['description']} ---")
        print(f"Feature shape: {X.shape}")
        
        for model_name, model in models.items():
            try:
                result = train_and_evaluate_model(X, y, f"{model_name}_{feature_name}", model)
                result['feature_set'] = feature_name
                result['feature_description'] = feature_info['description']
                results.append(result)
                
                print(f"{model_name}: MSE={result['mse']:.3f}, R²={result['r2']:.3f}, Error={result['error_pct']:.1f}%")
                
            except Exception as e:
                print(f"Error training {model_name} with {feature_name}: {e}")
    
    # Create comparison visualization
    create_model_comparison_plot(results, output_dir)
    
    # Find best model
    best_result = min(results, key=lambda x: x['mse'])
    print(f"\n--- Best Model ---")
    print(f"Model: {best_result['model_name']}")
    print(f"Features: {best_result['feature_description']}")
    print(f"MSE: {best_result['mse']:.3f}")
    print(f"R²: {best_result['r2']:.3f}")
    print(f"Error: {best_result['error_pct']:.1f}%")
    
    return results, best_result


def create_model_comparison_plot(results, output_dir):
    """Create visualization comparing different models and feature sets."""
    print("Creating model comparison visualization...")
    
    # Prepare data for plotting
    model_names = [r['model_name'] for r in results]
    mse_values = [r['mse'] for r in results]
    r2_values = [r['r2'] for r in results]
    error_pct_values = [r['error_pct'] for r in results]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # MSE comparison
    ax1.bar(range(len(results)), mse_values)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Root Mean Squared Error')
    ax1.set_title('MSE Comparison')
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # R² comparison
    ax2.bar(range(len(results)), r2_values)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score Comparison')
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Error percentage comparison
    ax3.bar(range(len(results)), error_pct_values)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Error Percentage (%)')
    ax3.set_title('Error Percentage Comparison')
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Feature set distribution
    feature_counts = {}
    for r in results:
        feature_set = r['feature_set']
        if feature_set not in feature_counts:
            feature_counts[feature_set] = 0
        feature_counts[feature_set] += 1
    
    ax4.pie(feature_counts.values(), labels=feature_counts.keys(), autopct='%1.1f%%')
    ax4.set_title('Distribution of Feature Sets')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison visualization saved to {output_dir}")


def create_prediction_visualization(best_result, pumpkins_df, output_dir):
    """Create visualization for the best model's predictions."""
    print("Creating prediction visualization for best model...")
    
    # If it's a simple day-of-year model, create a nice curve plot
    if 'day_only' in best_result['feature_set']:
        X = pumpkins_df[['DayOfYear']]
        y = pumpkins_df['Price']
        
        plt.figure(figsize=(12, 8))
        plt.scatter(X, y, alpha=0.6, label='Actual prices')
        
        # Create smooth curve
        X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        y_pred = best_result['model'].predict(X_range)
        plt.plot(X_range, y_pred, 'r-', linewidth=2, label='Best model prediction')
        
        plt.xlabel('Day of Year')
        plt.ylabel('Price ($)')
        plt.title(f'Best Model Predictions: {best_result["model_name"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'best_model_predictions.png'))
        plt.close()
    else:
        # For complex models, create residual plot
        plt.figure(figsize=(10, 6))
        residuals = best_result['y_test'] - best_result['pred']
        plt.scatter(best_result['pred'], residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price ($)')
        plt.ylabel('Residuals ($)')
        plt.title(f'Residual Plot: {best_result["model_name"]}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'best_model_residuals.png'))
        plt.close()
    
    print(f"Prediction visualization saved to {output_dir}")


def save_results_summary(results, best_result, output_dir):
    """Save results summary to a text file."""
    print("Saving results summary...")
    
    summary_path = os.path.join(output_dir, 'model_comparison_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("ML Pipeline Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("All Model Results:\n")
        f.write("-" * 30 + "\n")
        for result in sorted(results, key=lambda x: x['mse']):
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"Features: {result['feature_description']}\n")
            f.write(f"MSE: {result['mse']:.3f}\n")
            f.write(f"R²: {result['r2']:.3f}\n")
            f.write(f"MAE: {result['mae']:.3f}\n")
            f.write(f"Error %: {result['error_pct']:.1f}%\n")
            f.write(f"Train Score: {result['train_score']:.3f}\n")
            f.write("\n")
        
        f.write("Best Model:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Model: {best_result['model_name']}\n")
        f.write(f"Features: {best_result['feature_description']}\n")
        f.write(f"MSE: {best_result['mse']:.3f}\n")
        f.write(f"R²: {best_result['r2']:.3f}\n")
        f.write(f"MAE: {best_result['mae']:.3f}\n")
        f.write(f"Error %: {best_result['error_pct']:.1f}%\n")
        f.write(f"Train Score: {best_result['train_score']:.3f}\n")
    
    print(f"Results summary saved to {summary_path}")


def save_models(results, output_dir):
    """Save trained models to disk for later evaluation."""
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_metadata = {}
    
    # Results is a list of model results, not a dictionary
    for i, result in enumerate(results):
        model_name_parts = result['model_name'].split('_')
        if len(model_name_parts) >= 2:
            model_type = model_name_parts[0]
            feature_set = '_'.join(model_name_parts[1:])
        else:
            model_type = result['model_name']
            feature_set = f"feature_set_{i}"
        
        model_key = f"{feature_set}_{model_type}".lower()
        model_filename = f"{model_key}_model.joblib"
        model_path = os.path.join(models_dir, model_filename)
        
        # Save the model
        joblib.dump(result['model'], model_path)
        
        # Get feature names - need to extract from the model or result
        feature_names = []
        if hasattr(result['model'], 'feature_names_in_'):
            feature_names = list(result['model'].feature_names_in_)
        elif 'X_test' in result and hasattr(result['X_test'], 'columns'):
            feature_names = list(result['X_test'].columns)
        else:
            # Default feature names based on model name
            if 'day_only' in result['model_name'].lower():
                feature_names = ['DayOfYear']
            elif 'day_month' in result['model_name'].lower():
                feature_names = ['DayOfYear', 'Month']
            elif 'variety' in result['model_name'].lower():
                feature_names = ['Variety_PIE TYPE', 'Variety_MINIATURE', 'Variety_FAIRYTALE', 'Variety_MIXED HEIRLOOM VARIETIES']
            else:
                feature_names = [f'feature_{j}' for j in range(getattr(result.get('X_test', []), 'shape', [0, 1])[1])]
        
        # Store metadata
        model_metadata[model_key] = {
            'model_file': model_filename,
            'model_type': result['model_name'],
            'feature_names': feature_names,
            'performance': {
                'r2': result['r2'],
                'rmse': np.sqrt(result['mse']),
                'mae': result.get('mae', 0)
            },
            'training_data_shape': result.get('data_shape', 'unknown')
        }
        
        print(f"Saved model: {model_path}")
    
    # Save metadata as JSON
    import json
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"Model metadata saved to: {metadata_path}")
    return model_metadata


def main():
    parser = argparse.ArgumentParser(description='Complete ML pipeline for pumpkin price prediction')
    parser.add_argument('--data_path', default='examples/02-basic-ml/data/US-pumpkins.csv',
                       help='Path to the pumpkin CSV data file')
    parser.add_argument('--output_dir', default='outputs/02-basic-ml',
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Handle relative paths from current working directory
    original_cwd = os.environ.get('BUILD_WORKING_DIRECTORY', os.getcwd())
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(original_cwd, args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process data
    pumpkins_df = load_and_process_data(args.data_path)
    
    # Compare models and feature sets
    results, best_result = compare_models_and_features(pumpkins_df, args.output_dir)
    
    # Create visualization for best model
    create_prediction_visualization(best_result, pumpkins_df, args.output_dir)
    
    # Save results summary
    save_results_summary(results, best_result, args.output_dir)
    
    # Save trained models for evaluation
    model_metadata = save_models(results, args.output_dir)
    
    print("\nML Pipeline analysis complete!")
    print(f"Check {args.output_dir} for detailed results and visualizations.")
    print(f"Trained models saved in {args.output_dir}/models/")


if __name__ == "__main__":
    main()