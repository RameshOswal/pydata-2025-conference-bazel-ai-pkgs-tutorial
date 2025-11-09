"""
Model Evaluation Pipeline - Load saved models and evaluate performance.

This module demonstrates how to:
1. Load pre-trained models from disk
2. Evaluate model performance on new data
3. Compare multiple models
4. Generate evaluation reports and visualizations
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import joblib


def find_runfiles_path(relative_path):
    """Find a file in Bazel runfiles."""
    # Try runfiles first
    runfiles_dir = os.environ.get('RUNFILES_DIR')
    if runfiles_dir:
        full_path = os.path.join(runfiles_dir, '_main', relative_path)
        if os.path.exists(full_path):
            return full_path
    
    # Try current directory and parent directories
    current_dir = os.getcwd()
    for _ in range(5):  # Try up to 5 levels up
        test_path = os.path.join(current_dir, relative_path)
        if os.path.exists(test_path):
            return test_path
        current_dir = os.path.dirname(current_dir)
    
    # Return the original path as fallback
    return relative_path


def find_models_directory():
    """Find the models directory, trying different possible locations."""
    possible_paths = [
        "outputs/02-basic-ml/models",  # Relative to workspace root
        "../../../outputs/02-basic-ml/models",  # From bazel runfiles
        "outputs/02-basic-ml/models",  # Bazel runfiles path
    ]
    
    for path in possible_paths:
        resolved_path = find_runfiles_path(path)
        if os.path.exists(resolved_path):
            return resolved_path
    
    return None
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


def load_models_and_metadata(models_dir):
    """Load saved models and their metadata."""
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    models = {}
    for model_name, info in metadata.items():
        model_path = os.path.join(models_dir, info['model_file'])
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            models[model_name] = {
                'model': model,
                'metadata': info
            }
            print(f"Loaded model: {model_name} ({info['model_type']})")
        else:
            print(f"Warning: Model file not found: {model_path}")
    
    return models


def load_and_prepare_evaluation_data(data_path, models_info):
    """Load data and prepare features for evaluation."""
    # Read the data
    pumpkins = pd.read_csv(data_path)
    
    # Process data (similar to training pipeline)
    columns_to_select = ['City Name', 'Package', 'Variety', 'Date', 'Low Price', 'High Price']
    pumpkins = pumpkins[columns_to_select]
    
    # Filter to specific varieties for consistency
    pumpkins = pumpkins[pumpkins['Variety'].isin(['PIE TYPE', 'CARVING'])]
    
    # Convert date and create features
    pumpkins['Date'] = pd.to_datetime(pumpkins['Date'])
    pumpkins['Month'] = pumpkins['Date'].dt.month
    pumpkins['DayOfYear'] = pumpkins['Date'].dt.dayofyear
    
    # Create price
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2
    
    new_pumpkins = pd.DataFrame({
        'Month': pumpkins['Month'],
        'DayOfYear': pumpkins['DayOfYear'], 
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
    
    print(f"Evaluation data loaded: {len(new_pumpkins)} samples")
    
    # Prepare feature sets for each model
    feature_datasets = {}
    
    for model_name, model_info in models_info.items():
        feature_names = model_info['metadata']['feature_names']
        
        # Determine feature set based on model name or feature names
        if 'day_only' in model_name or (len(feature_names) == 1 and 'DayOfYear' in feature_names):
            X = new_pumpkins[['DayOfYear']]
        elif 'day_month' in model_name or (len(feature_names) == 2 and 'DayOfYear' in feature_names and 'Month' in feature_names):
            X = new_pumpkins[['DayOfYear', 'Month']]
        elif 'variety_only' in model_name or (any('Variety_' in col for col in feature_names) and len(feature_names) <= 5):
            X = pd.get_dummies(new_pumpkins['Variety'], prefix='Variety')
        elif 'complete' in model_name or len(feature_names) > 10:
            X = pd.get_dummies(new_pumpkins['Variety'], prefix='Variety') \
                .join(new_pumpkins[['Month', 'DayOfYear']]) \
                .join(pd.get_dummies(new_pumpkins['City'], prefix='City')) \
                .join(pd.get_dummies(new_pumpkins['Package'], prefix='Package'))
        else:
            # Default: use available features based on feature names
            available_features = []
            if 'DayOfYear' in feature_names:
                available_features.append('DayOfYear')
            if 'Month' in feature_names:
                available_features.append('Month')
            if available_features:
                X = new_pumpkins[available_features]
            else:
                # Fallback to simple features
                X = new_pumpkins[['DayOfYear', 'Month']]
        
        # Ensure feature columns match training data
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0  # Add missing columns with zeros
        
        X = X[feature_names]  # Reorder to match training
        
        feature_datasets[model_name] = {
            'X': X,
            'y': new_pumpkins['Price']
        }
    
    return feature_datasets, new_pumpkins


def evaluate_model_performance(model, X, y, model_name):
    """Evaluate a single model's performance."""
    predictions = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    
    # Additional metrics
    residuals = y - predictions
    residual_std = np.std(residuals)
    
    evaluation_results = {
        'model_name': model_name,
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'residual_std': residual_std,
        'predictions': predictions,
        'residuals': residuals,
        'n_samples': len(y)
    }
    
    print(f"\n{model_name} Evaluation Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Residual Std: {residual_std:.4f}")
    
    return evaluation_results


def perform_cross_validation(models_info, feature_datasets, cv_folds=5):
    """Perform cross-validation on loaded models."""
    cv_results = {}
    
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    
    for model_name, model_info in models_info.items():
        model = model_info['model']
        X = feature_datasets[model_name]['X']
        y = feature_datasets[model_name]['y']
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        cv_rmse_scores = cross_val_score(model, X, y, cv=cv_folds, 
                                       scoring='neg_mean_squared_error')
        cv_rmse_scores = np.sqrt(-cv_rmse_scores)
        
        cv_results[model_name] = {
            'r2_scores': cv_scores,
            'r2_mean': cv_scores.mean(),
            'r2_std': cv_scores.std(),
            'rmse_scores': cv_rmse_scores,
            'rmse_mean': cv_rmse_scores.mean(),
            'rmse_std': cv_rmse_scores.std()
        }
        
        print(f"\n{model_name} Cross-Validation:")
        print(f"  R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  RMSE: {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")
    
    return cv_results


def create_evaluation_visualizations(evaluation_results, cv_results, output_dir):
    """Create comprehensive evaluation visualizations."""
    
    # Performance comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = list(evaluation_results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # R² comparison
    r2_scores = [evaluation_results[name]['r2_score'] for name in model_names]
    cv_r2_means = [cv_results[name]['r2_mean'] for name in model_names]
    cv_r2_stds = [cv_results[name]['r2_std'] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    ax1.bar(x_pos - 0.2, r2_scores, 0.4, label='Holdout Test', color=colors[:len(model_names)], alpha=0.7)
    ax1.bar(x_pos + 0.2, cv_r2_means, 0.4, yerr=cv_r2_stds, 
            label='Cross-Validation', color=colors[:len(model_names)], alpha=0.5)
    ax1.set_title('Model Performance: R² Score')
    ax1.set_ylabel('R² Score')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE comparison
    rmse_scores = [evaluation_results[name]['rmse'] for name in model_names]
    cv_rmse_means = [cv_results[name]['rmse_mean'] for name in model_names]
    cv_rmse_stds = [cv_results[name]['rmse_std'] for name in model_names]
    
    ax2.bar(x_pos - 0.2, rmse_scores, 0.4, label='Holdout Test', color=colors[:len(model_names)], alpha=0.7)
    ax2.bar(x_pos + 0.2, cv_rmse_means, 0.4, yerr=cv_rmse_stds, 
            label='Cross-Validation', color=colors[:len(model_names)], alpha=0.5)
    ax2.set_title('Model Performance: RMSE')
    ax2.set_ylabel('RMSE')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Residual analysis for best model
    best_model = max(evaluation_results.keys(), key=lambda x: evaluation_results[x]['r2_score'])
    residuals = evaluation_results[best_model]['residuals']
    predictions = evaluation_results[best_model]['predictions']
    
    ax3.scatter(predictions, residuals, alpha=0.6, color='blue')
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals')
    ax3.set_title(f'Residual Plot - {best_model}')
    ax3.grid(True, alpha=0.3)
    
    # Q-Q plot for residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title(f'Q-Q Plot - {best_model}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_evaluation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation visualizations saved to {output_dir}/model_evaluation.png")


def generate_evaluation_report(models_info, evaluation_results, cv_results, output_dir):
    """Generate a comprehensive evaluation report."""
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Training performance summary
        f.write("TRAINING PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        for model_name, model_info in models_info.items():
            metadata = model_info['metadata']
            f.write(f"\n{model_name.upper()} ({metadata['model_type']}):\n")
            f.write(f"  Features: {len(metadata['feature_names'])}\n")
            f.write(f"  Training R²: {metadata['performance']['r2']:.4f}\n")
            f.write(f"  Training RMSE: {metadata['performance']['rmse']:.4f}\n")
            f.write(f"  Training MAE: {metadata['performance']['mae']:.4f}\n")
        
        # Evaluation performance
        f.write("\n\nEVALUATION PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        for model_name, results in evaluation_results.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  Evaluation R²: {results['r2_score']:.4f}\n")
            f.write(f"  Evaluation RMSE: {results['rmse']:.4f}\n")
            f.write(f"  Evaluation MAE: {results['mae']:.4f}\n")
            f.write(f"  Samples: {results['n_samples']}\n")
        
        # Cross-validation results
        f.write("\n\nCROSS-VALIDATION RESULTS\n")
        f.write("-" * 40 + "\n")
        for model_name, results in cv_results.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  CV R² Score: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}\n")
            f.write(f"  CV RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}\n")
        
        # Model ranking
        f.write("\n\nMODEL RANKING (by R² Score)\n")
        f.write("-" * 40 + "\n")
        ranked_models = sorted(evaluation_results.items(), 
                             key=lambda x: x[1]['r2_score'], reverse=True)
        for i, (model_name, results) in enumerate(ranked_models, 1):
            f.write(f"{i}. {model_name}: {results['r2_score']:.4f}\n")
        
        # Recommendations
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        best_model = ranked_models[0][0]
        f.write(f"Best performing model: {best_model}\n")
        f.write(f"This model achieved the highest R² score of {ranked_models[0][1]['r2_score']:.4f}\n")
        
        if len(ranked_models) > 1:
            second_best = ranked_models[1][0]
            performance_gap = ranked_models[0][1]['r2_score'] - ranked_models[1][1]['r2_score']
            f.write(f"Performance gap to second-best ({second_best}): {performance_gap:.4f}\n")
        
        f.write("\nConsider the trade-off between model complexity and performance.\n")
        f.write("Simpler models may be more interpretable and robust.\n")
    
    print(f"Evaluation report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved ML models')
    parser.add_argument('--models_dir', 
                       default=None,
                       help='Directory containing saved models (auto-detected if not provided)')
    parser.add_argument('--data_path', 
                       default='examples/02-basic-ml/data/US-pumpkins.csv',
                       help='Path to evaluation data')
    parser.add_argument('--output_dir', 
                       default='evaluation_results',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Auto-detect models directory if not provided
    if args.models_dir is None:
        args.models_dir = find_models_directory()
        if args.models_dir is None:
            print("Error: Could not find models directory. Please run training pipeline first or specify --models_dir")
            return
    
    # Handle relative paths
    original_cwd = os.getcwd()
    if not os.path.isabs(args.data_path):
        args.data_path = find_runfiles_path(args.data_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(original_cwd, args.output_dir)
    if not os.path.isabs(args.models_dir):
        args.models_dir = os.path.join(original_cwd, args.models_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting Model Evaluation Pipeline...")
    print(f"Models directory: {args.models_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Load models and metadata
        models_info = load_models_and_metadata(args.models_dir)
        
        if not models_info:
            print("No models found! Please run the training pipeline first.")
            return
        
        # Prepare evaluation data
        feature_datasets, raw_data = load_and_prepare_evaluation_data(args.data_path, models_info)
        
        # Evaluate each model
        evaluation_results = {}
        for model_name, model_info in models_info.items():
            model = model_info['model']
            X = feature_datasets[model_name]['X']
            y = feature_datasets[model_name]['y']
            
            results = evaluate_model_performance(model, X, y, model_name)
            evaluation_results[model_name] = results
        
        # Perform cross-validation
        cv_results = perform_cross_validation(models_info, feature_datasets)
        
        # Create visualizations
        create_evaluation_visualizations(evaluation_results, cv_results, args.output_dir)
        
        # Generate comprehensive report
        generate_evaluation_report(models_info, evaluation_results, cv_results, args.output_dir)
        
        print("\nModel evaluation complete!")
        print(f"Check {args.output_dir} for detailed evaluation results.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()