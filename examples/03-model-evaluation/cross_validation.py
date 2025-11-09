"""
Cross-Validation Pipeline - Advanced model validation techniques.

This module demonstrates:
1. K-fold cross-validation
2. Stratified cross-validation
3. Time series cross-validation
4. Learning curves
5. Validation curves
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
from sklearn.model_selection import (
    cross_val_score, cross_validate, KFold, StratifiedKFold,
    TimeSeriesSplit, learning_curve, validation_curve
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')


def load_data_for_cv(data_path):
    """Load and prepare data for cross-validation."""
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
        'Date': pumpkins['Date'],
        'Price': price
    })
    
    # Adjust prices based on package size
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/1.1
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price*2
    
    print(f"Cross-validation data loaded: {len(new_pumpkins)} samples")
    return new_pumpkins


def perform_kfold_cv(X, y, model, k=5, random_state=42):
    """Perform k-fold cross-validation."""
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring, 
                               return_train_score=True)
    
    results = {
        'method': 'K-Fold CV',
        'k': k,
        'test_r2': cv_results['test_r2'],
        'train_r2': cv_results['train_r2'],
        'test_rmse': np.sqrt(-cv_results['test_neg_mean_squared_error']),
        'train_rmse': np.sqrt(-cv_results['train_neg_mean_squared_error']),
        'test_mae': -cv_results['test_neg_mean_absolute_error'],
        'train_mae': -cv_results['train_neg_mean_absolute_error']
    }
    
    return results


def perform_stratified_cv(X, y, model, k=5, random_state=42):
    """Perform stratified cross-validation (binning continuous target)."""
    # Create bins for stratification
    y_binned = pd.cut(y, bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    cv_results = cross_validate(model, X, y, cv=skfold.split(X, y_binned), 
                               scoring=scoring, return_train_score=True)
    
    results = {
        'method': 'Stratified CV',
        'k': k,
        'test_r2': cv_results['test_r2'],
        'train_r2': cv_results['train_r2'],
        'test_rmse': np.sqrt(-cv_results['test_neg_mean_squared_error']),
        'train_rmse': np.sqrt(-cv_results['train_neg_mean_squared_error']),
        'test_mae': -cv_results['test_neg_mean_absolute_error'],
        'train_mae': -cv_results['train_neg_mean_absolute_error']
    }
    
    return results


def perform_time_series_cv(X, y, model, n_splits=5):
    """Perform time series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    cv_results = cross_validate(model, X, y, cv=tscv, scoring=scoring, 
                               return_train_score=True)
    
    results = {
        'method': 'Time Series CV',
        'n_splits': n_splits,
        'test_r2': cv_results['test_r2'],
        'train_r2': cv_results['train_r2'],
        'test_rmse': np.sqrt(-cv_results['test_neg_mean_squared_error']),
        'train_rmse': np.sqrt(-cv_results['train_neg_mean_squared_error']),
        'test_mae': -cv_results['test_neg_mean_absolute_error'],
        'train_mae': -cv_results['train_neg_mean_absolute_error']
    }
    
    return results


def generate_learning_curves(X, y, model, cv=5):
    """Generate learning curves to assess model performance vs training size."""
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv, 
        scoring='r2', shuffle=True, random_state=42
    )
    
    return {
        'train_sizes': train_sizes,
        'train_scores_mean': train_scores.mean(axis=1),
        'train_scores_std': train_scores.std(axis=1),
        'val_scores_mean': val_scores.mean(axis=1),
        'val_scores_std': val_scores.std(axis=1)
    }


def generate_validation_curves(X, y, param_name, param_range, cv=5):
    """Generate validation curves for polynomial degree."""
    if param_name == 'polynomialfeatures__degree':
        model = make_pipeline(PolynomialFeatures(), LinearRegression())
    else:
        model = LinearRegression()
    
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring='r2'
    )
    
    return {
        'param_range': param_range,
        'train_scores_mean': train_scores.mean(axis=1),
        'train_scores_std': train_scores.std(axis=1),
        'val_scores_mean': val_scores.mean(axis=1),
        'val_scores_std': val_scores.std(axis=1)
    }


def create_cv_visualizations(cv_results, learning_curves, validation_curves, output_dir):
    """Create comprehensive cross-validation visualizations."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # CV method comparison
    methods = [result['method'] for result in cv_results]
    test_r2_means = [result['test_r2'].mean() for result in cv_results]
    test_r2_stds = [result['test_r2'].std() for result in cv_results]
    train_r2_means = [result['train_r2'].mean() for result in cv_results]
    train_r2_stds = [result['train_r2'].std() for result in cv_results]
    
    x_pos = np.arange(len(methods))
    ax1.bar(x_pos - 0.2, test_r2_means, 0.4, yerr=test_r2_stds, 
            label='Test', alpha=0.7, color='blue')
    ax1.bar(x_pos + 0.2, train_r2_means, 0.4, yerr=train_r2_stds, 
            label='Train', alpha=0.7, color='red')
    ax1.set_title('Cross-Validation Methods Comparison')
    ax1.set_ylabel('R² Score')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE comparison
    test_rmse_means = [result['test_rmse'].mean() for result in cv_results]
    test_rmse_stds = [result['test_rmse'].std() for result in cv_results]
    train_rmse_means = [result['train_rmse'].mean() for result in cv_results]
    train_rmse_stds = [result['train_rmse'].std() for result in cv_results]
    
    ax2.bar(x_pos - 0.2, test_rmse_means, 0.4, yerr=test_rmse_stds, 
            label='Test', alpha=0.7, color='blue')
    ax2.bar(x_pos + 0.2, train_rmse_means, 0.4, yerr=train_rmse_stds, 
            label='Train', alpha=0.7, color='red')
    ax2.set_title('RMSE Comparison')
    ax2.set_ylabel('RMSE')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning curves
    if learning_curves:
        lc = learning_curves
        ax3.plot(lc['train_sizes'], lc['train_scores_mean'], 'o-', color='red', 
                label='Training score')
        ax3.fill_between(lc['train_sizes'], 
                        lc['train_scores_mean'] - lc['train_scores_std'],
                        lc['train_scores_mean'] + lc['train_scores_std'], 
                        alpha=0.1, color='red')
        
        ax3.plot(lc['train_sizes'], lc['val_scores_mean'], 'o-', color='blue', 
                label='Cross-validation score')
        ax3.fill_between(lc['train_sizes'], 
                        lc['val_scores_mean'] - lc['val_scores_std'],
                        lc['val_scores_mean'] + lc['val_scores_std'], 
                        alpha=0.1, color='blue')
        
        ax3.set_title('Learning Curves')
        ax3.set_xlabel('Training Set Size')
        ax3.set_ylabel('R² Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Validation curves
    if validation_curves:
        vc = validation_curves
        ax4.plot(vc['param_range'], vc['train_scores_mean'], 'o-', color='red', 
                label='Training score')
        ax4.fill_between(vc['param_range'], 
                        vc['train_scores_mean'] - vc['train_scores_std'],
                        vc['train_scores_mean'] + vc['train_scores_std'], 
                        alpha=0.1, color='red')
        
        ax4.plot(vc['param_range'], vc['val_scores_mean'], 'o-', color='blue', 
                label='Cross-validation score')
        ax4.fill_between(vc['param_range'], 
                        vc['val_scores_mean'] - vc['val_scores_std'],
                        vc['val_scores_mean'] + vc['val_scores_std'], 
                        alpha=0.1, color='blue')
        
        ax4.set_title('Validation Curves (Polynomial Degree)')
        ax4.set_xlabel('Polynomial Degree')
        ax4.set_ylabel('R² Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Cross-validation visualizations saved to {output_dir}/cross_validation_analysis.png")


def generate_cv_report(cv_results, learning_curves, validation_curves, output_dir):
    """Generate detailed cross-validation report."""
    report_path = os.path.join(output_dir, 'cross_validation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-VALIDATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # CV Results Summary
        f.write("CROSS-VALIDATION METHODS COMPARISON\n")
        f.write("-" * 50 + "\n")
        
        for result in cv_results:
            f.write(f"\n{result['method']}:\n")
            f.write(f"  Test R² Score: {result['test_r2'].mean():.4f} ± {result['test_r2'].std():.4f}\n")
            f.write(f"  Train R² Score: {result['train_r2'].mean():.4f} ± {result['train_r2'].std():.4f}\n")
            f.write(f"  Test RMSE: {result['test_rmse'].mean():.4f} ± {result['test_rmse'].std():.4f}\n")
            f.write(f"  Train RMSE: {result['train_rmse'].mean():.4f} ± {result['train_rmse'].std():.4f}\n")
            f.write(f"  Test MAE: {result['test_mae'].mean():.4f} ± {result['test_mae'].std():.4f}\n")
            f.write(f"  Train MAE: {result['train_mae'].mean():.4f} ± {result['train_mae'].std():.4f}\n")
            
            # Check for overfitting
            overfitting_gap = result['train_r2'].mean() - result['test_r2'].mean()
            if overfitting_gap > 0.1:
                f.write(f"  ⚠️  Potential overfitting detected (gap: {overfitting_gap:.4f})\n")
            else:
                f.write(f"  ✅ Good train/test balance (gap: {overfitting_gap:.4f})\n")
        
        # Learning curve analysis
        if learning_curves:
            f.write("\n\nLEARNING CURVE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            lc = learning_curves
            
            final_train_score = lc['train_scores_mean'][-1]
            final_val_score = lc['val_scores_mean'][-1]
            improvement = lc['val_scores_mean'][-1] - lc['val_scores_mean'][0]
            
            f.write(f"Final training score: {final_train_score:.4f}\n")
            f.write(f"Final validation score: {final_val_score:.4f}\n")
            f.write(f"Score improvement: {improvement:.4f}\n")
            
            if improvement > 0.05:
                f.write("✅ Model benefits from more training data\n")
            else:
                f.write("⚠️  Limited benefit from additional training data\n")
        
        # Validation curve analysis
        if validation_curves:
            f.write("\n\nVALIDATION CURVE ANALYSIS\n")
            f.write("-" * 32 + "\n")
            vc = validation_curves
            
            best_idx = np.argmax(vc['val_scores_mean'])
            best_param = vc['param_range'][best_idx]
            best_score = vc['val_scores_mean'][best_idx]
            
            f.write(f"Best polynomial degree: {best_param}\n")
            f.write(f"Best validation score: {best_score:.4f}\n")
            
            # Check for overfitting at best parameter
            train_score_at_best = vc['train_scores_mean'][best_idx]
            gap_at_best = train_score_at_best - best_score
            
            if gap_at_best > 0.1:
                f.write(f"⚠️  Overfitting at best parameter (gap: {gap_at_best:.4f})\n")
            else:
                f.write(f"✅ Good generalization at best parameter (gap: {gap_at_best:.4f})\n")
        
        # Recommendations
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        
        # Find best CV method
        best_cv_idx = np.argmax([result['test_r2'].mean() for result in cv_results])
        best_cv_method = cv_results[best_cv_idx]['method']
        
        f.write(f"Recommended CV method: {best_cv_method}\n")
        f.write("This method provides the most reliable performance estimates.\n\n")
        
        f.write("General recommendations:\n")
        f.write("- Use Time Series CV if data has temporal dependencies\n")
        f.write("- Use Stratified CV if target distribution is imbalanced\n")
        f.write("- Use K-Fold CV for general cases\n")
        f.write("- Monitor train/test gap to detect overfitting\n")
        f.write("- Use learning curves to determine optimal dataset size\n")
        f.write("- Use validation curves to tune hyperparameters\n")
    
    print(f"Cross-validation report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Advanced cross-validation analysis')
    parser.add_argument('--data_path', 
                       default='examples/02-basic-ml/data/US-pumpkins.csv',
                       help='Path to data file')
    parser.add_argument('--output_dir', 
                       default='output_cv',
                       help='Output directory for CV results')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds')
    
    args = parser.parse_args()
    
    # Handle relative paths
    original_cwd = os.getcwd()
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(original_cwd, args.data_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(original_cwd, args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting Cross-Validation Analysis...")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"CV folds: {args.cv_folds}")
    
    # Load and prepare data
    pumpkins_df = load_data_for_cv(args.data_path)
    
    # Use simple features for demonstration
    X = pumpkins_df[['DayOfYear', 'Month']]
    y = pumpkins_df['Price']
    
    # Create model
    model = LinearRegression()
    
    print("\nPerforming different CV methods...")
    
    # Perform different CV methods
    cv_results = []
    
    # K-Fold CV
    kfold_results = perform_kfold_cv(X, y, model, k=args.cv_folds)
    cv_results.append(kfold_results)
    print(f"K-Fold CV completed: R² = {kfold_results['test_r2'].mean():.4f}")
    
    # Stratified CV
    stratified_results = perform_stratified_cv(X, y, model, k=args.cv_folds)
    cv_results.append(stratified_results)
    print(f"Stratified CV completed: R² = {stratified_results['test_r2'].mean():.4f}")
    
    # Time Series CV
    ts_results = perform_time_series_cv(X, y, model, n_splits=args.cv_folds)
    cv_results.append(ts_results)
    print(f"Time Series CV completed: R² = {ts_results['test_r2'].mean():.4f}")
    
    # Generate learning curves
    print("\nGenerating learning curves...")
    learning_curves = generate_learning_curves(X, y, model, cv=args.cv_folds)
    
    # Generate validation curves (polynomial degree)
    print("Generating validation curves...")
    param_range = range(1, 6)
    validation_curves = generate_validation_curves(
        X, y, 'polynomialfeatures__degree', param_range, cv=args.cv_folds
    )
    
    # Create visualizations
    create_cv_visualizations(cv_results, learning_curves, validation_curves, args.output_dir)
    
    # Generate report
    generate_cv_report(cv_results, learning_curves, validation_curves, args.output_dir)
    
    print("\nCross-validation analysis complete!")
    print(f"Check {args.output_dir} for detailed results.")


if __name__ == "__main__":
    main()