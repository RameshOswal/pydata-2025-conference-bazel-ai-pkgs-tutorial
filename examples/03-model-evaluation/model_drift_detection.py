"""
Model Drift Detection - Monitor model performance over time.

This module demonstrates:
1. Data drift detection
2. Model performance monitoring
3. Statistical tests for distribution changes
4. Alerts and recommendations for model retraining
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')


def load_data_with_time_splits(data_path, split_date=None):
    """Load data and split into reference and current periods."""
    # Read the data
    pumpkins = pd.read_csv(data_path)
    
    # Process data
    columns_to_select = ['City Name', 'Package', 'Variety', 'Date', 'Low Price', 'High Price']
    pumpkins = pumpkins[columns_to_select]
    pumpkins = pumpkins[pumpkins['Variety'].isin(['PIE TYPE', 'CARVING'])]
    
    # Convert date and create features
    pumpkins['Date'] = pd.to_datetime(pumpkins['Date'])
    pumpkins['Month'] = pumpkins['Date'].dt.month
    pumpkins['DayOfYear'] = pumpkins['Date'].dt.dayofyear
    
    # Create price
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2
    
    processed_data = pd.DataFrame({
        'Month': pumpkins['Month'],
        'DayOfYear': pumpkins['DayOfYear'], 
        'Variety': pumpkins['Variety'], 
        'City': pumpkins['City Name'], 
        'Package': pumpkins['Package'], 
        'Date': pumpkins['Date'],
        'Price': price
    })
    
    # Adjust prices based on package size
    processed_data.loc[processed_data['Package'].str.contains('1 1/9'), 'Price'] = price/1.1
    processed_data.loc[processed_data['Package'].str.contains('1/2'), 'Price'] = price*2
    
    # Sort by date
    processed_data = processed_data.sort_values('Date')
    
    # Split data - use median date if no split_date provided
    if split_date is None:
        split_date = processed_data['Date'].median()
    else:
        split_date = pd.to_datetime(split_date)
    
    reference_data = processed_data[processed_data['Date'] <= split_date]
    current_data = processed_data[processed_data['Date'] > split_date]
    
    print(f"Data loaded and split:")
    print(f"  Reference period: {reference_data['Date'].min()} to {reference_data['Date'].max()}")
    print(f"  Current period: {current_data['Date'].min()} to {current_data['Date'].max()}")
    print(f"  Reference samples: {len(reference_data)}")
    print(f"  Current samples: {len(current_data)}")
    
    return reference_data, current_data, split_date


def detect_data_drift(reference_data, current_data, features, alpha=0.05):
    """Detect data drift using statistical tests."""
    drift_results = {}
    
    for feature in features:
        if feature in reference_data.columns and feature in current_data.columns:
            ref_values = reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()
            
            # Skip if not enough data
            if len(ref_values) < 10 or len(curr_values) < 10:
                drift_results[feature] = {
                    'ks_statistic': np.nan,
                    'ks_pvalue': np.nan,
                    'mw_statistic': np.nan,
                    'mw_pvalue': np.nan,
                    'drift_detected': False,
                    'reason': 'Insufficient data'
                }
                continue
            
            # Kolmogorov-Smirnov test for distribution difference
            ks_stat, ks_pvalue = ks_2samp(ref_values, curr_values)
            
            # Mann-Whitney U test for median difference
            mw_stat, mw_pvalue = mannwhitneyu(ref_values, curr_values, alternative='two-sided')
            
            # Detect drift if either test is significant
            drift_detected = (ks_pvalue < alpha) or (mw_pvalue < alpha)
            
            drift_results[feature] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'mw_statistic': mw_stat,
                'mw_pvalue': mw_pvalue,
                'drift_detected': drift_detected,
                'ref_mean': ref_values.mean(),
                'curr_mean': curr_values.mean(),
                'ref_std': ref_values.std(),
                'curr_std': curr_values.std(),
                'mean_change_pct': ((curr_values.mean() - ref_values.mean()) / ref_values.mean() * 100) if ref_values.mean() != 0 else 0
            }
    
    return drift_results


def simulate_model_performance_monitoring(models_info, reference_data, current_data):
    """Simulate model performance over time periods."""
    if not models_info:
        # Create a simple model for demonstration
        from sklearn.linear_model import LinearRegression
        X_ref = reference_data[['DayOfYear', 'Month']]
        y_ref = reference_data['Price']
        
        model = LinearRegression()
        model.fit(X_ref, y_ref)
        
        models_info = {
            'simple_model': {
                'model': model,
                'metadata': {
                    'model_type': 'LinearRegression',
                    'feature_names': ['DayOfYear', 'Month']
                }
            }
        }
    
    performance_monitoring = {}
    
    for model_name, model_info in models_info.items():
        model = model_info['model']
        feature_names = model_info['metadata']['feature_names']
        
        # Prepare features for both periods
        if len(feature_names) == 2 and 'DayOfYear' in feature_names and 'Month' in feature_names:
            X_ref = reference_data[['DayOfYear', 'Month']]
            X_curr = current_data[['DayOfYear', 'Month']]
        else:
            # Use simple features as fallback
            X_ref = reference_data[['DayOfYear', 'Month']]
            X_curr = current_data[['DayOfYear', 'Month']]
        
        y_ref = reference_data['Price']
        y_curr = current_data['Price']
        
        # Calculate performance on reference data
        ref_pred = model.predict(X_ref)
        ref_r2 = r2_score(y_ref, ref_pred)
        ref_rmse = np.sqrt(mean_squared_error(y_ref, ref_pred))
        
        # Calculate performance on current data
        curr_pred = model.predict(X_curr)
        curr_r2 = r2_score(y_curr, curr_pred)
        curr_rmse = np.sqrt(mean_squared_error(y_curr, curr_pred))
        
        # Calculate performance degradation
        r2_degradation = ref_r2 - curr_r2
        rmse_degradation = curr_rmse - ref_rmse
        
        performance_monitoring[model_name] = {
            'reference_performance': {
                'r2': ref_r2,
                'rmse': ref_rmse,
                'predictions': ref_pred
            },
            'current_performance': {
                'r2': curr_r2,
                'rmse': curr_rmse,
                'predictions': curr_pred
            },
            'degradation': {
                'r2_change': r2_degradation,
                'rmse_change': rmse_degradation,
                'r2_change_pct': (r2_degradation / ref_r2 * 100) if ref_r2 != 0 else 0,
                'rmse_change_pct': (rmse_degradation / ref_rmse * 100) if ref_rmse != 0 else 0
            },
            'needs_retraining': abs(r2_degradation) > 0.05 or abs(rmse_degradation / ref_rmse) > 0.2
        }
    
    return performance_monitoring


def create_drift_visualizations(reference_data, current_data, drift_results, 
                               performance_monitoring, output_dir):
    """Create comprehensive drift detection visualizations."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Feature distribution comparison
    features_to_plot = ['DayOfYear', 'Month', 'Price']
    colors = ['blue', 'red', 'green']
    
    for i, feature in enumerate(features_to_plot):
        if feature in reference_data.columns and feature in current_data.columns:
            ref_values = reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()
            
            ax1.hist(ref_values, bins=20, alpha=0.5, label=f'Reference {feature}', 
                    color=colors[i], density=True)
            ax1.hist(curr_values, bins=20, alpha=0.5, label=f'Current {feature}', 
                    color=colors[i], density=True, linestyle='--')
    
    ax1.set_title('Feature Distribution Comparison')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drift detection results
    drift_features = list(drift_results.keys())
    drift_scores = [drift_results[f]['ks_statistic'] for f in drift_features]
    drift_detected = [drift_results[f]['drift_detected'] for f in drift_features]
    
    colors_drift = ['red' if detected else 'green' for detected in drift_detected]
    
    ax2.bar(range(len(drift_features)), drift_scores, color=colors_drift, alpha=0.7)
    ax2.set_title('Data Drift Detection (KS Test)')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('KS Statistic')
    ax2.set_xticks(range(len(drift_features)))
    ax2.set_xticklabels(drift_features, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Model performance over time
    if performance_monitoring:
        model_names = list(performance_monitoring.keys())
        ref_r2 = [performance_monitoring[m]['reference_performance']['r2'] for m in model_names]
        curr_r2 = [performance_monitoring[m]['current_performance']['r2'] for m in model_names]
        
        x_pos = np.arange(len(model_names))
        ax3.bar(x_pos - 0.2, ref_r2, 0.4, label='Reference Period', alpha=0.7, color='blue')
        ax3.bar(x_pos + 0.2, curr_r2, 0.4, label='Current Period', alpha=0.7, color='red')
        ax3.set_title('Model Performance Comparison')
        ax3.set_ylabel('R² Score')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance degradation
        r2_changes = [performance_monitoring[m]['degradation']['r2_change'] for m in model_names]
        colors_perf = ['red' if change > 0.05 else 'orange' if change > 0.02 else 'green' 
                      for change in r2_changes]
        
        ax4.bar(range(len(model_names)), r2_changes, color=colors_perf, alpha=0.7)
        ax4.set_title('Model Performance Degradation')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('R² Score Change')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.axhline(y=0.05, color='red', linestyle='--', label='Retraining Threshold')
        ax4.axhline(y=0.02, color='orange', linestyle='--', label='Warning Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drift_detection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Drift detection visualizations saved to {output_dir}/drift_detection.png")


def generate_drift_report(drift_results, performance_monitoring, split_date, output_dir):
    """Generate comprehensive drift detection report."""
    report_path = os.path.join(output_dir, 'drift_detection_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL DRIFT DETECTION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Split Date: {split_date.strftime('%Y-%m-%d')}\n\n")
        
        # Data drift summary
        f.write("DATA DRIFT ANALYSIS\n")
        f.write("-" * 30 + "\n")
        
        drift_detected_count = sum(1 for result in drift_results.values() 
                                 if result['drift_detected'])
        total_features = len(drift_results)
        
        f.write(f"Features analyzed: {total_features}\n")
        f.write(f"Features with drift detected: {drift_detected_count}\n")
        f.write(f"Drift detection rate: {drift_detected_count/total_features*100:.1f}%\n\n")
        
        for feature, result in drift_results.items():
            f.write(f"{feature.upper()}:\n")
            if result['drift_detected']:
                f.write("  🚨 DRIFT DETECTED\n")
            else:
                f.write("  ✅ No drift detected\n")
            
            f.write(f"  KS Test: statistic={result['ks_statistic']:.4f}, p-value={result['ks_pvalue']:.4f}\n")
            f.write(f"  Mann-Whitney Test: statistic={result['mw_statistic']:.4f}, p-value={result['mw_pvalue']:.4f}\n")
            
            if 'ref_mean' in result:
                f.write(f"  Reference mean: {result['ref_mean']:.4f} ± {result['ref_std']:.4f}\n")
                f.write(f"  Current mean: {result['curr_mean']:.4f} ± {result['curr_std']:.4f}\n")
                f.write(f"  Mean change: {result['mean_change_pct']:.2f}%\n")
            
            f.write("\n")
        
        # Model performance monitoring
        if performance_monitoring:
            f.write("MODEL PERFORMANCE MONITORING\n")
            f.write("-" * 35 + "\n")
            
            models_needing_retraining = sum(1 for result in performance_monitoring.values() 
                                          if result['needs_retraining'])
            total_models = len(performance_monitoring)
            
            f.write(f"Models monitored: {total_models}\n")
            f.write(f"Models needing retraining: {models_needing_retraining}\n\n")
            
            for model_name, result in performance_monitoring.items():
                f.write(f"{model_name.upper()}:\n")
                
                if result['needs_retraining']:
                    f.write("  🚨 RETRAINING RECOMMENDED\n")
                else:
                    f.write("  ✅ Performance stable\n")
                
                ref_perf = result['reference_performance']
                curr_perf = result['current_performance']
                degradation = result['degradation']
                
                f.write(f"  Reference performance: R²={ref_perf['r2']:.4f}, RMSE={ref_perf['rmse']:.4f}\n")
                f.write(f"  Current performance: R²={curr_perf['r2']:.4f}, RMSE={curr_perf['rmse']:.4f}\n")
                f.write(f"  R² change: {degradation['r2_change']:.4f} ({degradation['r2_change_pct']:.2f}%)\n")
                f.write(f"  RMSE change: {degradation['rmse_change']:.4f} ({degradation['rmse_change_pct']:.2f}%)\n")
                f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        
        if drift_detected_count > 0:
            f.write("🚨 Data drift detected! Consider the following actions:\n")
            f.write("1. Investigate the root cause of distribution changes\n")
            f.write("2. Collect more recent training data\n")
            f.write("3. Retrain models with updated data\n")
            f.write("4. Implement online learning or model adaptation\n")
            f.write("5. Review data preprocessing pipeline\n\n")
        else:
            f.write("✅ No significant data drift detected.\n")
            f.write("Continue monitoring but no immediate action required.\n\n")
        
        if performance_monitoring and models_needing_retraining > 0:
            f.write("🚨 Model performance degradation detected!\n")
            f.write("Models requiring immediate retraining:\n")
            for model_name, result in performance_monitoring.items():
                if result['needs_retraining']:
                    f.write(f"  - {model_name}\n")
            f.write("\n")
        
        f.write("Monitoring best practices:\n")
        f.write("- Set up automated drift detection pipelines\n")
        f.write("- Define clear thresholds for retraining triggers\n")
        f.write("- Monitor both data drift and model performance\n")
        f.write("- Maintain historical performance baselines\n")
        f.write("- Implement gradual model rollout for updates\n")
    
    print(f"Drift detection report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Model drift detection and monitoring')
    parser.add_argument('--data_path', 
                       default='examples/02-basic-ml/data/US-pumpkins.csv',
                       help='Path to data file')
    parser.add_argument('--models_dir', 
                       default=None,
                       help='Directory containing saved models (optional)')
    parser.add_argument('--output_dir', 
                       default='output_drift',
                       help='Output directory for drift analysis results')
    parser.add_argument('--split_date', 
                       default=None,
                       help='Date to split reference/current data (YYYY-MM-DD)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level for drift detection')
    
    args = parser.parse_args()
    
    # Handle relative paths
    original_cwd = os.getcwd()
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(original_cwd, args.data_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(original_cwd, args.output_dir)
    if args.models_dir and not os.path.isabs(args.models_dir):
        args.models_dir = os.path.join(original_cwd, args.models_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting Model Drift Detection...")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Significance level: {args.alpha}")
    
    # Load data with time splits
    reference_data, current_data, split_date = load_data_with_time_splits(
        args.data_path, args.split_date
    )
    
    # Detect data drift
    features_to_monitor = ['DayOfYear', 'Month', 'Price']
    print(f"\nDetecting drift in features: {features_to_monitor}")
    
    drift_results = detect_data_drift(reference_data, current_data, 
                                    features_to_monitor, args.alpha)
    
    # Load models if available
    models_info = {}
    if args.models_dir and os.path.exists(args.models_dir):
        try:
            from model_evaluator import load_models_and_metadata
            models_info = load_models_and_metadata(args.models_dir)
            print(f"Loaded {len(models_info)} models for monitoring")
        except Exception as e:
            print(f"Could not load models: {e}")
            print("Proceeding with simulated model monitoring")
    
    # Monitor model performance
    print("\nMonitoring model performance...")
    performance_monitoring = simulate_model_performance_monitoring(
        models_info, reference_data, current_data
    )
    
    # Create visualizations
    create_drift_visualizations(reference_data, current_data, drift_results, 
                              performance_monitoring, args.output_dir)
    
    # Generate report
    generate_drift_report(drift_results, performance_monitoring, split_date, args.output_dir)
    
    # Summary
    drift_detected_count = sum(1 for result in drift_results.values() 
                             if result['drift_detected'])
    models_needing_retraining = sum(1 for result in performance_monitoring.values() 
                                  if result['needs_retraining'])
    
    print(f"\nDrift Detection Summary:")
    print(f"  Features with drift: {drift_detected_count}/{len(drift_results)}")
    print(f"  Models needing retraining: {models_needing_retraining}/{len(performance_monitoring)}")
    print(f"  Check {args.output_dir} for detailed analysis.")


if __name__ == "__main__":
    main()