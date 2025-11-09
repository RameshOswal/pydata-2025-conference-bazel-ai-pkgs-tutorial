# Model Evaluation Pipeline

This directory demonstrates advanced model evaluation techniques using Bazel for dependency management and reproducible builds.

## Overview

The 03-model-evaluation example showcases:

1. **Model Persistence and Loading**: Loading pre-trained models from disk
2. **Comprehensive Evaluation**: Multiple metrics and statistical analysis
3. **Cross-Validation**: Advanced validation techniques
4. **Drift Detection**: Monitoring model performance over time
5. **Automated Testing**: Unit tests for evaluation pipeline

## Components

### 1. Model Evaluator (`model_evaluator.py`)

Loads saved models from the 02-basic-ml pipeline and performs comprehensive evaluation:

- **Model Loading**: Deserializes joblib models with metadata
- **Performance Metrics**: R², RMSE, MAE, residual analysis
- **Cross-Validation**: K-fold validation with confidence intervals
- **Visualization**: Performance comparison plots and residual analysis
- **Reporting**: Detailed evaluation reports with recommendations

**Key Features:**
- Handles multiple models and feature sets
- Automatic feature alignment with training data
- Statistical significance testing
- Model ranking and selection recommendations

### 2. Cross-Validation Pipeline (`cross_validation.py`)

Advanced cross-validation techniques for robust model assessment:

- **K-Fold Cross-Validation**: Standard randomized validation
- **Stratified Cross-Validation**: Preserves target distribution
- **Time Series Cross-Validation**: Respects temporal order
- **Learning Curves**: Training size vs. performance analysis
- **Validation Curves**: Hyperparameter optimization guidance

**Key Features:**
- Multiple CV strategies comparison
- Overfitting detection
- Statistical confidence intervals
- Hyperparameter sensitivity analysis

### 3. Model Drift Detection (`model_drift_detection.py`)

Monitors model performance degradation and data distribution changes:

- **Data Drift Detection**: Statistical tests for distribution changes
- **Performance Monitoring**: Model accuracy tracking over time
- **Alert System**: Automated retraining recommendations
- **Temporal Analysis**: Time-based performance splitting

**Key Features:**
- Kolmogorov-Smirnov and Mann-Whitney U tests
- Performance degradation thresholds
- Visual drift indicators
- Retraining recommendations

### 4. Unit Tests (`test_evaluation.py`)

Comprehensive test suite for evaluation pipeline:

- **Model Loading Tests**: Metadata and serialization validation
- **Evaluation Tests**: Metric calculation verification
- **Drift Detection Tests**: Statistical test validation
- **Data Processing Tests**: Feature engineering verification

## Usage Examples

### Basic Model Evaluation

First, train models using the 02-basic-ml pipeline:

```bash
# Train models and save them
bazel run //examples/02-basic-ml:ml_pipeline
```

Then evaluate the saved models (models directory is auto-detected):

```bash
# Evaluate saved models - models directory auto-detected from Bazel runfiles
bazel run //examples/03-model-evaluation:model_evaluator

# Or specify custom output directory
bazel run //examples/03-model-evaluation:model_evaluator -- \
  --output_dir custom_evaluation_results

# Or specify custom models directory
bazel run //examples/03-model-evaluation:model_evaluator -- \
  --models_dir /path/to/custom/models \
  --output_dir evaluation_results
```

### Cross-Validation Analysis

```bash
# Perform comprehensive cross-validation
bazel run //examples/03-model-evaluation:cross_validation -- \
  --data_path examples/02-basic-ml/data/US-pumpkins.csv \
  --output_dir cv_analysis \
  --cv_folds 10
```

### Drift Detection

```bash
# Monitor for model drift - models directory auto-detected
bazel run //examples/03-model-evaluation:model_drift_detection -- \
  --data_path examples/02-basic-ml/data/US-pumpkins.csv \
  --output_dir drift_analysis \
  --split_date 2017-06-01

# Or specify custom models directory
bazel run //examples/03-model-evaluation:model_drift_detection -- \
  --data_path examples/02-basic-ml/data/US-pumpkins.csv \
  --models_dir /path/to/custom/models \
  --output_dir drift_analysis \
  --split_date 2017-06-01
```

### Running Tests

```bash
# Run evaluation pipeline tests
bazel test //examples/03-model-evaluation:test_evaluation

# Run all tests with detailed output
bazel test //examples/03-model-evaluation:test_evaluation --test_output=all
```

## Output Structure

Each evaluation run creates organized output directories:

```
evaluation_results/
├── model_evaluation.png          # Performance comparison visualizations
├── evaluation_report.txt         # Detailed evaluation report
└── models/                      # (if running from training pipeline)
    ├── day_only_model.joblib
    ├── complete_model.joblib
    └── model_metadata.json

cv_analysis/
├── cross_validation_analysis.png # CV method comparison plots
└── cross_validation_report.txt   # CV analysis report

drift_analysis/
├── drift_detection.png           # Drift visualization
└── drift_detection_report.txt    # Drift analysis report
```

## Integration with 02-basic-ml

The evaluation pipeline is designed to work seamlessly with models trained in 02-basic-ml:

1. **Model Compatibility**: Automatically loads joblib models with metadata
2. **Feature Alignment**: Handles feature engineering consistency
3. **Data Processing**: Uses same preprocessing pipeline
4. **Performance Tracking**: Compares evaluation vs. training performance

## Key Dependencies

- **joblib**: Model serialization and deserialization
- **scikit-learn**: Evaluation metrics and cross-validation
- **scipy**: Statistical tests for drift detection
- **matplotlib**: Visualization and reporting
- **pandas/numpy**: Data manipulation and analysis

## Best Practices Demonstrated

1. **Model Persistence**: Proper serialization with metadata
2. **Evaluation Rigor**: Multiple metrics and statistical validation
3. **Drift Monitoring**: Proactive performance tracking
4. **Test Coverage**: Comprehensive unit testing
5. **Reproducibility**: Bazel ensures consistent environments
6. **Documentation**: Clear reporting and recommendations

## Advanced Features

### Statistical Significance Testing

The evaluation pipeline includes statistical tests to determine if performance differences are significant:

- Confidence intervals for cross-validation scores
- Hypothesis testing for drift detection
- Performance degradation thresholds

### Automated Recommendations

The system provides actionable recommendations:

- Best model selection based on multiple criteria
- Retraining alerts when performance degrades
- Hyperparameter optimization suggestions
- Data collection priorities

### Production Readiness

The evaluation pipeline is designed for production deployment:

- Robust error handling and logging
- Configurable alert thresholds
- Automated report generation
- Integration-ready APIs

This evaluation pipeline demonstrates enterprise-grade ML model monitoring and provides a foundation for production ML systems with Bazel.