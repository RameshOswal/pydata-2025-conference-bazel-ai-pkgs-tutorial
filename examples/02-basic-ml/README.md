# 02-basic-ml: Machine Learning with Bazel

This example demonstrates how to build machine learning applications using Bazel, focusing on linear and polynomial regression for pumpkin price prediction.

## Overview

This example includes:
- **Linear Regression**: Simple linear regression using day of year to predict pumpkin prices
- **Polynomial Regression**: Non-linear regression using polynomial features
- **Complete ML Pipeline**: Feature engineering with categorical variables and model comparison

The examples are based on the [Microsoft ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/3-Linear/solution/notebook.ipynb) tutorial.

## Files

- `BUILD` - Bazel build configuration for ML examples
- `linear_regression.py` - Linear regression implementation
- `polynomial_regression.py` - Polynomial regression with degree comparison
- `ml_pipeline.py` - Complete ML pipeline with feature engineering
- `test_ml_examples.py` - Unit tests for ML functionality
- `data/US-pumpkins.csv` - Pumpkin price dataset

## Dependencies

The ML examples use these Python packages managed by Bazel:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `scikit-learn` - Machine learning algorithms

## Usage

### Linear Regression Example

```bash
# Run basic linear regression
bazel run //examples/02-basic-ml:linear_regression

# Specify custom parameters
bazel run //examples/02-basic-ml:linear_regression -- \
  --data_path examples/02-basic-ml/data/US-pumpkins.csv \
  --output_dir outputs/linear-regression \
  --predict_day 300
```

### Polynomial Regression Example

```bash
# Run polynomial regression (degree 2 by default)
bazel run //examples/02-basic-ml:polynomial_regression

# Compare different polynomial degrees
bazel run //examples/02-basic-ml:polynomial_regression -- \
  --degree 3 \
  --compare_degrees \
  --output_dir outputs/polynomial-regression
```

### Complete ML Pipeline

```bash
# Run complete ML pipeline with feature engineering
bazel run //examples/02-basic-ml:ml_pipeline -- \
  --output_dir outputs/ml-pipeline
```

### Run Tests

```bash
# Run unit tests
bazel test //examples/02-basic-ml:test_ml_examples
```

## Output

All examples generate:
- **Visualizations**: PNG plots showing data analysis and model results
- **Model metrics**: RMSE, R², MAE, and error percentages
- **Predictions**: Price predictions for specific days (e.g., programmer's day)

Output files are saved to the specified output directory (default: `outputs/02-basic-ml/`).

## Key Features Demonstrated

### 1. Data Processing
- Loading CSV data with pandas
- Filtering and cleaning data
- Feature extraction (day of year, month)
- Price calculation and normalization

### 2. Machine Learning
- Linear regression with scikit-learn
- Polynomial feature engineering
- Model training and evaluation
- Cross-validation and metrics calculation

### 3. Feature Engineering
- One-hot encoding for categorical variables
- Multiple feature set creation
- Feature importance and selection

### 4. Visualization
- Matplotlib integration with Bazel
- Headless plotting (`matplotlib.use('Agg')`)
- Data exploration visualizations
- Model comparison plots

### 5. Bazel Integration
- Python dependencies management
- Data files as Bazel resources
- Cross-platform builds
- Test integration

## Machine Learning Concepts

### Linear Regression
- Finds best-fit line through data points
- Assumes linear relationship between features and target
- Good baseline model for regression problems

### Polynomial Regression
- Extends linear regression with polynomial features
- Can capture non-linear relationships
- Higher degrees can lead to overfitting

### Feature Engineering
- Converting categorical variables to numerical (one-hot encoding)
- Creating new features from existing ones
- Feature selection and dimensionality considerations

### Model Evaluation
- **RMSE**: Root Mean Squared Error - lower is better
- **R²**: Coefficient of determination - higher is better (max 1.0)
- **MAE**: Mean Absolute Error - average prediction error

## Example Results

Typical results for PIE TYPE pumpkins:

```
Linear Regression:
- RMSE: ~2.77
- R²: ~0.07
- Error: ~17.2%

Polynomial Regression (degree 2):
- RMSE: ~2.40
- R²: ~0.20
- Error: ~14.8%

Complete Pipeline (all features):
- RMSE: ~2.23
- R²: ~0.97
- Error: ~8.3%
```

## Next Steps

This example can be extended with:
- Cross-validation for better model evaluation
- Hyperparameter tuning
- More sophisticated ML algorithms (Random Forest, XGBoost)
- Time series analysis for seasonal patterns
- Model deployment and serving

## Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Microsoft ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)