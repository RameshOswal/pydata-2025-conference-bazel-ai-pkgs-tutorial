# Data Processing with Bazel

This example demonstrates fundamental data processing techniques using Bazel for dependency management, based on the Microsoft ML-For-Beginners pumpkin price prediction tutorial.

## Overview

This tutorial shows how to:
- Set up Bazel for Python data processing projects
- Manage data science dependencies (pandas, matplotlib, numpy)
- Build reproducible data processing pipelines
- Handle CSV data with proper preprocessing
- Create both simple and advanced data processors

## Dataset

We use a subset of the US pumpkin market data from the [Microsoft ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners) repository. The dataset contains pumpkin price information by city and date, demonstrating:
- Data cleaning and filtering
- Feature engineering
- Price normalization by package size
- Time-based analysis

## Project Structure

```
01-data-processing/
├── BUILD                    # Bazel build configuration
├── data_processor.py        # Advanced data processing with pandas/matplotlib
├── simple_processor.py     # Simple data processing (built-in libraries only)
├── test_data_processing.py  # Unit tests (currently disabled)
├── data/
│   └── US-pumpkins.csv     # Sample pumpkin market data
└── README.md               # This file
```

## Building and Running

### Prerequisites

- Bazel 6.0+ with Bzlmod support
- Python 3.11+

### Build and Run

```bash
# Run the simple data processor (no external dependencies)
bazel run //examples/01-data-processing:simple_process_data

# Run the advanced data processor (uses pandas, matplotlib)
bazel run //examples/01-data-processing:process_data

# Build all targets
bazel build //examples/01-data-processing:all

# Generate data information
bazel build //examples/01-data-processing:show_data
```

### Run data processing

**Simple processor (built-in Python libraries only):**
```bash
bazel run //examples/01-data-processing:simple_process_data -- \
  --data_path examples/01-data-processing/data/US-pumpkins.csv
```

**Advanced processor (with pandas, numpy, matplotlib):**
```bash
# Create output directory
mkdir -p output

# Run advanced processing with visualizations
bazel run //examples/01-data-processing:advanced_process_data -- \
  --data_path examples/01-data-processing/data/US-pumpkins.csv \
  --output_dir output \
  --save_processed

# Or use the main target (same as advanced)
bazel run //examples/01-data-processing:process_data -- \
  --data_path examples/01-data-processing/data/US-pumpkins.csv \
  --output_dir output \
  --save_processed
```

### View data information

```bash
# Generate basic data info using shell commands
bazel build //examples/01-data-processing:show_data
cat bazel-bin/examples/01-data-processing/data_info.txt
```

## Key Bazel Concepts Demonstrated

### 1. Python Dependencies
The `BUILD` file shows how to declare Python dependencies:
```python
deps = [
    "@pypi//pandas",
    "@pypi//matplotlib",
    "@pypi//numpy",
],
```

### 2. Data Dependencies
Data files are managed as Bazel targets:
```python
filegroup(
    name = "pumpkin_data",
    srcs = ["data/US-pumpkins.csv"],
    visibility = ["//visibility:public"],
)
```

### 3. Binary Targets
Executable scripts are defined as `py_binary` targets:
```python
py_binary(
    name = "process_data",
    srcs = ["data_processor.py"],
    main = "data_processor.py",
    deps = [...],
    data = ["//examples/01-basic-ml-pipeline/data:pumpkin_data"],
)
```

### 4. Library Targets
Reusable code is packaged as `py_library` targets:
```python
py_library(
    name = "data_processor_lib",
    srcs = ["data_processor.py"],
    deps = [...],
    visibility = ["//visibility:public"],
)
```

## Output

The data processing script generates:
- `pumpkin_scatter.png`: Scatter plot of prices vs months
- `pumpkin_monthly_avg.png`: Bar chart of average prices by month  
- `processed_pumpkins.csv`: Cleaned and processed data (if --save_processed is used)

## Learning Objectives

After completing this example, you will understand:
- How to set up Bazel for Python ML projects
- Managing external Python dependencies with pip_parse
- Creating reproducible data processing pipelines
- Structuring ML code with proper separation of concerns
- Basic testing strategies for ML code

## Next Steps

### Adding External ML Dependencies

The current example uses only Python built-in libraries for maximum compatibility. To add external ML libraries like pandas, numpy, and scikit-learn:

1. **Set up rules_python in WORKSPACE**:
   ```starlark
   load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
   
   http_archive(
       name = "rules_python",
       # Use appropriate version for your Bazel
   )
   
   load("@rules_python//python:pip.bzl", "pip_parse")
   pip_parse(
       name = "pypi",
       requirements_lock = "//:requirements.txt",
   )
   ```

2. **Update BUILD file**:
   ```python
   py_binary(
       name = "advanced_process_data",
       srcs = ["data_processor.py"],  # The full-featured version
       deps = [
           "@pypi//pandas",
           "@pypi//matplotlib",
           "@pypi//numpy",
       ],
       data = [":pumpkin_data"],
   )
   ```

### Further Extensions

- Extend the pipeline to include model training
- Add more sophisticated testing
- Explore distributed processing with Bazel
- Integrate with ML frameworks like scikit-learn or TensorFlow