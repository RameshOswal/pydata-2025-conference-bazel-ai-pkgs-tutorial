# PyData 2025 Conference - Bazel AI Packages Tutorial

This repository contains the example codes and tutorials for the presentation on **Bazel for AI Package Management** presented at PyData 2025 Conference.

## 📚 Tutorial Overview

This tutorial demonstrates how to use Bazel for managing AI/ML packages and dependencies in Python projects. The examples and code samples provided here complement the main presentation repository.

## 🔗 Main Presentation Repository

For the complete presentation materials, slides, and detailed documentation, please visit:
**[https://github.com/RameshOswal/pydata-2025-conference-bazel-ai-pkgs](https://github.com/RameshOswal/pydata-2025-conference-bazel-ai-pkgs)**

## 📂 Repository Structure

```
├── README.md                   # This file
├── WORKSPACE                   # Bazel workspace configuration  
├── MODULE.bazel               # Bazel bzlmod configuration
├── BUILD                       # Root BUILD file with filegroups
├── .bazelrc                    # Bazel configuration
├── requirements.txt            # Python dependencies
├── requirements_lock.txt       # Locked dependency versions
├── examples/                   # Complete ML pipeline examples
│   ├── 01-data-processing/     # Data loading and preprocessing
│   ├── 02-basic-ml/           # Model training and evaluation  
│   └── 03-model-evaluation/   # Model evaluation and monitoring
├── outputs/                    # Generated outputs and trained models
└── evaluation_results/         # Model evaluation results
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

- Bazel 6.0+ (via Bazelisk)
- Basic understanding of Python package management

#### 🛠️ Bazel Installation (via Bazelisk)

Bazelisk is the recommended launcher for Bazel, ensuring you always use the correct Bazel version.

**Ubuntu (20.04/22.04/24.04):**

```bash
sudo apt update
sudo apt install curl unzip -y
curl -LO "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64"
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel
bazel --version
```

**macOS (with Homebrew):**

```bash
brew install bazelisk
brew link --overwrite bazelisk
bazel --version
```

**Windows (with Chocolatey):**

```powershell
choco install bazelisk
bazel --version
```

> After installation, use the `bazel` command as shown in the examples below.

### Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/RameshOswal/pydata-2025-conference-bazel-ai-pkgs-tutorial.git
   cd pydata-2025-conference-bazel-ai-pkgs-tutorial
   ```

2. Run the examples to verify your setup:

   ```bash
   # Run data processing example
   bazel run //examples/01-data-processing:simple_process_data -- --data_path examples/01-data-processing/data/US-pumpkins.csv
   
   # Train ML models
   bazel run //examples/02-basic-ml:ml_pipeline
   
   # Evaluate trained models
   bazel run //examples/03-model-evaluation:model_evaluator
   
   # Run tests to verify everything works
   bazel test //examples/02-basic-ml:test_ml_examples //examples/03-model-evaluation:test_evaluation
   ```

3. Follow the examples in order: data processing → ML training → model evaluation.

## 📋 Tutorial Contents

### Examples Included

1. **01-data-processing**: Data loading, cleaning, and preprocessing with Bazel
   - CSV data handling with pandas
   - Simple vs. advanced data processors
   - Bazel filegroups for data management

2. **02-basic-ml**: Complete ML training pipeline
   - Linear and polynomial regression models
   - Feature engineering and model comparison
   - Model persistence with joblib
   - Comprehensive unit testing

3. **03-model-evaluation**: Advanced model evaluation and monitoring
   - Model loading and performance evaluation
   - Cross-validation and statistical analysis
   - Drift detection and model monitoring
   - Automated evaluation reports

### Key Bazel Concepts Demonstrated

- **bzlmod Configuration**: Modern Bazel dependency management
- **Python Rules**: py_binary, py_library, py_test targets
- **Filegroups**: Managing data files and model artifacts  
- **Runfiles**: Handling file paths in Bazel environments
- **Testing**: Comprehensive test suites with proper dependencies

## 🎯 Learning Objectives

By the end of this tutorial, you will be able to:

- Set up Bazel for Python AI/ML projects
- Manage complex AI package dependencies using Bazel
- Optimize build processes for machine learning workflows
- Integrate Bazel with popular AI/ML frameworks

## 🤝 Contributing

If you find any issues or have suggestions for improvements, please:

1. Check the main repository: [pydata-2025-conference-bazel-ai-pkgs](https://github.com/RameshOswal/pydata-2025-conference-bazel-ai-pkgs)
2. Open an issue or submit a pull request

## 📧 Contact

For questions or feedback about this tutorial, please refer to the main presentation repository or contact the presenter through the PyData 2025 conference channels.

## 📖 Citation

If you use this work in your research or projects, please cite it as:

### BibTeX

```bibtex
@inproceedings{Oswal_Building_Bazel_Packages_2025,
author = {Oswal, Ramesh and Oswal, Jiten},
month = nov,
series = {PyData Seattle 2025},
title = {{Building Bazel Packages for AI/ML: SciPy, PyTorch, and Beyond}},
year = {2025}
}
```

### Citation File Format (CFF)

A `CITATION.cff` file is also available in this repository for automated citation tools.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This repository is specifically designed to accompany the PyData 2025 conference presentation. For the most up-to-date information and complete materials, always refer to the main repository linked above.