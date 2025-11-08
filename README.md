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
├── BUILD                       # Root BUILD file
├── .bazelrc                    # Bazel configuration
├── requirements.txt            # Python dependencies
├── examples/                   # Code examples from the tutorial
│   └── 01-basic-ml-pipeline/   # Basic ML data processing with Bazel
├── exercises/                  # Hands-on exercises
├── solutions/                  # Solutions to exercises
└── resources/                  # Additional resources and references
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Bazel 6.0+
- Basic understanding of Python package management

### Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/RameshOswal/pydata-2025-conference-bazel-ai-pkgs-tutorial.git
   cd pydata-2025-conference-bazel-ai-pkgs-tutorial
   ```

2. Run the first example to verify your setup:

   ```bash
   # Build the basic ML pipeline example
   bazel build //examples/01-basic-ml-pipeline:all
   
   # Run the data processing example
   bazel run //examples/01-basic-ml-pipeline:simple_process_data -- \
     --data_path examples/01-basic-ml-pipeline/data/US-pumpkins.csv
   ```

3. Follow the examples in order, starting with the basic setup in the `examples/` directory.

## 📋 Tutorial Contents

- **Basic Bazel Setup**: Introduction to Bazel for Python projects
- **AI Package Management**: Managing ML libraries with Bazel
- **Dependency Resolution**: Handling complex AI package dependencies
- **Build Optimization**: Optimizing builds for AI/ML workflows
- **Integration Examples**: Real-world integration scenarios

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

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This repository is specifically designed to accompany the PyData 2025 conference presentation. For the most up-to-date information and complete materials, always refer to the main repository linked above.