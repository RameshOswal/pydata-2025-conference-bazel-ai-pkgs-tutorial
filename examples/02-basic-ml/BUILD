load("@rules_python//python:defs.bzl", "py_binary", "py_test")

# Linear regression example
py_binary(
    name = "linear_regression",
    srcs = ["linear_regression.py"],
    deps = [
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//matplotlib",
        "@pypi//scikit_learn",
    ],
    data = [
        "data/US-pumpkins.csv",
    ],
)

# Polynomial regression example  
py_binary(
    name = "polynomial_regression",
    srcs = ["polynomial_regression.py"],
    deps = [
        "@pypi//pandas",
        "@pypi//numpy", 
        "@pypi//matplotlib",
        "@pypi//scikit_learn",
    ],
    data = [
        "data/US-pumpkins.csv",
    ],
)

# Complete ML pipeline example
py_binary(
    name = "ml_pipeline",
    srcs = ["ml_pipeline.py"],
    deps = [
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//matplotlib", 
        "@pypi//scikit_learn",
    ],
    data = [
        "data/US-pumpkins.csv",
    ],
)

# Unit tests for ML examples
py_test(
    name = "test_ml_examples",
    srcs = ["test_ml_examples.py"],
    deps = [
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//scikit_learn",
    ],
    data = [
        "data/US-pumpkins.csv",
    ],
)