"""
Bazel BUILD file for model evaluation examples.

This package demonstrates model evaluation and testing patterns using Bazel.
"""

load("@rules_python//python:defs.bzl", "py_binary", "py_test")

# Model evaluation and testing examples
py_binary(
    name = "model_evaluator",
    srcs = ["model_evaluator.py"],
    deps = [
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//matplotlib",
        "@pypi//scikit_learn",
        "@pypi//joblib",
        "@pypi//scipy",
    ],
    data = [
        "//examples/02-basic-ml:data/US-pumpkins.csv",
    ],
)

py_binary(
    name = "cross_validation",
    srcs = ["cross_validation.py"],
    deps = [
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//matplotlib",
        "@pypi//scikit_learn",
        "@pypi//joblib",
    ],
    data = [
        "//examples/02-basic-ml:data/US-pumpkins.csv",
    ],
)

py_binary(
    name = "model_drift_detection",
    srcs = ["model_drift_detection.py"],
    deps = [
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//matplotlib",
        "@pypi//scikit_learn",
        "@pypi//joblib",
        "@pypi//scipy",
    ],
    data = [
        "//examples/02-basic-ml:data/US-pumpkins.csv",
    ],
)

# Unit tests for evaluation functions
py_test(
    name = "test_evaluation",
    srcs = ["test_evaluation.py"],
    deps = [
        ":model_evaluator",
        ":cross_validation",
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//scikit_learn",
        "@pypi//joblib",
    ],
    data = [
        "//examples/02-basic-ml:data/US-pumpkins.csv",
    ],
)