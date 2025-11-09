"""
Bazel BUILD file for model serving examples.

This package demonstrates ML model serving patterns using FastAPI and OpenAPI.
"""

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

# Model serving library modules
py_library(
    name = "model_server_lib",
    srcs = ["model_server.py"],
    deps = [
        "@pypi//fastapi",
        "@pypi//uvicorn",
        "@pypi//pydantic",
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//scikit_learn",
        "@pypi//joblib",
    ],
)

py_library(
    name = "model_manager_lib",
    srcs = ["model_manager.py"],
    deps = [
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//scikit_learn",
        "@pypi//joblib",
    ],
)

py_library(
    name = "health_monitor_lib",
    srcs = ["health_monitor.py"],
    deps = [
        "@pypi//fastapi",
        "@pypi//uvicorn",
        "@pypi//pydantic",
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//scikit_learn",
        "@pypi//joblib",
    ],
)

py_library(
    name = "api_client_lib",
    srcs = ["api_client.py"],
    deps = [
        "@pypi//fastapi",
        "@pypi//pydantic",
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//requests",
    ],
)

# Main model serving API binary
py_binary(
    name = "model_server",
    srcs = ["model_server.py"],
    deps = [
        ":model_server_lib",
    ],
)

# Model management utilities binary
py_binary(
    name = "model_manager",
    srcs = ["model_manager.py"],
    deps = [
        ":model_manager_lib",
    ],
)

# Health check and monitoring binary
py_binary(
    name = "health_monitor",
    srcs = ["health_monitor.py"],
    deps = [
        ":health_monitor_lib",
    ],
)

# API client for testing binary
py_binary(
    name = "api_client",
    srcs = ["api_client.py"],
    deps = [
        ":api_client_lib",
    ],
)

# Unit tests for the API
py_test(
    name = "test_model_server",
    srcs = ["test_model_server.py"],
    deps = [
        ":model_server_lib",
        ":model_manager_lib",
        ":health_monitor_lib",
        ":api_client_lib",
        "@pypi//fastapi",
        "@pypi//pydantic",
        "@pypi//pandas",
        "@pypi//numpy",
        "@pypi//scikit_learn",
        "@pypi//joblib",
        "@pypi//requests",
        "@pypi//httpx",
    ],
)