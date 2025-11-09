"""
Root BUILD file for PyData 2025 Bazel AI/ML Tutorial
"""

# Export requirements.txt for pip_parse
exports_files(["requirements.txt"])

# Filegroup for trained models from 02-basic-ml
filegroup(
    name = "trained_models",
    srcs = glob([
        "outputs/02-basic-ml/models/*.joblib",
        "outputs/02-basic-ml/models/*.json",
    ]),
    visibility = ["//visibility:public"],
)

# Filegroup for all outputs
filegroup(
    name = "all_outputs",
    srcs = glob([
        "outputs/**/*",
    ]),
    visibility = ["//visibility:public"],
)