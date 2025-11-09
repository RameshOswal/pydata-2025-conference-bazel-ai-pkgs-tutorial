# Data filegroup - this should work without any external dependencies
filegroup(
    name = "pumpkin_data",
    srcs = ["data/US-pumpkins.csv"],
    visibility = ["//visibility:public"],
)

# Simple shell script to demonstrate basic data access
genrule(
    name = "show_data",
    srcs = [":pumpkin_data"],
    outs = ["data_info.txt"],
    cmd = "echo 'Pumpkin data file:' > $@ && echo '$(location :pumpkin_data)' >> $@ && wc -l $(location :pumpkin_data) >> $@",
)

# Python binary using only built-in libraries (no external dependencies)
py_binary(
    name = "simple_process_data",
    srcs = ["simple_processor.py"],
    data = [":pumpkin_data"],
    main = "simple_processor.py",
)

# Update the main process_data target to use the advanced processor
py_binary(
    name = "process_data",
    srcs = ["data_processor.py"],
    main = "data_processor.py",
    deps = [
        "@pypi//pandas",
        "@pypi//matplotlib",
        "@pypi//numpy",
    ],
    data = [":pumpkin_data"],
)
