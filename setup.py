#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import sys

from setuptools import find_packages, setup

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for cvnets.")

if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]

VERSION = 0.3


def do_setup(package_data):
    setup(
        name="cvnets",
        version=VERSION,
        description="CVNets: A library for training computer vision networks",
        url="https://github.com/apple/ml-cvnets.git",
        setup_requires=[
            'numpy<1.20.0; python_version<"3.7"',
            'numpy; python_version>="3.7"',
            "setuptools>=18.0",
        ],
        install_requires=[
            'numpy<1.20.0; python_version<"3.7"',
            'numpy; python_version>="3.7"',
            "torch",
            "tqdm",
        ],
        packages=find_packages(exclude=["config_files", "config_files.*"]),
        package_data=package_data,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "cvnets-train = main_train:main_worker",
                "cvnets-eval = main_eval:main_worker",
                "cvnets-eval-seg = main_eval:main_worker_segmentation",
                "cvnets-eval-det = main_eval:main_worker_detection",
                "cvnets-convert = main_conversion:main_worker_conversion",
                "cvnets-loss-landscape = main_loss_landscape:main_worker_loss_landscape",
            ],
        },
        zip_safe=False,
    )


def get_files(path, relative_to="."):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    package_data = {"cvnets": (get_files(os.path.join("cvnets", "config")))}
    do_setup(package_data)
