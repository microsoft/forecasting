#!/usr/bin/python

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script creates yaml files to build conda environments
# For generating a conda file for running only python code:
# $ python generate_conda_file.py
#
# For generating a conda file for running python gpu:
# $ python generate_conda_file.py --gpu


import argparse
import textwrap
from sys import platform


HELP_MSG = """
To create the conda environment:
$ conda env create -f {conda_env}.yaml

To update the conda environment:
$ conda env update -f {conda_env}.yaml

To register the conda environment in Jupyter:
$ conda activate {conda_env}
$ python -m ipykernel install --user --name {conda_env} \
--display-name "Python ({conda_env})"
"""


CHANNELS = ["defaults", "conda-forge"]

CONDA_BASE = {
    "python": "python==3.6.10",
    "pip": "pip>=19.1.1",
    "ipykernel": "ipykernel>=4.6.1",
    "jupyter": "jupyter>=1.0.0",
    "jupyter_nbextensions_configurator": "jupyter_nbextensions_configurator>=0.4.1",
    "numpy": "numpy>=1.16.2",
    "pandas": "pandas>=0.23.4",
    "pytest": "pytest>=3.6.4",
    "scipy": "scipy>=1.1.0",
    "xlrd": "xlrd>=1.1.0",
    "urllib3": "urllib3>=1.21.1",
    "scikit-learn": "scikit-learn>=0.20.3",
    "tqdm": "tqdm>=4.43.0",
    "pylint": "pylint>=2.4.4",
    "matplotlib": "matplotlib>=3.1.2",
    "r-base": "r-base>=3.3.0",
    "papermill": "papermill>=1.0.1",
}


CONDA_GPU = {}

PIP_BASE = {
    "azureml-sdk": "azureml-sdk[explain,automl]==1.0.85",
    "black": "black>=18.6b4",
    "nteract-scrapbook": "nteract-scrapbook>=0.3.1",
    "pre-commit": "pre-commit>=1.14.4",
    "tensorboard": "tensorboard==2.1.0",
    "tensorflow": "tensorflow==2.0",
    "flake8": "flake8>=3.3.0",
    "jupytext": "jupytext>=1.3.0",
    "lightgbm": "lightgbm==2.3.0",
    "statsmodels": "statsmodels==0.11.1",
    "pmdarima": "pmdarima==1.1.1",
    "gitpython": "gitpython==3.0.8",
}

PIP_GPU = {}

PIP_DARWIN = {}
PIP_DARWIN_GPU = {}

PIP_LINUX = {}
PIP_LINUX_GPU = {}

PIP_WIN32 = {}
PIP_WIN32_GPU = {}

CONDA_DARWIN = {}
CONDA_DARWIN_GPU = {}

CONDA_LINUX = {}
CONDA_LINUX_GPU = {}

CONDA_WIN32 = {}
CONDA_WIN32_GPU = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
        This script generates a conda file for different environments.
        Plain python is the default,
        but flags can be used to support GPU functionality."""
        ),
        epilog=HELP_MSG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", help="specify name of conda environment")
    parser.add_argument("--gpu", action="store_true", help="include packages for GPU support")
    args = parser.parse_args()

    # set name for environment and output yaml file
    conda_env = "forecasting_cpu"
    if args.gpu:
        conda_env = "forecasting_gpu"

    # overwrite environment name with user input
    if args.name is not None:
        conda_env = args.name

    # add conda and pip base packages
    conda_packages = CONDA_BASE
    pip_packages = PIP_BASE

    # update conda and pip packages based on flags provided
    if args.gpu:
        conda_packages.update(CONDA_GPU)
        pip_packages.update(PIP_GPU)

    # update conda and pip packages based on os platform support
    if platform == "darwin":
        conda_packages.update(CONDA_DARWIN)
        pip_packages.update(PIP_DARWIN)
        if args.gpu:
            conda_packages.update(CONDA_DARWIN_GPU)
            pip_packages.update(PIP_DARWIN_GPU)
    elif platform.startswith("linux"):
        conda_packages.update(CONDA_LINUX)
        pip_packages.update(PIP_LINUX)
        if args.gpu:
            conda_packages.update(CONDA_LINUX_GPU)
            pip_packages.update(PIP_LINUX_GPU)
    elif platform == "win32":
        conda_packages.update(CONDA_WIN32)
        pip_packages.update(PIP_WIN32)
        if args.gpu:
            conda_packages.update(CONDA_WIN32_GPU)
            pip_packages.update(PIP_WIN32_GPU)
    else:
        raise Exception("Unsupported platform. Must be Windows, Linux, or macOS")

    # write out yaml file
    conda_file = "{}.yaml".format(conda_env)
    with open(conda_file, "w") as f:
        for line in HELP_MSG.format(conda_env=conda_env).split("\n"):
            f.write("# {}\n".format(line))
        f.write("name: {}\n".format(conda_env))
        f.write("channels:\n")
        for channel in CHANNELS:
            f.write("- {}\n".format(channel))
        f.write("dependencies:\n")
        for conda_package in conda_packages.values():
            f.write("- {}\n".format(conda_package))
        f.write("- pip:\n")
        for pip_package in pip_packages.values():
            f.write("  - {}\n".format(pip_package))

    print("Generated conda file: {}".format(conda_file))
    print(HELP_MSG.format(conda_env=conda_env))
