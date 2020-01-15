# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# /* spell-checker: disable */
import os

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


# Package meta-data.
NAME = "forecasting_lib"
DESCRIPTION = "A library for forecasting"
URL = ""
EMAIL = "vapaunic@users.noreply.github.com"
AUTHOR = "AUTHORS.md"
LICENSE = "MIT"
LONG_DESCRIPTION = DESCRIPTION


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


here = os.path.abspath(os.path.dirname(__file__))

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)


setup(
    name=NAME,
    version=about["__version__"],
    url=URL,
    license=LICENSE,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    scripts=[],
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Data Scientists & Developers",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
    ],
)
