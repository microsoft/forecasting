# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fclib.common.utils import git_repo_path, module_exists, system_type, module_path


def test_git_repo_path():
    # implicitly test for no exception
    assert git_repo_path() is not None


def test_module_exists():
    assert module_exists("numpy")
    assert not module_exists("fakepkgxyz")


def test_system_type():
    assert system_type() in ["linux", "mac", "win"]


def test_module_path():
    # look for binaries we use in this repo
    assert module_path("forecasting_env", "python") != ""
    assert module_path("forecasting_env", "tensorboard") != ""
