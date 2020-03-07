import os
import pytest
from fclib.common.utils import git_repo_path


@pytest.fixture(scope="module")
def notebooks():
    repo_path = git_repo_path()
    examples_path = os.path.join(repo_path, "examples")
    quick_start_path = os.path.join(examples_path, "00_quick_start")

    # Path for the notebooks
    paths = {"lightgbm_quick_start": os.path.join(quick_start_path, "lightgbm_point_forecast.ipynb")}
    return paths
