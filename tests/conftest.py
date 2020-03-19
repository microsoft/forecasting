import os
import pytest
from fclib.common.utils import git_repo_path


@pytest.fixture(scope="module")
def notebooks():
    """Get paths of example notebooks.

    Returns:
        dict: Dictionary including paths of the example notebooks.
    """
    repo_path = git_repo_path()
    examples_path = os.path.join(repo_path, "examples")
    usecase_path = os.path.join(examples_path, "grocery_sales", "python")
    quick_start_path = os.path.join(usecase_path, "00_quick_start")

    # Path for the notebooks
    paths = {"lightgbm_quick_start": os.path.join(quick_start_path, "lightgbm_single_round.ipynb")}
    return paths
