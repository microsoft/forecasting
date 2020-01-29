# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from git import Repo


def git_repo_path():
    """Return the path of the forecasting repo"""

    repo = Repo(search_parent_directories=True)
    return repo.working_dir
