# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pytest
import papermill as pm
import scrapbook as sb

ABS_TOL = 5.0


@pytest.mark.integration
def test_lightgbm_quick_start(notebooks):
    notebook_path = notebooks["lightgbm_quick_start"]
    output_notebook_path = os.path.join(os.path.dirname(notebook_path), "output.ipynb")
    pm.execute_notebook(notebook_path, output_notebook_path, kernel_name="forecast_cpu")
    nb = sb.read_notebook(output_notebook_path)
    df = nb.scraps.dataframe
    assert df.shape[0] == 1
    mape = df.loc[df.name == "MAPE"]["data"][0]
    assert mape == pytest.approx(35.60, abs=ABS_TOL)


@pytest.mark.integration
def test_autoarima_quick_start(notebooks):
    notebook_path = notebooks["autoarima_quick_start"]
    output_notebook_path = os.path.join(os.path.dirname(notebook_path), "output.ipynb")
    pm.execute_notebook(
        notebook_path, output_notebook_path, kernel_name="forecast_cpu", parameters=dict(STORE_SUBSET=True),
    )
    nb = sb.read_notebook(output_notebook_path)
    df = nb.scraps.dataframe
    assert df.shape[0] == 1
    mape = df.loc[df.name == "MAPE"]["data"][0]
    print(mape)
    assert mape == pytest.approx(75.6, abs=ABS_TOL)


@pytest.mark.integration
def test_lightgbm_multi_round(notebooks):
    notebook_path = notebooks["lightgbm_multi_round"]
    output_notebook_path = os.path.join(os.path.dirname(notebook_path), "output.ipynb")
    pm.execute_notebook(
        notebook_path, output_notebook_path, kernel_name="forecast_cpu", parameters=dict(N_SPLITS=1),
    )
    nb = sb.read_notebook(output_notebook_path)
    df = nb.scraps.dataframe
    assert df.shape[0] == 1
    mape = df.loc[df.name == "MAPE"]["data"][0]
    assert mape == pytest.approx(36.0, abs=ABS_TOL)


@pytest.mark.integration
def test_dilatedcnn_multi_round(notebooks):
    notebook_path = notebooks["dilatedcnn_multi_round"]
    output_notebook_path = os.path.join(os.path.dirname(notebook_path), "output.ipynb")
    pm.execute_notebook(
        notebook_path, output_notebook_path, kernel_name="forecast_cpu", parameters=dict(N_SPLITS=2),
    )
    nb = sb.read_notebook(output_notebook_path)
    df = nb.scraps.dataframe
    assert df.shape[0] == 1
    mape = df.loc[df.name == "MAPE"]["data"][0]
    assert mape == pytest.approx(37.7, abs=ABS_TOL)


@pytest.mark.integration
def test_autoarima_multi_round(notebooks):
    notebook_path = notebooks["autoarima_multi_round"]
    output_notebook_path = os.path.join(os.path.dirname(notebook_path), "output.ipynb")
    pm.execute_notebook(
        notebook_path, output_notebook_path, kernel_name="forecast_cpu", parameters=dict(N_SPLITS=2, STORE_SUBSET=True),
    )
    nb = sb.read_notebook(output_notebook_path)
    df = nb.scraps.dataframe
    assert df.shape[0] == 1
    mape = df.loc[df.name == "MAPE"]["data"][0]
    assert mape == pytest.approx(74.35, abs=ABS_TOL)
