# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import warnings

from fclib.models.dilated_cnn import create_dcnn_model

# [pytest]
# filterwarnings = 'ignore:.*imp module is deprecated:DeprecationWarning'

def test_create_dcnn_model():
    # with pytest.deprecated_call():
    create_dcnn_model(
        seq_len=1,
        n_dyn_fea=1,
        n_outputs=2,
        n_dilated_layers=1,
        kernel_size=2,
        dropout_rate=0.05,
        max_cat_id=[30, 120]
    )

    create_dcnn_model(
        seq_len=1,
        n_dyn_fea=1,
        n_outputs=2,
        n_dilated_layers=2,
        kernel_size=2,
        dropout_rate=0.05,
        max_cat_id=[30, 120]
    )

