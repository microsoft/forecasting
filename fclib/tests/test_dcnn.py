# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fclib.models.dilated_cnn import create_dcnn_model

def test_create_dcnn_model():
    create_dcnn_model(seq_len=1) # default args

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

