# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This file contains utility functions for builing Dilated CNN model to solve time series forecasting problems.
"""


from math import ceil, log
from tensorflow.keras.layers import Input, Lambda, Embedding, Conv1D, Dropout, Flatten, Dense, concatenate
from tensorflow.keras.models import Model


def create_dcnn_model(
    seq_len,
    n_dyn_fea=1,
    n_outputs=1,
    n_dilated_layers=3,
    kernel_size=2,
    n_filters=3,
    dropout_rate=0.1,
    max_cat_id=[1e3, 1e3],
):
    """Create a Dilated CNN model.

    Args: 
        seq_len (int): Input sequence length
        n_dyn_fea (int): Number of dynamic features
        n_outputs (int): Number of outputs of the network
        kernel_size (int): Kernel size of each convolutional layer
        n_filters (int): Number of filters in each convolutional layer
        dropout_rate (float): Dropout rate in the network
        max_cat_id (list[int]): Each entry in the list represents the maximum value of the ID of a specific categorical variable. 

    Returns:
        object: Keras Model object
    """
    # Sequential input for dynamic features
    seq_in = Input(shape=(seq_len, n_dyn_fea))

    # Categorical input
    n_cat_fea = len(max_cat_id)
    cat_fea_in = Input(shape=(n_cat_fea,), dtype="uint8")
    cat_flatten = []
    for i, m in enumerate(max_cat_id):
        cat_fea = Lambda(lambda x, i: x[:, i, None], arguments={"i": i})(cat_fea_in)
        cat_fea_embed = Embedding(m + 1, ceil(log(m + 1)), input_length=1)(cat_fea)
        cat_flatten.append(Flatten()(cat_fea_embed))

    # Dilated convolutional layers
    conv1d_layers = []
    conv1d_layers.append(
        Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=1, padding="causal", activation="relu")(seq_in)
    )
    for i in range(1, n_dilated_layers):
        conv1d_layers.append(
            Conv1D(
                filters=n_filters, kernel_size=kernel_size, dilation_rate=2 ** i, padding="causal", activation="relu"
            )(conv1d_layers[i - 1])
        )

    # Skip connections
    if n_dilated_layers > 1:
        c = concatenate([conv1d_layers[0], conv1d_layers[-1]])
    else:
        c = conv1d_layers[0]

    # Output of convolutional layers
    conv_out = Conv1D(8, 1, activation="relu")(c)
    conv_out = Dropout(dropout_rate)(conv_out)
    conv_out = Flatten()(conv_out)

    # Concatenate with categorical features
    x = concatenate([conv_out] + cat_flatten)
    x = Dense(16, activation="relu")(x)
    output = Dense(n_outputs, activation="linear")(x)

    # Define model interface
    model = Model(inputs=[seq_in, cat_fea_in], outputs=output)

    return model
