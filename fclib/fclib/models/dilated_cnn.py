# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This file contains utility functions for builing Dilated CNN model to
solve time series forecasting problems.
"""


from math import ceil, log
from tensorflow.keras.layers import Input, Lambda, Embedding, Conv1D, Dropout, Flatten, Dense, concatenate
from tensorflow.keras.models import Model


def create_dcnn_model(
    seq_len, n_dyn_fea=1, n_cat_fea=2, n_outputs=1, kernel_size=2, n_filters=3, dropout_rate=0.1, max_cat_id=[1e3, 1e3],
):
    """Create a Dilated CNN model.

    Args: 
        seq_len (int): Input sequence length
        n_dyn_fea (int): Number of dynamic features
        n_cat_fea (int): Number of categorical features
        n_outputs (int): Number of outputs of the network
        kernel_size (int): Kernel size of each convolutional layer
        n_filters (int): Number of filters in each convolutional layer
        dropout_rate (float): Dropout rate in the network
        max_cat_id (list[int]): Maximum value of the grain1 ID

    Returns:
        object: Keras Model object
    """
    # Sequential input for dynamic features
    seq_in = Input(shape=(seq_len, n_dyn_fea))

    # Categorical input
    cat_fea_in = Input(shape=(n_cat_fea,), dtype="uint8")
    cat_flatten = []
    for i, m in enumerate(max_cat_id):
        cat_fea = Lambda(lambda x: x[:, i, None])(cat_fea_in)
        cat_fea_embed = Embedding(m + 1, ceil(log(m + 1)), input_length=1)(cat_fea)
        cat_flatten.append(Flatten()(cat_fea_embed))

    # Dilated convolutional layers
    c1 = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=1, padding="causal", activation="relu")(
        seq_in
    )
    c2 = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=2, padding="causal", activation="relu")(c1)
    c3 = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=4, padding="causal", activation="relu")(c2)

    # Skip connections
    c4 = concatenate([c1, c3])

    # Output of convolutional layers
    conv_out = Conv1D(8, 1, activation="relu")(c4)
    conv_out = Dropout(dropout_rate)(conv_out)
    conv_out = Flatten()(conv_out)

    # Concatenate with categorical features
    x = concatenate([conv_out] + cat_flatten)
    x = Dense(16, activation="relu")(x)
    output = Dense(n_outputs, activation="linear")(x)

    # Define model interface
    model = Model(inputs=[seq_in, cat_fea_in], outputs=output)

    return model
