# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import lightgbm as lgb

from fclib.models.lightgbm import predict


def test_predict(generate_ojdata, generate_data):
    data = pd.read_csv("fclib/tests/resources/ojdatagen.csv")
    newdata = generate_data.ojdata(61, 70)

    params = {"objective": "mape"}
    target = "logmove"

    lgb_data = lgb.Dataset(data.drop(columns=[target]), label=data[target])
    lgb_model = lgb.train(params, lgb_data, valid_sets=[lgb_data])

    predint = predict(newdata, lgb_model, target, ["store", "brand"], True)
    assert predint.logmove.dtype.name == "int64"
    predfloat = predict(newdata.drop(columns=[target]), lgb_model, "logmove", ["store", "brand"], False)
    assert predfloat.logmove.dtype.name == "float64"
