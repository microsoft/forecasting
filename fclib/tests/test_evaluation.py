# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from fclib.evaluation.evaluation_utils import MAPE, sMAPE, pinball_loss


y = np.array([1, 2, 3])
yhat = np.array([1.1, 2.2, 3.3])
TOLERANCE = 1e-5


def test_MAPE():
    assert abs(MAPE(yhat, y) - 0.1) < TOLERANCE


def test_sMAPE():
    assert abs(sMAPE(yhat, y) - 0.04761904) < TOLERANCE


def test_pinball_loss():
    df = pd.DataFrame({"yhat": yhat, "y": y})
    assert all(abs(pinball_loss(df.yhat, df.y, 0.5) - pd.Series([0.05, 0.1, 0.15])) < TOLERANCE)
