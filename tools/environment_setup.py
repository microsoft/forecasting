# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


conda update conda
conda env create -f tools/environment.yaml
eval "$(conda shell.bash hook)" && conda activate forecasting_env
pip install -e fclib
python -m ipykernel install --user --name forecasting_env
