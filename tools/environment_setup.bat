REM Copyright (c) Microsoft Corporation.
REM Licensed under the MIT License.

REM Please follow instructions in this link 
REM https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html 
REM to install Miniconda before running this script.


echo Update conda
call conda update conda --yes

echo Create conda environment
call conda env create -f tools/environment.yml

echo Activate conda environment
call conda activate forecasting_env

echo Install forecasting utility library
call pip install -e fclib

echo Register conda environment in Jupyter
call python -m ipykernel install --user --name forecasting_env

echo Environment setup is done!