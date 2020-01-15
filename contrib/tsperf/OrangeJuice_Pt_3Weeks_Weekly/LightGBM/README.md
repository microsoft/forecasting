# Implementation submission form

## Submission details

**Submission date**: 10/22/2018

**Benchmark name:** OrangeJuice_Pt_3Weeks_Weekly

**Submitter(s):** Chenhui Hu

**Submitter(s) email:** chenhhu@microsoft.com

**Submission name:** LightGBM

**Submission path:** retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/LightGBM

## Implementation description

### Modelling approach

In this submission, we implement boosted decision tree model using Python package `lightgbm`, which is a fast, distributed, high performance
gradient boosting framework based on decision tree algorithms.

### Feature engineering

The following features have been used in the implementation of the forecast method:
- datetime features including week, week of the month, and month
- weekly sales of each orange juice in recent weeks
- average sales of each orange juice in recent weeks
- other features including *store*, *brand*, *deal*, *feat* columns and price features


### Hyperparameter tuning

We tune the hyperparameters of the model with HyperDrive which is accessible through Azure ML SDK. A remote compute cluster with 16 CPU cores is created to distribute the computation. The hyperparameters tuned with HyperDrive and their ranges can be found in hyperparameter_tuning.ipynb.


### Description of implementation scripts

* `utils.py`: Python script including utility functions for building the model
* `train_score.py`: Python script that trains the model and generates forecast results for each round
* `train_score.ipynb` (optional): Jupyter notebook that trains the model and visualizes the results
* `train_validate.py` (optional): Python script that does training and validation with the 1st round training data
* `hyperparameter_tuning.ipynb` (optional): Jupyter notebook that tries different model configurations and selects the best model by running
`train_validate.py` script in a remote compute cluster with different sets of hyperparameters


### Steps to reproduce results

0. Follow the instructions [here](#resource-deployment-instructions) to provision a Linux virtual
machine and log into the provisioned VM.

1. Clone the Forecasting repo to the home directory of your machine

   ```bash
   cd ~
   git clone https://github.com/Microsoft/Forecasting.git
   ```
   Use one of the following options to securely connect to the Git repo:
   * [Personal Access Tokens](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/)  
   For this method, the clone command becomes
   ```bash
   git clone https://<username>:<personal access token>@github.com/Microsoft/Forecasting.git
   ```
   * [Git Credential Managers](https://github.com/Microsoft/Git-Credential-Manager-for-Windows)
   * [Authenticate with SSH](https://help.github.com/articles/connecting-to-github-with-ssh/)


2. Create a conda environment for running the scripts of data downloading, data preparation, and result evaluation. To do this, you need
to check if conda has been installed by runnning command `conda -V`. If it is installed, you will see the conda version in the terminal. Otherwise, please follow the instructions [here](https://conda.io/docs/user-guide/install/linux.html) to install conda. Then, you can go to `~/Forecasting` directory in the VM and create a conda environment named `tsperf` by

   ```bash
   conda env create --file ./common/conda_dependencies.yml
   ```

   This will create a conda environment with the Python and R packages listed in `conda_dependencies.yml` being installed. The conda
  environment name is also defined in the yml file.

3. Activate the conda environment and download the Orange Juice dataset. Use command `source activate tsperf` to activate the conda environment. Then, download the Orange Juice dataset by running the following command from `~/Forecasting` directory

   ```bash
   Rscript ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/common/download_data.r
   ```

   This will create a data directory `./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/data` and store the dataset in this directory. The dataset has two csv files - `yx.csv` and `storedemo.csv` which contain the sales information and store demographic information, respectively.

4. From `~/Forecasting` directory, run the following command to generate the training data and testing data for each forecast period:

   ```bash
   python ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/common/serve_folds.py --test --save
   ```

   This will generate 12 csv files named `train_round_#.csv` and 12 csv files named `test_round_#.csv` in two subfolders `/train` and
   `/test` under the data directory, respectively. After running the above command, you can deactivate the conda environment by running
   `source deactivate`.

5. Make sure Docker is installed
    
   You can check if Docker is installed on your VM by running

   ```bash
   sudo docker -v
   ```
   You will see the Docker version if Docker is installed. If not, you can install it by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/). Note that if you want to execute Docker commands without sudo as a non-root user, you need to create a Unix group and add users to it by following the instructions [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).  

6. Build a local Docker image by running the following command from `~/Forecasting` directory

   ```bash
   sudo docker build -t lightgbm_image:v1 ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/LightGBM
   ```

7. Choose a name for a new Docker container (e.g. lightgbm_container) and create it using command:   

   ```bash
   cd ~/Forecasting
   sudo docker run -it -v ~/Forecasting:/Forecasting --name lightgbm_container lightgbm_image:v1
   ```

   Note that option `-v ~/Forecasting:/Forecasting` allows you to mount `~/Forecasting` folder (the one you cloned) to the container so that you will have
   access to the source code in the container.

8. Train the model and make predictions from `/Forecasting` folder by running

   ```bash
   cd /Forecasting
   source ./common/train_score_vm ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/LightGBM Python3
   ```

   This will generate 5 `submission_seed_<seed number>.csv` files in the submission directory, where \<seed number\>
   is between 1 and 5. This command will also output 5 running times of train_score.py. The median of the times
   reported in rows starting with 'real' should be compared against the wallclock time declared in benchmark
   submission. After generating the forecast results, you can exit the Docker container by command `exit`.

9. Activate conda environment again by `source activate tsperf`. Then, evaluate the benchmark quality by running

   ```bash
   source ./common/evaluate ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/LightGBM ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly
   ```

   This command will output 5 benchmark quality values (MAPEs). Their median should be compared against the
   benchmark quality declared in benchmark submission.


## Implementation resources

**Platform:** Azure Cloud

**Resource location:** East US  

**Hardware:** Standard D2s v3 (2 vcpus, 8 GB memory, 16 GB temporary storage) Ubuntu Linux VM

**Data storage:** Premium SSD

**Dockerfile:** [retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/LightGBM/Dockerfile](https://github.com/Microsoft/Forecasting/blob/master/retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/LightGBM/Dockerfile)

**Key packages/dependencies:**  
  * Python
    - pandas==0.23.1
    - scikit-learn==0.19.1
    - lightgbm==2.1.2

## Resource deployment instructions

We use Azure Linux VM to develop the baseline methods. Please follow the instructions below to deploy the resource.
* Azure Linux VM deployment
  - Create an Azure account and log into [Azure portal](portal.azure.com/)
  - Refer to the steps [here](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) to deploy a Data
  Science Virtual Machine for Linux (Ubuntu). Select *D2s_v3* as the virtual machine size.


## Implementation evaluation

**Quality:**

*MAPE run 1: 35.91%*

*MAPE run 2: 36.28%*

*MAPE run 3: 35.99%*

*MAPE run 4: 36.49%*

*MAPE run 5: 36.57%*

*median MAPE: 36.28%*

**Time:**

*run time 1: 613.33 seconds*

*run time 2: 619.37 seconds*

*run time 3: 655.50 seconds*

*run time 4: 625.10 seconds*

*run time 5: 647.46 seconds*

*median run time: 625.10 seconds*

**Cost:** The hourly cost of the D2s v3 Ubuntu Linux VM in East US Azure region is 0.096 USD, based on the price at the submission date. Thus, the total cost is 625.10/3600 $\times$ 0.096 = $0.0167.
