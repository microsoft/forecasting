# Implementation submission form

## Submission information

**Submission date**: 10/26/2018

**Benchmark name:** GEFCom2017_D_Prob_MT_hourly

**Submitter(s):** Fang Zhou

**Submitter(s) email:** zhouf@microsoft.com

**Submission name:** Quantile Regression Neutral Network

**Submission branch:** [zhouf/energy_forecast_fnn_model_v1](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf?version=GBzhouf%2Fenergy_forecast_fnn_model_v1) and [zhouf/energy_forecast_fnn_cv_v1](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf?path=%2F&version=GBzhouf%2Fenergy_forecast_fnn_cv_v1)

**Pull request:** [GEFCom2017_D_Prob_MT_hourly - fnn submission](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf/pullrequest/181524?_a=overview)

**Submission path:** energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn


## Implementation description

### Modelling approach

In this submission, we implement a quantile regression neutral network model using the `qrnn` package in R.

### Feature engineering

The following features are used:  
**LoadLag**: Average load based on the same-day and same-hour load values of the same week, the week before the same week, and the week after the same week of the previous three years, i.e. 9 values are averaged to compute this feature.  
**DryBulbLag**:  Average DryBulb temperature based on the same-hour DryBulb values of the same day, the day before the same day, and the day after the same day of the previous three years, i.e. 9 values are averaged to compute this feature.  
**Weekly Fourier Series**: weekly_sin_1, weekly_cos_1,  weekly_sin_2, weekly_cos_2, weekly_sin_3, weekly_cos_3  
**Annual Fourier Series**: annual_sin_1, annual_cos_1, annual_sin_2, annual_cos_2, annual_sin_3, annual_cos_3  

### Model tuning

The data of January - April of 2016 were used as validation dataset for some minor model tuning. Based on the model performance on this validation dataset, a larger feature set was narrowed down to the features described above. The model hyperparameter tuning is done on the 6 train round data. The best model is selected by cross validation using these training data in the 6 forecast rounds. The set of hyperparameters which yield the best cross validation pinball loss will be used to train models and forecast energy load across all 6 forecast rounds.

### Description of implementation scripts

* `feature_engineering.py`: Python script for computing features and generating feature files.
* `train_validate.R`: R script that trains Quantile Regression Neutral Network models and evaluate the loss on validation data of each cross validation round and forecast round with a set of hyperparameters and calculate the average loss. This script is used for grid search on vm.
* `train_validate_aml.R`: R script that trains Quantile Regression Neutral Network models and evaluate the loss on validation data of each cross validation round and forecast round with a set of hyperparameters and calculate the average loss. This script is used as the entry script for hyperdrive. 
* `aml_estimator.py`: Python script that passes the inputs and outputs between hyperdrive and the entry script `train_validate_aml.R`. 
* `hyperparameter_tuning.ipynb`: Jupyter notebook that does hyperparameter tuning with azureml hyperdrive.
* `train_predict.R`: R script that trains Quantile Regression Neutral Network models and predicts on each round of test data.
* `train_validate_vm.sh`: Bash script that runs `feature_engineering.py` and `train_validate.R` multiple times to generate cross validation result files and measure model tuning time.
* `train_score_vm.sh`: Bash script that runs `feature_engineering.py` and `train_predict.R` five times to generate five submission files and measure model running time.

### Steps to reproduce results

0. Follow the instructions [here](#resource-deployment-instructions) to provision a Linux virtual machine and log into the provisioned
VM.

1. Clone the TSPerf repo to the home directory of your machine and check out the fnn model branch

   ```bash
   cd ~/TSPerf
   git clone https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf
   git checkout zhouf/energy_forecast_fnn_cv_v1
   ```
   Use one of the following options to securely connect to the Git repo:
   * [Personal Access Tokens](https://docs.microsoft.com/en-us/vsts/organizations/accounts/use-personal-access-tokens-to-authenticate?view=vsts)  
   For this method, the clone command becomes
   ```bash
   git clone https://<username>:<personal access token>@msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf
   ```
   * [Git Credential Managers](https://docs.microsoft.com/en-us/vsts/repos/git/set-up-credential-managers?view=vsts)
   * [Authenticate with SSH](https://docs.microsoft.com/en-us/vsts/repos/git/use-ssh-keys-to-authenticate?view=vsts)


2. Create a conda environment for running the scripts of data downloading, data preparation, and result evaluation.   
To do this, you need to check if conda has been installed by runnning command `conda -V`. If it is installed, you will see the conda version in the terminal. Otherwise, please follow the instructions [here](https://conda.io/docs/user-guide/install/linux.html) to install conda.  
Then, you can go to `TSPerf` directory in the VM and create a conda environment named `tsperf` by running

   ```bash
   cd ~/TSPerf
   conda env create --file ./common/conda_dependencies.yml
   ```

3. Download and extract data **on the VM**.

   ```bash
   source activate tsperf
   python energy_load/GEFCom2017_D_Prob_MT_hourly/common/download_data.py
   python energy_load/GEFCom2017_D_Prob_MT_hourly/common/extract_data.py
   ```

4. Prepare Docker container for model training and predicting.  
   > NOTE: To execute docker commands without
   sudo as a non-root user, you need to create a
   Unix group and add users to it by following the instructions
   [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user). Otherwise, simply change to a root user by the following command:

   ```bash
   sudo -i
   ```

   4.1 Log into Azure Container Registry (ACR)

   ```bash
   docker login --username tsperf --password <ACR Access Key> tsperf.azurecr.io
   ```

   The `<ACR Acccess Key>` can be found [here](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/ff18d7a8-962a-406c-858f-49acd23d6c01/resourceGroups/tsperf/providers/Microsoft.ContainerRegistry/registries/tsperf/accessKey).   

   4.2 Pull the Docker image from ACR to your VM

   ```bash
   docker pull tsperf.azurecr.io/energy_load/gefcom2017_d_prob_mt_hourly/fnn_image:v1
   ```

5. Tune Hyperparameters **within Docker container** or **with AzureML hyperdrive**.

   5.1.1 Start a Docker container from the image  

   ```bash
   docker run -it -v ~/TSPerf:/TSPerf --name fnn_cv_container tsperf.azurecr.io/energy_load/gefcom2017_d_prob_mt_hourly/fnn_image:v1
   ```

   Note that option `-v ~/TSPerf:/TSPerf` mounts the `~/TSPerf` folder (the one you cloned) to the container so that you can access the code and data on your VM within the container.

   5.1.2 Train and validate

   ```
   source activate tsperf
   bash /TSPerf/energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/train_validate_vm.sh > cv_out.txt &
   ```
   After generating the cross validation results, you can exit the Docker container by command `exit`. 

   5.2 Do hyperparameter tuning with AzureML hyperdrive
   
   To tune hyperparameters with AzureML hyperdrive, you don't need to create a local Docker container. You can do feature engineering on the VM by the command

   ```
   cd ~/TSPerf
   source activate tsperf
   python energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/feature_engineering.py
   ```
   and then run through the jupyter notebook `hyperparameter_tuning.ipynb` on the VM with the conda env `tsperf` as the jupyter kernel.

   Based on the average pinball loss obtained at each set of hyperparameters, you can choose the best set of hyperparameters and use it in the Rscript of `train_predict.R`.

6. Train and predict **within Docker container**.

   6.1 Start a Docker container from the image  

   ```bash
   docker run -it -v ~/TSPerf:/TSPerf --name fnn_container tsperf.azurecr.io/energy_load/gefcom2017_d_prob_mt_hourly/fnn_image:v1
   ```

   Note that option `-v ~/TSPerf:/TSPerf` mounts the `~/TSPerf` folder (the one you cloned) to the container so that you can access the code and data on your VM within the container.

   6.2 Train and predict  

   ```
   source activate tsperf
   bash /TSPerf/energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/train_score_vm.sh > out.txt &
   ```
   After generating the forecast results, you can exit the Docker container by command `exit`.

7. Model evaluation **on the VM**.

   ```bash
   source activate tsperf
   cd ~/TSPerf
   bash ./common/evaluate submissions/fnn energy_load/GEFCom2017_D_Prob_MT_hourly
   ```

## Implementation resources

**Platform:** Azure Cloud  
**Hardware:** Standard D8s v3 (8 vcpus, 32 GB memory) Linux Data Science Virtual Machine (DSVM)  
**Data storage:** Premium SSD  
**Docker image:** tsperf.azurecr.io/energy_load/gefcom2017_d_prob_mt_hourly/fnn_image:v1  

**Key packages/dependencies:**
  * Python
    - python==3.6    
  * R
    - r-base==3.5.1  
    - qrnn==2.0.2
    - data.table==1.10.4.3
    - rjson==0.2.20 (optional for cv)
    - doParallel==1.0.14 (optional for cv)

## Resource deployment instructions
Please follow the instructions below to deploy the Linux DSVM.
  - Create an Azure account and log into [Azure portal](portal.azure.com/)
  - Refer to the steps [here](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) to deploy a *Data Science Virtual Machine for Linux (Ubuntu)*.

## Implementation evaluation
**Quality:**  
Note there is no randomness in this fnn model, so the model quality is the same for all five runs.

* Pinball loss run 1: 80.27

* Pinball loss run 2: 80.24

* Pinball loss run 3: 80.25

* Pinball loss run 4: 80.24

* Pinball loss run 5: 80.22

* Median Pinball loss: 80.24

**Time:**

* Run time 1: 5187 seconds

* Run time 2: 5132 seconds

* Run time 3: 5046 seconds

* Run time 4: 5048 seconds

* Run time 5: 5095 seconds

* Median run time: 5095 seconds

**Cost:**  
The hourly cost of the Standard D8s DSVM is 0.3840 USD based on the price at the submission date.   
Thus, the total cost is 5095/3600 * 0.3840 = $0.5435.
