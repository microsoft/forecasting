# Implementation submission form

## Submission information

**Submission date**: 10/26/2018

**Benchmark name:** GEFCom2017_D_Prob_MT_hourly

**Submitter(s):** Fang Zhou

**Submitter(s) email:** zhouf@microsoft.com

**Submission name:** Quantile Regression Neural Network

**Submission path:** benchmarks/GEFCom2017_D_Prob_MT_hourly/fnn


## Implementation description

### Modelling approach

In this submission, we implement a quantile regression neural network model using the `qrnn` package in R.

### Feature engineering

The following features are used:  
**LoadLag**: Average load based on the same-day and same-hour load values of the same week, the week before the same week, and the week after the same week of the previous three years, i.e. 9 values are averaged to compute this feature.  
**DryBulbLag**:  Average DryBulb temperature based on the same-hour DryBulb values of the same day, the day before the same day, and the day after the same day of the previous three years, i.e. 9 values are averaged to compute this feature.  
**Weekly Fourier Series**: weekly_sin_1, weekly_cos_1,  weekly_sin_2, weekly_cos_2, weekly_sin_3, weekly_cos_3  
**Annual Fourier Series**: annual_sin_1, annual_cos_1, annual_sin_2, annual_cos_2, annual_sin_3, annual_cos_3  

### Model tuning

The data of January - April of 2016 were used as validation dataset for some minor model tuning. Based on the model performance on this validation dataset, a larger feature set was narrowed down to the features described above. The model hyperparameter tuning is done on the 6 train round data using 4 cross validation folds with 6 forecasting rounds in each fold. The set of hyperparameters which yield the best cross validation pinball loss will be used to train models and forecast energy load across all 6 forecast rounds.

### Description of implementation scripts

Train and Predict:
* `compute_features.py`: Python script for computing features and generating feature files.
* `train_predict.R`: R script that trains Quantile Regression Neural Network models and predicts on each round of test data.
* `train_score_vm.sh`: Bash script that runs `compute_features.py` and `train_predict.R` five times to generate five submission files and measure model running time.

Tune hyperparameters using R:
* `cv_settings.json`: JSON script that sets cross validation folds.
* `train_validate.R`: R script that trains Quantile Regression Neural Network models and evaluate the loss on validation data of each cross validation round and forecast round with a set of hyperparameters and calculate the average loss. This script is used for grid search on vm.
* `train_validate_vm.sh`: Bash script that runs `compute_features.py` and `train_validate.R` multiple times to generate cross validation result files and measure model tuning time.

Tune hyperparameters using AzureML HyperDrive:
* `cv_settings.json`: JSON script that sets cross validation folds.
* `train_validate_aml.R`: R script that trains Quantile Regression Neural Network models and evaluate the loss on validation data of each cross validation round and forecast round with a set of hyperparameters and calculate the average loss. This script is used as the entry script for hyperdrive.
* `aml_estimator.py`: Python script that passes the inputs and outputs between hyperdrive and the entry script `train_validate_aml.R`.
* `hyperparameter_tuning.ipynb`: Jupyter notebook that does hyperparameter tuning with azureml hyperdrive.

### Steps to reproduce results

1. Follow the instructions [here](#resource-deployment-instructions) to provision a Linux virtual machine and log into the provisioned
VM.

2. Clone the Forecasting repo to the home directory of your machine

   ```bash
   cd ~
   git clone https://github.com/Microsoft/Forecasting.git
   cd Forecasting
   ```
   Use one of the following options to securely connect to the Git repo:
   * [Personal Access Tokens](https://docs.microsoft.com/en-us/vsts/organizations/accounts/use-personal-access-tokens-to-authenticate?view=vsts)  
   For this method, the clone command becomes
   ```bash
   git clone https://<username>:<personal access token>@github.com/Microsoft/Forecasting.git
   ```
   * [Git Credential Managers](https://docs.microsoft.com/en-us/vsts/repos/git/set-up-credential-managers?view=vsts)
   * [Authenticate with SSH](https://docs.microsoft.com/en-us/vsts/repos/git/use-ssh-keys-to-authenticate?view=vsts)

3. Create a conda environment for running the scripts of data downloading, data preparation, and result evaluation.   
To do this, you need to check if conda has been installed by runnning command `conda -V`. If it is installed, you will see the conda version in the terminal. Otherwise, please follow the instructions [here](https://conda.io/docs/user-guide/install/linux.html) to install conda.  
Then, you can go to `~/Forecasting` directory in the VM and create a conda environment named `tsperf` by running

   ```bash
   cd ~/Forecasting
   conda env create --file tsperf/benchmarking/conda_dependencies.yml
   ```

4. Download and extract data **on the VM**.

   ```bash
   source activate tsperf
   python tsperf/benchmarking/GEFCom2017_D_Prob_MT_hourly/download_data.py
   python tsperf/benchmarking/GEFCom2017_D_Prob_MT_hourly/extract_data.py
   ```

5. Prepare Docker container for model training and predicting.

   > NOTE: To execute docker commands without sudo as a non-root user, you need to create a Unix group and add users to it by following the instructions
   [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user). Otherwise, simply prefix all docker commands with sudo.

   5.1 Make sure Docker is installed
    
   You can check if Docker is installed on your VM by running

   ```bash
   sudo docker -v
   ```
   You will see the Docker version if Docker is installed. If not, you can install it by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

   5.2 Build a local Docker image

   ```bash
   sudo docker build -t fnn_image benchmarks/GEFCom2017_D_Prob_MT_hourly/fnn
   ```

6. Tune Hyperparameters **within Docker container** or **with AzureML hyperdrive**.

   6.1.1 Start a Docker container from the image  

   ```bash
   sudo docker run -it -v ~/Forecasting:/Forecasting --name fnn_cv_container fnn_image
   ```

   Note that option `-v ~/Forecasting:/Forecasting` mounts the `~/Forecasting` folder (the one you cloned) to the container so that you can access the code and data on your VM within the container.

   6.1.2 Train and validate

   ```
   source activate tsperf
   cd /Forecasting
   nohup bash benchmarks/GEFCom2017_D_Prob_MT_hourly/fnn/train_validate_vm.sh >& cv_out.txt &
   ```
   After generating the cross validation results, you can exit the Docker container by command `exit`.

   6.2 Do hyperparameter tuning with AzureML hyperdrive

   To tune hyperparameters with AzureML hyperdrive, you don't need to create a local Docker container. You can do feature engineering on the VM by the command

   ```
   cd ~/Forecasting
   source activate tsperf
   python benchmarks/GEFCom2017_D_Prob_MT_hourly/fnn/compute_features.py
   ```
   and then run through the jupyter notebook `hyperparameter_tuning.ipynb` on the VM with the conda env `tsperf` as the jupyter kernel.

   Based on the average pinball loss obtained at each set of hyperparameters, you can choose the best set of hyperparameters and use it in the Rscript of `train_predict.R`.

7. Train and predict **within Docker container**.

   7.1 Start a Docker container from the image  

   ```bash
   sudo docker run -it -v ~/Forecasting:/Forecasting --name fnn_container fnn_image
   ```

   Note that option `-v ~/Forecasting:/Forecasting` mounts the `~/Forecasting` folder (the one you cloned) to the container so that you can access the code and data on your VM within the container.

   7.2 Train and predict  

   ```
   source activate tsperf
   cd /Forecasting
   nohup bash benchmarks/GEFCom2017_D_Prob_MT_hourly/fnn/train_score_vm.sh >& out.txt &
   ```
   The last command will take about 7 hours to complete. You can monitor its progress by checking out.txt file. Also during the run you can disconnect from VM. After reconnecting to VM, use the command  

   ```
   sudo docker exec -it fnn_container /bin/bash
   tail out.txt
   ```
   to connect to the running container and check the status of the run.  
   After generating the forecast results, you can exit the Docker container by command `exit`.

8. Model evaluation **on the VM**.

   ```bash
   source activate tsperf
   cd ~/Forecasting
   bash tsperf/benchmarking/evaluate fnn tsperf/benchmarking/GEFCom2017_D_Prob_MT_hourly
   ```

## Implementation resources

**Platform:** Azure Cloud  
**Resource location:** East US region   
**Hardware:** Standard D8s v3 (8 vcpus, 32 GB memory) Ubuntu Linux VM  
**Data storage:** Premium SSD  
**Dockerfile:** [energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/Dockerfile](https://github.com/Microsoft/Forecasting/blob/master/energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/Dockerfile)  

**Key packages/dependencies:**
  * Python
    - python==3.7    
  * R
    - r-base==3.5.3  
    - qrnn==2.0.2
    - data.table==1.10.4.3
    - rjson==0.2.20 (optional for cv)
    - doParallel==1.0.14 (optional for cv)

## Resource deployment instructions
Please follow the instructions below to deploy the Linux DSVM.
  - Create an Azure account and log into [Azure portal](portal.azure.com/)
  - Refer to the steps [here](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) to deploy a *Data Science Virtual Machine for Linux (Ubuntu)*. Select *D8s_v3* as the virtual machine size.  

## Implementation evaluation
**Quality:**  

* Pinball loss run 1: 79.54

* Pinball loss run 2: 78.32

* Pinball loss run 3: 80.06

* Pinball loss run 4: 80.12

* Pinball loss run 5: 80.13

* Median Pinball loss: 80.06

**Time:**

* Run time 1: 1092 seconds

* Run time 2: 1085 seconds

* Run time 3: 1062 seconds

* Run time 4: 1083 seconds

* Run time 5: 1110 seconds

* Median run time: 1085 seconds

**Cost:**  
The hourly cost of the Standard D8s Ubuntu Linux VM in East US Azure region is 0.3840 USD, based on the price at the submission date.   
Thus, the total cost is 1085/3600 * 0.3840 = $0.1157.

**Average relative improvement (in %) over GEFCom2017 benchmark model**  (measured over the first run)  
Round 1: 6.13  
Round 2: 19.20  
Round 3: 18.86   
Round 4: 3.84  
Round 5: 2.76  
Round 6: 11.10  

**Ranking in the qualifying round of GEFCom2017 competition**  
4
