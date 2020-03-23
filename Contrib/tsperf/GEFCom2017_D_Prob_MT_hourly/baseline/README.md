# Implementation submission form

## Submission information

**Submission date**: 09/14/2018

**Benchmark name:** GEFCom2017_D_Prob_MT_hourly

**Submitter(s):** Hong Lu

**Submitter(s) email:** honglu@microsoft.com

**Submission name:** baseline

**Submission path:** benchmarks/GEFCom2017_D_Prob_MT_hourly/baseline


## Implementation description

### Modelling approach

In this submission, we implement a simple quantile regression model using the `quantreg` package in R.

### Feature engineering

The following features are used:  
**LoadLag**: Average load based on the same-day and same-hour load values of the same week, the week before the same week, and the week after the same week of the previous three years, i.e. 9 values are averaged to compute this feature.  
**DryBulbLag**:  Average DryBulb temperature based on the same-hour DryBulb values of the same day, the day before the same day, and the day after the same day of the previous three years, i.e. 9 values are averaged to compute this feature.  
**Weekly Fourier Series**: weekly_sin_1, weekly_cos_1,  weekly_sin_2, weekly_cos_2, weekly_sin_3, weekly_cos_3  
**Annual Fourier Series**: annual_sin_1, annual_cos_1, annual_sin_2, annual_cos_2, annual_sin_3, annual_cos_3  

### Model tuning

The data of January - April of 2016 were used as validation dataset for some minor model tuning. Based on the model performance on this validation dataset, a larger feature set was narrowed down to the features described above.
No parameter tuning was done.

### Description of implementation scripts

* `compute_features.py`: Python script for computing features and generating feature files.
* `train_predict.R`: R script that trains Quantile Regression models and predicts on each round of test data.
* `train_score_vm.sh`: Bash script that runs `compute_features.py` and `train_predict.R` five times to generate five submission files and measure model running time.

### Steps to reproduce results

1. Follow the instructions [here](#resource-deployment-instructions) to provision a Linux virtual machine and log into the provisioned
VM.

2. Clone the Forecasting repo to the home directory of your machine

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

   5.1 Make sure Docker is installed
    
   You can check if Docker is installed on your VM by running

   ```bash
   sudo docker -v
   ```
   You will see the Docker version if Docker is installed. If not, you can install it by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/). Note that if you want to execute Docker commands without sudo as a non-root user, you need to create a Unix group and add users to it by following the instructions [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).  

   5.2 Build a local Docker image

   ```bash
   sudo docker build -t baseline_image benchmarks/GEFCom2017_D_Prob_MT_hourly/baseline
   ```

6. Train and predict **within Docker container**

   6.1 Start a Docker container from the image  

   ```bash
   sudo docker run -it -v ~/Forecasting:/Forecasting --name baseline_container baseline_image
   ```

   Note that option `-v ~/Forecasting:/Forecasting` mounts the `~/Forecasting` folder (the one you cloned) to the container so that you can access the code and data on your VM within the container.

   6.2 Train and predict  

   ```
   source activate tsperf
   cd /Forecasting
   bash benchmarks/GEFCom2017_D_Prob_MT_hourly/baseline/train_score_vm.sh
   ```
   After generating the forecast results, you can exit the Docker container by command `exit`.
   
7. Model evaluation **on the VM**

   ```bash
   source activate tsperf
   cd ~/Forecasting
   bash tsperf/benchmarking/evaluate baseline tsperf/benchmarking/GEFCom2017_D_Prob_MT_hourly
   ```

## Implementation resources

**Platform:** Azure Cloud   
**Resource location:** East US region   
**Hardware:** Standard D8s v3 (8 vcpus, 32 GB memory) Ubuntu Linux VM    
**Data storage:** Premium SSD  
**Dockerfile:** [energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/baseline/Dockerfile](https://github.com/Microsoft/Forecasting/blob/master/energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/baseline/Dockerfile)  

**Key packages/dependencies:**
  * Python
    - python==3.7    
  * R
    - r-base==3.5.3  
    - quantreg==5.34
    - data.table==1.10.4.3

## Resource deployment instructions
Please follow the instructions below to deploy the Linux DSVM.
  - Create an Azure account and log into [Azure portal](portal.azure.com/)
  - Refer to the steps [here](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) to deploy a *Data Science Virtual Machine for Linux (Ubuntu)*. Select *D8s_v3* as the virtual machine size.  

## Implementation evaluation
**Quality:**  
Note there is no randomness in this baseline model, so the model quality is the same for all five runs.

* Pinball loss run 1: 84.12

* Pinball loss run 2: 84.12

* Pinball loss run 3: 84.12

* Pinball loss run 4: 84.12

* Pinball loss run 5: 84.12

* Median Pinball loss: 84.12

**Time:**

* Run time 1: 188 seconds

* Run time 2: 185 seconds

* Run time 3: 185 seconds

* Run time 4: 189 seconds

* Run time 5: 189 seconds

* Median run time:  **188 seconds**

**Cost:**  
The hourly cost of the Standard D8s Ubuntu Linux VM in East US Azure region is 0.3840 USD, based on the price at the submission date.   
Thus, the total cost is 188/3600 * 0.3840 = $0.0201.

**Average relative improvement (in %) over GEFCom2017 benchmark model**  (measured over the first run)  
Round 1: -6.67  
Round 2: 20.26  
Round 3: 20.05  
Round 4: -5.61  
Round 5: -6.45  
Round 6: 11.21  

**Ranking in the qualifying round of GEFCom2017 competition**  
10
