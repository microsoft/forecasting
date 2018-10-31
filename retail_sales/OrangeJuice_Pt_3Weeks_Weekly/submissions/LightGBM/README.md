# Implementation submission form

## Submission details

**Submission date**: 10/22/2018

**Benchmark name:** OrangeJuice_Pt_3Weeks_Weekly

**Submitter(s):** Chenhui Hu

**Submitter(s) email:** chenhhu@microsoft.com

**Submission name:** LightGBM

**Submission branch:** [chenhui/boosted_decision_tree](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf?version=GBchenhui%2Fboosted_decision_tree)

**Pull request:** [Added boosted decision tree method for retail sales forecasting](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf/pullrequest/159654?_a=overview)

**Submission path:** [/retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/LightGBM](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf?path=%2Fretail_sales%2FOrangeJuice_Pt_3Weeks_Weekly%2Fsubmissions%2FLightGBM&version=GBchenhui%2Fboosted_decision_tree)


## Implementation description

### Modelling approach

In this submission, we implement boosted decision tree method using Python package `lightgbm`, which is a fast, distributed, high performance 
gradient boosting framework based on decision tree algorithms. 

### Feature engineering

The following features have been used in the implementation of the forecast method:
- datetime features including week number, week of the month, and day of the month
- weekly sales of each orange juice in recent weeks
- average sales of each orange juice during recent weeks
- other features including *store*, *brand*, *profit*, *deal*, *feat* columns


### Hyperparameter tuning

This model mainly involves the following hyperparameters:  
- Hyperparameters of gradient boosting machine (GBM): *num_leaves* (maximum number of leaves in one tree), *min_data_in_leaf* (minimum number of data in one leaf), *learning_rate* (learning rate), *feature_fraction* (ratio of the randomly selected features), *bagging_fraction* (ratio of the randomly sampled data), *bagging_freq* (frequency for bagging), *num_threads* (number of threads), *num_boost_round* (number of training iterations)
- Lag values for computing the weekly sales in recent weeks
- Window size and starting point for computing the average sales 

We have conducted a grid search to tune these parameters. The range of each hyperparameter is selected based on its importance reported in other places and our intuition about the data. The ranges of a few important hyperparameters are as follows
- *num_leaves*: [50, 80, 100, 200]
- *min_data_in_leaf*: [100, 200]
- *learning_rate*: [0.001, 0.002, 0.01, 0.02, 0.1]
- *num_boost_round*: [100, 400, 1000, 2000]


### Description of implementation scripts

* `train_score.py`: Python script that trains the model and evaluates its performance
* `train_score.ipynb` (optional): Jupyter notebook that trains the model and visualizes the results


### Steps to reproduce results

0. Follow the instructions [here](#resource-deployment-instructions) to provision a Linux virtual
machine and log into the provisioned VM. 

1. Choose submission branch and clone the Git repo to home directory of your machine:

   ```bash
   cd ~
   git clone https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf
   cd ~/TSPerf
   git checkout chenhui/boosted_decision_tree
   ```

   Please use the recommended [Git Credential Managers](https://docs.microsoft.com/en-us/vsts/repos/git/set-up-credential-managers?view=vsts) or [Personal Access Tokens](https://docs.microsoft.com/en-us/vsts/organizations/accounts/use-personal-access-tokens-to-authenticate?view=vsts) to securely 
   connect to Git repos via HTTPS authentication. If these don't work, you can try to [connect through SSH](https://docs.microsoft.com/en-us/vsts/repos/git/use-ssh-keys-to-authenticate?view=vsts). The above commands will download the 
   source code of the submission branch into a local folder named TSPerf. Note that you will not need to run `git checkout chenhui/boosted_decision_tree` once the submission branch is merged into the master branch.

2. Create a conda environment for running the scripts of data downloading, data preparation, and result evaluation. To do this, you need 
to check if conda has been installed by runnning command `conda -V`. If it is installed, you will see the conda version in the terminal. Otherwise, please follow the instructions [here](https://conda.io/docs/user-guide/install/linux.html) to install conda. Then, you can go to `TSPerf` directory in the VM and create a conda environment named `tsperf` by

   ```bash
   conda env create --file ./common/conda_dependencies.yml
   ```
  
   This will create a conda environment with the Python and R packages listed in `conda_dependencies.yml` being installed. The conda 
  environment name is also defined in the yml file. 

3. Activate the conda environment and download the Orange Juice dataset. Use command `source activate tsperf` to activate the conda environment. Then, download the Orange Juice dataset by running the following command from `/TSPerf` directory 

   ```bash
   Rscript ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/common/download_data.r
   ```

   This will create a data directory `./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/data` and store the dataset in this directory. The dataset has two csv files - `yx.csv` and `storedemo.csv` which contain the sales information and store demographic information, respectively. 

4. From `/TSPerf` directory, run the following command to generate the training data and testing data for each forecast period:

   ```bash
   python ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/common/serve_folds.py --test --save
   ```

   This will generate 12 csv files named `train_round_#.csv` and 12 csv files named `test_round_#.csv` in two subfolders `/train` and 
   `/test` under the data directory, respectively. After running the above command, you can deactivate the conda environment by running 
   `source deactivate`.

5. Log into Azure Container Registry (ACR):
   
   ```bash
   docker login --username tsperf --password <ACR Access Key> tsperf.azurecr.io
   ```
   
   The `<ACR Acccess Key>` can be found [here](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/ff18d7a8-962a-406c-858f-49acd23d6c01/resourceGroups/tsperf/providers/Microsoft.ContainerRegistry/registries/tsperf/accessKey). If want to execute docker commands without 
   sudo as a non-root user, you need to create a 
   Unix group and add users to it by following the instructions 
   [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

6. Pull a Docker image from ACR using the following command   

   ```bash
   docker pull tsperf.azurecr.io/retail_sales/orangejuice_pt_3weeks_weekly/lightgbm_image:v1
   ```

7. Choose a name for a new Docker container (e.g. lightgbm_container) and create it using command:   
   
   ```bash
   cd ~/TSPerf
   docker run -it -v $(pwd):/TSPerf --name lightgbm_container tsperf.azurecr.io/retail_sales/orangejuice_pt_3weeks_weekly/lightgbm_image:v1
   ```
   
   Note that option `-v $(pwd):/TSPerf` allows you to mount `/TSPerf` folder (the one you cloned) to the container so that you will have 
   access to the source code in the container. 

8. Inside the Docker container, train the model and make predictions by running the following command from `/TSPerf` folder

   ```bash
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

**Resource location:** West US 2

**Hardware:** Standard D2s v3 (2 vcpus, 8 GB memory, 16 GB temporary storage) Ubuntu Linux VM

**Data storage:** Premium SSD

**Docker image:** tsperf.azurecr.io/retail_sales/orangejuice_pt_3weeks_weekly/baseline_image:v1

**Key packages/dependencies:**  
  * Python
    - pandas==0.23.1
    - lightgbm==2.1.2

## Resource deployment instructions

We use Azure Linux VM to develop the baseline methods. Please follow the instructions below to deploy the resource.
* Azure Linux VM deployment
  - Create an Azure account and log into [Azure portal](portal.azure.com/)
  - Refer to the steps [here](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) to deploy a Data 
  Science Virtual Machine for Linux (Ubuntu). Select *D2s_v3* as the virtual machine size.


## Implementation evaluation

**Quality:** 

*MAPE run 1: 52.08%*

*MAPE run 2: 52.05%*

*MAPE run 3: 52.01%*

*MAPE run 4: 52.05%*

*MAPE run 5: 51.99%*

*median MAPE: 52.04%*

**Time:** 

*run time 1: 543.30 seconds*

*run time 2: 530.38 seconds*

*run time 3: 535.10 seconds*

*run time 4: 533.15 seconds*

*run time 5: 541.32 seconds*

*median run time: 536.65 seconds*

**Cost:** The total cost is 182.10/3600 $\times$ 0.096 = $0.0143.
