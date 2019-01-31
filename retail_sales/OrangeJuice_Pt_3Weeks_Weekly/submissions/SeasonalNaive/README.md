# Implementation submission form

## Submission details

**Submission date**: 09/01/2018

**Benchmark name:** OrangeJuice_Pt_3Weeks_Weekly

**Submitter(s):** Chenhui Hu

**Submitter(s) email:** chenhhu@microsoft.com

**Submission name:** SeasonalNaive

**Submission branch:** [chenhui/seasonal_naive](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf?_a=contents&version=GBchenhui%2Fseasonal_naive)

**Pull request:** [Added seasonal naive method for retail sales forecasting](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf/pullrequest/150386?_a=overview)

**Submission path:** [/retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/SeasonalNaive](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf?path=%2Fretail_sales%2FOrangeJuice_Pt_3Weeks_Weekly%2Fsubmissions%2FSeasonalNaive&version=GBchenhui%2Fseasonal_naive)


## Implementation description

### Modelling approach

In this submission, we implement seasonal naive forecast method using R package `forecast`. 

### Feature engineering

Only the weekly sales of each orange juice has been used in the implementation of the forecast method.

### Hyperparameter tuning

Default hyperparameters of the forecasting algorithm are used. Additionally, the frequency of the weekly sales time series is set to be 52, 
since there are approximately 52 weeks in a year. 

### Description of implementation scripts

* `train_score.r`: R script that trains the model and evaluate its performance
* `seasonal_naive.Rmd` (optional): R markdown that trains the model and visualizes the results
* `seasonal_naive.nb.html` (optional): Html file associated with the R markdown file

### Steps to reproduce results

0. Follow the instructions [here](#resource-deployment-instructions) to provision a Linux virtual machine and log into the provisioned 
VM. 

1. Choose submission branch and clone the Git repo to home directory of your machine:

   ```bash
   cd ~
   git clone https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf
   cd ~/TSPerf
   ```

   Please use the recommended [Git Credential Managers](https://docs.microsoft.com/en-us/vsts/repos/git/set-up-credential-managers?view=vsts) or [Personal Access Tokens](https://docs.microsoft.com/en-us/vsts/organizations/accounts/use-personal-access-tokens-to-authenticate?view=vsts) to securely 
   connect to Git repos via HTTPS authentication. If these don't work, you can try to [connect through SSH](https://docs.microsoft.com/en-us/vsts/repos/git/use-ssh-keys-to-authenticate?view=vsts). The above commands will download the 
   source code of the submission branch into a local folder named TSPerf. Note that you will not need to run `git checkout chenhui/seasonal_naive` once the submission branch is merged into the master branch.

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
   docker pull tsperf.azurecr.io/retail_sales/orangejuice_pt_3weeks_weekly/baseline_image:v1
   ```

7. Choose a name for a new Docker container (e.g. snaive_container) and create it using command:   
   
   ```bash
   docker run -it -v $(pwd):/TSPerf --name snaive_container tsperf.azurecr.io/retail_sales/orangejuice_pt_3weeks_weekly/baseline_image:v1
   ```
   
   Note that option `-v $(pwd):/TSPerf` allows you to mount `/TSPerf` folder (the one you cloned) to the container so that you will have 
   access to the source code in the container. 

8. Inside `/TSPerf` folder, train the model and make predictions by running

   ```bash
   source ./common/train_score_vm ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/SeasonalNaive R
   ``` 
 
   This will generate 5 `submission_seed_<seed number>.csv` files in the submission directory, where \<seed number\> 
   is between 1 and 5. This command will also output 5 running times of train_score.py. The median of the times 
   reported in rows starting with 'real' should be compared against the wallclock time declared in benchmark 
   submission. After generating the forecast results, you can exit the Docker container by command `exit`. 

9. Activate conda environment again by `source activate tsperf`. Then, evaluate the benchmark quality by running
   
   ```bash
   source ./common/evaluate ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/SeasonalNaive ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly
   ```

   This command will output 5 benchmark quality values (MAPEs). Their median should be compared against the 
   benchmark quality declared in benchmark submission.


## Implementation resources

**Platform:** Azure Cloud 

**Resource location:** East US  

**Hardware:** Standard D2s v3 (2 vcpus, 8 GB memory, 16 GB temporary storage) Ubuntu Linux VM

**Data storage:** Premium SSD

**Docker image:** tsperf.azurecr.io/retail_sales/orangejuice_pt_3weeks_weekly/baseline_image:v1

**Key packages/dependencies:**  
  * R 
    - r-base==3.5.1  
    - forecast==8.1

## Resource deployment instructions

We use Azure Linux VM to develop the baseline methods. Please follow the instructions below to deploy the resource.
* Azure Linux VM deployment
  - Create an Azure account and log into [Azure portal](portal.azure.com/)
  - Refer to the steps [here](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) to deploy a Data 
  Science Virtual Machine for Linux (Ubuntu). Select *D2s_v3* as the virtual machine size.


## Implementation evaluation

**Quality:** 

*MAPE run 1: 165.06%*

*MAPE run 2: 165.06%*

*MAPE run 3: 165.06%*

*MAPE run 4: 165.06%*

*MAPE run 5: 165.06%*

*median MAPE: 165.06%*

**Time:** 

*run time 1: 159.44 seconds*

*run time 2: 160.45 seconds*

*run time 3: 162.24 seconds*

*run time 4: 158.73 seconds*

*run time 5: 160.83 seconds*

*median run time: 160.45 seconds*

**Cost:** The hourly cost of the D2s v3 Ubuntu Linux VM in East US Azure region is 0.096 USD, based on the price at the submission date. Thus, the total cost is 160.45/3600 $\times$ 0.096 = $0.0043.

Note that there is no randomness in the forecasts obtained by the above method. Thus, quality values do not change over 
different runs.
