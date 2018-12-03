# Implementation submission form

## Submission details

**Submission date**: 11/29/2018

**Benchmark name:** OrangeJuice_Pt_3Weeks_Weekly

**Submitter(s):** Chenhui Hu

**Submitter(s) email:** chenhhu@microsoft.com

**Submission name:** DilatedCNN

**Submission branch:** [chenhui/wavenet](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf?version=GBchenhui%2Fwavenet)

**Pull request:** [Added Dilated CNN method for retail sales forecasting](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf/pullrequest/150743?_a=overview)

**Submission path:** [/retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/DilatedCNN](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/TSPerf?path=%2Fretail_sales%2FOrangeJuice_Pt_3Weeks_Weekly%2Fsubmissions%2FDilatedCNN&version=GBchenhui%2Fwavenet)


## Implementation description

### Modelling approach

In this submission, we implement a Dilated Convolutional Neural Network (CNN) model using Keras package. Dilated CNN is a class of CNN that was initially 
proposed to improve audio waveform generation in [this paper](https://arxiv.org/abs/1609.03499) by Oord et al in 2016. Later this model has shown great 
performance in solving time series forecasting problems of several recent machine learning competitions. 

### Feature engineering

The following features have been used in the implementation of the forecast method:

- datetime features including week of the month and month number
- weekly sales of each orange juice in recent weeks 
- other dynamic features including *deal* and  *feat* columns 
- static features including store index and brand index

### Hyperparameter tuning

We tune the hyperparameters of the model with HyperDrive which is accessible through Azure ML SDK. A Batch AI cluster with GPU support is created 
to distribute the computation. The hyperparameters tuned with HyperDrive and their ranges are as follows
- input sequence length: [6, 8, 10, 12, 14, 16, 18, 20]
- batch size: [16, 32, 64]
- learning rate: [0.01, 0.015, 0.02, 0.025]
- number of epochs: [3,4,5,6,8]

### Description of implementation scripts

* `train_score.py`: Python script that trains the model and generates forecast results for each round
* `train_score.ipynb` (optional): Jupyter notebook that trains the model and visualizes the results
* `train_validate.py` (optional): Python script that does training and validation with the 1st round training data 
* `hyperparameter_tuning.ipynb` (optional): Jupyter notebook that tries different model configurations and selects the best model by running 
`train_validate.py` script in a Batch AI cluster with different sets of hyperparameters

### Steps to reproduce results

0. Follow the instructions [here](#resource-deployment-instructions) to provision a Linux virtual machine and log into the provisioned 
VM. 

1. Choose submission branch and clone the Git repo to home directory of your machine:

   ```bash
   cd ~
   git clone https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf
   cd ~/TSPerf
   git checkout chenhui/wavenet
   ```

   Please use the recommended [Git Credential Managers](https://docs.microsoft.com/en-us/vsts/repos/git/set-up-credential-managers?view=vsts) or [Personal Access Tokens](https://docs.microsoft.com/en-us/vsts/organizations/accounts/use-personal-access-tokens-to-authenticate?view=vsts) to securely 
   connect to Git repos via HTTPS authentication. If these don't work, you can try to [connect through SSH](https://docs.microsoft.com/en-us/vsts/repos/git/use-ssh-keys-to-authenticate?view=vsts). The above commands will download the 
   source code of the submission branch into a local folder named TSPerf. Note that you will not need to run `git checkout chenhui/wavenet` once the submission branch is merged into the master branch.

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
   docker pull tsperf.azurecr.io/retail_sales/orangejuice_pt_3weeks_weekly/dcnn_image:v1
   ```

7. Choose a name for a new Docker container (e.g. dcnn_container) and create it using command:   
   
   ```bash
   docker run -it -v $(pwd):/TSPerf --runtime=nvidia --name dcnn_container tsperf.azurecr.io/retail_sales/orangejuice_pt_3weeks_weekly/dcnn_image:v1
   ```
   
   Note that option `-v $(pwd):/TSPerf` allows you to mount `/TSPerf` folder (the one you cloned) to the container so that you will have 
   access to the source code in the container. 

8. Inside `/TSPerf` folder, train the model and make predictions by running

   ```bash
   source ./common/train_score_vm ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/DilatedCNN Python3
   ``` 
 
   This will generate 5 `submission_seed_<seed number>.csv` files in the submission directory, where \<seed number\> 
   is between 1 and 5. This command will also output 5 running times of train_score.py. The median of the times 
   reported in rows starting with 'real' should be compared against the wallclock time declared in benchmark 
   submission. After generating the forecast results, you can exit the Docker container by command `exit`. 

9. Activate conda environment again by `source activate tsperf`. Then, evaluate the benchmark quality by running
   
   ```bash
   source ./common/evaluate ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/DilatedCNN ./retail_sales/OrangeJuice_Pt_3Weeks_Weekly
   ```

   This command will output 5 benchmark quality values (MAPEs). Their median should be compared against the 
   benchmark quality declared in benchmark submission.


## Implementation resources

**Platform:** Azure Cloud 

**Resource location:** South Central US 

**Hardware:** Standard NC12 (2 GPUs, 12 vCPUs, 112 GB memory, 680 GB temporary storage) Ubuntu Linux VM

**Data storage:** Standard HDD

**Docker image:** tsperf.azurecr.io/retail_sales/orangejuice_pt_3weeks_weekly/dcnn_image:v1

**Key packages/dependencies:**  
  * Python 
    - pandas==0.23.1  
    - tensorflow-gpu==1.12.0
    - keras==2.2.4

## Resource deployment instructions

We use Azure Linux VM to develop the baseline methods. Please follow the instructions below to deploy the resource.
* Azure Linux VM deployment
  - Create an Azure account and log into [Azure portal](portal.azure.com/)
  - Refer to the steps [here](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) to deploy a Data 
  Science Virtual Machine for Linux (Ubuntu). Select *NC12* as the virtual machine size.


## Implementation evaluation

**Quality:** 

*MAPE run 1: 37.27%*

*MAPE run 2: 39.23%*

*MAPE run 3: 40.28%*

*MAPE run 4: 39.38%*

*MAPE run 5: 38.42%*

*median MAPE: 39.23%*

**Time:** 

*run time 1: 1168.20 seconds*

*run time 2: 1161.48 seconds*

*run time 3: 1162.72 seconds*

*run time 4: 1161.84 seconds*

*run time 5: 1161.57 seconds*

*median run time: 1161.84 seconds*

**Cost:** The total cost is 1161.84/3600 $\times$ 2.160 = $0.6971.
