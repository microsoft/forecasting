# TSPerf Rules

## Contents

1. [Introduction](#introduction)  
   1.1 [Vision](#vision)  
   1.2 [Goals](#goals)   
2. [Framework](#framework)    
    2.1 [Definitions](#definitions)  
    2.2 [Structure and hierarchy of documents](#structure-and-hierarchy-of-documents)  
    2.3 [Structure of repository](#structure-of-repository)   
3. [Benchmarks](#benchmarks)  
    3.1 [Probabilistic electricity load forecasting](#probabilistic-electricity-load-forecasting)  
    3.2 [Retail sales forecasting](#retail-sales-forecasting)  
4. [Development of benchmark implementation](#development-of-benchmark-implementation)  
    4.1 [Feature engineering](#feature-engineering)  
    4.2 [Guideline for creating Docker images](#guideline-for-creating-docker-images)  
    4.2 [Guideline for measuring performance](#guideline-for-measuring-performance) 
5. [Submission of benchmark implementation](#submission-of-benchmark-implementation)  
    5.1 [Guideline for submitting reproduction instructions](#guideline-for-submitting-reproduction-instructions)  
    5.2 [Guideline for submitting the code](#guideline-for-submitting-the-code)   
    5.3 [Pull request process](#pull-request-process)
6. [Review of submissions](#review-of-submissions)  
    6.1 [Standalone VM](#standalone-vm)  
    6.2 [Batch AI](#batch-ai)   
7. [Performance board](#performance-board)  
8. [Selection of reference implementation](#selection-of-reference-implementation)
    
## Introduction

### Vision

Our vision it to establish a leading framework that allows discovery and comparison of various time-series forecasting algorithms and architectures on a cloud-based environment. This framework will allow data scientists or customers to discover the best approach that fits their use case from cost, time and quality perspective.
TSPerf framework is designed to facilitate data science community participation and contribution through the development of implementations against a given set of forecasting problems and datasets. Once submitted, implementations will be measured in terms of standard metrics of model accuracy, training cost and model training time. Each implementation will include all the necessary instruction and tools that will ensure reproducibility on Azure customer's subscription.
Note: The TSPerf vision shared common principles with the [MLPerf](https://mlperf.org/) vision and designed to be proposed as a new track after an internal validation of the concept. 


### Goals:

We are planning on the following 3 phase goals:
* **Phase 1** – Internal only frame work that is used by internal Microsoft data scientists for submission and retrieval of implementation with the objective to reduce customer deployment time 
* **Phase 2**- Enable external customers to access, discover and clone implementations with the objective of enabling forecasting implementations on Azure in short time
* **Phase 3** Propose TSPerf as a new track for time-series forecasting in [MLPerf](https://mlperf.org/)


## Framework

### Definitions 
We adopt several definitions from [MLPerf](https://github.com/mlperf/policies/blob/master/rules.adoc) 
and also add a number of new ones:  

A **use case** is a broad family of abstract problems in a specific vertical. Examples include energy 
demand forecasting, retail sales forecasting and medical image classification.

A **benchmark** is a specific abstract problem that can be solved using ML by training a model based on a 
specific dataset.

**Benchmark rules** is a set of rules for implementing a benchmark to produce a class of comparable 
results, such as training and test set partition, performance evaluation metric and process.

A **architecture** consists of a defined set of cloud hardware resources and services such as processors, 
memories, disks, clusters and interconnect. It also includes specific versions of all software such as 
operating system, compilers, libraries, and drivers that significantly influences the running time, 
excluding the software framework for training ML models.

A **software framework** is a specific version of a software library or set of related libraries, possibly with 
associated offline compiler, for training ML models using an architecture. Examples include specific versions 
of Caffe2, MXNet, CNTK, PaddlePaddle, pyTorch, or TensorFlow.

A **benchmark implementation** is an implementation of a benchmark in a particular software framework by a user 
under the benchmark rules.

**Benchmark quality** is performance metric value of a benchmark implementation, measured according to 
benchmark rules.

A **run** is a complete execution of a benchmark implementation on an architecture, training a model from 
initialization to the specified quality target and scoring the model over the specified test set.

A **run result**  is a wall-clock time to execute a complete run and the cost of that execution.

A **benchmark result**  is the median of five run results, where median is taken over the quality of five models.

A **benchmark submission** is a source code of benchmark implementation, declared benchmark result and corresponding run result, along with reproduction instructions. 

A **validated benchmark submission** is a benchmark submission that passed review process.

A **benchmark performance board** is a table of validated benchmark submissions.

A **benchmark reference implementation** is a specific validated benchmark submission chosen from the performance board. 

The following diagram summarizes the relations between different terms and the workflow:

![Definitions](images/definitions.png =900x450)

### Structure and hierarchy of documents

This document serves as the master document and includes the generic concepts and scope of the benchmarks.
In addition to it, there will be a specific benchmark submission guidelines documents that are part of each benchmark. These documents will include detailed benchmark description, as well as benchmark-specific implementation and submission instructions. 

### Structure of repository

We use Git repo to maintain the source code and relevant files. The repository has three levels of folders: use case, benchmark, and benchmark implementation.
The top-level directory `/TSPerf` consists of folders for all the existing use cases, a folder storing common utility scripts, a folder storing internal 
docs, and a Markdown file with an overview of TSPerf framework. 

* Use case folders: Each such folder is named after a specific use case and contains scripts of the implementations/submissions for every benchmark of this 
use case. Currently we have a folder `/TSPerf/energy_load` for the energy load forecasting use case and another folder `/TSPerf/retail_sales` for the 
retail sales forecasting use case. 

  Under each use case folder, we have subfolders for different benchmarks and a Markdown file listing all benchmarks of this use case. For example, 
  `/TSPerf/energy_load/GEFCom2017-D_Prob_MT_hourly` contains all the submissions for a probabilistic forecasting problem defined upon GEFCom2017-D dataset 
  and `/TSPerf/energy_load/GEFCom2014_Pt_1Month_Hourly` includes all the submissions for a point forecasting problem defined upon GEFCom2014 dataset. In 
  addition, `/TSPerf/energy_load/README.md` summarizes all the benchmarks of the energy load forecasting use case. 

  Under each benchmark folder, there are a subfolder containing source code of reference implementation, a subfolder containing source code of all 
  submissions, a subfolder storing common utility scripts, and a Markdown file specifying the benchmark. The description of each item under the benchmark 
  folder is as follows

    * `/reference` folder: This folder contains all the necessary scripts and the submission form for reproducing the reference implementation. For 
    instance, `/TSPerf/energy_load/GEFCom2014_Pt_1Month_Hourly/reference` includes the required submission files of the reference implementation for 
    GEFCom2014_Pt_1Month_Hourly.

    * `/submissions` folder: This folder contains multiple subfolders where each subfolder includes all the necessary scripts and 
    the submission form for reproducing a certain submission. For instance, `/submission1` folder under 
    `/TSPerf/energy_load/GEFCom2014_Pt_1Month_Hourly/submissions` includes the required submission files of submission1.
    
    * `/common` folder: This folder includes utility scripts for a benchmark. As an example, `/TSPerf/energy_load/GEFCom2014_Pt_1Month_Hourly/common` 
    contains the scripts that could be commonly used for GEFCom2014_Pt_1Month_Hourly, such as Python scripts that download the data, prepare training and 
    scoring data, and evaluate performance of the benchmark implementation. 

    * `/README.md`: This Markdown file provides detailed instructions about a certain benchmark. For instance, 
    `/TSPerf/energy_load/GEFCom2014_Pt_1Month_Hourly/README.md` describes benchmark and provides benchmark-specific guidance for evaluating model 
    performance and creating benchmark submission. 

* `/TSPerf/common` folder: This folder has the scripts that could be used across different use cases, such as Python scripts which compute the evaluation 
metrics of the forecasting results.

* `/TSPerf/internal_docs` folder: This folder contains the internal documents that we create during the development of TSPerf. 

* `/TSPerf/README.md` file: This Markdown file describes the TSPerf framework in general. It introduces the goal and vision, specifies the use cases and 
benchmarks, and points to performance boards and more detailed documentation.

## Benchmarks  

The following table summarizes benchmarks that are currently included in TSPerf:

| **Benchmark** | **Dataset** | **Benchmark directory** |  
| --------------------- | ----|---------------- |  
| Probabilistic electricity load forecasting | GEFCom2017 |`TSPerf\energy_load\GEFCom2017-D_Prob_MT_Hourly` |
| Retail sales forecasting | Orange Juice dataset | `TSPerf\retail_sales\OrangeJuice_Pt_3Weeks_Weekly` |

Next sections provide a high-level description of the benchmarks. A more detailed description of the benchmarks is in README files in benchmark directories.

### Probabilistic electricity load forecasting

Probabilistic load forecasting (PLF) has become increasingly important in
power systems planning and operations in recent years. The applications of PLF
include energy production planning, reliability analysis, probabilistic price
forecasting, etc.

The objective of this benchmark is to generate probabilistic forecasting of
electricity load on the GEFCom2017 competition qualifying match data. We use
about 6 years of data for training. Forecast is done at 6 time points, with
horizon of 1 ~ 2 months and granularity of 1 hour. The forecasts should be in
the form of 9 quantiles, i.e. the 10th, 20th, ... 90th percentiles, following
the format of the provided template file.  There are 10 time series (zones) to
forecast, including the 8 ISO New England zones, the Massachusetts (sum of
three zones under Massachusetts), and the total (sum of the first 8 zones).
The quality metric of this benchmark is the Pinball loss function.

### Retail sales forecasting

Sales forecasting is a key objective for the management of retail stores. With the projection of future sales, store managers will be able to optimize 
the inventory based on their business goals. This will generate more profitable order fulfillment and reduce the inventory cost. 

The objective of this benchmark is to forecast orange juice sales of different brands for multiple stores with the Orange Juice dataset from R package 
`bayesm`. The forecast type is point forecasting. Forecast is done at 12 time points, with horizon of 3 weeks and granularity of 1 week. The forecasts should  
include the predicted sales during the target periods for each brand and each store. The output file of the forecast results should follow the format of the 
provided template file. There are in total 913 time series to forecast. The quality metric of this benchmark is the mean average percentage error 
averaged over all time series. 

## Development of benchmark implementation

In this section we provide a number of guidelines for developing reproducible results.

### Feature engineering

To accelerate the development of benchamrk implementations, we provide a number of  common feature engineering functions. Benchmark-independent feature engineering functions are stored in `common/feature_utils.py` file. Benchmark-specific feature engineering functions are stored in `<benchmark vertical>/<benchmark dataset>/common/feature_engineering.py` files. Currently these are `energy_load/GEFCom2017_D_Prob_MT_hourly/common/feature_engineering.py` and `retail_sales/OrangeJuice_Pt_3Weeks_Weekly/common/feature_engineering.py` files. When developing a new benchmark implementation, submitter should reuse provided feature engineering functions as much as possible.

### Guideline for creating Docker images

We recommend that the submitter provides a Docker image for the reproducibility of the submission. Docker is a platform for developers to develop, deploy,
and run applications with containers. Please refer to [this tutorial](https://docs.docker.com/get-started/) to learn basics of Docker. Here we introduce the 
steps of creating and publishing the Docker image.

A new Docker image can be created by modifying the Dockerfile of the baseline implementation of a given benchmark. For instance, there is a Dockerfile in 
`retail_sales/OrangeJuice_Pt_3Weeks_Weekly/baseline/Naive` folder used in the baseline model of retail sales forecasting. In the beginning of this Dockerfile 
we load [ubuntu:16.04](https://hub.docker.com/_/ubuntu/) image as the base image. Then, we install a list of basic Linux packages which are required 
for installing other packages or needed in the model development. Afterwards, we install an R environment with r-base version 3.5.1. Finally, we install R 
dependencies with the following commands
```bash
RUN echo 'options(repos = list(CRAN = "http://mran.revolutionanalytics.com/snapshot/2018-08-27/"))' >> /etc/R/Rprofile.site
ADD ./install_R_dependencies.r /tmp
RUN Rscript install_R_dependencies.r
```
where `install_R_dependencies.r` is an R script that specifies and installs a list of R packages (See the R dependency file in `retail_sales/OrangeJuice_Pt_3Weeks_Weekly/baseline/Naive` folder as an example). You can modify the listed packages based on your 
need. To ensure the same R package version is installed, we use a MRAN snapshot URL to download packages archived on a specific date which can also be 
customized. In case you need to install Python packages, we suggest you first update your `pip` via adding the following command to the Dockerfile
```bash
RUN pip3 install --upgrade pip
```
if you use Python 3. Then, you can mount a Python dependency file into the Docker container and install Python dependencies using the following commands
```bash
WORKDIR /tmp
ADD ./python_dependencies.txt /tmp
RUN pip3 install -r python_dependencies.txt
```
where `python_dependencies.txt` is a file specifying the Python packages and versions (See the Python dependency file in 
`retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/LightGBM` folder as an example). Again you can update the listed packages there if necessary. Note that you will need to use `RUN pip` command if you are working with Python 2.

After customizing the Dockerfile and dependency files, you can build a local Docker image in a Linux VM by following the steps below:

1. Make sure Docker is installed. You can check if Docker is installed in your VM by running
    ```bash
    docker -v
    ```
    You will see the Docker version if Docker is installed. If not, you can install it by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/). Note that if you want to execute Docker commands as a non-root user, you need to create a Unix group and add users to it by following the instructions [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user). Otherwise, you need to run the commands with sudo. 

2. Build Docker image by running
    ```bash
    docker build -t <image name> .
    ```
    from the submission folder where the Dockerfile and dependency files reside. Here `<image name>` is the name of the local Docker image. An example name  is `lightgbm_image:v1`, where `v1` indicates the version of the Docker image. It may take tens of minutes to build the Docker image for the first time. But the process could be much faster if you rebuild the image after applying small changes to the Dockerfile or dependency files, since previous Docker building steps will be cached and most of them will not be repeated.  
    
3. After the Docker image is built, you may need to test your model training and scoring script inside a Docker container created from this image. To do this, you will need to
    * 3.1 Choose a name for a new Docker container and create it by running the following command from `/TSPerf` folder (assuming that you've cloned TSPerf repository):
        ```bash
        docker run -it -v $(pwd):/TSPerf --name <container name> <image name>
        ```
        Note that option `-v $(pwd):/TSPerf` allows you to mount `/TSPerf` folder (the one you cloned) to the container so that you will have access to the source code and data in the container. Here `<container name>` is the name of the Docker container, e.g. `lightgbm_container`. You will automatically enter the Docker container after executing the above command. 
    * 3.2 Inside the Docker container, train the model and make predictions by running the following command from `/TSPerf` folder
        ```bash
        source ./common/train_score_vm <submission path> <script type> 
        ```
        where `train_score_vm` is a bash script that invokes the model training and scoring script; `<submission path>` and `<script type>` are the path of the submission folder (e.g., `./retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/LightGBM`) and type of the script (R, Python, or Python3), respectively. 

4. If the above test goes smoothly, we can push the Docker image to the Azure Container Registry (ACR) with the following steps:
    * 4.1 Log into Azure Container Registry (ACR)
    ```bash
    docker login --username tsperf --password <ACR Access Key> tsperf.azurecr.io
    ``` 
    where `<ACR Acccess Key>` can be found [here](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/ff18d7a8-962a-406c-858f-49acd23d6c01/resourceGroups/tsperf/providers/Microsoft.ContainerRegistry/registries/tsperf/accessKey).
    * 4.2 Create a tag that refers to the Docker image
    ```bash
    docker tag <image name> <tag name>
    ```
    where `<tag name>` is the name of the Docker image in the ACR. We recommend to name the tag using the convention `tsperf.azurecr.io/<benchmark directory>/<image name>`, e.g. `tsperf.azurecr.io/retail_sales/orangejuice_pt_3weeks_weekly/lightgbm_image:v1`.
    * 4.3 Push the Docker image to ACR
    ```bash
    docker push <tag name>
    ```
    with `<tag name>` being the one that you picked in the last step. Make sure that your tag name starts with `tsperf.azurecr.io/`, otherwise the docker push command will not work. You can find the Docker image [here](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/ff18d7a8-962a-406c-858f-49acd23d6c01/resourceGroups/tsperf/providers/Microsoft.ContainerRegistry/registries/tsperf/repository), after it is successfully pushed to ACR.

### Non-determinism restrictions
This section is aligned with [MLPerf](https://mlperf.org/). Some more detailed instructions are added.  
The following forms of non-determinism are acceptable in [MLPerf](https://mlperf.org/).
- Floating point operation order. For example, certain functions in cuDNN do not guarantee reproducibility across runs.

- Random initialization of the weights and/or biases.

- Random traversal of the inputs.  

In order to avoid any other sources of non-determinisms, we recommend setting random seeds whenever a package/software framework provides a function for setting random seed, e.g. `numpy.random.seed()`, `random.seed()` and `tf.set_random_seed()`.  

### Measuring performance

A *benchmark result* is the median of five run results produced using the integer random number generator seeds 1 through 5.  All five run results must also be reported. The following measurements should be included:
  * quality of the model
  * wall-clock running time
  * cloud cost  

The median should be computed over 5 values of the quality of the model.

Submission guidelines for [Standalone VM](#standalone-vm) and [Batch AI](#batch-ai) have detailed instructions for measuing quality of the model and wall-clock running time. In the next section we provide instructions for measuring cloud costs.

#### Measuring cloud cost

The cloud cost is the total cost of obtaining the benchmark result using fixed prices for the general public at the time the result is collected.  The total cost should be computed as the product of wall-clock time and the sum of the costs of all Azure services used by benchmark implementation. 
The total cost can be computed using [Azure pricing calculator](https://azure.microsoft.com/en-us/pricing/calculator/).  When computing the costs, do not use spot pricing. Also, do not include in the cloud costs the
* costs of uploading the data to the cloud  
* costs of developing model and tuning hyperparameters

## Submission of benchmark implementation

### Guideline for submitting reproduction instructions

#### System and software framework availability
This section is aligned with [MLPerf](https://mlperf.org/).  
If you are using a publicly available architecture components or software framework, you must use their publicly available and widely-used used versions.  
If you are using an experimental software framework or architecture components, you must make the architecture components and software framework you use available upon request for replication.

#### Benchmark implementation source code
This section is aligned with [MLPerf](https://mlperf.org/).  
Source code used for the benchmark implementations must be open-sourced under a license that permits a commercial entity to freely use the implementation for benchmarking. The code must be available as long as the results are actively used.

#### Environment setup
1. Parallel/distributed computation environment setup  
If you are using multiple machines for parallel/distributed computation, you must provide a script for automatically creating the cluster (preferred) or instructions for manual creation.
2. Virtual machine or Docker image setup  
You need to provide instructions for setting up the implementation from a plain VM, or a Docker file/image for creating the container needed to execute the implementation.
3. Virtual environment setup  
If your implementation is light-weight and does not have any system dependency, a YAML file for creating a conda environment is also acceptable.
4. Software framework and package version report  
The submitter needs to submit a report summarizing all the software framework and package versions used for producing the reported result. This is to prevent the newer version of a software framework or package significantly changing the implementation result.

#### Reporting benchmark results
This section is aligned with [MLPerf](https://mlperf.org/).  
The submitter needs to run the benchmark implementation five times using the integer random number generator seeds 1 through 5 and report all five benchmark results (quality of the model, 
wall-clock running time and cloud cost).  The variance of the five run results should be reasonable, otherwise, it's an indicator of instability of the implementation. The median of the five 
results is reported as the performance of the submitted implementation. 

#### Hyperparameter tuning
Submitter should justify choice of hyperparameter values if they are not the default ones. Example of justifications are improvement in validation error, reduction in running time or cost. 
Detailed instructions for hyperparameter tuning are optional. However, it's **highly recommended** to provide details of your hyperparameter tuning process, which will make it easier to adopt an implementation to a new dataset.

### Guideline for submitting the code

New benchmark submissions to TSPerf should be made through pull requests in the Git repo by adding a completed submission form and supporting code in a new 
submission folder. 

The submission process depends on the architecture and is described in the following two subsections.

#### Standalone VM

0. Provision a virtual machine and log into the provisioned VM. 

1. Clone the Git repo and create a new git branch
   ```bash
   git clone https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf   
   git checkout -b <branch name>
   ```

2. Create a new submission folder under `<benchmark directory>/submissions` where `<benchmark directory>` is a root benchmark directory, e.g., 
`/LSTM_3layers` under `energy_load/<benchmark name>/submissions`. Please name your submission folder by following a similar naming convention.

3. Download the data using the following commands
   ```bash
   python <benchmark directory>/common/get_data.py
   ``` 

4. Implement a benchmark with the downloaded data to solve an associated problem defined by TSPerf. During the implementation, please use only the data 
specified in the problem description to train your model and evaluate model performance based on the specified testing data. Always check the common utility 
folder under `<benchmark directory>` to see if there is a module which creates the training and testing data for each forecast period, e.g. 
`energy_load/problem1/common/serve_folds.py`. Use this module as much as possible. Forecasts **must not** be generated from models that have been trained on 
data of the forecast period or later. Submission code will be inspected to enforce this.

5. To submit your solution you must create a script (in any language) that includes all code necessary to train your model and produce predictions for all 
forecasted periods in the required format. This script should be named as `train_score.*` with * indicating the file type. It should accept an input argument which is an integer random number generator seed. For example, it should run as `python <submission directory>/train_score.py <seed value>` if it is in Python. This command will generate a CSV file named `submission_seed_<seed value>.csv` which includes the predictions. Please include 
`train_score.*` script and `submission_seed_<seed value>.csv` files in your submission directory. 

6. Report five run results produced using the integer random number generator seeds 1 through 5. This can be done by running the model 
training and scoring script as follows
   ```bash
   time -p python <submission directory>/train_score.py <seed value>
   ```
   where `<seed value>` is an integer between 1 and 5. This command also computes the running time of each run. 

7. Evaluate the performance of each run with evaluate.py and compute the median. Once you have generated your submission files, you can evaluate the quality of the predictions with
   ```bash
   python <benchmark directory>/common/evaluate.py <submission directory>/submission_seed_<seed value>.csv 
   ```
   This command will output a benchmark quality value (e.g. MAPE). Evaluate the quality of each run by changing `<seed value>` above and compute the median of the quality values out of five runs.

8. Include other scripts that are necessary for reproducing the submitted benchmark results. For example, you should include a Dockerfile containing all 
dependencies for running your benchmark submission. The Dockerfile can point to a `.txt` file which contains a list of necessary packages. 

9. Create a Docker image and push it to the ACR   
   To create your Docker image, for example you can go to `/submissions/submission1/` folder and run the following command   
   ```bash
   docker build -t submission1_image .
   ```
   Then, you can push the image to ACR by executing
   ```bash
   docker tag submission1_image tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1
   docker push tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1
   ```
   Note that you will need to log into the ACR before publishing the image.


10. Include a submission form in the submission folder as README.md file. The submission form documents the submitter's information, method utlized in the 
benchmark implementation, information about the scripts, obtained results, and steps of reproducing the results. An example submission form can be found 
here (TODO: Add link). Specifically, it should include
    * name of the branch with submission code
    * benchmark path, e.g. `/TSPerf/energy_load/problem1`
    * path to submission directory, e.g. `/TSPerf/energy_load/problem1/submissions/submission1`
    * instructions for provisioning the architecture components (e.g. DSVM, Batch AI)
    * name of Docker image stored in tsperf registry, for example
      tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1
    * benchmark quality values obtained with random seeds 1 through 5
    * running time in each run
    * cost in each run 

11. Create pull request for review by following the process in the next section.

#### Batch AI

[Batch AI](https://azure.microsoft.com/services/batch-ai/) is an Azure product which enables the training of machine learning models in parallel on a cluster of VMs. In the context of TSPerf, it can be used in two ways:

- Train models and generate forecasts for multiple time series in parallel. For example, in retail sales forecasting benchmarks, it could be used to train/score models for multiple products concurrently.
- For benchmark implementations where an ensemble of forecasts is generated, Batch AI can be used to parallelize the training of these models.

Note that it is **not** permissible to use Batch AI to parallelize the generation forecasts across test folds. In order to be realistic, test predictions must be made sequentially for each test fold.

To make a submission that utilizes Batch AI, complete points 0 to 4 as in the *Standalone VM* section, and then complete the following steps:

1. Create your model training and scoring script(s) to be run on the Batch AI cluster. These script(s) can be in any language and must include all code necessary to train your models and maeke predictions on the test periods. You may have multiple scripts (one for each model in an ensemble for example) to be executed on separate nodes of the cluster. Alternatively, you may create a single script which can run differently on separate nodes based on the value of a script parameter. Each execution of these scripts will be a single Batch AI job to be run on a single node. If the output of the model(s) is non-deterministic (if it varies based on weight initialization for example), the script must accept the seed value as a parameter.

2. Create a job execution script named `execute_jobs.*`. This script performs several functions:

    - Create the job.json files to define each Batch AI job. 
    - Create the Batch AI experiment
    - Trigger each job
    - Download the job scripts' output files from blob storage
    - Combine results into a `submission_seed_<seed value>.csv` which includes the test fold predictions in the required format.

    This script can be written in any language but Python is recommended due to the availability of [utilities](https://github.com/Azure/BatchAI/tree/master/utilities) for Batch AI in this language.

3. Report five run results produced using the integer random number generator seeds 1 through 5. If your script is written in Python, this can be done by running the following
    ```bash
    time -p python <submission directory>/execute_jobs.py <seed value>
    ```
    where `<seed value>` is an integer between 1 and 5.

Complete the submission by following steps 7-11 in the *Standalone VM* section. For step 9, you will need a docker image for running the `execute_jobs.*` script and a docker image for running the training/scoring scripts on each cluster node. For simplicity, you may choose to use the same docker image for both sets of scripts.

### Pull request process

This section describes the Pull Request Process that we (the framework team) would like to use for the TSPerf development process as well as for the benchmark implementation development. The process applies for both document and code development.

#### Terms Definition

* Approver  - A person who is designated to approve PRs
* Developer - A person who develops code or documents
* Pull Request - A request to merge code or documents into the master branch
* Reviewer - A person who is assigned to review a PR by the developer

#### Pull Request PR Process

1. Code/docs development is performed by the developers on a separated branch outside of the master branch
2. Upon completion of a given task the developer will issue a PR that includes the following elements:
   * Code/Doc to be reviewed
   * List of reviewers (at least one reviewer)
   * Designated approver
3. Each of the listed reviewers should review and provide comments for the PR
4. Comments could be of 2 types:
   * General notes that don't require change or update of the submitted code/doc
   * Comment with a request to change the code/doc
5. The designated approver should:
   * Collect all comments and verify implementation
   * Review the entire code/doc for validity
   * Approve the PR (after all comments are processed and completed)
6. After the PR approval, the developer should merge the relevant code/doc into the master branch and resolve conflicts (if exist)

#### Resource Planing Implications

Since reviewing and approving PRs could be time consuming, it is important to plan and allocate resources in advance for that. Therefore, the following guidelines should be considered:

* At the sprint planing, all expected PRs should be discussed based on inputs from all developers
* At the sprint planning, the designated approver should be chosen
* The designated approver should estimate the required effort for reviewing all PRs and allocate the required time for the next sprint accordingly
* A 2nd approver should be assigned in case of possible conflicts or time constraints
* Developers should notify the reviewers in advance at the sprint planning 
* Other developers who take dependency on the PR's code should be included as reviewers
* Reviewers should allocate time for the next sprint for reviewing 

## Review of submissions
The goal of the review is to validate the declared

* quality of the model 
* wall-clock running time
* cloud cost  

We will explain below how to validate these quantities. Additionally the reviewer should check that the 
evaluation of the quality of the model is done using a standard `evaluate.py` script and that the code

* does not use test data for training
* has a good quality 
* is well documented

We do not have specific guidelines for checking these items. Reviewer should use his/her own judgement 
to decide if there is no test data leakage and if the code or documentation need improvement. In particular, reviewer should verify that hyperparameters were chosen based on the training and validation set (see examples [here](#hyperparameter-tuning)) and were not chosen to optimize performance over the test set.

Reviewer should set up execution environment before running benchmark implementation. Initially the 
reviewer needs to complete the following three steps:

0. Verify that the submission has README.md file with  
    * name of the branch with submission code
    * benchmark path, for example /TSPerf/energy_load/problem1
    * path to submission directory, for example  /TSPerf/energy_load/problem1/benchmarks/submission1
    * instructions for provisioning the architecture components (e.g. DSVM, Batch AI)
    * name of Docker image stored in tsperf registry, for example 
    tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1

In the following sections all occurences of "README file" refer to README file in the submission, unless 
specified otherwise.

The next steps depend on the architecture and are described in the following two subsections.

### Standalone VM

1. Follow the instructions in the README file and provision the virtual machine that was used to generate 
benchmark results. Then log into the provisioned VM.

2. Choose submission branch and clone the Github repo to your machine:

        git clone https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf
        git checkout <branch name>

3. Download the data using the following commands

        python <benchmark path>/common/get_data.py

    where \<benchmark path\> is a root benchmark directory, for example energy_load/problem1

4. Log into Azure Container Registry:
   
       docker login --username tsperf --password <ACR Access Key> tsperf.azurecr.io
   
   If want to execute docker commands without sudo as a non-root user, you need to create a Unix group and 
   add users to it by following the instructions 
   [here](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

5. Pull a Docker image from ACR, using image name that is specified in README file:   
      
       docker pull <image name>

6. Choose a name for a new Docker container and create it using command:   
   
       docker run -it -v ~/TSPerf:/TSPerf --name <container name> <image name>
   
   Note that you need to mount `/TSPerf` folder (the one you cloned) to the container so that you will 
   have access to the source code in the container. 

7. Inside Docker container, run the following command:  

       source <benchmark directory>/common/train_score_vm <submission directory> 

   This will generate 5 `submission_seed_<seed number>.csv` files in the submission directory, where \<seed number\> 
   is between 1 and 5. This command will also output 5 running times of train_score.py. The median of the times 
   reported in rows starting with 'real' should be compared against the wallclock time declared in benchmark 
   submission.
   
8. Evaluate the benchmark quality by running

       source <benchmark directory>/common/evaluate <submission directory> <benchmark directory>

    This command will output 5 benchmark quality values (e.g. MAPEs). Their median should be compared against the 
    benchmark quality declared in benchmark submission.

### Batch AI

1. Provision Linux Data Science VM with DS4v2 configuration. 

2. Choose submission branch and clone the Github repo to your machine:

        git clone https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_git/TSPerf
        git checkout <branch name>

3. Download the data using the following commands

        python <benchmark path>/common/get_data.py

    where \<benchmark path\> is a root benchmark directory, for example energy_load/problem1

4. Follow the instructions in the README file and provision the resource group, Batch AI workspace, Batch AI cluster 
and storage account with file share and blob storage. When creating resources, please record *resource group name*, 
*storage account name*, *storage account key* and *file share name*. You will need to provide these parameters in the 
later steps.

5. Upload \<submission directory\>/train_score.py script to file share account by following the instructions in README 
file.

6. Upload the dataset to blob storage by following the instructions in README file.

7. Create Batch AI experiment:

       az batchai experiment create -g <resource group name> -w <workspace name > -n <submission name>

   where resource group name and workspace name are the ones used in step 4. Submission name is rightmost directory in 
   submission path. 

8. Run 5 Batch AI jobs

        source  train_score_batchai <resource group name> <workspace name> <cluster name> <submission directory> 
        <storage account name>

    This command will create 5 `submission_seed_<seed number>.csv` files in the local directory, where \<seed number\> is 
    between 1 and 5. This command will also output 5 running times of Batch AI jobs. The median of these times should 
    be compared against the wallclock time declared in benchmark submission. 

9. Evaluate the benchmark quality by running

       source <benchmark directory>/common/evaluate <submission directory> <benchmark directory>

    This command will output 5 benchmark quality values (e.g. MAPEs). Their median should be compared against the 
    benchmark quality declared in the benchmark submission.

## Performance board
Each benchmark will have a separate performance board. All performance boards will have the following columns:
* submission name
* URL of submission folder in VSTS
* benchmark quality (e.g. MAPE)
* training and scoring time
* training and scoring cost
* architecture (e.g. DSVM)
* framework (e.g. Tensorflow)
* algorithm (e.g. LSTM)  

Each row will be a validated benchmark submission. Performance board will be updated by reviewer, after validating a benchmark 
submission. Every validated benchmark submission will be shown in the performance board. Since benchmark submission are measured 
by three parameters (quality, running time and cost), there will be no ranking between performance board entries.

## Selection of reference implementation
 
We will select from a benchmark performance board a reference implementation that optimizes quality, running time and cost tradeoffs. The selection will be done manually, after examining all performance board entries. We target to have two reference implementations, one for each benchmark.
