# Implementation submission form

Fill out this form before making a submission. Save it as README.md in the submission directory.

## Submission details

**Benchmark name:** <*e.g. GEFCom2017-D_Prob_MT_hourly*>

**Submitter(s):** <*Name of the person(s) submitting the implementation*>

**Submitter(s) email:**

**Submission name** <*An abbreviated name of the method or model used up to a maximum of 24 characters e.g. LSTM_3layers*>

**Submission date:** <*MM/DD/YYYY*>

**Submission branch:** <*Name of the git branch with the submission e.g. LSTM_3layers*>

**Submission PR:** <*Name of the pull request of the submission e.g. LSTM_3layers*>

**Submission path:** <*e.g. /energy_load/GEFCom2017-D_Prob_MT_hourly/submissions/submission1*>

## Implementation description

### Modelling approach

<*Give a brief description of the key elements of the implementation including the modelling approach and the algorithms used*>

**Implentation diagram** <*Optional. Include if the description of the implementation would be clearer with a diagram*>

### Feature engineering

<*Describe any feature engineering used to transform the data for your model*>

### Hyperparameter tuning

<*Describe efforts to tune your model, the hyperparameter tuning approach taken and the set of key hyperparameters of your final model>

### Description of implementation scripts

<*Briefly describe the scripts in this submission and how they are related*>

### Steps to reproduce

<*Outline the detailed steps of how to reproduce the results of this benchmark implementation*>

## Implementation resources

**Platform** <*e.g. Azure, AWS, Google Cloud, or On-Prem*>

**Hardware** <*e.g. Standard NC6 (6 vcpus, 56 GB memory), 2xK80 GPUs, SSD* or *Batch AI cluster of 6x Standard NC6*>

**Data storage:** <*e.g. Premium SSD, blob storage*>

**Docker image:** <Name and location of the implementation docker image e.g. tsperf.azurecr.io/common/image:v1>

**Key packages/dependencies:**
List the key packages, deep learning frameworks included in the docker image. Only include the packages that are important for implementing the model. Exclude packages used for data manipulation, feature engineering etc. E.g.:
    
    - python==3.5
    - keras==2.2.1
    - tensorflow==1.0

## Resource deployment instructions

<*Provide detailed instructions on how to deploy the cloud resources to run your implementation. Ideally, this will include a script, using for example the Azure CLI, for automated deployment.*>


## Implementation evaluation

**Quality:** <*Call your train_score.\* script using one of the train_score bash scripts included in /TSPerf/common. Provide the performance value for each of the five runs and also report the median value. E.g.:*

*sMAPE run 1: 15.6%*

*sMAPE run 2: 15.3%*

*sMAPE run 3: 15.1%*

*sMAPE run 4: 15.2%*

*sMAPE run 5: 15.6%*

*median sMAPE: 15.3%*>

**Time:** <*Call your train_score.\* script using one of the train_score bash scripts included in /TSPerf/common. Provide the total running time for each of the five runs and also report the median value. E.g.:*

*run time 1: 46 minutes*

*run time 2: 43 minutes*

*run time 3: 41 minutes*

*run time 4: 42 minutes*

*run time 5: 46 minutes*

*median run time: 43 minutes*>

**Cost:** <*Provide an estimate of the total cost to run your train_score.\* script using cloud-based resources. Refer to the [performance measurement guide](/internal_docs/performance_measurement_guide.md) for how to measure the cloud cost*>