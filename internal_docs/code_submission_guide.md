## Guideline for Submitting Code and Results

New benchmark submissions to TSPerf should be made through pull requests in the Git repo by adding a completed submission form and supporting code in a new 
submission folder. 

The submission process depends on the system and is described in the following two subsections.

### Standalone VM

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
    * instructions for provisioning the system (e.g. DSVM, Batch AI)
    * name of Docker image stored in tsperf registry, for example
      tsperf.azurecr.io/energy_load/problem1/submission1/submission1_image:v1
    * benchmark quality values obtained with random seeds 1 through 5
    * running time in each run
    * cost in each run 

11. Create pull request for review

### Batch AI

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