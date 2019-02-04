# Forecasting with ES-RNNs
The Exponential-Smoothing Recurrent Neural Network (ES-RNN), developed by Uber Engineer and former Microsoft Engineer Slawek Smyl, is quite powerful and carries a lot of potential for forecasting. 

This Readme will cover three tasks that the ES-RNN can handle presently with regards to TSPerf. The first of these tasks is to run the ES-RNN with the M4 competition dataset that it was initially built for. The second task involves extending the ES-RNN to a subset of the Orange Juice Sales dataset from the *bayesm* R package. The third task involves expanding the ES-RNN to train on a greater portion of the same Orange Juice Sales dataset (more features). 

## 1. ES-RNN for M4
The repository for this task can be found in `./M4`. The structure of this repository is identical to the structure of Slawek's M4 competition submission (<https://github.com/M4Competition/M4-methods/tree/master/118%20-%20slaweks17>). The link only has a zip file, which you must unzip in order to access the contents. 

Once unzipped, follow the instructions in `/ES_RNN/github/c++/readme.txt` to install dependencies (mainly Dynet). Once Dynet is installed and built, we can run the ES_RNN model. 

Before we run it, however, we must change file paths in ES_RNN.cc (and other ES_RNN files in the repo that you wish to use). The lines where we initialize the directory paths that we wish to use are lines 72 through 75. The data directory, which is originally located in the root of the M4 repo, has been conveniently relocated to `./M4/data`. `./M4/data/Train` is the path we want to use for `DATA_DIR` (line 72). Create the directory `./M4/data/output` if it is not already present, and set OUTPUT_DIR (line 74) to point to this directory. 

**Remember that file paths in Linux use single forward slashes ('/') while file paths in Windows use double backslashes ('\\'). Also note that in addition to M4-info.csv in `./M4/data/Train`, the file that provides the training data corresponds to the value of the constant `VARIABLE` in line 81 of `ES.RNN_cc`. 

We should now be ready to run the script. We can simply use Slawek's build_mkl file (also in the c++ folder), which compiles `ES_RNN.cc` and prepares an executable named `ES_RNN` that you can use. Note that you will have to open the file and modify the terminal command so that the file paths used point to your installation of dynet. Once done, simply initiate the following command on your terminal while in the c++ folder: 
```
./build_mkl ES_RNN
```

There will be several warnings printed, but as long as there are no errors, we are safe to proceed. We have two options for how we can run the `ES_RNN` executable. The first is simply to run the executable directly. We do this with the following command: 
```
./ES_RNN <seed_for_chunks> <chunk_number>
```
Note that `seed_for_chunks` and `chunk_number` are both integers. We must pass in a minimum of two integer arguments. We can pass a third integer argument for ibigOffset, making our command look like: 
```
./ES_RNN <seed_for_chunks> <chunk_number> <ibigOffset>
```
This parameter indicates where to continue a run that ended prematurely. Note that we can also simply execute the command without any arguments (the defaults are 10, 1, 0, respectively). 

The second method to run the ES_RNN executable is to run multiple processes at once. We can use Slawek's `run18` file (which is also located in the `c++` folder). This file triggers 18 different ES_RNN calls in parallel (don't worry, each process creates a different output repo). Note that these processes must come in pairs. The first call in each pair gets a `chunk_number` of 1 while the second call in each pair gets a `chunk_number` of 2. 

Notes about the `ES_RNN.cc` file: 
- parameters of interest can be found from lines 72 through 106 and then lines 187 through 218. 
- Command line arguments are parsed at the start of the main method (lines 380 through 399). 
- The sales-info dataset is processed at lines 553 through 564.
- The actual training data is read at lines 566 through 586. 
- The code for the actual model can be found starting at line 617
- The trainer object is declared at line 611, but the actual training process starts at line 670
- The test code starts at line 914. 
- The code for saving the forecasts is located at 1115 to 1125.
- The `NUMBER_OF_TRAINING_EPOCHS` should be greater than or equal to the `AVERAGING_LEVEL`, or the code will run into a segmentation fault.

Once the model is completed, the outputted forecasts can be found in `./M4/data/output`. To evaluate and sanity check the results, an iPython notebook, `./M4/data/scripts/ES_RNN M4 Sanity Check.ipynb`. 

## 2. ES-RNN for Limited Sales Dataset
The second phase of the project involves using the ES-RNN on the orange juice sales dataset from the *bayesm* R package. This is the first attempt at using the ES-RNN beyond its initial purpose, the M4 competition. As a result, we approach this problem by applying minimal changes to the model's code itself and by keeping the dataset as similar as possible to the M4 dataset. The original dataset, which can be found in the `./sales_limited/data` directory, has several features, including store, week, brand, (the log of) the number of juice cartons sold, prices for each brand, advertisement information, and so forth. 

We transform the dataset so that it only uses information about number of orange juice cartons of a given brand sold at a given store at a given week. We accomplish this using either the iPython notebook, `preprocessing.ipynb` or the python script `preprocessing.py`. The former can transform one file at a time (be sure to change the appropriate file path in the notebook to point to the file you wish to transform). The latter can transform multiple files in a single process. Just modify `./sales-limited/data/Train/training-files.txt` to have the number of files you wish to transform in the first line, followed by the file paths of the files you wish to transform (one per line). Then simply run the following command from `./sales-limited/data/scripts`.  
```
python preprocessing.py <path to training-files.txt>
```
Like the M4 dataset, we have two main dataframes. The first shows time-series codes and store and brand information while the second shows time-series codes to the actual number of juice cartons sold per week (the time series). You can see examples of the tranformed dataframes in the iPython notebooks.

The transformed data should have been created in `./sales-limited/data/Train`. Open the flie `./sales-limited/github/c++/ES_RNN.cc`, and make the following modifications: 
- modify the DATA_DIR variable to point to `./sales-limited/data/Train`. 
- modify `OUTPUT_DIR` to point to `./sales-limited/data/output`. 
- modify `INPUT_PATH` to point to the correct training file. 
- modiy `INFO_INPUT_PATH` to point to `DATA_DIR + "sales-info.csv"`. 

We now run the model the same way we did for the M4 competition. We first use build_mkl to generate the `ES_RNN` executable and then either run it manually or use the `run2` script to execute multiple commands in parallel. Note that information about the time taken by the processes is outputted in `./sales-limited/data/output/time`. 

The data outputted looks similar to the output of the M4. competition, which is not the format we wish to have it in. We can reformat it either using postprocessing.ipynb or by using postprocessing.py with the following command from `./sales-limited/data/scripts`: 
```
python postprocessing.py <path_to_file>
```
If we ran the model on the training rounds in TSPerf, then we can run evaluate.py as is in the TSPerf package to explore the MAPE for each round. 

IF THE EXISTING `ES_RNN.cc` FILE DOESN'T WORK, THEN PLEASE PULL MY GUARANTEED WORKING COPY FROM MY VM: 
```
scp -r deegup@workhorse.westus2.cloudapp.azure.com:~/M4/ES_RNN/github/c++/backups/ES_RNN_salesnaive_working.cc
```

## 3. Using External Features from the Sales Dataset
There are several features that have gone unused in our usage of the ES_RNN model to forecast on our transformed sales dataset. To accommodate these features, we transform our dataset by first filtering out the columns that we wish to use and then by adding a new column to the left of the dataframe for time-series IDs. Just like in the limited version of the problem (part 2 in this readme), we will also have an "info" dataframe where each time-series ID maps to store and brand information. This transformation is handled by the notebook `./sales_extfeatures/data/scripts/extfeatures_transformer.ipynb`. 

The process of accommodating a the other features has the following steps (based on my vision of it): 
1. Modify the data parsing code to read the data and store all the features. 
2. The time-series data contains weekly information. Instead of having a single entry for the number of orange juice containers sold in a given week, we will have a vector for each feature we wish to include during training. For all weeks with the same time-series ID, we will concatenate the vectors together, in chronological order. 
3. We must update the input size and the output size (which should be the forecasting horizon times the length of each vector for each week for a given time series). 

An alternative, cleaner (but more involved) way we might be able to proceed this is to write a new struct for the new type of time-series described above. The default code simply uses a struct called M4TS. We can model the code for the new struct off of the code for the M4TS object, but it may be tough figuring out how to store as much data as is present in the retail sales dataset. 

I was in the process of modifying the `ES_RNN.cc` file to include the code for my approach for handling external features. If you ssh into `deegup@workhorse.westus2.cloudapp.azure.com`, my in-progress file is located at `~/M4/ES_RNN/github/c++/ES_RNN.cc`. 