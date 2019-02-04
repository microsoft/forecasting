When the c++ workers run, they output results (forecasts) to a directory or two. 
(Sorry occasionally two directories are filled, in such case first "manually" put all the output files to a single dir)
These scripts merge them into one file and save it, show a sample of graphs, and if this is backtesting run (LBACK>0), calculate some accuracy metrics.

Both scripts needs to be updated with your input, output dirs, and other params, see inside, there are a lot of comments there.

merge.R is meant to be used for point forecst runs, so for ES_RNN and ES_RNN_E programs.
mergePI.R - for Prediction Interval runs, so for ES_RNN_PI and ES_RNN_E_PI programs.
