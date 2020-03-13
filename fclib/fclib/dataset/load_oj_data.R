# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script retrieves the orangeJuice dataset from the bayesm R package and saves the data as csv

args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args)<2) {
  stop("At least two arguments must be supplied (.rda file path and data directory).", call.=FALSE)
} else if (length(args)==2) {
  RDA_PATH <- args[1]
  DATA_DIR <- args[2]
}

# Load the data from bayesm library
load(RDA_PATH)
yx <- orangeJuice[[1]]
storedemo <- orangeJuice[[2]]

# Create a data directory
fpath <- file.path(DATA_DIR)
if(!dir.exists(fpath)) dir.create(fpath)

# Write the data to csv files
write.csv(yx, file = file.path(fpath, "yx.csv"), quote = FALSE, na = " ", row.names = FALSE)
write.csv(storedemo, file = file.path(fpath, "storedemo.csv"), quote = FALSE, na = " ", row.names = FALSE)

print(paste("Data download completed. Data saved to ", DATA_DIR))
