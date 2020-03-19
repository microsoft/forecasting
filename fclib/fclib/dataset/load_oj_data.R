# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script retrieves the orangeJuice dataset from the bayesm R package and saves the data as csv.
#
# Two arguments must be supplied to this script:
#
# RDA_PATH - path to the local .rda file containing the data
# DATA_DIR - destination directory for saving processed .csv files

args = commandArgs(trailingOnly=TRUE)

# Test if there are at least two arguments: if not, return an error
if (length(args)==2) {
  RDA_PATH <- args[1]
  DATA_DIR <- args[2]
} else {
   stop("Two arguments must be supplied - path to .rda file and destination data directory).", call.=FALSE)
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
