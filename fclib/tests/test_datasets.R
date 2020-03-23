#!/usr/bin/Rscript 
#
# Test download retail data
#
# Note that we define a function to test download_data.r file in retail benchmarking,
# based on output checking.

## Define a function to test download_data.r file in retail benchmarking.
test_download_retail_data <- function() {
  # Test download_data.r file in retail benchmarking
  #
  # Args:
  #   NULL.
  #
  # Returns:
  #   NULL. 
  BENCHMARK_DIR <- file.path('./retail_sales', 'OrangeJuice_Pt_3Weeks_Weekly')
  DATA_DIR <- file.path(BENCHMARK_DIR, 'data')
  SCRIPT_PATH <- file.path(BENCHMARK_DIR, 'common', 'download_data.r')
  # Call data download script
  source(SCRIPT_PATH)
  # Check downloaded data
  sales <- read.csv(file.path(DATA_DIR, 'yx.csv'))
  if(all(dim(sales) == c(106139, 19)) == FALSE) {
    stop("There is something wrong")
  }
  column_names <- c('store', 'brand', 'week', 'logmove', 'constant', 
                    'price1', 'price2', 'price3', 'price4', 'price5', 
                    'price6', 'price7', 'price8', 'price9', 'price10', 
                    'price11', 'deal', 'feat', 'profit')
  if(all(colnames(sales) == column_names) == FALSE) {
    stop("There is something wrong")
  }
  storedemo <- read.csv(file.path(DATA_DIR, 'storedemo.csv'))
  if(all(dim(storedemo) == c(83, 12)) == FALSE) {
    stop("There is something wrong")
  }
  column_names <- c('STORE', 'AGE60', 'EDUC', 'ETHNIC', 'INCOME', 
                    'HHLARGE', 'WORKWOM', 'HVAL150', 'SSTRDIST', 
                    'SSTRVOL', 'CPDIST5', 'CPWVOL5')
  if(all(colnames(storedemo) == column_names) == FALSE) { 
    stop("There is something wrong")
  } 
}

## Test download_data.r file in retail benchmarking.
test_download_retail_data()