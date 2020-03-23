#!/usr/bin/Rscript 
#
# Source entire R testing files
#
# Note that we first define a function to source entire folder including R files. Then, 
# we simply source all .R files within the specified folder.

## Define a function to source entire folder
source_entire_folder <- function(folderName, verbose=FALSE, showWarnings=TRUE) { 
  # Find all .R files within a folder and soruces them
  #
  # Args:
  #   folderName: Name of the folder including R files to be sourced.
  #   verbose: If TRUE, print message; if not, not. Default is FALSE.
  #
  # Returns:
  #   NULL.
  files <- list.files(folderName, full.names=FALSE)
  # Grab only R files that start with the word 'test'
  files <- files[grepl("\\.[rR]$", files)]
  files <- files[grepl("^test", files)]
  if (!length(files) && showWarnings)
    warning("No R test files in ", folderName)
  for (f in files) {
    if (verbose)
      cat("sourcing: ", f, "\n")
    ## TODO:  add caught whether error or not and return that
    try(source(paste(folderName, f, sep='/'), local=FALSE, echo=FALSE), silent=!verbose)
  }
  return(invisible(NULL))
}

## Source all .R files within the folder of tests/unit
source_entire_folder('./tests/unit', verbose=TRUE)