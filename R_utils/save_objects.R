# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#' Loads serialised objects relating to a given forecasting example into the current workspace
#'
#' @param example The particular forecasting example.
#' @param file The name of the file (with extension).
#' @return
#' This function is run for its side effect, namely loading the given file into the global environment.
load_objects <- function(example, file)
{
    examp_dir <- here::here("examples", example, "R")
    load(file.path(examp_dir, file), envir=globalenv())
}

#' Saves R objects for a forecasting example to a file
#'
#' @param ... Objects to save, as unquoted names.
#' @param example The particular forecasting example.
#' @param file The name of the file (with extension).
save_objects <- function(..., example, file)
{
    examp_dir <- here::here("examples", example, "R")
    save(..., file=file.path(examp_dir, file))
}
