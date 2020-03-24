# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#' Creates a local background cluster for parallel computations
#'
#' @param ncores The number of nodes (cores) for the cluster. The default is 2 less than the number of physical cores.
#' @param libs The packages to load on each node, as a character vector.
#' @param useXDR For most platforms, this can be left at its default `FALSE` value.
#' @return
#' A cluster object. 
make_cluster <- function(ncores=NULL, libs=character(0), useXDR=FALSE)
{
    if(is.null(ncores))
        ncores <- max(2, parallel::detectCores(logical=FALSE) - 2)
    cl <- parallel::makeCluster(ncores, type="PSOCK", useXDR=useXDR)
    res <- try(parallel::clusterCall(
        cl,
        function(libs)
        {
            for(lib in libs) library(lib, character.only=TRUE)
        },
        libs
    ), silent=TRUE)
    if(inherits(res, "try-error"))
        parallel::stopCluster(cl)
    else cl
}


#' Deletes a local background cluster
#'
#' @param cl The cluster object, as returned from `make_cluster`.
destroy_cluster <- function(cl)
{
    try(parallel::stopCluster(cl), silent=TRUE)
}
