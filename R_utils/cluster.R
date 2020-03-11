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

destroy_cluster <- function(cl)
{
    try(parallel::stopCluster(cl), silent=TRUE)
}
