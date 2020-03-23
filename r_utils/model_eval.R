# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#' Computes forecast values on a dataset
#'
#' @param mable A mable (model table) as returned by `fabletools::model`.
#' @param newdata The dataset for which to compute forecasts.
#' @param ... Further arguments to `fabletools::forecast`.
#' @return
#' A tsibble, with one column per model type in `mable`, and one column named `.response` containing the response variable from `newdata`.
get_forecasts <- function(mable, newdata, ...)
{
    fcast <- forecast(mable, new_data=newdata, ...)
    keyvars <- key_vars(fcast)
    keyvars <- keyvars[-length(keyvars)]
    indexvar <- index_var(fcast)
    fcastvar <- as.character(attr(fcast, "response")[[1]])
    fcast <- fcast %>%
        as_tibble() %>%
        pivot_wider(
            id_cols=all_of(c(keyvars, indexvar)),
            names_from=.model,
            values_from=all_of(fcastvar))
    select(newdata, !!keyvars, !!indexvar, !!fcastvar) %>%
        rename(.response=!!fcastvar) %>%
        inner_join(fcast)
}


#' Evaluate quality of forecasts given a criterion
#'
#' @param fcast_df A tsibble as returned from `get_forecasts`.
#' @param gof A goodness-of-fit function. The default is to use `fabletools::MAPE`, which computes the mean absolute percentage error.
#' @return
#' A single-row data frame with the computed goodness-of-fit statistic for each model.
eval_forecasts <- function(fcast_df, gof=fabletools::MAPE)
{
    if(!is.function(gof))
        gof <- get(gof, mode="function")
    resp <- fcast_df$.response
    keyvars <- key_vars(fcast_df)
    indexvar <- index_var(fcast_df)
    fcast_df %>%
        as_tibble() %>%
        select(-all_of(c(keyvars, indexvar, ".response"))) %>%
        summarise_all(
            function(x, .actual) gof(x - .actual, .actual=.actual),
            .actual=resp
        )
}
