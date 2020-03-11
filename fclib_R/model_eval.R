# returns tibble of response and predicted values
get_forecasts <- function(mable, newdata=NULL, h=NULL, ...)
{
    fcast <- forecast(mable, new_data=newdata, h=h, ...)
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
