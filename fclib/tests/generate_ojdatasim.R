#!/usr/bin/Rscript

ojdata <- local({
    # simulated data
    set.seed(12345)
    df <- expand.grid(store=1:2, brand=1:3, week=50:60)
    n <- nrow(df)
    df$logmove <- rnorm(n, 9, 1)
    df$constant <- 1
    df$price1 <- rnorm(n, 0.55, 0.003)
    df$price2 <- rnorm(n, 0.55, 0.003)
    df$price3 <- rnorm(n, 0.55, 0.003)
    df$price4 <- rnorm(n, 0.55, 0.003)
    df$price5 <- rnorm(n, 0.55, 0.003)
    df$price6 <- rnorm(n, 0.55, 0.003)
    df$price7 <- rnorm(n, 0.55, 0.003)
    df$price8 <- rnorm(n, 0.55, 0.003)
    df$price9 <- rnorm(n, 0.55, 0.003)
    df$price10 <- rnorm(n, 0.55, 0.003)
    df$price11 <- rnorm(n, 0.55, 0.003)
    df$deal <- rbinom(n, 1, 0.5)
    df$feat <- rbinom(n, 1, 0.25)
    df$profit <- rnorm(n, 30, 7.5)
    df
})
write.csv(ojdata, "fclib/tests/resources/ojdatasimR.csv", row.names=FALSE)
