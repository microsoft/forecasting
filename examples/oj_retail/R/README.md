The Rmarkdown notebooks in this directory are as follows. Each notebook also has a corresponding HTML file, which is the rendered output from running the code.

- [`01_dataprep.Rmd`](01_dataprep.Rmd) creates the training and test datasets
- [`02_basic_models.Rmd`](02_basic_models.Rmd) fits a range of simple time series models to the data, including ARIMA.
- [`02a_ets_models.Rmd`](02a_ets_models.Rmd) fits a basic ETS model to the data, using the output from the previous ARIMA model to impute missing values.
- [`02b_reg_models.Rmd`](02b_reg_models.Rmd) adds independent variables as regressors to the ARIMA model.
- [`02c_prophet_models.Rmd`](02c_prophet_models.Rmd) fits some simple models using the Prophet algorithm.

If you want to run the code in the notebooks interactively, you must start from `01_dataprep.Rmd` and proceed in sequence, as the earlier notebooks will generate artifacts (datasets/model objects) that are used by later ones.

The following packages are required to run the notebooks in this directory:

- bayesm (the source of the data)
- dplyr
- tidyr
- ggplot2
- prophet
- tsibble
- urca
- fable
- feasts
- yaml
- here

The easiest way to install them is to run

```r
install.packages("bayesm")
install.packages("here")
install.packages("tidyverse") # installs all tidyverse packages
install.packages(c("fable", "feasts", "urca"))
```

