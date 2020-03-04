## Orange juice dataset

### Package installation

You'll need the following packages to run the notebooks in this directory:

- bayesm (the source of the data)
- ggplot2
- dplyr
- tidyr
- jsonlite
- tsibble
- urca
- fable
- fabletools
- feasts

The easiest way to install them is to run

```r
install.packages("bayesm")
install.packages("tidyverse") # installs all tidyverse packages
install.packages(c("fable", "feasts", "urca"))
```

The Rmarkdown notebooks in this directory are as follows. You should run them in sequence, as each will create output objects (datasets/models) that are used in later notebooks.

- [`01_dataprep.Rmd`](01_dataprep.Rmd) creates the training and test datasets
- [`02_simplemodels.Rmd`](02_simplemodels.Rmd) fits a range of simple time series models to the data, including ARIMA and ETS models.
- [`02a_simplereg_models.Rmd`](02a_simplereg_models.Rmd) adds independent variables as regressors to the ARIMA model.
- [`03_model_eval.Rmd`](03_model_eval.Rmd) evaluates the goodness of fit of the models on the test data.

