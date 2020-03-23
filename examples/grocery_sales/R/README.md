# Forecasting examples in R: orange juice retail sales

The Rmarkdown notebooks in this directory are as follows. Each notebook also has a corresponding HTML file, which is the rendered output from running the code.

- [`01_dataprep.Rmd`](01_dataprep.Rmd) creates the training and test datasets
- [`02_basic_models.Rmd`](02_basic_models.Rmd) fits a range of simple time series models to the data, including ARIMA and ETS.
- [`02a_reg_models.Rmd`](02a_reg_models.Rmd) adds independent variables as regressors to the ARIMA model.
- [`02b_prophet_models.Rmd`](02b_prophet_models.Rmd) fits some simple models using the Prophet algorithm.

If you want to run the code in the notebooks interactively, you must start from `01_dataprep.Rmd` and proceed in sequence, as the earlier notebooks will generate artifacts (datasets/model objects) that are used by later ones.

## Package installation

The following packages are needed to run the basic analysis notebooks in this directory:

- rmarkdown
- dplyr
- tidyr
- ggplot2
- tsibble
- fable
- feasts
- yaml
- here

It's likely that you will already have many of these (particularly the [Tidyverse](https://tidyverse.org) packages) installed, if you use R for data science tasks. The main exceptions are the packages in the [Tidyverts](https://tidyverts.org) family, which is a modern framework for time series analysis building on the Tidyverse.

```r
install.packages("tidyverse") # installs all tidyverse packages
install.packages("rmarkdown")
install.packages("here")
install.packages(c("tsibble", "fable", "feasts"))
```

The following packages are needed to run the Prophet analysis notebook:

- prophet
- fable.prophet

While prophet is available from CRAN, its frontend for the tidyverts framework, fable.prophet, is currently on GitHub only. You can install these packages with

```r
install.packages("prophet")
install.packages("https://github.com/mitchelloharawild/fable.prophet/archive/master.tar.gz", repos=NULL)
```
