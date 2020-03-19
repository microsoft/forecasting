# Forecasting examples

This folder contains Python and R examples for building forecasting solutions on the Orange Juice dataset which is part of the [Dominick's dataset](https://www.chicagobooth.edu/research/kilts/datasets/dominicks). The examples are presented in Python Jupyter notebooks and R Markdown files, respectively.


## Orange Juice Dataset

In this scenario, we will use the Orange Juice (OJ) dataset to forecast its sales. The OJ dataset is from R package [bayesm](https://cran.r-project.org/web/packages/bayesm/index.html) and is part of the [Dominick's dataset](https://www.chicagobooth.edu/research/kilts/datasets/dominicks).

This dataset contains the following two tables:
- **yx.cs.** - Weekly sales of refrigerated orange juice at 83 stores. This table has 106139 rows and 19 columns. It includes weekly sales and prices of 11 orange juice brands as well as information about profit, deal, and advertisement for each brand. Note that the weekly sales is captured by a column named `logmove` which corresponds to the natural logarithm of the number of units sold. To get the number of units sold, you need to apply an exponential transform to this column.
- **storedemo.csv** -  Demographic information on those stores. This table has 83 rows and 13 columns. For every store, the table describes demographic information of its consumers, distance to the nearest warehouse store, average distance to the nearest 5 supermarkets, ratio of its sales to the nearest warehouse store, and ratio of its sales to the average of the nearest 5 stores.

Note that the week number starts from 40 in this dataset, while the full Dominick's dataset has data starting from week 1 to week 400. According to [Dominick's Data Manual](https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset/dominicks-manual-and-codebook_kiltscenter.aspx), week 1 starts on 09/14/1989. Please see pages 40 and 41 of the [bayesm reference manual](https://cran.r-project.org/web/packages/bayesm/bayesm.pdf) and the [Dominick's Data Manual](https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset/dominicks-manual-and-codebook_kiltscenter.aspx) for more details about the data.



## Summary

The following summarizes each directory of the forecasting examples.

| Directory | Content | Description |
| --- | --- | --- |
| [python](./python)| [00_quick_start/](./python/00_quick_start) <br>[01_prepare_data/](./python/01_prepare_data) <br> [02_model/](./python/02_model) <br> [03_model_tune_deploy/](./python/03_model_tune_deploy/) | <ul> <li> Quick start examples for single-round training </li> <li> Data exploration and preparation notebooks </li> <li> Multi-round training examples </li> <li> Model tuning and deployment example </li> </ul> |
| [R](./R) | [01_dataprep.Rmd](R/01_dataprep.Rmd) <br> [02_basic_models.Rmd](R/02_basic_models.Rmd) <br> [02a_reg_models.Rmd](R/02a_reg_models.Rmd) <br> [02b_prophet_models.Rmd](R/02b_prophet_models.Rmd) | <ul> <li>Data preparation</li> <li>Basic time series models</li> <li>ARIMA-regression models</li> <li>Prophet models</li> </ul> |

