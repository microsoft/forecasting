# TSPerf

TSPerf is a framework that allows discovery and comparison of various time-series forecasting algorithms and architectures on a cloud-based environment. This framework allows data scientists to discover the best approach that fits their use case from cost, time and quality perspective.
TSPerf framework is designed to facilitate data science community participation and contribution through the development of benchmark implementations against a given set of forecasting problems and datasets. Benchmark implementations are measured in terms of standard metrics of model accuracy, training cost and model training time. Each implementation includes all the necessary instructions and tools that ensure its reproducibility on Azure customer's subscription. We plan to leverage TSPerf to propose a new time-series forecasting track in [MLPerf](https://mlperf.org/).

The following table summarizes benchmarks that are currently included in TSPerf. Source code of the models and instructions for reproducing their performance can be found in submission folders.

| **Benchmark** | **Dataset** | **Benchmark directory** |  
| --------------------- | ----|---------------- |  
| Probabilistic electricity load forecasting | GEFCom2017 |`energy_load\GEFCom2017-D_Prob_MT_Hourly` |
| Retail sales forecasting | Orange Juice dataset | `retail_sales\OrangeJuice_Pt_3Weeks_Weekly` |

A complete documentation of TSPerf, along with the instructions for submitting and reviewing benchmark implementations, can be found [here](./internal_docs/tsperf_rules.md). The tables below 
list the models developed so far.

### Probabilistic energy forecasting

| Submission Name | Submission Folder URL | Pinball Loss | Training and Scoring Time | Training and Scoring Cost | Architecture | Framework | Algorithm |
| -------------- | --------------------- | --------- | ---- | --- | -- | -- | -- |
| Baseline | energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/baseline | 84.66 | 446 sec | $0.05 | Linux DSVM (Standard D8s v3, Premium SSD) | quantreg package of R | Linear Quantile Regression  |

### Retail sales forecasting

| Submission Name | Submission Folder URL | MAPE (%) | Training and Scoring Time | Training and Scoring Cost | Architecture | Framework | Algorithm |
| -------------- | --------------------- | --------- | ---- | --- | -- | -- | -- |
| Baseline |retail_sales/OrangeJuice_Pt_3Weeks_Weekly/baseline | 109.67 | 114.06 sec | $0.003 | Linux DSVM (Standard D2s v3, Premium SSD) | forecast package of R | Naive Forecast  |
| AutoARIMA | retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/AutoARIMA | 77.66 | 2214.93 sec | $0.06 | Linux DSVM (Standard D2s v3, Premium SSD) | forecast package of R | Auto ARIMA  |
| ETS | retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/ETS | 70.99 | 277.01 sec | $0.01 | Linux DSVM (Standard D2s v3, Premium SSD) | forecast package of R | ETS  |
| MeanForecast | retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/MeanForecast | 70.74 | 69.88 sec | $0.002 | Linux DSVM (Standard D2s v3, Premium SSD) | forecast package of R | Mean forecast  |
| SeasonalNaive | retail_sales/OrangeJuice_Pt_3Weeks_Weekly/submissions/SeasonalNaive | 165.06 | 160.45 sec | $0.004 | Linux DSVM (Standard D2s v3, Premium SSD) | forecast package of R | Seasonal Naive  |





