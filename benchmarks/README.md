### Benchmarks

TODO: Modify below based on the benchmarks we include.

Each entry below describes a forecasting benchmark.

| **Benchmark** | **Description** |
| ----------- | --------------- |
|[GEFCom2017_D_Prob_MT_hourly](./GEFCom2017_D_Prob_MT_hourly)| Energy load forecasting on GEFCom2017-D competition data (qualifying match, defined-data track)<br/>Forecast hourly values for 1~2 months ahead<br/>Probabilistic forecasts<br/>Historic temperature, load data, calendar data, and the US Federal Holidays |
|[OrangeJuice_Pt_3Weeks_Weekly](./OrangeJuice_Pt_3Weeks_Weekly)| Sales forecasting on Orange Juice data from R package `bayesm` <br> Forecast weekly values for 3 weeks ahead <br> Orange juice sales data and store demographic information |

### Abbreviations in benchmark names
**Forecast type**  
- Prob: Probabilistic
- Pt: Point

**Forecast horizon type**  
- ST: short-term, <= 2 days ahead
- MT: medium-term, 2 days ~ 3 months ahead
- LT: long-term, more than three months ahead
