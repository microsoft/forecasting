# Retrieves the orangeJuice dataset from the bayesm R package
# and saves as csv

install.packages("bayesm", repos = "http://mran.revolutionanalytics.com/snapshot/2018-08-27/")

library(bayesm)

data("orangeJuice")

yx <- orangeJuice[[1]]
storedemo <- orangeJuice[[2]]

fpath <- file.path("retail_sales", "OrangeJuice_Pt_3Weeks_Weekly", "data")

write.csv(yx, file = file.path(fpath, "yx.csv"), quote = FALSE, na = " ", row.names = FALSE)
write.csv(storedemo, file = file.path(fpath, "storedemo.csv"), quote = FALSE, na = " ", row.names = FALSE)
