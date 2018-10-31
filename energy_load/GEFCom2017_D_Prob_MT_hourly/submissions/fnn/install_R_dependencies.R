# Identify the required packages.

packages <- c("data.table", "magrittr")

# Determine which need to be installed.

install <- packages[!(packages %in% installed.packages()[, "Package"])]

# Install the packages.

if (length(install))
{
  install.packages(install)
} else
{
  cat("\nNo additional generic R packages need to be installed.")
}
cat("\n\n")

# Configure ssl

library(httr)
set_config(config(ssl_verifypeer=0L))

# Additional specific packages, often as an interim measure.

cat("We also need to install these specific packages...\n")

# Identify the required packages from github repo.

dev_packages <- c("rstudio/tfruns", "rstudio/reticulate", "rstudio/keras")

# Determine which need to be installed.

dev_install <- dev_packages[!(dev_packages %in% installed.packages()[, "Package"])]

# Install the packages into the local R library.

if (length(dev_install))
{
  devtools::install_github(dev_install)
} else
{
  cat("\nNo additional generic R packages need to be installed.")
}
cat("\n")
