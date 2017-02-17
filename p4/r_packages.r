# Configure MRO to use packages from snapshot in January 2017
#library(checkpoint)
#checkpoint("2017-01-27")
options(repos = c(CRAN = "https://mran.revolutionanalytics.com/snapshot/2017-01-27"))

# Install essential R packages
install.packages("devtools", dependencies = T)

# Install tidyverse meta package including:
# ggplot2, dplyr, tidyr, readr, purrr, tibble
install.packages("tidyverse", dependencies = T)

# Install additional R packages for nanodegree
install.packages(c(
  "alr3",
  "gridExtra",
  "GGally",
  "scales",
  "memisc",
  "lattice",
  "MASS",
  "car"))