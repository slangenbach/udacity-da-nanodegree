# Configure MRO to use packages from snapshot in January 2017
library(checkpoint)
checkpoint("2017-01-27")

# Install essential R packages
install.packages("devtools", dependencies = T)

# Install tidyverse meta package including:
# ggplot2, dplyr, tidyr, readr, purrr, tibble
install.packages("tidyverse", dependencies = T)

# Install additional R packages for nanodegree
install.packages("alr3", dependencies = T)
install.packages("gridExtra")
