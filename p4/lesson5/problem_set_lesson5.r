# setup environment
library(tidyverse)

# load data
data("diamonds")
  
# histogram of diamond prices
ggplot(data = diamonds, aes(x = price, color = cut)) + 
  geom_histogram() + 
  facet_wrap( ~ color)

# scatterplot of diamond prices vs table
ggplot(data = diamonds, aes(x = table, y = price, color = cut)) +
  geom_point()

# scatterplot of diamond prices vs volume
diamonds_top99 <- subset(diamonds, top_99 <- quantile(top_99))

ggplot(data = diamonds_top99)
       aes(x = x*y*z,
           y = price,
           color = clarity)) +
  geom_point() +
  scale_x_log10()
