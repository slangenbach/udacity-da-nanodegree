# setup environment
library(ggplot2)

# load data
data("diamonds")

# explore data
str(diamonds)

# histogram of price of all diamonds
qplot(x = price, data = diamonds)

# get summary stats on diamonds price
summary(diamonds$price)

# number of diamonds with price < 500
sum(diamonds$price < 500)

# number of diamonds with price < 250
sum(diamonds$price < 250)

# number of diamonds with price >= 15000
sum(diamonds$price >= 15000)

# histogram of price of all diamonds with formatting
qplot(x = price, data = diamonds)  +
  scale_x_continuous(breaks = seq(0, 5000, 100), limits = c(0, 5000))
