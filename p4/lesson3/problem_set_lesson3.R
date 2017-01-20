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

# histogram of diamond prices by cut
qplot(x = price, data = diamonds) +
  facet_wrap(~cut)

# prices of diamonds by cut
by(diamonds$price, diamonds$cut, summary, digits = max(getOption('digits')))

# histogram of diamond prices by cut with free scales
qplot(x = price, data = diamonds) +
  facet_wrap(~cut, scales = "free_y")

# histogram of diamond prices per carat faceted by cut
qplot(x = price/carat, data = diamonds) +
  scale_x_log10() +
  facet_wrap(~cut)

# boxplot of diamond price by clarity
qplot(x = clarity, y = price, data = diamonds, geom = "boxplot")

# prices of diamonds by color
by(diamonds$price, diamonds$color, summary)

# IQR of diamonds price by color
by(diamonds$price, diamonds$color, IQR)

# boxplot of price/carat by color
qplot(x = color, y = price/carat, data = diamonds, geom = "boxplot") +
  coord_cartesian(ylim = c(0,7500))

# frequency polygon of carat
qplot(x = carat, data = diamonds, color = carat, geom = "freqpoly", binwidth = 0.25) + 
  scale_x_continuous(breaks = seq(0, 5, 0.1))

# table of counts per carat
table(diamonds$carat)

# work with gapminder data
# gm <- read...
