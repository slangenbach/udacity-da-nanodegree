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
diamonds_top99 <- subset(diamonds, price < quantile(diamonds$price, 0.99))

ggplot(data = diamonds_top99,
       aes(x = x*y*z,
           y = price,
           color = clarity)) +
  geom_point() +
  scale_x_log10()

# load pf data
pf <- read.delim("./lesson3/pseudo_facebook.tsv")

# create proportion of friendships initiated variable
pf$prob_initiated <- ifelse(
  is.nan(pf$friendships_initiated / pf$friend_count),
  0,
  pf$friendships_initiated / pf$friend_count)

# create year_joined.bucket variable
pf$year_joined <- floor(2014 - (pf$tenure / 365))
pf$year_joined.bucket <- cut(pf$year_joined, breaks = c(2004, 2009, 2011, 2012, 2014))

# line graph of median proportion of friendships initiated
ggplot(data = pf,
       aes(x = tenure, y = prob_initiated, color = year_joined.bucket)) +
  geom_line(stat = "summary", fun.y =  median) +
  geom_smooth(method = "lm")

# mean proportion of friendships initiated by > 2012 group
test <- subset(pf, year_joined > 2012)
mean(test$prob_initiated, na.rm = T)

# scatterplot of diamonds price/carat ratio
ggplot(data = diamonds,
       aes(x = cut,
           y = price/carat,
           color = color)) +
  geom_point() +
  facet_wrap(~ clarity)

# gapminder data
