# setup enviroment
library(tidyverse)

# load data
ds <- diamonds

# scatterplot of price vs x
ggplot(data = ds, aes(y = price, x = x)) +
  geom_point()

# correlations between price and x/y/z
cor.test(ds$price, ds$x)
cor.test(ds$price, ds$y)
cor.test(ds$price, ds$z)

# scatterplot of price vs depth
ggplot(data = ds, aes(y = price, x = depth)) +
  geom_point(alpha = 1/100) +
  scale_x_continuous(breaks = 2)

# correlation between price and depth
cor.test(ds$price, ds$depth)

# scatterpolt between price and carat without top 1%
ggplot(data = ds, aes(y = price, x = carat)) +
  geom_point() +
  scale_x_continuous(limits = c(0, quantile(ds$carat, 0.99))) +
  scale_y_continuous(limits = c(0, quantile(ds$price, 0.99))) +
  labs(title = "Price vs Carat", x = "Carat", y = "Price")

# create volume variable
ds$volume <- (ds$x * ds$y * ds$z)

# scatterplot of price vs volume
ggplot(data = ds, aes(y = price, x = volume)) +
  geom_point()

# correlation between price and volume without diamonds with volume = 0 or >= 800
ds_subset <- subset(ds, volume > 0 & volume < 800)
cor.test(ds_subset$price, ds_subset$volume)

# scatterplot of price vs volume on ds subset
ggplot(data = ds_subset, aes(y = price, x = volume)) +
  geom_point(alpha = 1/100) +
  geom_smooth(method = "lm")

# create new data frame with summary statistics of ds
diamondsByClarity <- ds %>%
  group_by(clarity) %>%
  summarise(mean_price = mean(price),
            median_price = median(price),
            min_price = min(price),
            max_price = max(price),
            n = n())

# create new dataframe by clarity
diamonds_by_clarity <- group_by(ds, clarity)
diamonds_mp_by_clarity <- summarise(diamonds_by_clarity, mean_price = mean(price))

# create new dataframe by color
diamonds_by_color <- group_by(ds, color)
diamonds_mp_by_color <- summarise(diamonds_by_color, mean_price = mean(price))

# barplots of mean diamond prices by clarity and color
bp_by_clarity <- ggplot(data = diamonds_mp_by_clarity, aes(x = clarity, y = mean_price)) + geom_boxplot()
bp_by_color <- ggplot(data = diamonds_mp_by_color, aes(x = color, y = mean_price)) + geom_boxplot()

# display barplots on single canvas
library(gridExtra)
grid.arrange(bp_by_clarity, bp_by_color)

# work with gapminder data
gm <- readxl::read_excel("./lesson3/gapminder.xlsx")