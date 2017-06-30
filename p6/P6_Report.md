# Create a Tableau Story
## Udacity data analyst nanodegree
Link to [my subsmission](https://public.tableau.com/profile/stefan.langenbach#!/vizhome/Udacitydataanalystnanodegree/Story1) on Tableau Public

## Summary
Within this project the well-known [Titanic](https://www.kaggle.com/c/titanic) dataset is analyzed. The focus of the analysis is to investigate
why certain passengers survived the accident any why other did not. My visualization provides an overview of
the main feature of the data set, i.e. gender, age, social status (measured by passenger class and ticket fare), city of embarkment, family on board (measured by number of parents/spouses/children/siblings) and highlights some interesting attributes of survivors and non-survivors.

## Design
In order to make the visualization appealing and effective, I took the following design decisions
* Chart types (bar charts for comparisons between items with few categories, histograms for distributions, scatter plots for relationships)
* Focus on position, size and color saturation as main design elements
* Usage of a color palette suitable for color-blinds
* Usage of the same color palette across all visualizations

My idea was to introduce the dataset to the audience via stacked bar charts. Doing so allowed me to to focus on the difference between genders right from the start and made it easy to show the distribution of data in relation to some factor (i.e. passenger class)

## Feedback
In order to improve my visualization I requested feedback from two coworkers ands somebody totally unfamiliar with data analysis and visualization. Besides that I also implemented feedback from Udactiy reviews.

The first coworker did provide feedback related to design elements, i.e.:
* Usage of a consistent color palette (my first iteration did not honor my own design concept)
* Improved format of axes, i.e. rotation of labels for better readability (see P6_Visualization_before_feedback.png)
* Omitting dedicated legends if labels and color palette were already conveying all necessary information: Some of my charts included legends on which color was referring to which gender (see P6_Visualization_before_feedback.png), although this information is clearly visible from the label of top-left bubble chart
* Limits on the number of objects on dashboards. I used to have all overview visualizations on one dashboard: That seemed good from an information density perspective, but apparently confused viewers (see P6_Visualization_before_feedback.png)
* Omitting NULL values: The bar chart showing the distribution of ages (in bins) was including NULL values, resulting in a skewed distributions (see P6_Visualization_before_feedback.png).

I did implement all of his advice.

The Second coworker focused on an entirely different aspect of the visualization, the story.
In essence she advised me to:
* Use visualizations as filters to easily differentiate between sex.
* Design the story in a better way

I added a filter to easily focus on either males and females. Furthermore I this filter was explicitly added to the story (after Udacity review feedback) to make it as easy as possible for viewers to use it

The third person providing feedback was very clear about the title of charts. She urged me to convey the main message of the chart directly in the title. Seems obvious, but I had not implemented that for all charts (see P6_Visualization_before_feedback.png)

I followed that advice and edited the titles to all visualizations to tell a story

## Resources
* [Tableau documentation](http://onlinehelp.tableau.com/current/pro/desktop/en-us/help.htm)
* [Chart suggestion by Andrew Abela](http://extremepresentation.typepad.com/.shared/image.html?/photos/uncategorized/choosing_a_good_chart.jpg)
* [Graph Selection matrix by Stephen Few](https://www.perceptualedge.com/articles/misc/Graph_Selection_Matrix.pdf)
