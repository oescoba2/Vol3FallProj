# Happiness Report

## Abstract
We model happiness based on country-specific variables in order to understand the observed correlation between a country's economic development and average happiness. We rely on the World Happiness Report and data from the world bank to create several models to predict average happiness. We hope to better understand factors which affect how people rank their own happiness.

## Intro 
Happiness is a ethereal treasure, and one that is hard to define, yet we believe there is value in at least pursuing it.
And in order to pursue happiness, there first needs to be some measure of it.
Then we can examine which features contribute most to experiencing happiness.

To that end, the World Happiness Report produces data on a worldwide scale to approximate the happiness of individuals. 
The World Happiness Report determines happiness based on the Cantril ladder: respondents are asked to think of a ladder, with the best possible life for them being a 10 and the worst possible life being a 0. 
They are then asked to rate their own current lives on that 0 to 10 scale.

We examine country wide averages of happiness scores and fit models to predict happiness based on country specific data from the world bank. 
The world bank mantains an extensive database on countries with over one hundred features ranging from rates of domestic violence to gross domestic product (GDP).
The data cover multiple years and enough countries to account for a large majority of the world's land mass which humans inhabit. 

## Data cleaning

Depending on two separate databanks naturally leads to problems with missing data and data cleaning.
It will be apparent that a swath of major African countries right in the center of Africa are missing from out models.
Though we cannot make up for the amount of data missing on those countries, thanks to the number of features in the world bank database, we successfully selected features for the remaining countries with at most 10% of missing values.
To fill in remaining missing data, we used a K-Nearest Neighbors Imputer. 

Convieniently, all features from the world bank were numeric, and none were categorical.
Since all of our models are regression based, our KNN imputer was sufficient for cleaning our data.

