# ML Question

> Describe how you would build a model to suggest trending search queries on Leboncoin?

Let's define a trending query as a query with a significant increase in searches for a significant amount of time, but also with a sufficiant amount of searches: going from 1 query per day to 4 is a huge increase in percentage but does not make this query a trend.

We could use the ongoing top N queries, but some query might be always on top and thus not really reflect a trend.

## Global trending searches

Let's assume that we want to construct the same trend for every user of Leboncoin.

Definition of the query time series analyzed:

> number of unique users searching this query during the last `sliding time window`

The time window can be adjust if we want to update the trend every day or every minute.

With this data we are able to construct time series for every query and try to detect an increasing spike which last.

A first simple approach could be:
- for every time series compute the confidence interval for centered moving average
  - keep the query where the value is above the interval more than X computation (we compare only to the first interval)
- sort the queries by the total amount of unique users (the last saw)
- take the top queries 

To detect a spike we could also try to predict the evolution of the time series and spot an unusual increasing difference.

We do not want to remove the seasonnality as it is part of what we want to capture. During Halloween, the trending queries could be around costumes.

## Split per category / User's recommendation

This approach can be apply for each category and a first personnalized trend could be a sampling of each category trend according to user's search distribution in these category.


