---
title: 'FastAI Lesson 5'
date: '2023-11-28'
categories: ['fastai', 'Part 1']
description: 'Recap, quiz, and sharing post for lesson 5'
execute: 
  message: false
  warning: false
editor_options: 
  chunk_output_type: console
---

# FastAI Lesson 5

## Tenacious Animal

<p align="center">
  <img src="polar_bear.jpg" alt="Polar bear"/>
  <figcaption align="center">Polar bears can wait for hours or even days to catch their prey.</figcaption>
</p>

## Recap

This lesson was about tabular data. Firstly we looked at recreating a simple linear model in python. This used sum product with a series of coefficients broadcast across our dataset. We used SGD to optimize this. Then we looked at creating a simple neural network to do the same thing. This was a single hidden layer with a single output layer. We used pytorch for this and optimized with SGD using the gradients on the tensors. Finally we looked at constructing the most basic "deep" neural network. This was a series of hidden layers, a relu, and an output layer. This was very fiddly in terms of parameters, the size of things, how they were initialized. This motivated the need for a higher level library like fastai, which was demonstrated in the second notebook. This took about 5 lines of code to do the same thing as the previous notebook.

Finally Jeremy introduced a few new concepts for machine learning, specifically binary splits and random forests. The idea of a binary split is to find some fact about your data that divides it in two. For example with the titanic dataset, is the person male or female? Then we examine the dependent variable with respect to that split. Given a series of binary splits, getting finer and finer granularity we can build a decision tree. The random forest is a collection of decision trees, each one trained on a subset of the data. The random forest then takes the average of all the trees to make a prediction.

Jeremy recommended using random forests as a way to understand datasets and as an initial baseline when working with tabular data. This is because they are fast to train and can give you a good idea of what is going on with your data, including things like which features are of particular importance (in the titanic dataset, sex is hugely predictive of survival, for example).

## Quiz

1. What is a continuous variable?
A variable that can take on any value in a range.
1. What is a categorical variable?
A variable that can take on a limited number of values.
1. Provide two of the words that are used for the possible values of a categorical variable.
Levels or classes.
1. What is a "dense layer"?
A layer where each input is connected to each output.
1. How do entity embeddings reduce memory usage and speed up neural networks?
They replace a categorical variable with a lookup table
1. What kinds of datasets are entity embeddings especially useful for?
Those with high cardinality categorical variables.
1. What are the two main families of machine learning algorithms?
Ensemble of decision trees and neural networks. The former is more interpretable, the latter is more flexible. The former is useful for structured data, the latter is useful for unstructured data.
1. Why do some categorical columns need a special ordering in their classes? How do you do this in Pandas?
Because the categorical variable has an intrinsic ordering. For example, the days of the week. We can do this in pandas by using the `ordered=True` argument when creating the categorical variable.
1. Summarize what a decision tree algorithm does.
It finds the best binary split for each column in the dataset, and then repeats this process on each of the two resulting datasets, and so on, until it reaches some stopping criterion.
1. Why is a date different from a regular categorical or continuous variable, and how can you preprocess it to allow it to be used in a model?
Dates *can* be treated as continuous or categorical variables, but it is often useful to preprocess them into additional categorical variables such as year, month, day, holidays, and so forth. This is because dates often have special meanings that are not captured by treating them as pure numbers.
1. Should you pick a random validation set in the bulldozer competition? If no, what kind of validation set should you pick?
No, because the data is ordered by date. Instead
1. What is pickle and what is it useful for?
Pickle is a python library for saving python objects to disk. It is useful for saving models, so that they can be loaded later.
1. How are `mse`, `samples`, and `values` calculated in the decision tree drawn in this chapter?
`mse` is the mean squared error of the dependent variable in that group of rows. `samples` is the number of rows in that group. `values` is the mean of the dependent variable in that group.
1. How do we deal with outliers, before building a decision tree?
We can remove them, or cap them at some maximum or minimum value. Or we can use a log transform to bring them into a more reasonable range.
1. How do we handle categorical variables in a decision tree?
We can use one-hot encoding (which isn't very necessary) or just keep them as is, making split decisions based on whether a row has a particular value or not.
1. What is bagging?
Bagging is the process of training a model on a subset of the data, and then averaging the results. This is useful for reducing overfitting.
1. What is the difference between `max_samples` and `max_features` when creating a random forest?
`max_samples` is the number of rows to use for each tree. `max_features` is the number of columns to use for each split.
1. If you increase `n_estimators` to a very high value, can that lead to overfitting? Why or why not?
Yes, because the model will be able to memorize the training set given so many trees. It is important to use a validation set to find the optimal number of trees and to make sure the model will generalize.
1. In the section "Creating a Random Forest", just after <<max_features>>, why did `preds.mean(0)` give the same result as our random forest?
Because the random forest is just the average of the trees, and `preds.mean(0)` is the average of the predictions of the trees.
1. What is "out-of-bag-error"?
The error of the model on the rows that were not used for training that particular tree.
1. Make a list of reasons why a model's validation set error might be worse than the OOB error. How could you test your hypotheses?
Because the validation set might be distributed differently than the training set. We can test this by looking at the feature importance of each tree.
1. Explain why random forests are well suited to answering each of the following question:
   - How confident are we in our predictions using a particular row of data?
      - Because we can use the predictions of each tree to calculate the standard deviation of the predictions.
   - For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?
      - Because we can look at the feature importance of each tree.
   - Which columns are the strongest predictors?
      - Because we can look at the feature importance of each tree.
   - How do predictions vary as we vary these columns?
      - Because we can look at the feature importance of each tree.
1. What's the purpose of removing unimportant variables?
To make the model simpler and faster to train, and to reduce overfitting.
1. What's a good type of plot for showing tree interpreter results?
A waterfall plot.
1. What is the "extrapolation problem"?
When a model is asked to make predictions for data that is outside the range of the training data.
1. How can you tell if your test or validation set is distributed in a different way than your training set?
By looking at the feature importance of each tree.
1. Why do we ensure `saleElapsed` is a continuous variable, even although it has less than 9,000 distinct values?
Because it is a date, and dates have a special meaning that is not captured by treating them as pure numbers.
1. What is "boosting"?
Boosting is the process of training a series of models, each one trying to correct the errors of the previous one.
1. How could we use embeddings with a random forest? Would we expect this to help?
We could use embeddings with a random forest by replacing each categorical variable with a one-hot encoded vector, and then using the embeddings of those vectors as the input to the random forest. This would probably not help, because random forests are already very good at dealing with categorical variables.
1. Why might we not always use a neural net for tabular modeling?
Because neural nets are more complex and take longer to train than random forests, and random forests are often just as accurate.