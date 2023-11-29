---
title: 'FastAI lesson 7'
date: '2023-11-30'
draft: false
categories: ['fastai', 'Part 1']
description: 'Recap, quiz, and sharing post for lesson 7'
execute: 
  message: false
  warning: false
editor_options: 
  chunk_output_type: console
---

# FastAI Lesson 7

## Tenacious Animal

<p align="center">
  <img src="saltwater_croc.jpg" alt="Saltie"/>
  <figcaption align="center">These apex predators live over 70 years and grow as long as 23 feet.</figcaption>
</p>

## Recap

This lesson was another lesson that covered a couple domains: vision and collaborative filtering. The vision part of the lesson worked through a couple more iterations of the rice paddy doctor competitions. First, jeremy demonstrated how to do gradient accumulution, showing how to write it into a training loop and then demonstrating the fastai callback that does it for you. Gradient accumulation is a technique that allows you to use a larger batch size than your GPU can handle by accumulating the gradients over multiple batches and then updating the weights after a certain number of batches. This allows you to use a larger batch size than you would otherwise be able to use, which can be useful for training on smaller datasets. 

He then worked through how we would alter our model to predict multiple categories as an output. Beginning with noting that the model is not actually producing one output but 10 outputs that represent the probabilities of various diseases. He then coded the model by hand to have twenty activations and showed how we alter our loss and metric functions to account for the fact of multiple categories in the dataloader's `blocks` parameter (1 input, 2 outputs). Given these multiple outputs, our loss and metrics callbacks now receive the outputs of the model in a tuple alongside the disease and rice variety labels. The loss function then calculates the loss for each output and returns the sum of the two losses. The metrics callback then calculates the error rate (1-accuracy, accuracy takes the mean of matching input to validations set as a float ). 

The loss function itself is "cross entropy loss", which starts with using softmax to convert the activations into probabilities. Softmax is a function that takes a vector of numbers and converts them into a vector of probabilities that sum to 1. The cross entropy loss function then takes the softmax probabilities and compares them to the target labels, calculating the loss as the negative log of the probability of the correct label. The point of using the negative log is that it penalizes the model more for being more wrong. 

The last part of the lesson looked at collaborative filtering using movie reviews and introduced this problem domain as well as the concept of encodings. Collaborative filtering is a technique for making recommendations based on the preferences of other users. The idea is that if you have a large enough dataset of user preferences, you can use that data to predict what a user might like based on what other users with similar preferences have liked. The first step in this process is to create a matrix of users and items (movies in this case) and then fill in the matrix with the ratings that each user has given to each movie. This matrix is called a utility matrix. The problem with this matrix is that it is sparse, meaning that most of the cells are empty. This is because most users have not rated most movies. The goal of collaborative filtering is to fill in the empty cells with predicted ratings.

One method for doing the more accurately (making a better model) is to use something called latent factors. Latent factors are a way of representing the preferences of users and the characteristics of items in a lower dimensional space aka an embedding. The idea is that you can represent the preferences of users and the characteristics of items in a lower dimensional space and then use that representation to predict the ratings of users for items. The way that you do this is by creating two matrices, one for users and one for items, and then multiplying them together to get a matrix of predicted ratings. The user matrix is a matrix of users and their preferences for each latent factor. The item matrix is a matrix of items and their characteristics for each latent factor. The predicted ratings matrix is the product of the user matrix and the item matrix. Finally, to improve our model further we want to add a bias to the predicted ratings. The bias is a number that is added to the predicted rating to account for the fact that some users tend to rate movies higher than others and some movies tend to be rated higher than others. In mathematical terms bias allows us to shift the predicted ratings up or down on the intercept. Finally jeremy added weight decay to the model to improve the model further. Weight decay is a technique for penalizing the model for having large weights. It does this by adding the sum of the squares of the weights to the loss function. This encourages the model to have smaller weights, which can help prevent overfitting. Essentially SGD is causing the model to balance the loss and the weight decay term.

## Quiz
1. How do you do gradient accumulation in fastai?
You divide the loss by the number of accumulation steps in your training loop and then you pass a callback to Learner that accumulates the gradients, using the desired effective batch size as an argument.
1. What is ensembling and how is it different with nueral networks vs random forests?
Ensembling is the process of combining the predictions of multiple models to get a better prediction. With random forests, you can train multiple trees on different subsets of the data and then average their predictions. With neural networks, you can train multiple models on the same data and then average their predictions.
1. What is TTA?
TTA stands for test time augmentation. The idea is to do augmentation on the actual test input during inference and then average that. Augmentation in this case is doing the same thing that would happen during data/training augmentation (rotation, etc). This can make the model more robust to different inputs.
1. What is k-fold cross validation? Is it better than doing straight ensembling?
K-fold cross validation is a technique for evaluating a model by splitting the data into k folds and then training the model on k-1 folds and evaluating it on the remaining fold. This is repeated k times, each time using a different fold as the validation set. The results are then averaged. This is better than straight ensembling because it allows you to evaluate the model on more data.
1. How do you use it to get a better result from your test set?
You can use it to get a better result from your test set by training your model on the entire dataset and then using k-fold cross validation to evaluate it. This allows you to use the entire dataset for training and testing.
1. What is cross entropy loss?
Cross entropy loss is a loss function that is used for classification problems. It is the negative log of the probability of the correct label. The probability is calculated using softmax.
1. what is softmax?
Softmax is a function that takes a vector of numbers and converts them into a vector of probabilities that sum to 1. It is used to convert the activations of the last layer of a neural network into probabilities.
1. Why can't you use MSE as a loss function when trying to create a classifier?
MSE works for loss in cases where the prediction is a quantitative output (for example predicting house prices). Its not ideal for classification problems because it doesn't really make sense to think of probabilities as a quantitative output.
1. What is binary cross entropy loss? What case is it for?
This is a special case of cross entropy loss where the categories are binary (0 or 1).
1. Why is it that a multi-target model can have a higher accuracy than a single-target model, for the same data categorization task?
Conceptually this can happen because the model learns more things about the problem space that might be relevant - for example in the case of rice diseases, the breed of rice may actually determine what diseases it is susceptible to. Mathematically, this can happen because the model is learning to predict multiple outputs, which means that it is learning more about the problem space.
1. What is the last layer of most neural networks used for? What is are the activations of the last layer of a neural network?
The last layer of most neural networks is used to convert the activations of the previous layer into the desired output. The activations of the last layer are the outputs of the model.
1. What is an embedding and what is it used for?
An embedding is a representation of a categorical variable in a lower dimensional space. It is used to represent the categorical variable in a way that is more useful for the model.
1. What are latent factors? Why are they used?
Latent factors are the factors that are used to represent the categorical variable in a lower dimensional space. They are used to represent the categorical variable in a way that is more useful for the model. They can be used without our explicit knowledge of what they are (which is amazing!)
1. Implment a basic dot product model in PyTorch.
```python
class DotProduct(nn.Module):
    def __init__(self, n_users, n_movies, n_factors):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        
    def forward(self, user, movie):
        user_factors = self.user_factors(user)
        movie_factors = self.movie_factors(movie)
        user_bias = self.user_bias(user)
        movie_bias = self.movie_bias(movie)
        return (user_factors * movie_factors).sum(1, keepdim=True) + user_bias + movie_bias
```
1. How can you use a sigmoid to create a range of predictions?
You can use a sigmoid to create a range of predictions by applying it to the output of the model. This will convert the output of the model into a probability between 0 and 1 (which can be scaled to a range of values).
1. What is the use of the bias in a dot product model? Explain conceptually and mathematically.
The bias in a dot product model is used to shift the predicted ratings up or down on the intercept. Conceptually, this is used to account for the fact that some users tend to rate movies higher than others and some movies tend to be rated higher than others (in this case). Mathematically, this is done by adding a bias term to the predicted ratings.
1. What is another name for weight decay? What does weight decay do?
Weight decay is also called L2 regularization. It penalizes the model for having large weights by adding the sum of the squares of the weights to the loss function. This encourages the model to have smaller weights, which can help prevent overfitting.
1. Why do we use it? What is the equation for it?
We use weight decay to prevent overfitting. The equation for it is `loss_with_weight_decay = loss + wd * (weights**2).sum()`. Or in terms of the gradients, `weight.grad += wd * 2 * weights` (we can just use a larger or smaller value for wd and ignore the 2).
1. What is weight decay balancing?
Weight decay balancing is the process of finding the right value for weight decay. The idea is that you want to find a value that is large enough to prevent overfitting but not so large that it prevents the model from learning. Larger weights cause the model to be oversensitive to small changes in the input, which reduces its ability to generalize.
1. What is regularization?
Regularization is a technique for preventing overfitting. It is done by adding a term to the loss function that penalizes the model for having large weights. Basically this is another way of talking about weight decay. A model that is "simpler" is regularized and would have smaller weights.
