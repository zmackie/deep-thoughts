---
title: 'FastAI Lesson 3'
date: '2023-11-15'
categories: ['fastai', 'Part 1']
description: 'Recap, quiz, and sharing post for lesson 3'
execute: 
  message: false
  warning: false
editor_options: 
  chunk_output_type: console
---

# Recap

This was a tough lesson. I started calling this the "week 3 wall" because I felt just totally stopped in my tracks by this one. Conceptually it just took me a a long time to wrap my head around this one. One of the things I've had to embrace with this course to make progress is just allowing myself to not understand all the details of something. Like for example I get the idea of using a derivative to optimize a quadratic function. Somehow intuitively I can make sense of the image of a tangent line moving down a function and slowly flattening out at the bottom of the graph. But putting all the pieces together has been a struggle. I worked through the titanic example on my own, and that helped it make a bit more sense. What helped there was going from the linear function to the super simple two-layer nueral net. And applying matrix multiplication there was helpful to cement the concept.

As Jeremy has said time and again, tenacity is super important. I wasn't going to let this week stall me. To inspire myself I generated some fun pictures in Dall-e of especially tenacious animals.
Here's a honey-badger for you viewing pleasure. I encourage you to adopt it as your `fastai` spirit animal.

<p align="center">
  <img src="honey_badger.jpg" alt="Sublime's custom image"/>
</p>

One of the crucial things that took me a long time to understand is the relationship between the loss function and the model. For some reason I had the idea that the  loss function was related to the model somehow, like in some kind of mathematical sense. But the crucial thing here to understand is that the model (which includes the architecture and the parameters) is tangential to the loss function in SGD. The loss function is a way to understand the *performance* of the model WRT a known answer. SGD is a method of using the loss function (which is somewhat arbitrary - at least that's what it seems like a the moment) to optimize or train the values of the weights of the model. The "gradient" is actually a calculation of the slope of the loss function WRT the input parameters, and gradients might be many gradients - one for each parameter. The gradient is calculated using the derivative of the loss function, which says for a change in the input, how will the loss be changed. What is mysterious to me is the backwards/back propogation step. What's ultimately confusing to me is the fact that the loss function helps the input parameters adjust, but the input parameters are not directly involved in the loss function at all! They loss function only uses the predictions and the known answers. So how does the loss function help the parameters adjust? I think this is the backwards/back propogation step, but I'm not sure. I think I need to do some more reading on this. The key seems to be the "chain rule".

>The chain rule is a fundamental principle in calculus that states how to compute the derivative of a composite function. Concisely, it can be stated as:

> "If a function $( y )$ is a composite of another function $(u)$, such that $(y=f(u))$ and $(u=g(x))$, then the derivative of $( y )$ with respect to $( x )$ is the product of the derivative of ( f ) with respect to ( u ) and the derivative of ( g ) with respect to ( x ). In mathematical terms, it is expressed as $( \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} )$."


Conceptually I can start to grasp that there is function composition going on, but the details are still a bit fuzzy. Ultimately the way I'm thinking about it is that the prediction is itself the output of a function. But thinking about the prediction as an arbitary set of values isn't quite correct. Because I started thinking, well if the prediction is just an array of values, how is it possible to know anything about what produced those values? And in the general sense its not. But this is what `pytorch.requires_grad` allows us to do. Setting that parameter on the input actually causes the library to records a computation graph and the prediction (or the output of running the params through that graph) will be statefully associated with the input parameters via that graph, which is what allows us to call `.backward()` on the prediction and have the gradients calculated. Because the prediction has knowledge of what produced it, its possible to then follow that chain of operations backwards, calculating the gradients of each step, and then using those gradients to adjust the parameters. I think this is the key to understanding the backwards/back propogation step. 

Getting the [breed recognizer model deployed](https://zmackie-dog-cat-breeds.hf.space/) was a bit trickier than the last model. Due to the use of some of the models from `timm`, I entered into some fiddly dependency dancing. Luckily other people had figured this out for me and so a quick google (which turned up an answer on the fastai forums) turned up a fix. One of the weirder things that I'm struggling with in terms of deployment is how to manage dependencies for an app built in this way. Running the app from jupyterhub and running a `pip install` puts the packages into the conda environment that jupyterhub is running in. This makes it slightly hard to understand the dependencies and respective versions that need to go into the `requirements.txt` file. Ultimately I had to use `--report` which gives an output of package versions and then use that to populate the `requirements.txt` file. I'm not sure if this is the best way to do this, but it's what I've got for now. Probably there's a a better way to use poetry or `venv` or something to manage this.

I also found it valuable to run through pytorchs [basic tensor tutorial](https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html) to wrap my head around this datatype a bit more.
I found myself reading the [torch/tensor Docs](https://pytorch.org/docs/stable/tensors.html) a lot, and noticed that the `torch.method()` docs usually have examples, which makese it easier to understand things. However playing with the stuff in my terminal was always the most illuminating.

```python
>>> t = torch.randint(high=10,size=(2,3))
>>> t
tensor([[8, 0, 7],
        [0, 7, 3]])
>>> t.T
tensor([[8, 0],
        [0, 7],
        [7, 3]])
>>> t.T.shape
torch.Size([3, 2])
>>> t.shape
torch.Size([2, 3])
```


Its still totally magical to me, though, that I can actually make these things!

## Quiz

1. How is a grayscale image represented on a computer? How about a color image?
Images are represented as matrixes in a computer, with every cell in the matrix representing a pixel. In a grayscale image, the value of the cell is the intensity of the pixel. In a color image, the value of the cell is a vector of three values, representing the intensity of the red, green, and blue channels of the pixel.

1. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?
This dataset has a csv of labels and `Train` and 'valid' folder. Within each folder, there is a subfolder for each label, eg `/train/3`.
The data is organized this way because this is a common structure and allows for easy loading (`dataLoader` can read the parent folder and infer the labels from the subfolders).

1. Explain how the "pixel similarity" approach to classifying digits works.
This approach approach takes the matrix of threes and calculates a mean matrix. We then compare a given image to this mean image. The comparison is performed using something a function that handles negative numbers but taking the absolute values or squaring the squaring the values. This is necessary because simple summing of the matrix resulting from taking the difference will cause some numbers to cancel each other out. The operation to deal with negative numbers (squaring or absolute value) is performed before calculating the mean. The result of this operation is a single number, which is the "distance" between the image and the mean image. The image is then classified as the label of the mean image with the smallest distance.

1. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
A list comprehension is a handy way of dealing with lists in python. It allows you to perform an operation on each element of a list and return a new list. For example, `[x^2 for x in range(10) if x%2==1]`will return a list of the odd numbers from 0 to 9, doubled.

1. What is a "rank-3 tensor"?
A rank-3 tensor is a tensor with 3 dimensions. A rank-0 tensor is a single number, also called a scalar. A rank-1 tensor is a vector. A rank-2 tensor is a matrix. A rank-3 tensor is a 3-dimensional matrix, meaning a matrix of matrixes.

1. What is the difference between tensor rank and shape? How do you get the rank from the shape?
Rank is the number of dimensions in a tensor. Shape is the size of each dimension. You can get the rank from the shape by counting the number of elements in the shape.
For example `Tensor([1,2,3,4]).shape` will return `torch.Size([4])` which has a rank 1.

```python
>>> tensor([1],[1],[1]).shape
torch.Size([3, 1]) # rank 2, three rows, one column
# stacked_threes is a a rank 3 tensor
# it has shape torch.Size([6131, 28, 28]) 
stacked_threes[0][0][0].ndim # rank 0, ie scalar, ie raw value
stacked_threes[0][0].ndim # rank 1, ie vector
stacked_threes[0].ndim # rank 2, ie matrix
stacked_threes.ndim # rank 3, ie 3-dimensional matrix
```
  
7. What are RMSE and L1 norm?
These are loss functions. RMSE is the root mean squared error. It is the square root of the mean of the squared differences between the predictions and the actual values. The L1 norm is the mean of the absolute value of the differences between the predictions and the actual values.
```python
# RMSE
def rmse(preds, targets):
    return ((preds-targets)**2).mean().sqrt()
# L1 norm
def l1(preds, targets):
    return (preds-targets).abs().mean()
```
8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
Using matrix multiplication. Matrix multiplication is a way of performing a calculation on a matrix of numbers. It is much faster than a python loop because it is implemented in C, which is much faster than python.
The operation happens on the GPU, which is designed for this kind of operation.
1. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
```python
>>> a = torch.tensor(range(1,10)).reshape(3,3)*2
tensor([[ 2,  4,  6],
        [ 8, 10, 12],
        [14, 16, 18]])
>>> a[1:,1:]
tensor([[10, 12],
        [16, 18]])
```
10. What is broadcasting?
Broadcasting is the ability of pytorch to perform operations on tensors of different shapes. For example, if you have a rank-1 tensor and a rank-2 tensor, pytorch will automatically expand the rank-1 tensor to match the shape of the rank-2 tensor. This is useful because it allows you to perform operations on tensors of different shapes without having to manually reshape them.
```python
>>> a = torch.tensor([1,2,3])
>>> b = torch.tensor([[1,2,3],[4,5,6]])
>>> a+b
tensor([[2, 4, 6],
        [5, 7, 9]])
>>> a.broadcast_to((2,3))
tensor([[1, 2, 3],
        [1, 2, 3]])
>>> b.broadcast_to((1,2,3))
tensor([[[1, 2, 3],
         [4, 5, 6]]])
>>> b.broadcast_to((2,2,3))
tensor([[[1, 2, 3],
         [4, 5, 6]],
        [[1, 2, 3],
         [4, 5, 6]]])

```
11. Are metrics generally calculated using the training set, or the validation set? Why?
Metrics are calculated using the validation set. We do this because we want to make sure the model is training to become generalized. If we used the training set, we would be optimizing the model to perform well on the training set, but we wouldn't know if it was generalizing well.
1. What is SGD?
SGD is a way of optimizing a function, say for examples $y=x^2$. SGD is a way of finding the minimum of this function. 
It does this by using the derivative to calculate the gradient, ie the slope at a particular point. The gradient tells us how changing the input parameters will change the output. If the gradient is positive, then increasing the input will increase the output. If the gradient is negative, then increasing the input will decrease the output. The gradient is calculated at a particular point, and then the input is adjusted by a small amount in the opposite direction of the gradient. This is repeated until the gradient is zero, which means that the input is at the minimum of the function. We are trying to minimize the function in our case because the function represents the loss of our model ie the gap between our model's prediction and the actual value of the input.
1. Why does SGD use mini-batches?
We use mini-batches in order to speed up the calculation in SGD. When we are optimizing, we are usually doing so over a large number of parameters. If we were to calculate the gradient for each parameter, it would take a long time. Instead, we can calculate the gradient for a small batch of parameters (as an average of the items in the  batch), and then use that to adjust the parameters in the batch. This is much faster than calculating the gradient for each parameter individually.
1. What are the seven steps in SGD for machine learning?
    1. Initialize the parameters with random values
    1. Calculate the predictions
    1. Calculate the loss
    1. Calculate the gradients, which approximates how the paramters need to change to reduce the loss
    1. Adjust the parameters by a small amount in the opposite direction of the gradient. ie If the gradient is positive, then decrease the parameter. If the gradient is negative, then increase the parameter.
    1. Repeat steps 2-5 until the loss is small enough for our purposes
1. How do we initialize the weights in a model?
Randomly. That's the "stochastic" part of SGD. We initialize the weights randomly because we don't know what the optimal weights are. We are trying to find them. So we start with random weights and then use SGD to adjust them.
1. What is "loss"?
Loss is a calculation of the difference between the model's prediction and the actual value. It is a way of measuring how well the model is performing.
1. Why can't we always use a high learning rate?
If the learning rate is too high, we will bounce our parameters around, possible never arriving at the minimum. If the learning rate is too low, it will take a long time to arrive at the minimum.
1. What is a "gradient"?
A gradient is a calculation of the slope of a function at a particular point. It tells us how changing the input parameters will change the output. Its the actual specific value of the derivative at a particular point in the loss function.
1. Do you need to know how to calculate gradients yourself?
Nope, pytorch does it for us.
1. Why can't we use accuracy as a loss function?
Accuracy is generally steppy. Meaning that an improvement of 0.1% of our model a big deal, but might not immediately result in flipping one incorrect prediction to correct. But we want to be able to make small adjustments to our parameters. So we need a loss function that is smooth, meaning that small changes in the parameters will result in small changes in the loss. Smoothness also allows us to easily find gradients and derivatives.
1. Draw the sigmoid function. What is special about its shape?
Sigmoid takes all inputs and normalizes them into a value between 0 and 1. This allows easy optimization using SGD.
```python
def sigmoid(x):
    return 1/(1+torch.exp(-x))
```


22. What is the difference between a loss function and a metric?
A loss function has to be smooth, meaning that small changes in the parameters cause a response in the loss. Smoothness also allows us to easily find gradients and derivatives. A metric doesn't have to be smooth. It just has to be a way of measuring how well the model is performing. A metric, on the other hand, is really what *we* care about. We care about accuracy, but this might not be amendable to optimization using SGD. As a human, we want to focus on this, rather than loss, in judging our model's performance. The computer doing the optimizing, on the other hand, will use the loss to do its work. Interestingly loss is sometimes a compromise between two needs: our goal with the model and the ability of the function to be optimized using its gradient.
23. What is the function to calculate new weights using a learning rate?
```python
new_weight = old_weight - gradient*learning_rate
```
24. What does the `DataLoader` class do?
A `DataLoader` allows any python collection to be treated as an iterator. It has built in functionality to create batches and allows for shuffling, which improves the performance of training, as it gives our model variety. Notice below that listing the `dl` multiple times will return different batches.
```python
from torch.utils.data import DataLoader
>>> coll = range(2,26)
>>> dl = DataLoader(coll, batch_size=4, shuffle=True)
>>> list(dl)
[tensor([13, 17, 23, 16]), tensor([ 4,  2, 12, 15]), tensor([ 5, 22, 10, 19]), tensor([ 7, 25, 21, 24]), tensor([ 6, 20,  8,  9]), tensor([ 3, 11, 18, 14])]
>>> list(dl)
[tensor([17, 16, 22, 24]), tensor([21,  6, 12,  5]), tensor([14,  8,  4, 20]), tensor([15, 13, 18,  9]), tensor([11,  7, 23, 25]), tensor([19,  2,  3, 10])]
>>> list(dl)
[tensor([12, 25, 24,  4]), tensor([ 5, 22,  2,  3]), tensor([13, 21, 23, 14]), tensor([10, 17,  6, 18]), tensor([15,  8, 19, 20]), tensor([11, 16,  7,  9])]
```
25. Write pseudocode showing the basic steps taken in each epoch for SGD.
```python
for each opoch:
  - for each batch
    - make a prediction
    - calculate the loss
    - calculate the gradients
    - updates the parameters based on the gradients * lr
```

26. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?
```python
def special(a, b):
  return list(zip(a,b))
```
This structure is how pythorch expects to receive datasets. The first item in the tuple is the independant variable. The second item is the dependent variable. In other words these are the inputs and the targets of the model.

27. What does `view` do in PyTorch?
View allows pytorch to change the shape of a tensor without changing its contents. The new shape has to be compatible with the existing contents.
For example a `tensor([1,2,3,4])` can be reshaped into a `tensor([[1,2],[3,4]])` because the number of elements is the same.
```python
>>> a = torch.tensor(range(1,10))
>>> a
tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> a.view(3,3)
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
```

28. What are the "bias" parameters in a neural network? Why do we need them?
The formula for a line is $y=w*x+b$. Bias is the $b$ in this formula. This allows our function to be more flexible, as in input of zero won't always have to result in an output of zero.
1. What does the `@` operator do in Python?
The `@` operator performs matrix multiplication.
1. What does the `backward` method do?
The `backward` method calculates the gradients input parameters WRT the output of loss function. It does this by following the chain rule. It calculates the gradient of the loss function with respect to the output of the model, then the gradient of the output of the model with respect to the input parameters. It does this by following the computation graph that was created when the model was run. In order to be able to do this, the input params tensor need to have `requires_grad` set to `True`.
1. Why do we have to zero the gradients?
We do this after we update the input parameters. This is because the gradients are accumulated. In other words, if we don't zero them, they will continue to increase. We want to start fresh with each batch.
1. What information do we have to pass to `Learner`?
We have to pass the data, the model, the loss function, and the optimizer function (ie SGD), and optionally a metric function.
1. Show Python or pseudocode for the basic steps of a training loop.
```python
- unpack our dl
- calculate the loss, which implies making a prediciton with the model and comparing it to the actual label
- calculate the gradients
- update the parameters by param -= gradients * lr
- zero the gradients
```

34. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.
A nonlinear function. For negative values its output is zero. For positive values its output is the input value.
1. What is an "activation function"?
These are the outputs of a layer in a nueral network.
1. What's the difference between `F.relu` and `nn.ReLU`?
`F.relu` is a function. `nn.ReLU` is a class. They both do the same thing, but `nn.ReLU` is an object that can be used in a model.
1. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
We use more because it allows us to have fewer input parameters, which makese the model faster to train.


extra credit 😏:
<details>
1. Create your own implementation of `Learner` from scratch, based on the training loop shown in this chapter.
[kaggle notebook](https://www.kaggle.com/zanadar/learner-from-scratch)
</details>