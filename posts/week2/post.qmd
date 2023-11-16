---
title: 'FastAI Lesson 2'
date: '2023-11-13'
categories: ['fastai', 'Part 1']
description: 'Recap, quiz, and sharing post for lesson 2'
execute: 
  message: false
  warning: false
editor_options: 
  chunk_output_type: console
---

# Recap

This lesson was all about getting a model deployed to production. It was an occasion for me to make sure I had `jupyter` working locally and to understand how that flow would works.
I used Jeremy's technique of building an app in notebook and exporting it with `nbdev` directives, which is super neat! I also set myself up on paperspace, which I'd highly endorese. Eight dollars well spent.

This lesson's clean notebook was basically a recap of lesson 1, with bears instead of cats and dogs or birds.

I deployed my model to a [Huggingface Space](https://zmackie-bean-lesions.hf.space/) using Gradio. This is based on a model I trained last week for recognizing bean lesions. So if you happen to be a farmer wanting to evaluated the health of your beans...you know who to call 😸. I skipped the step of using the api for this app. Gradio now has an API client that has to be installed with `npm`. Which...Im not trying to deploy anything that needs a build step. I wish they hadn't done that.

I hit a few hiccups which were easily navigated with some googling. Mainly these were API changes in various libraries. Inevitable as things move along but bumps nonetheless.

## Quiz

1. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
Not great at images that differ in stylisitically. The camera to recognize the bears might be mounted upside down.
1. Where do text models currently have a major deficiency?
They're plausible, but are they correct?!
1. What are possible negative societal implications of text generation models?
Spreading false information that's very convincing
1. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
Human in the loop.
1. What kind of tabular data is deep learning particularly good at?
where the column data might be very diverse
1. What's a key downside of directly using a deep learning model for recommendation systems?
The recommendations might not actually be helpful. eg. they might recommend books you already have
1. What are the steps of the Drivetrain Approach?
objectives, levers, data, model (how the levers influence the objectives)
1. How do the steps of the Drivetrain Approach map to a recommendation system?
objective:  drive more sales
levers: ranking of the recommendations
data: past choices of the user
model: two models that are contingent on seeing or not seeing the recommendation
1. Create an image recognition model using data you curate, and deploy it on the web.
1. What is `DataLoaders`?
Generic class for getting data into a learner
1. What four things do we need to tell fastai to create `DataLoaders`?
  1. input and output types
  1. how get get the items
  1. how to label the items
  1. how to create a validaton set
1. What does the `splitter` parameter to `DataBlock` do?
telling fai how to split of a validation set
1. How do we ensure a random split always gives the same validation set?
seed will set the randomness
1. What letters are often used to signify the independent and dependent variables?
x for independent 
y for dependent
1. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
crop = cut
pad = fill in with black
squish = distort to fit
chose might be dictated by the particulars of the data
1. What is data augmentation? Why is it needed?
creating random variations that seem different (to a model) but don't actually change in meaning
1. What is the difference between `item_tfms` and `batch_tfms`?
item is single, batch is a group
1. What is a confusion matrix?
a nXn matrix that plots what the model was predicted vs the correct labels.  the center diagonal is correct guess. others a re off somehow. lets us see issues in data or model.
uses the validation set.
1. What does `export` save?
a pkl file of the trained model
1. What is it called when we use a model for getting predictions, instead of training?
inference
1. What are IPython widgets?
1. When might you want to use CPU for deployment? When might GPU be better?
cpu is more cost effective, easier to manage, more available, and perfectly suitable for running inferences.
1. What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
network issues, latency, sensitive data, complexity of runnning infra
1. What are three examples of problems that could occur when rolling out a bear warning system in practice?
data is video vs pictures, bears in are in novel positions or lighting, night pictures, speed of results
1. What is "out-of-domain data"?
data that differs a lot to what was seen in training
1. What is "domain shift"?
type of data changes over time so the initial model doesnt apply so much. the use of the model actually changes things, so the model has to be adjusted.
1. What are the three steps in the deployment process?
  1. manual steps, human checks it all
  1. limited scope. time or geography limited. careful supervisions
  1. gradual expansion. need reporting. need to consider what can go wrong.
