---
title: "A Checkin on what's working"
date: '2023-11-17'
categories: ['fastai', 'Part 1']
description: 'Some observations and thoughts about my learning process so far'
execute: 
  message: false
  warning: false
editor_options: 
  chunk_output_type: console
---
# Observations on my learning process

### Tenacious animal
<p align="center">
  <img src="hippo.jpg" alt="A hippo" width="500" height="500"/>
  <figcaption align="center">Dall-e can't spell</figcaption>
</p>



### The learning process

I thought it might be useful a this point to note a few things about my process of going through the course, so far. In addtiton to the content of the course, fastai is just brilliant in encorporating practices of learning that are miles more effective than the typical bottom up approach. There are also small ways the course does this, which are fascinating to try and pick out.

At the outset, knowing that my time was limited, it was important to me to optimize my learning. There are a number of resource that I drew on for this:

- Barbara Oakley's course on learning how to learn
- Marcin, a fastai alums book on learning
- Scott Young's book on Ultralearning
- The book "Make it Stick"
- The book "The First 20 Hours"
- The paper ["Teaching the Science of Learning"](https://cognitiveresearchjournal.springeropen.com/articles/10.1186/s41235-017-0087-y)

That latter paper talks about 6 techniques that are effective for learning, which I'll elaborate on below:

- Spaced practice
- Retrieval practice
- Elaboration
- Interleaving
- Concrete examples
- Dual coding

**Spaced practice** is relatively known (anki). 

**Retrieval practice** is suprising, in some ways. We learn by trying to retrieve information. And in fact actually quizzing yourself on information that you know *nothing* about somehow primes your brain to learn better. Jeremy has a number of structural elements in the course that are designed to do this. The quizzes at the end of each lesson are one example. The other is the "fastai" library itself. It's a library that is designed to be used by people who don't know anything about deep learning.

I've been trying to use chatgpt to quiz me about the chapters...and its not great at specifics. In general (higher level) it gives good questions. Quizzing is a big part of it. I know try and go through the quizzes before, during, and after lessons. I also try and do the clean notebooks, quzzing myeslef on the outputs. Another technique I haven't tried, but would like to, is doing a sort of "flipped" approach to the clean notebooks, where I try and create the code based on the prose descriptions.

**Interleaving** is a technique I haven't tried. Basically you weave subjects together. Like I could weave vision and tabular DL learning practice.

**Concrete examples** are obvious, and are how the course is rooted.

**Dual coding** is about presenting the information in multiple forms, like visual and written. It isn't about learning styles, but about complementary information.

I've also been experimenting with using various LLM tools to both enhance my learning and my coding productivity. For example the learner I developed for the last lesson used copilot and chatgpt to help create the classes, do some refactoring, and some debugging. I've also been trying to use a remote jupyter kernel with a local notebook instance so I can utilize copilot as part of my authoring process. I think I'll be experimenting with nbdev next. 

Anyway this is a quick dump of some techniques and experiments I've been trying. I'll write later to see how they are working.