# Crash Course in Statistics for Machine Learning
You do not need to know statistics before you can start learning and applying machine learning. You can start today.

Nevertheless, knowing some statistics can be very helpful to understand the language used in machine learning. Knowing some statistics will eventually be required when you want to start making strong claims about your results.

In this README, you will discover a few key concepts from statistics that will give you the confidence you need to get started and make progress in machine learning.

## Statistical Inference
There are processes in the real world that we would like to understand.

For example human behaviours like clicking on an add or buy a product.

They are not straightforward to understand. There are complexities and uncertainties. The process has an element of randomness to it (it is stochastic).

We understand these processes by making observation and collecting data. The data is not the process, it is a proxy for the process that gives us something to work with to understand the process.

The methods we use to make observations and collect or sample data also introduce uncertainties into the data. Together with the inherent randomness in the real-world process, we now have two sources of randomness in our data.

Given the data we have collected, we clean it up, create a model and try to say something about the process in the real world.

For example, we may make a prediction or describe the relationships between elements within the process.

This is called statistical inference. We go from a real world stochastic process, collect and model the process in data, and come back to the process in the world and say something about it.

## Statistical Population
Data belongs to a population (N). A data population is all possible observations that could be made. The population is abstract, an ideal.

When you make observations or work with data, you are working with a sample of the population (n).

If you are working on a prediction problem, you are seeking to best leverage n to characterize N so that you minimize the errors in the predictions you make from other n your system will encounter.

You must be careful in your selection and handling of your sample. The size and qualities of the data will affect your ability to effectively characterize the problem, to make predictions or describe the data. The randomness (biases) introduced during the collection of must be considered and even manipulated, managed or corrected.

## Big Data
The promise of big data is that you no longer need to worry about sampling data, that you can work with all the data.

That you are working with N and not n. This is false and dangerous thinking.

You are still working with a sample. You can see how this is the case. For example if you are modeling customer data in a SaaS business, you are working with a sample of the population that found and signed up for the service prior to your modeling. Those caveats bias the data you are working with.

You must be careful to not over generalize your findings, to be cautious about claims beyond that data you have observed. For example, the trends of all users of twitter do not represent the trends of all humans.

In the other direction, big data allows you model each individual entities, such as one customer (n=1), using all data collected on that entity to date. This is a powerful, exciting, and computationally demanding frontier.

## Statistical Models
The world is complicated and we need to simplify it with assumptions in order to understand it.

A model is a simplification of a process in the real world. It will always be wrong, but it might be useful.

A statistical model describes the relationship between data attributes, such as a dependent variable with independent variables.

You can think about your data before hand and propose a model that describes relationships between the data.

You can also run machine learning algorithms that assume a type of model of a specific form will describe the relationship and find the parameters to fit the model to the data. This is where notions of a fit, overfitting and underfitting come from, where the model is too specific or not specific enough in its ability to generalize beyond observed data.

Simpler models are easier to understand and use than more complex models. As such, it is a good idea to start with the simplest models for a problem and increase complexity as you need. For example assume a linear form for your model before considering a non-linear, or a parametric before a non-parametric model.

## Summary
In this README, you took a brief crash course in key concepts in statistics that you need when getting started in machine learning.

Specifically, the ideas of statistical inference, statistical populations, how ideas from big data fit in, and statistical models.

Take it slow, statistics is a big field and you do not need to know it all.

Don’t rush out and purchase an undergraduate textbook on statistics, at least, not yet. It is too much, too soon.
