---
title: Creating a complex pandas dataframe
categories: data science
---

Over the holiday break, I set out to create a customized Python scatter plot function. (Yes, I know that's a weird thing to do over the holidays. I hid the nerd-ness from my family.) While creating this function, I realized that there were cases where the plot and/or legend rendering could go out of whack. This could happen if the feature where I wanted marker size to be represented has a skewed distributions, contain 0, or contain negative values. Therefore, creating a data frame that has features that range in complexity was important for evaluating my custom scatterplot function.

I created a data frame with 1000 samples (AKA as m or training examples). To evaluate the robustness of my scatter plot visualization, I made up features that have regression variables representing nine distributions: uniform, Gaussian, bimodal, lognormal, Poisson, negative binomial, chi-square, a distribution containing a 6 log-order range of numbers, and a left-skewed distribution. There are [tons more](https://en.wikipedia.org/wiki/List_of_probability_distributions) probability distributions than I ever thought, but I settled on using some distributions that were [common](http://blog.cloudera.com/blog/2015/12/common-probability-distributions-the-data-scientists-crib-sheet/) for data science. (That last link also does a nice job explaining the various distributions.) Some of the features will contain values that are positive, negative, fractions, or zero.

I also created features that serve as categorical variables. (I suppose they could also be labels in a supervised machine learning paradigm, but remember this is a figment of my imagination.) One classification feature has 10 unique classes evenly distributed among the 1000 samples (group 1 will be in the first 100 samples, group 2 will be in samples 101-200, etc.). Another feature directly mirrors the Poisson distribution feature so that I have classification feature that can be skewed. While I don't have a Bernoulli distribution in my data frame, here are the heads and tails of tit. (Who's good at nerdy dad jokes? This guy.)

```
df.tail()
```

You can see the code that created this data frame of my imagination, along with the scatter plot function, on my Github page here.
