---
title: "Why the Spiderman meme is relevant to Bayesian adaptive priors and fixed effects"
mathjax: True
toc: true
toc_sticky: true
categories: [data science, statistics]
---

For a while, I've wondered about the different approches for multilevel modeling, also known as mixed effects modeling. My initial understanding is with a Bayesian perspective since I learned about it from Statistical Rethinking. But when hearing others talk about "fixed effects", "varying effects", "random effects", and "mixed effects", I had trouble connecting my own understanding of the concept to theirs. Even more perplexing, I wasn't sure what the *source(s)* of the differences were:
- It it a frequentist vs. Bayesian thing?
- Is it a statistical package thing?
- Is it because there are five different definitions of "fixed and random effects", [infamously observed by Andrew Gelman](https://statmodeling.stat.columbia.edu/2005/01/25/why_i_dont_use/) and why he avoids using those terms?

I decided to take a deep dive to resolve my confusion, with much help from numerous sources. Please check out the [Acknowledgments and references](#acknowledgements-and-references) section!

In this post, I'll be comparing an example of mixed effects modeling across statistical philosophies and across statistical languages. As a bonus, a meme awaits.

| method | approach  |  language | package |
| -- |-- | ------ | ----- |
| 1 | frequentist  |  R | `lme4` | 
| 2 | Bayesian  |  Python | `pymc` | 

Note that the default language in the code blocks is Python. A cell running R will have `%%R` designated at the top. A variable can be inputted (`-i`) or outputted (`-o`) on that same line if it is used between the two languages.

*Special thanks to Patrick Robotham for providing a lot of feedback.*


```python
from aesara import tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymc as pm
import xarray as xr
```


```python
%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(1234)
az.rcParams["stats.hdi_prob"] = 0.95 

def standardize(x):
    x = (x - np.mean(x)) / np.std(x)
    return x
```


```python
# Enable running of R code
%load_ext rpy2.ipython
```


```r
%%R
suppressMessages(library(tidyverse))
suppressMessages(library(lme4))
suppressMessages(library(arm))
suppressMessages(library(merTools))
```

# Create synthetic `cafe` dataset

The dataset I am using is created from a scenario described in Statistical Rethinking.

Here are a few more details of the dataset from Dr. McElreath's book:
> Begin by defining the population of cafés that the robot might visit. This means we’ll define the average wait time in the morning and the afternoon, as well as the correlation between them. These numbers are sufficient to define the average properties of the cafés. Let’s define these properties, then we’ll sample cafés from them.

Nearly all Python code is taken from the [Statistical Rethinking pymc repo](https://github.com/pymc-devs/pymc-resources/blob/main/Rethinking_2/Chp_14.ipynb) with some minor alterations.


```python
a = 3.5  # average morning wait time
b = -1.0  # average difference afternoon wait time
sigma_a = 1.0  # std dev in intercepts
sigma_b = 0.5  # std dev in slopes
rho = -0.7  # correlation between intercepts and slopes

Mu = [a, b]

sigmas = [sigma_a, sigma_b]
Rho = np.matrix([[1, rho], [rho, 1]])
Sigma = np.diag(sigmas) * Rho * np.diag(sigmas)  # covariance matrix

N_cafes = 20
vary_effects = np.random.multivariate_normal(mean=Mu, cov=Sigma, size=N_cafes)
a_cafe = vary_effects[:, 0]
b_cafe = vary_effects[:, 1]
```

Now simulate the observations.


```python
N_visits = 10
afternoon = np.tile([0, 1], N_visits * N_cafes // 2)
cafe_id = np.repeat(np.arange(0, N_cafes), N_visits)

mu = a_cafe[cafe_id] + b_cafe[cafe_id] * afternoon
sigma = 0.5  # std dev within cafes

wait = np.random.normal(loc=mu, scale=sigma, size=N_visits * N_cafes)
df_cafes = pd.DataFrame(dict(cafe=cafe_id, afternoon=afternoon, wait=wait))
```

To get a sense of the data structure we just created, let's take a look at the first and last 5 rows.


```python
df_cafes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cafe</th>
      <th>afternoon</th>
      <th>wait</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2.724888</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1.951626</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>2.488389</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1.188077</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>2.026425</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_cafes.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cafe</th>
      <th>afternoon</th>
      <th>wait</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>195</th>
      <td>19</td>
      <td>1</td>
      <td>3.394933</td>
    </tr>
    <tr>
      <th>196</th>
      <td>19</td>
      <td>0</td>
      <td>4.544430</td>
    </tr>
    <tr>
      <th>197</th>
      <td>19</td>
      <td>1</td>
      <td>2.719524</td>
    </tr>
    <tr>
      <th>198</th>
      <td>19</td>
      <td>0</td>
      <td>3.379111</td>
    </tr>
    <tr>
      <th>199</th>
      <td>19</td>
      <td>1</td>
      <td>2.459750</td>
    </tr>
  </tbody>
</table>
</div>



Note that this dataset is balanced, meaning that each group (cafe) has the same number of observations. Mixed effects / multilevel models shine with unbalanced data where it can leverage partial pooling.

# Visualize data

Let's plot the raw data and see how the effect of afternoon influences wait time. Instead of plotting in order of the arbitrarily named cafes (0 to 19), I'll show in order of increasing average morning wait time so that we can appreciate the variability across the dataset.


```python
df_cafes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cafe</th>
      <th>afternoon</th>
      <th>wait</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2.644592</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2.126485</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>2.596465</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>2.250297</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>3.310709</td>
    </tr>
  </tbody>
</table>
</div>




```r
%%R -i df_cafes

# credit to TJ Mahr for a template of this code
xlab <- "Afternoon"
ylab <- "Wait time"
titlelab <- "Wait times for each cafe (ordered by increasing average time)"

# order by increasing average morning wait time (intercept only)
cafe_ordered_by_avgwaittime <- df_cafes %>%
              filter(afternoon==0) %>%
              group_by(cafe) %>%
              summarize(mean = mean(wait)) %>%
              arrange(mean)

# Turn the gear column from a numeric in a factor with a certain order
df_cafes$cafe <- factor(df_cafes$cafe, levels=cafe_ordered_by_avgwaittime$cafe)

ggplot(df_cafes) + 
  aes(x = afternoon, y = wait) + 
  geom_boxplot(aes(fill=factor(afternoon))) +
  stat_summary(fun.y="mean", geom="line") +
  facet_wrap("cafe") +
  labs(x = xlab, y = ylab, title=titlelab)
```


    
![png](/assets/2022-09-13-mixed_effects_freqvsbayes_cafes_files/2022-09-13-mixed_effects_freqvsbayes_cafes_15_0.png)
    


One pattern is that as we increase morning wait time (e.g. the intercept) the difference in wait time in the afternoon (the slope) gets bigger. In other words, when we simulated this dataset, we included a *co-variance* structure between the intercept and slope. When we develop an inferential model with this data, we want to be able to reveal this co-variance.

# Definitions of mixed effects modeling

## Equation set 1: both fixed and random effects terms in linear model

[Galecki and Burzykowski](https://link.springer.com/book/10.1007/978-1-4614-3900-4), [Wikipedia](https://en.wikipedia.org/wiki/Mixed_model), and [this page from UCLA](https://stats.oarc.ucla.edu/other/mult-pkg/introduction-to-linear-mixed-models/) all describe a linear mixed model with an equation similar to equation 1 below.

*I rely heavily on the UCLA page since it is the one that helped me the most. In fact, if you don't care about how it connects to the Bayesian approach, stop reading this and check that out instead!*

In contrast to the Bayesian set of equations, the fixed effects and random effects are in the same equation here.

$$ \textbf{y} = \textbf{X} \boldsymbol{\beta} + \textbf{Z} \textbf{u} + \boldsymbol{\epsilon}  \tag{1}$$

The left side of the equation $\textbf{y}$ represents all of our observations (or the wait time in the cafe example). The $\boldsymbol{\beta}$ in the first term of the equation represents a vector of coefficients across the population of cafes. These are the fixed effects. The $\textbf{u}$ in the second term of equation 1 represents a matrix of coefficients for *each individual cafe*. These are the random effects. Both $\textbf{X}$ and $\textbf{Z}$ are the design matrix of covariates. Finally, there's a residual error term $\boldsymbol{\epsilon}$.

When relating this equation all back to the cafe dataset we just created, I needed to dig deeper to how terms represented an individual observation versus the group (cafe) level. Doing a dimensional analysis helped.

| Equation 1 variable | Dimensions  |  Effects type | Comment |
| -- |-- | ------ | ----- |
| $\textbf{y}$ | 200 x 1  |  n/a |  This vector represents the wait time for all 200 observations. I'll refer to this as $w_i$ later in equation 2. | 
| $\textbf{X}$ | 200 x 2  |  associated with fixed |  The first column of each observation is 1 since it is multiplied by the intercept term. The second column is $A$, which will be 0 or 1 for `afternoon`. | 
| $\boldsymbol{\beta}$ | 2 x 1 |  fixed | The two elements in the $\boldsymbol{\beta}$ (bold font beta) are what I'll refer to as the intercept $\alpha$ and the slope $\beta$ (unbolded beta) across all cafes in equation 2. | 
| $\textbf{Z}$ | 200 x (2x20)  |  associated with random |  The first 20 columns representing intercepts for each cafe and the second 20 for the covariate (`afternoon`). See visual below.  |
| $\textbf{u}$ | (2x20) x 1  |  random |  $\textbf{u}$ holds each of the 20 cafes' intercept $a_\text{cafe}$ and slope $b_\text{cafe}$. There's an implied correlation structure between them. |
| $\boldsymbol{\epsilon}$| 200 x 1  |  n/a |  Normally distributed residual error. |

To better understand what $\textbf{Z}$ looks like we can create an alternate representation of `df_cafes`. Each row of the matrix $\textbf{Z}$ is for an individual observation. The first 20 columns of a row are the 20 intercepts of a cafe (column 1 is cafe 1, column 2 is cafe 2, etc.) All of the first 20 columns will contain a 0 *except* for the column that represents the cafe that observation is associated with which will be a 1. The next 20 columns (columns 21-40) will represent `afternoon`. All of this second group of columns will be 0 *except* for the column that represents the cafe that observation is associated with *and* if the observation is associated with an afternon observation.

To be clear, the structure of `df_cafes`, where each row is an observation with the cafe, afternoon status, and wait time, is already in a form to be understood by the `lmer` and `pymc` packages. What I'm showing below is to help understand what the matrix $\textbf{Z}$ looks like in the above equations.


```python
Z = np.zeros((200, 40))
for i in df_cafes.index:
    cafe = df_cafes.loc[i, 'cafe']
    afternoon = df_cafes.loc[i, 'afternoon']
    Z[i, cafe] = 1
    Z[i, 20+cafe] = afternoon
```

We can take a look at the first 12 rows of Z. The first 10 are for the first cafe and observations alternate morning and afternoon, hence what's displayed in column 20. I included the first two rows of the second cafe to show how the `1` moves over a row after the first 10 rows. I'll use `pandas` to better display the values.


```python
pd.set_option('display.max_columns', 40)
(
    pd.DataFrame(Z[0:12, :])
    .astype(int)
    .style
    .highlight_max(axis=1, props='color:navy; background-color:yellow;')
    .highlight_min(axis=1, props='color:white; background-color:#3E0B51;')
)
```




<style type="text/css">
#T_7cfc2_row0_col0, #T_7cfc2_row1_col0, #T_7cfc2_row1_col20, #T_7cfc2_row2_col0, #T_7cfc2_row3_col0, #T_7cfc2_row3_col20, #T_7cfc2_row4_col0, #T_7cfc2_row5_col0, #T_7cfc2_row5_col20, #T_7cfc2_row6_col0, #T_7cfc2_row7_col0, #T_7cfc2_row7_col20, #T_7cfc2_row8_col0, #T_7cfc2_row9_col0, #T_7cfc2_row9_col20, #T_7cfc2_row10_col1, #T_7cfc2_row11_col1, #T_7cfc2_row11_col21 {
  color: navy;
  background-color: yellow;
}
#T_7cfc2_row0_col1, #T_7cfc2_row0_col2, #T_7cfc2_row0_col3, #T_7cfc2_row0_col4, #T_7cfc2_row0_col5, #T_7cfc2_row0_col6, #T_7cfc2_row0_col7, #T_7cfc2_row0_col8, #T_7cfc2_row0_col9, #T_7cfc2_row0_col10, #T_7cfc2_row0_col11, #T_7cfc2_row0_col12, #T_7cfc2_row0_col13, #T_7cfc2_row0_col14, #T_7cfc2_row0_col15, #T_7cfc2_row0_col16, #T_7cfc2_row0_col17, #T_7cfc2_row0_col18, #T_7cfc2_row0_col19, #T_7cfc2_row0_col20, #T_7cfc2_row0_col21, #T_7cfc2_row0_col22, #T_7cfc2_row0_col23, #T_7cfc2_row0_col24, #T_7cfc2_row0_col25, #T_7cfc2_row0_col26, #T_7cfc2_row0_col27, #T_7cfc2_row0_col28, #T_7cfc2_row0_col29, #T_7cfc2_row0_col30, #T_7cfc2_row0_col31, #T_7cfc2_row0_col32, #T_7cfc2_row0_col33, #T_7cfc2_row0_col34, #T_7cfc2_row0_col35, #T_7cfc2_row0_col36, #T_7cfc2_row0_col37, #T_7cfc2_row0_col38, #T_7cfc2_row0_col39, #T_7cfc2_row1_col1, #T_7cfc2_row1_col2, #T_7cfc2_row1_col3, #T_7cfc2_row1_col4, #T_7cfc2_row1_col5, #T_7cfc2_row1_col6, #T_7cfc2_row1_col7, #T_7cfc2_row1_col8, #T_7cfc2_row1_col9, #T_7cfc2_row1_col10, #T_7cfc2_row1_col11, #T_7cfc2_row1_col12, #T_7cfc2_row1_col13, #T_7cfc2_row1_col14, #T_7cfc2_row1_col15, #T_7cfc2_row1_col16, #T_7cfc2_row1_col17, #T_7cfc2_row1_col18, #T_7cfc2_row1_col19, #T_7cfc2_row1_col21, #T_7cfc2_row1_col22, #T_7cfc2_row1_col23, #T_7cfc2_row1_col24, #T_7cfc2_row1_col25, #T_7cfc2_row1_col26, #T_7cfc2_row1_col27, #T_7cfc2_row1_col28, #T_7cfc2_row1_col29, #T_7cfc2_row1_col30, #T_7cfc2_row1_col31, #T_7cfc2_row1_col32, #T_7cfc2_row1_col33, #T_7cfc2_row1_col34, #T_7cfc2_row1_col35, #T_7cfc2_row1_col36, #T_7cfc2_row1_col37, #T_7cfc2_row1_col38, #T_7cfc2_row1_col39, #T_7cfc2_row2_col1, #T_7cfc2_row2_col2, #T_7cfc2_row2_col3, #T_7cfc2_row2_col4, #T_7cfc2_row2_col5, #T_7cfc2_row2_col6, #T_7cfc2_row2_col7, #T_7cfc2_row2_col8, #T_7cfc2_row2_col9, #T_7cfc2_row2_col10, #T_7cfc2_row2_col11, #T_7cfc2_row2_col12, #T_7cfc2_row2_col13, #T_7cfc2_row2_col14, #T_7cfc2_row2_col15, #T_7cfc2_row2_col16, #T_7cfc2_row2_col17, #T_7cfc2_row2_col18, #T_7cfc2_row2_col19, #T_7cfc2_row2_col20, #T_7cfc2_row2_col21, #T_7cfc2_row2_col22, #T_7cfc2_row2_col23, #T_7cfc2_row2_col24, #T_7cfc2_row2_col25, #T_7cfc2_row2_col26, #T_7cfc2_row2_col27, #T_7cfc2_row2_col28, #T_7cfc2_row2_col29, #T_7cfc2_row2_col30, #T_7cfc2_row2_col31, #T_7cfc2_row2_col32, #T_7cfc2_row2_col33, #T_7cfc2_row2_col34, #T_7cfc2_row2_col35, #T_7cfc2_row2_col36, #T_7cfc2_row2_col37, #T_7cfc2_row2_col38, #T_7cfc2_row2_col39, #T_7cfc2_row3_col1, #T_7cfc2_row3_col2, #T_7cfc2_row3_col3, #T_7cfc2_row3_col4, #T_7cfc2_row3_col5, #T_7cfc2_row3_col6, #T_7cfc2_row3_col7, #T_7cfc2_row3_col8, #T_7cfc2_row3_col9, #T_7cfc2_row3_col10, #T_7cfc2_row3_col11, #T_7cfc2_row3_col12, #T_7cfc2_row3_col13, #T_7cfc2_row3_col14, #T_7cfc2_row3_col15, #T_7cfc2_row3_col16, #T_7cfc2_row3_col17, #T_7cfc2_row3_col18, #T_7cfc2_row3_col19, #T_7cfc2_row3_col21, #T_7cfc2_row3_col22, #T_7cfc2_row3_col23, #T_7cfc2_row3_col24, #T_7cfc2_row3_col25, #T_7cfc2_row3_col26, #T_7cfc2_row3_col27, #T_7cfc2_row3_col28, #T_7cfc2_row3_col29, #T_7cfc2_row3_col30, #T_7cfc2_row3_col31, #T_7cfc2_row3_col32, #T_7cfc2_row3_col33, #T_7cfc2_row3_col34, #T_7cfc2_row3_col35, #T_7cfc2_row3_col36, #T_7cfc2_row3_col37, #T_7cfc2_row3_col38, #T_7cfc2_row3_col39, #T_7cfc2_row4_col1, #T_7cfc2_row4_col2, #T_7cfc2_row4_col3, #T_7cfc2_row4_col4, #T_7cfc2_row4_col5, #T_7cfc2_row4_col6, #T_7cfc2_row4_col7, #T_7cfc2_row4_col8, #T_7cfc2_row4_col9, #T_7cfc2_row4_col10, #T_7cfc2_row4_col11, #T_7cfc2_row4_col12, #T_7cfc2_row4_col13, #T_7cfc2_row4_col14, #T_7cfc2_row4_col15, #T_7cfc2_row4_col16, #T_7cfc2_row4_col17, #T_7cfc2_row4_col18, #T_7cfc2_row4_col19, #T_7cfc2_row4_col20, #T_7cfc2_row4_col21, #T_7cfc2_row4_col22, #T_7cfc2_row4_col23, #T_7cfc2_row4_col24, #T_7cfc2_row4_col25, #T_7cfc2_row4_col26, #T_7cfc2_row4_col27, #T_7cfc2_row4_col28, #T_7cfc2_row4_col29, #T_7cfc2_row4_col30, #T_7cfc2_row4_col31, #T_7cfc2_row4_col32, #T_7cfc2_row4_col33, #T_7cfc2_row4_col34, #T_7cfc2_row4_col35, #T_7cfc2_row4_col36, #T_7cfc2_row4_col37, #T_7cfc2_row4_col38, #T_7cfc2_row4_col39, #T_7cfc2_row5_col1, #T_7cfc2_row5_col2, #T_7cfc2_row5_col3, #T_7cfc2_row5_col4, #T_7cfc2_row5_col5, #T_7cfc2_row5_col6, #T_7cfc2_row5_col7, #T_7cfc2_row5_col8, #T_7cfc2_row5_col9, #T_7cfc2_row5_col10, #T_7cfc2_row5_col11, #T_7cfc2_row5_col12, #T_7cfc2_row5_col13, #T_7cfc2_row5_col14, #T_7cfc2_row5_col15, #T_7cfc2_row5_col16, #T_7cfc2_row5_col17, #T_7cfc2_row5_col18, #T_7cfc2_row5_col19, #T_7cfc2_row5_col21, #T_7cfc2_row5_col22, #T_7cfc2_row5_col23, #T_7cfc2_row5_col24, #T_7cfc2_row5_col25, #T_7cfc2_row5_col26, #T_7cfc2_row5_col27, #T_7cfc2_row5_col28, #T_7cfc2_row5_col29, #T_7cfc2_row5_col30, #T_7cfc2_row5_col31, #T_7cfc2_row5_col32, #T_7cfc2_row5_col33, #T_7cfc2_row5_col34, #T_7cfc2_row5_col35, #T_7cfc2_row5_col36, #T_7cfc2_row5_col37, #T_7cfc2_row5_col38, #T_7cfc2_row5_col39, #T_7cfc2_row6_col1, #T_7cfc2_row6_col2, #T_7cfc2_row6_col3, #T_7cfc2_row6_col4, #T_7cfc2_row6_col5, #T_7cfc2_row6_col6, #T_7cfc2_row6_col7, #T_7cfc2_row6_col8, #T_7cfc2_row6_col9, #T_7cfc2_row6_col10, #T_7cfc2_row6_col11, #T_7cfc2_row6_col12, #T_7cfc2_row6_col13, #T_7cfc2_row6_col14, #T_7cfc2_row6_col15, #T_7cfc2_row6_col16, #T_7cfc2_row6_col17, #T_7cfc2_row6_col18, #T_7cfc2_row6_col19, #T_7cfc2_row6_col20, #T_7cfc2_row6_col21, #T_7cfc2_row6_col22, #T_7cfc2_row6_col23, #T_7cfc2_row6_col24, #T_7cfc2_row6_col25, #T_7cfc2_row6_col26, #T_7cfc2_row6_col27, #T_7cfc2_row6_col28, #T_7cfc2_row6_col29, #T_7cfc2_row6_col30, #T_7cfc2_row6_col31, #T_7cfc2_row6_col32, #T_7cfc2_row6_col33, #T_7cfc2_row6_col34, #T_7cfc2_row6_col35, #T_7cfc2_row6_col36, #T_7cfc2_row6_col37, #T_7cfc2_row6_col38, #T_7cfc2_row6_col39, #T_7cfc2_row7_col1, #T_7cfc2_row7_col2, #T_7cfc2_row7_col3, #T_7cfc2_row7_col4, #T_7cfc2_row7_col5, #T_7cfc2_row7_col6, #T_7cfc2_row7_col7, #T_7cfc2_row7_col8, #T_7cfc2_row7_col9, #T_7cfc2_row7_col10, #T_7cfc2_row7_col11, #T_7cfc2_row7_col12, #T_7cfc2_row7_col13, #T_7cfc2_row7_col14, #T_7cfc2_row7_col15, #T_7cfc2_row7_col16, #T_7cfc2_row7_col17, #T_7cfc2_row7_col18, #T_7cfc2_row7_col19, #T_7cfc2_row7_col21, #T_7cfc2_row7_col22, #T_7cfc2_row7_col23, #T_7cfc2_row7_col24, #T_7cfc2_row7_col25, #T_7cfc2_row7_col26, #T_7cfc2_row7_col27, #T_7cfc2_row7_col28, #T_7cfc2_row7_col29, #T_7cfc2_row7_col30, #T_7cfc2_row7_col31, #T_7cfc2_row7_col32, #T_7cfc2_row7_col33, #T_7cfc2_row7_col34, #T_7cfc2_row7_col35, #T_7cfc2_row7_col36, #T_7cfc2_row7_col37, #T_7cfc2_row7_col38, #T_7cfc2_row7_col39, #T_7cfc2_row8_col1, #T_7cfc2_row8_col2, #T_7cfc2_row8_col3, #T_7cfc2_row8_col4, #T_7cfc2_row8_col5, #T_7cfc2_row8_col6, #T_7cfc2_row8_col7, #T_7cfc2_row8_col8, #T_7cfc2_row8_col9, #T_7cfc2_row8_col10, #T_7cfc2_row8_col11, #T_7cfc2_row8_col12, #T_7cfc2_row8_col13, #T_7cfc2_row8_col14, #T_7cfc2_row8_col15, #T_7cfc2_row8_col16, #T_7cfc2_row8_col17, #T_7cfc2_row8_col18, #T_7cfc2_row8_col19, #T_7cfc2_row8_col20, #T_7cfc2_row8_col21, #T_7cfc2_row8_col22, #T_7cfc2_row8_col23, #T_7cfc2_row8_col24, #T_7cfc2_row8_col25, #T_7cfc2_row8_col26, #T_7cfc2_row8_col27, #T_7cfc2_row8_col28, #T_7cfc2_row8_col29, #T_7cfc2_row8_col30, #T_7cfc2_row8_col31, #T_7cfc2_row8_col32, #T_7cfc2_row8_col33, #T_7cfc2_row8_col34, #T_7cfc2_row8_col35, #T_7cfc2_row8_col36, #T_7cfc2_row8_col37, #T_7cfc2_row8_col38, #T_7cfc2_row8_col39, #T_7cfc2_row9_col1, #T_7cfc2_row9_col2, #T_7cfc2_row9_col3, #T_7cfc2_row9_col4, #T_7cfc2_row9_col5, #T_7cfc2_row9_col6, #T_7cfc2_row9_col7, #T_7cfc2_row9_col8, #T_7cfc2_row9_col9, #T_7cfc2_row9_col10, #T_7cfc2_row9_col11, #T_7cfc2_row9_col12, #T_7cfc2_row9_col13, #T_7cfc2_row9_col14, #T_7cfc2_row9_col15, #T_7cfc2_row9_col16, #T_7cfc2_row9_col17, #T_7cfc2_row9_col18, #T_7cfc2_row9_col19, #T_7cfc2_row9_col21, #T_7cfc2_row9_col22, #T_7cfc2_row9_col23, #T_7cfc2_row9_col24, #T_7cfc2_row9_col25, #T_7cfc2_row9_col26, #T_7cfc2_row9_col27, #T_7cfc2_row9_col28, #T_7cfc2_row9_col29, #T_7cfc2_row9_col30, #T_7cfc2_row9_col31, #T_7cfc2_row9_col32, #T_7cfc2_row9_col33, #T_7cfc2_row9_col34, #T_7cfc2_row9_col35, #T_7cfc2_row9_col36, #T_7cfc2_row9_col37, #T_7cfc2_row9_col38, #T_7cfc2_row9_col39, #T_7cfc2_row10_col0, #T_7cfc2_row10_col2, #T_7cfc2_row10_col3, #T_7cfc2_row10_col4, #T_7cfc2_row10_col5, #T_7cfc2_row10_col6, #T_7cfc2_row10_col7, #T_7cfc2_row10_col8, #T_7cfc2_row10_col9, #T_7cfc2_row10_col10, #T_7cfc2_row10_col11, #T_7cfc2_row10_col12, #T_7cfc2_row10_col13, #T_7cfc2_row10_col14, #T_7cfc2_row10_col15, #T_7cfc2_row10_col16, #T_7cfc2_row10_col17, #T_7cfc2_row10_col18, #T_7cfc2_row10_col19, #T_7cfc2_row10_col20, #T_7cfc2_row10_col21, #T_7cfc2_row10_col22, #T_7cfc2_row10_col23, #T_7cfc2_row10_col24, #T_7cfc2_row10_col25, #T_7cfc2_row10_col26, #T_7cfc2_row10_col27, #T_7cfc2_row10_col28, #T_7cfc2_row10_col29, #T_7cfc2_row10_col30, #T_7cfc2_row10_col31, #T_7cfc2_row10_col32, #T_7cfc2_row10_col33, #T_7cfc2_row10_col34, #T_7cfc2_row10_col35, #T_7cfc2_row10_col36, #T_7cfc2_row10_col37, #T_7cfc2_row10_col38, #T_7cfc2_row10_col39, #T_7cfc2_row11_col0, #T_7cfc2_row11_col2, #T_7cfc2_row11_col3, #T_7cfc2_row11_col4, #T_7cfc2_row11_col5, #T_7cfc2_row11_col6, #T_7cfc2_row11_col7, #T_7cfc2_row11_col8, #T_7cfc2_row11_col9, #T_7cfc2_row11_col10, #T_7cfc2_row11_col11, #T_7cfc2_row11_col12, #T_7cfc2_row11_col13, #T_7cfc2_row11_col14, #T_7cfc2_row11_col15, #T_7cfc2_row11_col16, #T_7cfc2_row11_col17, #T_7cfc2_row11_col18, #T_7cfc2_row11_col19, #T_7cfc2_row11_col20, #T_7cfc2_row11_col22, #T_7cfc2_row11_col23, #T_7cfc2_row11_col24, #T_7cfc2_row11_col25, #T_7cfc2_row11_col26, #T_7cfc2_row11_col27, #T_7cfc2_row11_col28, #T_7cfc2_row11_col29, #T_7cfc2_row11_col30, #T_7cfc2_row11_col31, #T_7cfc2_row11_col32, #T_7cfc2_row11_col33, #T_7cfc2_row11_col34, #T_7cfc2_row11_col35, #T_7cfc2_row11_col36, #T_7cfc2_row11_col37, #T_7cfc2_row11_col38, #T_7cfc2_row11_col39 {
  color: white;
  background-color: #3E0B51;
}
</style>
<table id="T_7cfc2">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_7cfc2_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_7cfc2_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_7cfc2_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_7cfc2_level0_col3" class="col_heading level0 col3" >3</th>
      <th id="T_7cfc2_level0_col4" class="col_heading level0 col4" >4</th>
      <th id="T_7cfc2_level0_col5" class="col_heading level0 col5" >5</th>
      <th id="T_7cfc2_level0_col6" class="col_heading level0 col6" >6</th>
      <th id="T_7cfc2_level0_col7" class="col_heading level0 col7" >7</th>
      <th id="T_7cfc2_level0_col8" class="col_heading level0 col8" >8</th>
      <th id="T_7cfc2_level0_col9" class="col_heading level0 col9" >9</th>
      <th id="T_7cfc2_level0_col10" class="col_heading level0 col10" >10</th>
      <th id="T_7cfc2_level0_col11" class="col_heading level0 col11" >11</th>
      <th id="T_7cfc2_level0_col12" class="col_heading level0 col12" >12</th>
      <th id="T_7cfc2_level0_col13" class="col_heading level0 col13" >13</th>
      <th id="T_7cfc2_level0_col14" class="col_heading level0 col14" >14</th>
      <th id="T_7cfc2_level0_col15" class="col_heading level0 col15" >15</th>
      <th id="T_7cfc2_level0_col16" class="col_heading level0 col16" >16</th>
      <th id="T_7cfc2_level0_col17" class="col_heading level0 col17" >17</th>
      <th id="T_7cfc2_level0_col18" class="col_heading level0 col18" >18</th>
      <th id="T_7cfc2_level0_col19" class="col_heading level0 col19" >19</th>
      <th id="T_7cfc2_level0_col20" class="col_heading level0 col20" >20</th>
      <th id="T_7cfc2_level0_col21" class="col_heading level0 col21" >21</th>
      <th id="T_7cfc2_level0_col22" class="col_heading level0 col22" >22</th>
      <th id="T_7cfc2_level0_col23" class="col_heading level0 col23" >23</th>
      <th id="T_7cfc2_level0_col24" class="col_heading level0 col24" >24</th>
      <th id="T_7cfc2_level0_col25" class="col_heading level0 col25" >25</th>
      <th id="T_7cfc2_level0_col26" class="col_heading level0 col26" >26</th>
      <th id="T_7cfc2_level0_col27" class="col_heading level0 col27" >27</th>
      <th id="T_7cfc2_level0_col28" class="col_heading level0 col28" >28</th>
      <th id="T_7cfc2_level0_col29" class="col_heading level0 col29" >29</th>
      <th id="T_7cfc2_level0_col30" class="col_heading level0 col30" >30</th>
      <th id="T_7cfc2_level0_col31" class="col_heading level0 col31" >31</th>
      <th id="T_7cfc2_level0_col32" class="col_heading level0 col32" >32</th>
      <th id="T_7cfc2_level0_col33" class="col_heading level0 col33" >33</th>
      <th id="T_7cfc2_level0_col34" class="col_heading level0 col34" >34</th>
      <th id="T_7cfc2_level0_col35" class="col_heading level0 col35" >35</th>
      <th id="T_7cfc2_level0_col36" class="col_heading level0 col36" >36</th>
      <th id="T_7cfc2_level0_col37" class="col_heading level0 col37" >37</th>
      <th id="T_7cfc2_level0_col38" class="col_heading level0 col38" >38</th>
      <th id="T_7cfc2_level0_col39" class="col_heading level0 col39" >39</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_7cfc2_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_7cfc2_row0_col0" class="data row0 col0" >1</td>
      <td id="T_7cfc2_row0_col1" class="data row0 col1" >0</td>
      <td id="T_7cfc2_row0_col2" class="data row0 col2" >0</td>
      <td id="T_7cfc2_row0_col3" class="data row0 col3" >0</td>
      <td id="T_7cfc2_row0_col4" class="data row0 col4" >0</td>
      <td id="T_7cfc2_row0_col5" class="data row0 col5" >0</td>
      <td id="T_7cfc2_row0_col6" class="data row0 col6" >0</td>
      <td id="T_7cfc2_row0_col7" class="data row0 col7" >0</td>
      <td id="T_7cfc2_row0_col8" class="data row0 col8" >0</td>
      <td id="T_7cfc2_row0_col9" class="data row0 col9" >0</td>
      <td id="T_7cfc2_row0_col10" class="data row0 col10" >0</td>
      <td id="T_7cfc2_row0_col11" class="data row0 col11" >0</td>
      <td id="T_7cfc2_row0_col12" class="data row0 col12" >0</td>
      <td id="T_7cfc2_row0_col13" class="data row0 col13" >0</td>
      <td id="T_7cfc2_row0_col14" class="data row0 col14" >0</td>
      <td id="T_7cfc2_row0_col15" class="data row0 col15" >0</td>
      <td id="T_7cfc2_row0_col16" class="data row0 col16" >0</td>
      <td id="T_7cfc2_row0_col17" class="data row0 col17" >0</td>
      <td id="T_7cfc2_row0_col18" class="data row0 col18" >0</td>
      <td id="T_7cfc2_row0_col19" class="data row0 col19" >0</td>
      <td id="T_7cfc2_row0_col20" class="data row0 col20" >0</td>
      <td id="T_7cfc2_row0_col21" class="data row0 col21" >0</td>
      <td id="T_7cfc2_row0_col22" class="data row0 col22" >0</td>
      <td id="T_7cfc2_row0_col23" class="data row0 col23" >0</td>
      <td id="T_7cfc2_row0_col24" class="data row0 col24" >0</td>
      <td id="T_7cfc2_row0_col25" class="data row0 col25" >0</td>
      <td id="T_7cfc2_row0_col26" class="data row0 col26" >0</td>
      <td id="T_7cfc2_row0_col27" class="data row0 col27" >0</td>
      <td id="T_7cfc2_row0_col28" class="data row0 col28" >0</td>
      <td id="T_7cfc2_row0_col29" class="data row0 col29" >0</td>
      <td id="T_7cfc2_row0_col30" class="data row0 col30" >0</td>
      <td id="T_7cfc2_row0_col31" class="data row0 col31" >0</td>
      <td id="T_7cfc2_row0_col32" class="data row0 col32" >0</td>
      <td id="T_7cfc2_row0_col33" class="data row0 col33" >0</td>
      <td id="T_7cfc2_row0_col34" class="data row0 col34" >0</td>
      <td id="T_7cfc2_row0_col35" class="data row0 col35" >0</td>
      <td id="T_7cfc2_row0_col36" class="data row0 col36" >0</td>
      <td id="T_7cfc2_row0_col37" class="data row0 col37" >0</td>
      <td id="T_7cfc2_row0_col38" class="data row0 col38" >0</td>
      <td id="T_7cfc2_row0_col39" class="data row0 col39" >0</td>
    </tr>
    <tr>
      <th id="T_7cfc2_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_7cfc2_row1_col0" class="data row1 col0" >1</td>
      <td id="T_7cfc2_row1_col1" class="data row1 col1" >0</td>
      <td id="T_7cfc2_row1_col2" class="data row1 col2" >0</td>
      <td id="T_7cfc2_row1_col3" class="data row1 col3" >0</td>
      <td id="T_7cfc2_row1_col4" class="data row1 col4" >0</td>
      <td id="T_7cfc2_row1_col5" class="data row1 col5" >0</td>
      <td id="T_7cfc2_row1_col6" class="data row1 col6" >0</td>
      <td id="T_7cfc2_row1_col7" class="data row1 col7" >0</td>
      <td id="T_7cfc2_row1_col8" class="data row1 col8" >0</td>
      <td id="T_7cfc2_row1_col9" class="data row1 col9" >0</td>
      <td id="T_7cfc2_row1_col10" class="data row1 col10" >0</td>
      <td id="T_7cfc2_row1_col11" class="data row1 col11" >0</td>
      <td id="T_7cfc2_row1_col12" class="data row1 col12" >0</td>
      <td id="T_7cfc2_row1_col13" class="data row1 col13" >0</td>
      <td id="T_7cfc2_row1_col14" class="data row1 col14" >0</td>
      <td id="T_7cfc2_row1_col15" class="data row1 col15" >0</td>
      <td id="T_7cfc2_row1_col16" class="data row1 col16" >0</td>
      <td id="T_7cfc2_row1_col17" class="data row1 col17" >0</td>
      <td id="T_7cfc2_row1_col18" class="data row1 col18" >0</td>
      <td id="T_7cfc2_row1_col19" class="data row1 col19" >0</td>
      <td id="T_7cfc2_row1_col20" class="data row1 col20" >1</td>
      <td id="T_7cfc2_row1_col21" class="data row1 col21" >0</td>
      <td id="T_7cfc2_row1_col22" class="data row1 col22" >0</td>
      <td id="T_7cfc2_row1_col23" class="data row1 col23" >0</td>
      <td id="T_7cfc2_row1_col24" class="data row1 col24" >0</td>
      <td id="T_7cfc2_row1_col25" class="data row1 col25" >0</td>
      <td id="T_7cfc2_row1_col26" class="data row1 col26" >0</td>
      <td id="T_7cfc2_row1_col27" class="data row1 col27" >0</td>
      <td id="T_7cfc2_row1_col28" class="data row1 col28" >0</td>
      <td id="T_7cfc2_row1_col29" class="data row1 col29" >0</td>
      <td id="T_7cfc2_row1_col30" class="data row1 col30" >0</td>
      <td id="T_7cfc2_row1_col31" class="data row1 col31" >0</td>
      <td id="T_7cfc2_row1_col32" class="data row1 col32" >0</td>
      <td id="T_7cfc2_row1_col33" class="data row1 col33" >0</td>
      <td id="T_7cfc2_row1_col34" class="data row1 col34" >0</td>
      <td id="T_7cfc2_row1_col35" class="data row1 col35" >0</td>
      <td id="T_7cfc2_row1_col36" class="data row1 col36" >0</td>
      <td id="T_7cfc2_row1_col37" class="data row1 col37" >0</td>
      <td id="T_7cfc2_row1_col38" class="data row1 col38" >0</td>
      <td id="T_7cfc2_row1_col39" class="data row1 col39" >0</td>
    </tr>
    <tr>
      <th id="T_7cfc2_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_7cfc2_row2_col0" class="data row2 col0" >1</td>
      <td id="T_7cfc2_row2_col1" class="data row2 col1" >0</td>
      <td id="T_7cfc2_row2_col2" class="data row2 col2" >0</td>
      <td id="T_7cfc2_row2_col3" class="data row2 col3" >0</td>
      <td id="T_7cfc2_row2_col4" class="data row2 col4" >0</td>
      <td id="T_7cfc2_row2_col5" class="data row2 col5" >0</td>
      <td id="T_7cfc2_row2_col6" class="data row2 col6" >0</td>
      <td id="T_7cfc2_row2_col7" class="data row2 col7" >0</td>
      <td id="T_7cfc2_row2_col8" class="data row2 col8" >0</td>
      <td id="T_7cfc2_row2_col9" class="data row2 col9" >0</td>
      <td id="T_7cfc2_row2_col10" class="data row2 col10" >0</td>
      <td id="T_7cfc2_row2_col11" class="data row2 col11" >0</td>
      <td id="T_7cfc2_row2_col12" class="data row2 col12" >0</td>
      <td id="T_7cfc2_row2_col13" class="data row2 col13" >0</td>
      <td id="T_7cfc2_row2_col14" class="data row2 col14" >0</td>
      <td id="T_7cfc2_row2_col15" class="data row2 col15" >0</td>
      <td id="T_7cfc2_row2_col16" class="data row2 col16" >0</td>
      <td id="T_7cfc2_row2_col17" class="data row2 col17" >0</td>
      <td id="T_7cfc2_row2_col18" class="data row2 col18" >0</td>
      <td id="T_7cfc2_row2_col19" class="data row2 col19" >0</td>
      <td id="T_7cfc2_row2_col20" class="data row2 col20" >0</td>
      <td id="T_7cfc2_row2_col21" class="data row2 col21" >0</td>
      <td id="T_7cfc2_row2_col22" class="data row2 col22" >0</td>
      <td id="T_7cfc2_row2_col23" class="data row2 col23" >0</td>
      <td id="T_7cfc2_row2_col24" class="data row2 col24" >0</td>
      <td id="T_7cfc2_row2_col25" class="data row2 col25" >0</td>
      <td id="T_7cfc2_row2_col26" class="data row2 col26" >0</td>
      <td id="T_7cfc2_row2_col27" class="data row2 col27" >0</td>
      <td id="T_7cfc2_row2_col28" class="data row2 col28" >0</td>
      <td id="T_7cfc2_row2_col29" class="data row2 col29" >0</td>
      <td id="T_7cfc2_row2_col30" class="data row2 col30" >0</td>
      <td id="T_7cfc2_row2_col31" class="data row2 col31" >0</td>
      <td id="T_7cfc2_row2_col32" class="data row2 col32" >0</td>
      <td id="T_7cfc2_row2_col33" class="data row2 col33" >0</td>
      <td id="T_7cfc2_row2_col34" class="data row2 col34" >0</td>
      <td id="T_7cfc2_row2_col35" class="data row2 col35" >0</td>
      <td id="T_7cfc2_row2_col36" class="data row2 col36" >0</td>
      <td id="T_7cfc2_row2_col37" class="data row2 col37" >0</td>
      <td id="T_7cfc2_row2_col38" class="data row2 col38" >0</td>
      <td id="T_7cfc2_row2_col39" class="data row2 col39" >0</td>
    </tr>
    <tr>
      <th id="T_7cfc2_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_7cfc2_row3_col0" class="data row3 col0" >1</td>
      <td id="T_7cfc2_row3_col1" class="data row3 col1" >0</td>
      <td id="T_7cfc2_row3_col2" class="data row3 col2" >0</td>
      <td id="T_7cfc2_row3_col3" class="data row3 col3" >0</td>
      <td id="T_7cfc2_row3_col4" class="data row3 col4" >0</td>
      <td id="T_7cfc2_row3_col5" class="data row3 col5" >0</td>
      <td id="T_7cfc2_row3_col6" class="data row3 col6" >0</td>
      <td id="T_7cfc2_row3_col7" class="data row3 col7" >0</td>
      <td id="T_7cfc2_row3_col8" class="data row3 col8" >0</td>
      <td id="T_7cfc2_row3_col9" class="data row3 col9" >0</td>
      <td id="T_7cfc2_row3_col10" class="data row3 col10" >0</td>
      <td id="T_7cfc2_row3_col11" class="data row3 col11" >0</td>
      <td id="T_7cfc2_row3_col12" class="data row3 col12" >0</td>
      <td id="T_7cfc2_row3_col13" class="data row3 col13" >0</td>
      <td id="T_7cfc2_row3_col14" class="data row3 col14" >0</td>
      <td id="T_7cfc2_row3_col15" class="data row3 col15" >0</td>
      <td id="T_7cfc2_row3_col16" class="data row3 col16" >0</td>
      <td id="T_7cfc2_row3_col17" class="data row3 col17" >0</td>
      <td id="T_7cfc2_row3_col18" class="data row3 col18" >0</td>
      <td id="T_7cfc2_row3_col19" class="data row3 col19" >0</td>
      <td id="T_7cfc2_row3_col20" class="data row3 col20" >1</td>
      <td id="T_7cfc2_row3_col21" class="data row3 col21" >0</td>
      <td id="T_7cfc2_row3_col22" class="data row3 col22" >0</td>
      <td id="T_7cfc2_row3_col23" class="data row3 col23" >0</td>
      <td id="T_7cfc2_row3_col24" class="data row3 col24" >0</td>
      <td id="T_7cfc2_row3_col25" class="data row3 col25" >0</td>
      <td id="T_7cfc2_row3_col26" class="data row3 col26" >0</td>
      <td id="T_7cfc2_row3_col27" class="data row3 col27" >0</td>
      <td id="T_7cfc2_row3_col28" class="data row3 col28" >0</td>
      <td id="T_7cfc2_row3_col29" class="data row3 col29" >0</td>
      <td id="T_7cfc2_row3_col30" class="data row3 col30" >0</td>
      <td id="T_7cfc2_row3_col31" class="data row3 col31" >0</td>
      <td id="T_7cfc2_row3_col32" class="data row3 col32" >0</td>
      <td id="T_7cfc2_row3_col33" class="data row3 col33" >0</td>
      <td id="T_7cfc2_row3_col34" class="data row3 col34" >0</td>
      <td id="T_7cfc2_row3_col35" class="data row3 col35" >0</td>
      <td id="T_7cfc2_row3_col36" class="data row3 col36" >0</td>
      <td id="T_7cfc2_row3_col37" class="data row3 col37" >0</td>
      <td id="T_7cfc2_row3_col38" class="data row3 col38" >0</td>
      <td id="T_7cfc2_row3_col39" class="data row3 col39" >0</td>
    </tr>
    <tr>
      <th id="T_7cfc2_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_7cfc2_row4_col0" class="data row4 col0" >1</td>
      <td id="T_7cfc2_row4_col1" class="data row4 col1" >0</td>
      <td id="T_7cfc2_row4_col2" class="data row4 col2" >0</td>
      <td id="T_7cfc2_row4_col3" class="data row4 col3" >0</td>
      <td id="T_7cfc2_row4_col4" class="data row4 col4" >0</td>
      <td id="T_7cfc2_row4_col5" class="data row4 col5" >0</td>
      <td id="T_7cfc2_row4_col6" class="data row4 col6" >0</td>
      <td id="T_7cfc2_row4_col7" class="data row4 col7" >0</td>
      <td id="T_7cfc2_row4_col8" class="data row4 col8" >0</td>
      <td id="T_7cfc2_row4_col9" class="data row4 col9" >0</td>
      <td id="T_7cfc2_row4_col10" class="data row4 col10" >0</td>
      <td id="T_7cfc2_row4_col11" class="data row4 col11" >0</td>
      <td id="T_7cfc2_row4_col12" class="data row4 col12" >0</td>
      <td id="T_7cfc2_row4_col13" class="data row4 col13" >0</td>
      <td id="T_7cfc2_row4_col14" class="data row4 col14" >0</td>
      <td id="T_7cfc2_row4_col15" class="data row4 col15" >0</td>
      <td id="T_7cfc2_row4_col16" class="data row4 col16" >0</td>
      <td id="T_7cfc2_row4_col17" class="data row4 col17" >0</td>
      <td id="T_7cfc2_row4_col18" class="data row4 col18" >0</td>
      <td id="T_7cfc2_row4_col19" class="data row4 col19" >0</td>
      <td id="T_7cfc2_row4_col20" class="data row4 col20" >0</td>
      <td id="T_7cfc2_row4_col21" class="data row4 col21" >0</td>
      <td id="T_7cfc2_row4_col22" class="data row4 col22" >0</td>
      <td id="T_7cfc2_row4_col23" class="data row4 col23" >0</td>
      <td id="T_7cfc2_row4_col24" class="data row4 col24" >0</td>
      <td id="T_7cfc2_row4_col25" class="data row4 col25" >0</td>
      <td id="T_7cfc2_row4_col26" class="data row4 col26" >0</td>
      <td id="T_7cfc2_row4_col27" class="data row4 col27" >0</td>
      <td id="T_7cfc2_row4_col28" class="data row4 col28" >0</td>
      <td id="T_7cfc2_row4_col29" class="data row4 col29" >0</td>
      <td id="T_7cfc2_row4_col30" class="data row4 col30" >0</td>
      <td id="T_7cfc2_row4_col31" class="data row4 col31" >0</td>
      <td id="T_7cfc2_row4_col32" class="data row4 col32" >0</td>
      <td id="T_7cfc2_row4_col33" class="data row4 col33" >0</td>
      <td id="T_7cfc2_row4_col34" class="data row4 col34" >0</td>
      <td id="T_7cfc2_row4_col35" class="data row4 col35" >0</td>
      <td id="T_7cfc2_row4_col36" class="data row4 col36" >0</td>
      <td id="T_7cfc2_row4_col37" class="data row4 col37" >0</td>
      <td id="T_7cfc2_row4_col38" class="data row4 col38" >0</td>
      <td id="T_7cfc2_row4_col39" class="data row4 col39" >0</td>
    </tr>
    <tr>
      <th id="T_7cfc2_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_7cfc2_row5_col0" class="data row5 col0" >1</td>
      <td id="T_7cfc2_row5_col1" class="data row5 col1" >0</td>
      <td id="T_7cfc2_row5_col2" class="data row5 col2" >0</td>
      <td id="T_7cfc2_row5_col3" class="data row5 col3" >0</td>
      <td id="T_7cfc2_row5_col4" class="data row5 col4" >0</td>
      <td id="T_7cfc2_row5_col5" class="data row5 col5" >0</td>
      <td id="T_7cfc2_row5_col6" class="data row5 col6" >0</td>
      <td id="T_7cfc2_row5_col7" class="data row5 col7" >0</td>
      <td id="T_7cfc2_row5_col8" class="data row5 col8" >0</td>
      <td id="T_7cfc2_row5_col9" class="data row5 col9" >0</td>
      <td id="T_7cfc2_row5_col10" class="data row5 col10" >0</td>
      <td id="T_7cfc2_row5_col11" class="data row5 col11" >0</td>
      <td id="T_7cfc2_row5_col12" class="data row5 col12" >0</td>
      <td id="T_7cfc2_row5_col13" class="data row5 col13" >0</td>
      <td id="T_7cfc2_row5_col14" class="data row5 col14" >0</td>
      <td id="T_7cfc2_row5_col15" class="data row5 col15" >0</td>
      <td id="T_7cfc2_row5_col16" class="data row5 col16" >0</td>
      <td id="T_7cfc2_row5_col17" class="data row5 col17" >0</td>
      <td id="T_7cfc2_row5_col18" class="data row5 col18" >0</td>
      <td id="T_7cfc2_row5_col19" class="data row5 col19" >0</td>
      <td id="T_7cfc2_row5_col20" class="data row5 col20" >1</td>
      <td id="T_7cfc2_row5_col21" class="data row5 col21" >0</td>
      <td id="T_7cfc2_row5_col22" class="data row5 col22" >0</td>
      <td id="T_7cfc2_row5_col23" class="data row5 col23" >0</td>
      <td id="T_7cfc2_row5_col24" class="data row5 col24" >0</td>
      <td id="T_7cfc2_row5_col25" class="data row5 col25" >0</td>
      <td id="T_7cfc2_row5_col26" class="data row5 col26" >0</td>
      <td id="T_7cfc2_row5_col27" class="data row5 col27" >0</td>
      <td id="T_7cfc2_row5_col28" class="data row5 col28" >0</td>
      <td id="T_7cfc2_row5_col29" class="data row5 col29" >0</td>
      <td id="T_7cfc2_row5_col30" class="data row5 col30" >0</td>
      <td id="T_7cfc2_row5_col31" class="data row5 col31" >0</td>
      <td id="T_7cfc2_row5_col32" class="data row5 col32" >0</td>
      <td id="T_7cfc2_row5_col33" class="data row5 col33" >0</td>
      <td id="T_7cfc2_row5_col34" class="data row5 col34" >0</td>
      <td id="T_7cfc2_row5_col35" class="data row5 col35" >0</td>
      <td id="T_7cfc2_row5_col36" class="data row5 col36" >0</td>
      <td id="T_7cfc2_row5_col37" class="data row5 col37" >0</td>
      <td id="T_7cfc2_row5_col38" class="data row5 col38" >0</td>
      <td id="T_7cfc2_row5_col39" class="data row5 col39" >0</td>
    </tr>
    <tr>
      <th id="T_7cfc2_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_7cfc2_row6_col0" class="data row6 col0" >1</td>
      <td id="T_7cfc2_row6_col1" class="data row6 col1" >0</td>
      <td id="T_7cfc2_row6_col2" class="data row6 col2" >0</td>
      <td id="T_7cfc2_row6_col3" class="data row6 col3" >0</td>
      <td id="T_7cfc2_row6_col4" class="data row6 col4" >0</td>
      <td id="T_7cfc2_row6_col5" class="data row6 col5" >0</td>
      <td id="T_7cfc2_row6_col6" class="data row6 col6" >0</td>
      <td id="T_7cfc2_row6_col7" class="data row6 col7" >0</td>
      <td id="T_7cfc2_row6_col8" class="data row6 col8" >0</td>
      <td id="T_7cfc2_row6_col9" class="data row6 col9" >0</td>
      <td id="T_7cfc2_row6_col10" class="data row6 col10" >0</td>
      <td id="T_7cfc2_row6_col11" class="data row6 col11" >0</td>
      <td id="T_7cfc2_row6_col12" class="data row6 col12" >0</td>
      <td id="T_7cfc2_row6_col13" class="data row6 col13" >0</td>
      <td id="T_7cfc2_row6_col14" class="data row6 col14" >0</td>
      <td id="T_7cfc2_row6_col15" class="data row6 col15" >0</td>
      <td id="T_7cfc2_row6_col16" class="data row6 col16" >0</td>
      <td id="T_7cfc2_row6_col17" class="data row6 col17" >0</td>
      <td id="T_7cfc2_row6_col18" class="data row6 col18" >0</td>
      <td id="T_7cfc2_row6_col19" class="data row6 col19" >0</td>
      <td id="T_7cfc2_row6_col20" class="data row6 col20" >0</td>
      <td id="T_7cfc2_row6_col21" class="data row6 col21" >0</td>
      <td id="T_7cfc2_row6_col22" class="data row6 col22" >0</td>
      <td id="T_7cfc2_row6_col23" class="data row6 col23" >0</td>
      <td id="T_7cfc2_row6_col24" class="data row6 col24" >0</td>
      <td id="T_7cfc2_row6_col25" class="data row6 col25" >0</td>
      <td id="T_7cfc2_row6_col26" class="data row6 col26" >0</td>
      <td id="T_7cfc2_row6_col27" class="data row6 col27" >0</td>
      <td id="T_7cfc2_row6_col28" class="data row6 col28" >0</td>
      <td id="T_7cfc2_row6_col29" class="data row6 col29" >0</td>
      <td id="T_7cfc2_row6_col30" class="data row6 col30" >0</td>
      <td id="T_7cfc2_row6_col31" class="data row6 col31" >0</td>
      <td id="T_7cfc2_row6_col32" class="data row6 col32" >0</td>
      <td id="T_7cfc2_row6_col33" class="data row6 col33" >0</td>
      <td id="T_7cfc2_row6_col34" class="data row6 col34" >0</td>
      <td id="T_7cfc2_row6_col35" class="data row6 col35" >0</td>
      <td id="T_7cfc2_row6_col36" class="data row6 col36" >0</td>
      <td id="T_7cfc2_row6_col37" class="data row6 col37" >0</td>
      <td id="T_7cfc2_row6_col38" class="data row6 col38" >0</td>
      <td id="T_7cfc2_row6_col39" class="data row6 col39" >0</td>
    </tr>
    <tr>
      <th id="T_7cfc2_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_7cfc2_row7_col0" class="data row7 col0" >1</td>
      <td id="T_7cfc2_row7_col1" class="data row7 col1" >0</td>
      <td id="T_7cfc2_row7_col2" class="data row7 col2" >0</td>
      <td id="T_7cfc2_row7_col3" class="data row7 col3" >0</td>
      <td id="T_7cfc2_row7_col4" class="data row7 col4" >0</td>
      <td id="T_7cfc2_row7_col5" class="data row7 col5" >0</td>
      <td id="T_7cfc2_row7_col6" class="data row7 col6" >0</td>
      <td id="T_7cfc2_row7_col7" class="data row7 col7" >0</td>
      <td id="T_7cfc2_row7_col8" class="data row7 col8" >0</td>
      <td id="T_7cfc2_row7_col9" class="data row7 col9" >0</td>
      <td id="T_7cfc2_row7_col10" class="data row7 col10" >0</td>
      <td id="T_7cfc2_row7_col11" class="data row7 col11" >0</td>
      <td id="T_7cfc2_row7_col12" class="data row7 col12" >0</td>
      <td id="T_7cfc2_row7_col13" class="data row7 col13" >0</td>
      <td id="T_7cfc2_row7_col14" class="data row7 col14" >0</td>
      <td id="T_7cfc2_row7_col15" class="data row7 col15" >0</td>
      <td id="T_7cfc2_row7_col16" class="data row7 col16" >0</td>
      <td id="T_7cfc2_row7_col17" class="data row7 col17" >0</td>
      <td id="T_7cfc2_row7_col18" class="data row7 col18" >0</td>
      <td id="T_7cfc2_row7_col19" class="data row7 col19" >0</td>
      <td id="T_7cfc2_row7_col20" class="data row7 col20" >1</td>
      <td id="T_7cfc2_row7_col21" class="data row7 col21" >0</td>
      <td id="T_7cfc2_row7_col22" class="data row7 col22" >0</td>
      <td id="T_7cfc2_row7_col23" class="data row7 col23" >0</td>
      <td id="T_7cfc2_row7_col24" class="data row7 col24" >0</td>
      <td id="T_7cfc2_row7_col25" class="data row7 col25" >0</td>
      <td id="T_7cfc2_row7_col26" class="data row7 col26" >0</td>
      <td id="T_7cfc2_row7_col27" class="data row7 col27" >0</td>
      <td id="T_7cfc2_row7_col28" class="data row7 col28" >0</td>
      <td id="T_7cfc2_row7_col29" class="data row7 col29" >0</td>
      <td id="T_7cfc2_row7_col30" class="data row7 col30" >0</td>
      <td id="T_7cfc2_row7_col31" class="data row7 col31" >0</td>
      <td id="T_7cfc2_row7_col32" class="data row7 col32" >0</td>
      <td id="T_7cfc2_row7_col33" class="data row7 col33" >0</td>
      <td id="T_7cfc2_row7_col34" class="data row7 col34" >0</td>
      <td id="T_7cfc2_row7_col35" class="data row7 col35" >0</td>
      <td id="T_7cfc2_row7_col36" class="data row7 col36" >0</td>
      <td id="T_7cfc2_row7_col37" class="data row7 col37" >0</td>
      <td id="T_7cfc2_row7_col38" class="data row7 col38" >0</td>
      <td id="T_7cfc2_row7_col39" class="data row7 col39" >0</td>
    </tr>
    <tr>
      <th id="T_7cfc2_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_7cfc2_row8_col0" class="data row8 col0" >1</td>
      <td id="T_7cfc2_row8_col1" class="data row8 col1" >0</td>
      <td id="T_7cfc2_row8_col2" class="data row8 col2" >0</td>
      <td id="T_7cfc2_row8_col3" class="data row8 col3" >0</td>
      <td id="T_7cfc2_row8_col4" class="data row8 col4" >0</td>
      <td id="T_7cfc2_row8_col5" class="data row8 col5" >0</td>
      <td id="T_7cfc2_row8_col6" class="data row8 col6" >0</td>
      <td id="T_7cfc2_row8_col7" class="data row8 col7" >0</td>
      <td id="T_7cfc2_row8_col8" class="data row8 col8" >0</td>
      <td id="T_7cfc2_row8_col9" class="data row8 col9" >0</td>
      <td id="T_7cfc2_row8_col10" class="data row8 col10" >0</td>
      <td id="T_7cfc2_row8_col11" class="data row8 col11" >0</td>
      <td id="T_7cfc2_row8_col12" class="data row8 col12" >0</td>
      <td id="T_7cfc2_row8_col13" class="data row8 col13" >0</td>
      <td id="T_7cfc2_row8_col14" class="data row8 col14" >0</td>
      <td id="T_7cfc2_row8_col15" class="data row8 col15" >0</td>
      <td id="T_7cfc2_row8_col16" class="data row8 col16" >0</td>
      <td id="T_7cfc2_row8_col17" class="data row8 col17" >0</td>
      <td id="T_7cfc2_row8_col18" class="data row8 col18" >0</td>
      <td id="T_7cfc2_row8_col19" class="data row8 col19" >0</td>
      <td id="T_7cfc2_row8_col20" class="data row8 col20" >0</td>
      <td id="T_7cfc2_row8_col21" class="data row8 col21" >0</td>
      <td id="T_7cfc2_row8_col22" class="data row8 col22" >0</td>
      <td id="T_7cfc2_row8_col23" class="data row8 col23" >0</td>
      <td id="T_7cfc2_row8_col24" class="data row8 col24" >0</td>
      <td id="T_7cfc2_row8_col25" class="data row8 col25" >0</td>
      <td id="T_7cfc2_row8_col26" class="data row8 col26" >0</td>
      <td id="T_7cfc2_row8_col27" class="data row8 col27" >0</td>
      <td id="T_7cfc2_row8_col28" class="data row8 col28" >0</td>
      <td id="T_7cfc2_row8_col29" class="data row8 col29" >0</td>
      <td id="T_7cfc2_row8_col30" class="data row8 col30" >0</td>
      <td id="T_7cfc2_row8_col31" class="data row8 col31" >0</td>
      <td id="T_7cfc2_row8_col32" class="data row8 col32" >0</td>
      <td id="T_7cfc2_row8_col33" class="data row8 col33" >0</td>
      <td id="T_7cfc2_row8_col34" class="data row8 col34" >0</td>
      <td id="T_7cfc2_row8_col35" class="data row8 col35" >0</td>
      <td id="T_7cfc2_row8_col36" class="data row8 col36" >0</td>
      <td id="T_7cfc2_row8_col37" class="data row8 col37" >0</td>
      <td id="T_7cfc2_row8_col38" class="data row8 col38" >0</td>
      <td id="T_7cfc2_row8_col39" class="data row8 col39" >0</td>
    </tr>
    <tr>
      <th id="T_7cfc2_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_7cfc2_row9_col0" class="data row9 col0" >1</td>
      <td id="T_7cfc2_row9_col1" class="data row9 col1" >0</td>
      <td id="T_7cfc2_row9_col2" class="data row9 col2" >0</td>
      <td id="T_7cfc2_row9_col3" class="data row9 col3" >0</td>
      <td id="T_7cfc2_row9_col4" class="data row9 col4" >0</td>
      <td id="T_7cfc2_row9_col5" class="data row9 col5" >0</td>
      <td id="T_7cfc2_row9_col6" class="data row9 col6" >0</td>
      <td id="T_7cfc2_row9_col7" class="data row9 col7" >0</td>
      <td id="T_7cfc2_row9_col8" class="data row9 col8" >0</td>
      <td id="T_7cfc2_row9_col9" class="data row9 col9" >0</td>
      <td id="T_7cfc2_row9_col10" class="data row9 col10" >0</td>
      <td id="T_7cfc2_row9_col11" class="data row9 col11" >0</td>
      <td id="T_7cfc2_row9_col12" class="data row9 col12" >0</td>
      <td id="T_7cfc2_row9_col13" class="data row9 col13" >0</td>
      <td id="T_7cfc2_row9_col14" class="data row9 col14" >0</td>
      <td id="T_7cfc2_row9_col15" class="data row9 col15" >0</td>
      <td id="T_7cfc2_row9_col16" class="data row9 col16" >0</td>
      <td id="T_7cfc2_row9_col17" class="data row9 col17" >0</td>
      <td id="T_7cfc2_row9_col18" class="data row9 col18" >0</td>
      <td id="T_7cfc2_row9_col19" class="data row9 col19" >0</td>
      <td id="T_7cfc2_row9_col20" class="data row9 col20" >1</td>
      <td id="T_7cfc2_row9_col21" class="data row9 col21" >0</td>
      <td id="T_7cfc2_row9_col22" class="data row9 col22" >0</td>
      <td id="T_7cfc2_row9_col23" class="data row9 col23" >0</td>
      <td id="T_7cfc2_row9_col24" class="data row9 col24" >0</td>
      <td id="T_7cfc2_row9_col25" class="data row9 col25" >0</td>
      <td id="T_7cfc2_row9_col26" class="data row9 col26" >0</td>
      <td id="T_7cfc2_row9_col27" class="data row9 col27" >0</td>
      <td id="T_7cfc2_row9_col28" class="data row9 col28" >0</td>
      <td id="T_7cfc2_row9_col29" class="data row9 col29" >0</td>
      <td id="T_7cfc2_row9_col30" class="data row9 col30" >0</td>
      <td id="T_7cfc2_row9_col31" class="data row9 col31" >0</td>
      <td id="T_7cfc2_row9_col32" class="data row9 col32" >0</td>
      <td id="T_7cfc2_row9_col33" class="data row9 col33" >0</td>
      <td id="T_7cfc2_row9_col34" class="data row9 col34" >0</td>
      <td id="T_7cfc2_row9_col35" class="data row9 col35" >0</td>
      <td id="T_7cfc2_row9_col36" class="data row9 col36" >0</td>
      <td id="T_7cfc2_row9_col37" class="data row9 col37" >0</td>
      <td id="T_7cfc2_row9_col38" class="data row9 col38" >0</td>
      <td id="T_7cfc2_row9_col39" class="data row9 col39" >0</td>
    </tr>
    <tr>
      <th id="T_7cfc2_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_7cfc2_row10_col0" class="data row10 col0" >0</td>
      <td id="T_7cfc2_row10_col1" class="data row10 col1" >1</td>
      <td id="T_7cfc2_row10_col2" class="data row10 col2" >0</td>
      <td id="T_7cfc2_row10_col3" class="data row10 col3" >0</td>
      <td id="T_7cfc2_row10_col4" class="data row10 col4" >0</td>
      <td id="T_7cfc2_row10_col5" class="data row10 col5" >0</td>
      <td id="T_7cfc2_row10_col6" class="data row10 col6" >0</td>
      <td id="T_7cfc2_row10_col7" class="data row10 col7" >0</td>
      <td id="T_7cfc2_row10_col8" class="data row10 col8" >0</td>
      <td id="T_7cfc2_row10_col9" class="data row10 col9" >0</td>
      <td id="T_7cfc2_row10_col10" class="data row10 col10" >0</td>
      <td id="T_7cfc2_row10_col11" class="data row10 col11" >0</td>
      <td id="T_7cfc2_row10_col12" class="data row10 col12" >0</td>
      <td id="T_7cfc2_row10_col13" class="data row10 col13" >0</td>
      <td id="T_7cfc2_row10_col14" class="data row10 col14" >0</td>
      <td id="T_7cfc2_row10_col15" class="data row10 col15" >0</td>
      <td id="T_7cfc2_row10_col16" class="data row10 col16" >0</td>
      <td id="T_7cfc2_row10_col17" class="data row10 col17" >0</td>
      <td id="T_7cfc2_row10_col18" class="data row10 col18" >0</td>
      <td id="T_7cfc2_row10_col19" class="data row10 col19" >0</td>
      <td id="T_7cfc2_row10_col20" class="data row10 col20" >0</td>
      <td id="T_7cfc2_row10_col21" class="data row10 col21" >0</td>
      <td id="T_7cfc2_row10_col22" class="data row10 col22" >0</td>
      <td id="T_7cfc2_row10_col23" class="data row10 col23" >0</td>
      <td id="T_7cfc2_row10_col24" class="data row10 col24" >0</td>
      <td id="T_7cfc2_row10_col25" class="data row10 col25" >0</td>
      <td id="T_7cfc2_row10_col26" class="data row10 col26" >0</td>
      <td id="T_7cfc2_row10_col27" class="data row10 col27" >0</td>
      <td id="T_7cfc2_row10_col28" class="data row10 col28" >0</td>
      <td id="T_7cfc2_row10_col29" class="data row10 col29" >0</td>
      <td id="T_7cfc2_row10_col30" class="data row10 col30" >0</td>
      <td id="T_7cfc2_row10_col31" class="data row10 col31" >0</td>
      <td id="T_7cfc2_row10_col32" class="data row10 col32" >0</td>
      <td id="T_7cfc2_row10_col33" class="data row10 col33" >0</td>
      <td id="T_7cfc2_row10_col34" class="data row10 col34" >0</td>
      <td id="T_7cfc2_row10_col35" class="data row10 col35" >0</td>
      <td id="T_7cfc2_row10_col36" class="data row10 col36" >0</td>
      <td id="T_7cfc2_row10_col37" class="data row10 col37" >0</td>
      <td id="T_7cfc2_row10_col38" class="data row10 col38" >0</td>
      <td id="T_7cfc2_row10_col39" class="data row10 col39" >0</td>
    </tr>
    <tr>
      <th id="T_7cfc2_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_7cfc2_row11_col0" class="data row11 col0" >0</td>
      <td id="T_7cfc2_row11_col1" class="data row11 col1" >1</td>
      <td id="T_7cfc2_row11_col2" class="data row11 col2" >0</td>
      <td id="T_7cfc2_row11_col3" class="data row11 col3" >0</td>
      <td id="T_7cfc2_row11_col4" class="data row11 col4" >0</td>
      <td id="T_7cfc2_row11_col5" class="data row11 col5" >0</td>
      <td id="T_7cfc2_row11_col6" class="data row11 col6" >0</td>
      <td id="T_7cfc2_row11_col7" class="data row11 col7" >0</td>
      <td id="T_7cfc2_row11_col8" class="data row11 col8" >0</td>
      <td id="T_7cfc2_row11_col9" class="data row11 col9" >0</td>
      <td id="T_7cfc2_row11_col10" class="data row11 col10" >0</td>
      <td id="T_7cfc2_row11_col11" class="data row11 col11" >0</td>
      <td id="T_7cfc2_row11_col12" class="data row11 col12" >0</td>
      <td id="T_7cfc2_row11_col13" class="data row11 col13" >0</td>
      <td id="T_7cfc2_row11_col14" class="data row11 col14" >0</td>
      <td id="T_7cfc2_row11_col15" class="data row11 col15" >0</td>
      <td id="T_7cfc2_row11_col16" class="data row11 col16" >0</td>
      <td id="T_7cfc2_row11_col17" class="data row11 col17" >0</td>
      <td id="T_7cfc2_row11_col18" class="data row11 col18" >0</td>
      <td id="T_7cfc2_row11_col19" class="data row11 col19" >0</td>
      <td id="T_7cfc2_row11_col20" class="data row11 col20" >0</td>
      <td id="T_7cfc2_row11_col21" class="data row11 col21" >1</td>
      <td id="T_7cfc2_row11_col22" class="data row11 col22" >0</td>
      <td id="T_7cfc2_row11_col23" class="data row11 col23" >0</td>
      <td id="T_7cfc2_row11_col24" class="data row11 col24" >0</td>
      <td id="T_7cfc2_row11_col25" class="data row11 col25" >0</td>
      <td id="T_7cfc2_row11_col26" class="data row11 col26" >0</td>
      <td id="T_7cfc2_row11_col27" class="data row11 col27" >0</td>
      <td id="T_7cfc2_row11_col28" class="data row11 col28" >0</td>
      <td id="T_7cfc2_row11_col29" class="data row11 col29" >0</td>
      <td id="T_7cfc2_row11_col30" class="data row11 col30" >0</td>
      <td id="T_7cfc2_row11_col31" class="data row11 col31" >0</td>
      <td id="T_7cfc2_row11_col32" class="data row11 col32" >0</td>
      <td id="T_7cfc2_row11_col33" class="data row11 col33" >0</td>
      <td id="T_7cfc2_row11_col34" class="data row11 col34" >0</td>
      <td id="T_7cfc2_row11_col35" class="data row11 col35" >0</td>
      <td id="T_7cfc2_row11_col36" class="data row11 col36" >0</td>
      <td id="T_7cfc2_row11_col37" class="data row11 col37" >0</td>
      <td id="T_7cfc2_row11_col38" class="data row11 col38" >0</td>
      <td id="T_7cfc2_row11_col39" class="data row11 col39" >0</td>
    </tr>
  </tbody>
</table>




We can visualize all of $\textbf{Z}$ here.


```python
plt.imshow(Z, aspect='auto')
plt.text(10, 220, s='intercept (cafe)', ha='center', fontsize=14)
plt.text(30, 220, s='covariate (afternoon)', ha='center', fontsize=14)
plt.ylabel('observations')
plt.title('Visual representation of Z')
```




    Text(0.5, 1.0, 'Visual representation of Z')




    
![png](/assets/2022-09-13-mixed_effects_freqvsbayes_cafes_files/2022-09-13-mixed_effects_freqvsbayes_cafes_25_1.png)
    


The vector in $\textbf{u}$ is really where the mixed effects model takes advantage of the covariance structure of the data. In our dataset, the first 20 elements of the vector represent the random intercepts of the cafes and the next 20 are the random slopes. A cafe's random effects can be thought of as an offset from the populations (the fixed effects). Accordingly, a random effect will be multivariate normally distributed, with mean 0 and a co-variance matrix S.

$$ \textbf{u} \sim \text{Normal}(0, \textbf{S}) \tag{2}$$ 
   
Remember that the $\textbf{u}$ is a (2x20) x 1 matrix, where each cafe's intercept $a_\text{cafe}$ and slope $b_\text{cafe}$ are contained. Therefore, we can also write this as.

$$ \textbf{u} = \begin{bmatrix} a_{\text{cafe}} \\ b_{\text{cafe}} \end{bmatrix} \sim \text{MVNormal} \left( \begin{bmatrix} 0 \\ 0 \end{bmatrix} , \textbf{S} \right)   \tag{3}$$ 

In other words, in Equation 1, both the random intercept and random slope are both expected to lie at 0. With regards to $\textbf{S}$, [my prior post](https://benslack19.github.io/data%20science/statistics/cov_matrix_weirdness/) talked about covariance matrixes so I won't elaborate here. The key conceptual point of relevance in this problem is that the covariance matrix $\textbf{S}$ can reflect the correlation ($\rho$) that the intercept (average morning wait time) and slope (difference between morning and afternoon wait time). 

$$ \textbf{S} = \begin{pmatrix} \sigma_{\alpha}^2 & \rho\sigma_{\alpha}\sigma_{\beta} \\ 
                \rho\sigma_{\alpha}\sigma_{\beta} & \sigma_{\beta}^2 \end{pmatrix} \tag{4}$$ 

 We know there is a correlation because (a) we generated the data that way and (b) we can directly observe this when we [visualized the data](#Visualize-data).

Finally, the role of $\boldsymbol{\epsilon}$ is to capture any residual variance. Between observations, it is assumed to be homogenous and independent.

### Non-linear algebra form

Equation 1 is written concisely in linear algebra form. However, since our dataset is relatively simple (only one predictor variable), equation 1 can be written in an expanded, alternative form as equation 2. This might make it easier to understand (at least it did for me). The notation will start to get hairy with subscripts and so I will explicitly rename some variables for this explanation. It will also better match with the Bayesian set of equations described in the McElreath text. Equation 2 is written at the level of a single observation $i$. I'll repeat Equation 1 here so it's easier to see the conversion.

$$ \textbf{y} = \textbf{X} \boldsymbol{\beta} + \textbf{Z} \textbf{u} + \boldsymbol{\epsilon}  \tag{5}$$

$$ W_i = (\alpha + \beta \times A_i) + (a_{\text{cafe}[i]} + b_{\text{cafe}[i]} \times A_i) + \epsilon_i \tag{6} $$

Let's start off with the left side where we can see that $\textbf{y}$ will now be $W_i$ for wait time. On the right side, I have segmented the fixed and random effects with parentheses. For both, I've deconstructed the linear algebra expression form to a simpler form. After re-arrangement, we can obtain the following form in equation 3.

$$ W_i = (\alpha + a_{\text{cafe}[i]}) + (\beta + b_{\text{cafe}[i]}) \times A_i + \epsilon_{\text{cafe}} \tag{7} $$

Here, we can better appreciate how a cafe's random effects intercept can be thought of as an offset from the population intercept. The same logic of an offset can be applied to its slope. We will come back to equation 3 after covering Equation set 2, the Bayesian approach.

## Equation set 2: fixed effects as an adaptive prior, varying effects in the linear model

The following equations are taken from Chapter 14 in Statistical Rethinking. These set of equations look like a beast, but to be honest, they're more intuitive to me, probably because I learned this approach initially. I'll state the equations before comparing them directly with Equation set 1 but you may already start seeing the relationship. Essentially what is going on is a re-writing of the above equations in a Bayesian way such that the fixed effects can act as an adaptive prior. 

$$ W_i \sim \text{Normal}(\mu_i, \sigma) \tag{8} $$
$$ \mu_i = \alpha_{\text{cafe}[i]} + \beta_{\text{cafe}[i]} \times A_{i} \tag{9}$$
$$ \sigma \sim \text{Exp}(1) \tag{10}$$

Equation 8 is stating how wait time is normally distributed around $\mu$ and $\sigma$. By making $w_i$ stochastic instead of deterministic (using a ~ instead of =), the $\sigma$ replaces $\epsilon_i$. In equation 10, the prior for $\sigma$ is exponentially distributed and paramaterized with 1. The expected value parameter $\mu$ comes from the linear model in equation 9. You can start to see the similarities with equation 7 above.

$$ \begin{bmatrix}\alpha_{\text{cafe}} \\ \beta_{\text{cafe}} \end{bmatrix} \sim \text{MVNormal} \left( \begin{bmatrix}{\alpha} \\ {\beta} \end{bmatrix} , \textbf{S} \right)    \tag{11}$$

The $\alpha_{\text{cafe}}$ and $\beta_{\text{cafe}}$ terms come from sampling of a multivariate normal distribution as shown in equation 11. **Note the very subtle difference in placement of the subscript `cafe` when compared to equation 6 and 7. This is an important point I'll discuss later.** On the right side, the two-dimensional normal distribution's expected values are $\alpha$ and $\beta$. The rest of the equations shown below are our priors for each parameter we're trying to estimate.

$$ \alpha \sim \text{Normal}(5, 2) \tag{13}$$  

$$ \beta \sim \text{Normal}(-1, 0.5)  \tag{14}$$  

$$ \textbf{S} = \begin{pmatrix} \sigma_{\alpha}^2 & \rho\sigma_{\alpha}\sigma_{\beta} \\ 
                \rho\sigma_{\alpha}\sigma_{\beta} & \sigma_{\beta}^2 \end{pmatrix} = \begin{pmatrix} \sigma_{\alpha} & 0 \\ 0 & \sigma_{\beta} \end{pmatrix} \textbf{R} \begin{pmatrix} \sigma_{\alpha} & 0 \\ 0 & \sigma_{\beta} \end{pmatrix}  \tag{12}$$


$$ \sigma, \sigma_{\alpha}, \sigma_{\beta} \sim \text{Exp}(1) \tag{15}$$
$$ \textbf{R} \sim \text{LKJCorr}(2) \tag{16}$$

## Comparison of equation sets

To recap, the first equation set has an explicit fixed effects term and varying effects term in the linear model. In the second equation, the linear model is already "mixed". It contains both the fixed and varying effects terms implicitly. The fixed effects estimates can be seen in equation 5.

I think you can think of these $\alpha_{\text{cafe}}$ and $\beta_{\text{cafe}}$ terms as already incorporating the information from the fixed and random effects simultaneously.

Now that we have the dataset, we can run the two models, one with `lmer` and one with `pymc`. Here are the equations that these packages run.

# Running equation set 1 with `lmer` (frequentist)

The `lmer` and by extension (`brms`) syntax was confusing to me. Thanks to @probot from the Discord channel for helping me understand this.

`lmer(wait ~ 1 + afternoon + (1 + afternoon | cafe), df_cafes)`

The `1` corresponds to inclusion of the intercept term. A `0` would exclude it. The `1 + 
wait` corresponds to the "fixed effects" portion of the model ($\alpha + \beta \times A_i$) while the `(1 + wait | cafe)` is the "varying effects" ($a_{\text{cafe}} + b_{\text{cafe}} \times A_i$).


```r
%%R -i df_cafes -o m -o df_fe_estimates -o df_fe_ci -o df_fe_summary

# m df_fe_summary
m <- lmer(wait ~ 1 + afternoon + (1 + afternoon | cafe), df_cafes)
arm::display(m)

# get fixed effects coefficients
df_fe_estimates <- data.frame(summary(m)$coefficients)
# get fixed effects coefficient CIs
df_fe_ci <- data.frame(confint(m))

df_fe_summary <- merge(
    df_fe_estimates,
    df_fe_ci[c('(Intercept)', 'afternoon'), ],
    by.x=0,
    by.y=0
)
rownames(df_fe_summary) <- df_fe_summary[, 1]
```

    lmer(formula = wait ~ 1 + afternoon + (1 + afternoon | cafe), 
        data = df_cafes)
                coef.est coef.se
    (Intercept)  3.64     0.23  
    afternoon   -1.04     0.11  
    
    Error terms:
     Groups   Name        Std.Dev. Corr  
     cafe     (Intercept) 0.99           
              afternoon   0.39     -0.74 
     Residual             0.48           
    ---
    number of obs: 200, groups: cafe, 20
    AIC = 369.9, DIC = 349.2
    deviance = 353.5 


    R[write to console]: Computing profile confidence intervals ...
    


Can we get the partial pooling results from the `lmer` output and see how it compares with the unpooled estimates? Let's export it for use later.


```r
%%R -i m -o df_partial_pooling -o random_sims

# Make a dataframe with the fitted effects
df_partial_pooling <- coef(m)[["cafe"]] %>% 
  rownames_to_column("cafe") %>% 
  as_tibble() %>% 
  rename(Intercept = `(Intercept)`, Slope_Days = afternoon) %>% 
  add_column(Model = "Partial pooling")

# estimate confidence interval
random_sims <- REsim(m, n.sims = 1000)
#plotREsim(random_sims)
```


```python
random_sims
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>groupFctr</th>
      <th>groupID</th>
      <th>term</th>
      <th>mean</th>
      <th>median</th>
      <th>sd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>cafe</td>
      <td>0</td>
      <td>(Intercept)</td>
      <td>-1.277651</td>
      <td>-1.283341</td>
      <td>0.379761</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cafe</td>
      <td>1</td>
      <td>(Intercept)</td>
      <td>0.164935</td>
      <td>0.162715</td>
      <td>0.420411</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cafe</td>
      <td>2</td>
      <td>(Intercept)</td>
      <td>-1.047076</td>
      <td>-1.043646</td>
      <td>0.387153</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cafe</td>
      <td>3</td>
      <td>(Intercept)</td>
      <td>0.474320</td>
      <td>0.500552</td>
      <td>0.400053</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cafe</td>
      <td>4</td>
      <td>(Intercept)</td>
      <td>-1.473647</td>
      <td>-1.468940</td>
      <td>0.394707</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cafe</td>
      <td>5</td>
      <td>(Intercept)</td>
      <td>0.086072</td>
      <td>0.082010</td>
      <td>0.408971</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cafe</td>
      <td>6</td>
      <td>(Intercept)</td>
      <td>-0.640217</td>
      <td>-0.628944</td>
      <td>0.412642</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cafe</td>
      <td>7</td>
      <td>(Intercept)</td>
      <td>1.507154</td>
      <td>1.516430</td>
      <td>0.391119</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cafe</td>
      <td>8</td>
      <td>(Intercept)</td>
      <td>-0.657831</td>
      <td>-0.659448</td>
      <td>0.394984</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cafe</td>
      <td>9</td>
      <td>(Intercept)</td>
      <td>0.332758</td>
      <td>0.331037</td>
      <td>0.388295</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cafe</td>
      <td>10</td>
      <td>(Intercept)</td>
      <td>-1.018611</td>
      <td>-1.025387</td>
      <td>0.389930</td>
    </tr>
    <tr>
      <th>12</th>
      <td>cafe</td>
      <td>11</td>
      <td>(Intercept)</td>
      <td>0.925071</td>
      <td>0.913997</td>
      <td>0.397095</td>
    </tr>
    <tr>
      <th>13</th>
      <td>cafe</td>
      <td>12</td>
      <td>(Intercept)</td>
      <td>-1.407149</td>
      <td>-1.403259</td>
      <td>0.384820</td>
    </tr>
    <tr>
      <th>14</th>
      <td>cafe</td>
      <td>13</td>
      <td>(Intercept)</td>
      <td>-0.412975</td>
      <td>-0.414958</td>
      <td>0.412863</td>
    </tr>
    <tr>
      <th>15</th>
      <td>cafe</td>
      <td>14</td>
      <td>(Intercept)</td>
      <td>1.346380</td>
      <td>1.343109</td>
      <td>0.403694</td>
    </tr>
    <tr>
      <th>16</th>
      <td>cafe</td>
      <td>15</td>
      <td>(Intercept)</td>
      <td>0.336807</td>
      <td>0.346523</td>
      <td>0.390567</td>
    </tr>
    <tr>
      <th>17</th>
      <td>cafe</td>
      <td>16</td>
      <td>(Intercept)</td>
      <td>0.747439</td>
      <td>0.735906</td>
      <td>0.413094</td>
    </tr>
    <tr>
      <th>18</th>
      <td>cafe</td>
      <td>17</td>
      <td>(Intercept)</td>
      <td>-0.046579</td>
      <td>-0.035018</td>
      <td>0.396795</td>
    </tr>
    <tr>
      <th>19</th>
      <td>cafe</td>
      <td>18</td>
      <td>(Intercept)</td>
      <td>1.659019</td>
      <td>1.646634</td>
      <td>0.393909</td>
    </tr>
    <tr>
      <th>20</th>
      <td>cafe</td>
      <td>19</td>
      <td>(Intercept)</td>
      <td>0.323375</td>
      <td>0.327348</td>
      <td>0.392401</td>
    </tr>
    <tr>
      <th>21</th>
      <td>cafe</td>
      <td>0</td>
      <td>afternoon</td>
      <td>0.498557</td>
      <td>0.501401</td>
      <td>0.182594</td>
    </tr>
    <tr>
      <th>22</th>
      <td>cafe</td>
      <td>1</td>
      <td>afternoon</td>
      <td>-0.336036</td>
      <td>-0.337360</td>
      <td>0.193462</td>
    </tr>
    <tr>
      <th>23</th>
      <td>cafe</td>
      <td>2</td>
      <td>afternoon</td>
      <td>0.395379</td>
      <td>0.391621</td>
      <td>0.189140</td>
    </tr>
    <tr>
      <th>24</th>
      <td>cafe</td>
      <td>3</td>
      <td>afternoon</td>
      <td>0.296956</td>
      <td>0.293144</td>
      <td>0.191710</td>
    </tr>
    <tr>
      <th>25</th>
      <td>cafe</td>
      <td>4</td>
      <td>afternoon</td>
      <td>0.059611</td>
      <td>0.055121</td>
      <td>0.189680</td>
    </tr>
    <tr>
      <th>26</th>
      <td>cafe</td>
      <td>5</td>
      <td>afternoon</td>
      <td>-0.033068</td>
      <td>-0.036143</td>
      <td>0.194723</td>
    </tr>
    <tr>
      <th>27</th>
      <td>cafe</td>
      <td>6</td>
      <td>afternoon</td>
      <td>0.236107</td>
      <td>0.237904</td>
      <td>0.192575</td>
    </tr>
    <tr>
      <th>28</th>
      <td>cafe</td>
      <td>7</td>
      <td>afternoon</td>
      <td>-0.473485</td>
      <td>-0.479199</td>
      <td>0.185549</td>
    </tr>
    <tr>
      <th>29</th>
      <td>cafe</td>
      <td>8</td>
      <td>afternoon</td>
      <td>0.408039</td>
      <td>0.411507</td>
      <td>0.194145</td>
    </tr>
    <tr>
      <th>30</th>
      <td>cafe</td>
      <td>9</td>
      <td>afternoon</td>
      <td>-0.402131</td>
      <td>-0.393931</td>
      <td>0.186868</td>
    </tr>
    <tr>
      <th>31</th>
      <td>cafe</td>
      <td>10</td>
      <td>afternoon</td>
      <td>0.316072</td>
      <td>0.309198</td>
      <td>0.189218</td>
    </tr>
    <tr>
      <th>32</th>
      <td>cafe</td>
      <td>11</td>
      <td>afternoon</td>
      <td>-0.335749</td>
      <td>-0.340427</td>
      <td>0.186644</td>
    </tr>
    <tr>
      <th>33</th>
      <td>cafe</td>
      <td>12</td>
      <td>afternoon</td>
      <td>0.521558</td>
      <td>0.519243</td>
      <td>0.184606</td>
    </tr>
    <tr>
      <th>34</th>
      <td>cafe</td>
      <td>13</td>
      <td>afternoon</td>
      <td>-0.006800</td>
      <td>-0.014344</td>
      <td>0.199548</td>
    </tr>
    <tr>
      <th>35</th>
      <td>cafe</td>
      <td>14</td>
      <td>afternoon</td>
      <td>-0.277165</td>
      <td>-0.281127</td>
      <td>0.188748</td>
    </tr>
    <tr>
      <th>36</th>
      <td>cafe</td>
      <td>15</td>
      <td>afternoon</td>
      <td>-0.234501</td>
      <td>-0.235683</td>
      <td>0.192804</td>
    </tr>
    <tr>
      <th>37</th>
      <td>cafe</td>
      <td>16</td>
      <td>afternoon</td>
      <td>-0.182673</td>
      <td>-0.185997</td>
      <td>0.194017</td>
    </tr>
    <tr>
      <th>38</th>
      <td>cafe</td>
      <td>17</td>
      <td>afternoon</td>
      <td>-0.017126</td>
      <td>-0.023784</td>
      <td>0.187302</td>
    </tr>
    <tr>
      <th>39</th>
      <td>cafe</td>
      <td>18</td>
      <td>afternoon</td>
      <td>-0.364424</td>
      <td>-0.364049</td>
      <td>0.187532</td>
    </tr>
    <tr>
      <th>40</th>
      <td>cafe</td>
      <td>19</td>
      <td>afternoon</td>
      <td>-0.028883</td>
      <td>-0.032691</td>
      <td>0.185824</td>
    </tr>
  </tbody>
</table>
</div>



OK, now let's try the Bayesian approach and compare answers.

# Running equation set 2 with `pymc` (Bayesian)


```python
n_cafes = df_cafes['cafe'].nunique()
cafe_idx = pd.Categorical(df_cafes["cafe"]).codes

with pm.Model() as m14_1:
    # can't specify a separate sigma_a and sigma_b for sd_dist but they're equivalent here
    chol, Rho_, sigma_cafe = pm.LKJCholeskyCov(
        "chol_cov", n=2, eta=2, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
    )
    
    a_bar = pm.Normal("a_bar", mu=5, sigma=2.0)  # prior for average intercept
    b_bar = pm.Normal("b_bar", mu=-1, sigma=0.5)  # prior for average slope

    
    ab_subject = pm.MvNormal(
        "ab_subject", mu=at.stack([a_bar, b_bar]), chol=chol, shape=(n_cafes, 2)
    )  # population of varying effects
    # shape needs to be (n_cafes, 2) because we're getting back both a and b for each cafe

    mu = ab_subject[cafe_idx, 0] + ab_subject[cafe_idx, 1] * df_cafes["afternoon"].values  # linear model
    sigma_within = pm.Exponential("sigma_within", 1.0)  # prior stddev within cafes (in the top line)

    wait = pm.Normal("wait", mu=mu, sigma=sigma_within, observed=df_cafes["wait"].values)  # likelihood

    idata_m14_1 = pm.sample(1000, target_accept=0.9)

```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [chol_cov, a_bar, b_bar, ab_subject, sigma_within]




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 02:03&lt;00:00 Sampling 4 chains, 1 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 140 seconds.
    There was 1 divergence after tuning. Increase `target_accept` or reparameterize.



```python
# take a glimpse at the head and tail of the summary table
pd.concat(
    [
        az.summary(idata_m14_1).head(10),
        az.summary(idata_m14_1).tail(10)
    ]
)
```

    /Users/blacar/opt/anaconda3/envs/pymc_env2/lib/python3.10/site-packages/arviz/stats/diagnostics.py:586: RuntimeWarning: invalid value encountered in double_scalars
      (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
    /Users/blacar/opt/anaconda3/envs/pymc_env2/lib/python3.10/site-packages/arviz/stats/diagnostics.py:586: RuntimeWarning: invalid value encountered in double_scalars
      (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_2.5%</th>
      <th>hdi_97.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a_bar</th>
      <td>3.654</td>
      <td>0.223</td>
      <td>3.203</td>
      <td>4.074</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>4802.0</td>
      <td>3140.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b_bar</th>
      <td>-1.049</td>
      <td>0.109</td>
      <td>-1.265</td>
      <td>-0.844</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3446.0</td>
      <td>3200.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ab_subject[0, 0]</th>
      <td>2.380</td>
      <td>0.200</td>
      <td>1.996</td>
      <td>2.785</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>4271.0</td>
      <td>2783.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ab_subject[0, 1]</th>
      <td>-0.587</td>
      <td>0.245</td>
      <td>-1.071</td>
      <td>-0.119</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>3077.0</td>
      <td>2833.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ab_subject[1, 0]</th>
      <td>3.820</td>
      <td>0.199</td>
      <td>3.442</td>
      <td>4.220</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>3988.0</td>
      <td>3167.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ab_subject[1, 1]</th>
      <td>-1.402</td>
      <td>0.248</td>
      <td>-1.897</td>
      <td>-0.945</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>3165.0</td>
      <td>3182.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ab_subject[2, 0]</th>
      <td>2.606</td>
      <td>0.199</td>
      <td>2.210</td>
      <td>2.988</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>4702.0</td>
      <td>3450.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ab_subject[2, 1]</th>
      <td>-0.681</td>
      <td>0.240</td>
      <td>-1.156</td>
      <td>-0.218</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>3696.0</td>
      <td>3014.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ab_subject[3, 0]</th>
      <td>4.120</td>
      <td>0.203</td>
      <td>3.739</td>
      <td>4.532</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>3475.0</td>
      <td>2800.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ab_subject[3, 1]</th>
      <td>-0.707</td>
      <td>0.266</td>
      <td>-1.213</td>
      <td>-0.184</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>2482.0</td>
      <td>2921.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_cov[0]</th>
      <td>0.988</td>
      <td>0.163</td>
      <td>0.710</td>
      <td>1.328</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>5207.0</td>
      <td>3263.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_cov[1]</th>
      <td>-0.226</td>
      <td>0.105</td>
      <td>-0.442</td>
      <td>-0.033</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2769.0</td>
      <td>3178.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_cov[2]</th>
      <td>0.299</td>
      <td>0.093</td>
      <td>0.120</td>
      <td>0.481</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>1379.0</td>
      <td>1308.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_within</th>
      <td>0.482</td>
      <td>0.027</td>
      <td>0.431</td>
      <td>0.534</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3773.0</td>
      <td>2542.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_cov_corr[0, 0]</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>4000.0</td>
      <td>4000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>chol_cov_corr[0, 1]</th>
      <td>-0.579</td>
      <td>0.192</td>
      <td>-0.898</td>
      <td>-0.196</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>3196.0</td>
      <td>2983.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_cov_corr[1, 0]</th>
      <td>-0.579</td>
      <td>0.192</td>
      <td>-0.898</td>
      <td>-0.196</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>3196.0</td>
      <td>2983.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_cov_corr[1, 1]</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>4087.0</td>
      <td>4000.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_cov_stds[0]</th>
      <td>0.988</td>
      <td>0.163</td>
      <td>0.710</td>
      <td>1.328</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>5207.0</td>
      <td>3263.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_cov_stds[1]</th>
      <td>0.386</td>
      <td>0.107</td>
      <td>0.182</td>
      <td>0.605</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>1541.0</td>
      <td>1201.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


# Comparison of `lmer` and `pymc` outputs

While `pymc` returns posterior estimates for each parameter, including $\rho$, for this post, we are interested in comparing the output comparable to the "fixed effects" and "varying effects" from `lmer`. Having the equations above can help us piece together the relevant bits of information. The fixed intercept and slope are easy because we've used the same characters $\alpha$ and $\beta$ in equation set 2 as we did in Equation set 1.

However, when identifying the "varying effects", we'll have to do some arithmetic with the `pymc` output. In contrast with the `lmer` output, the `pymc` outputs have the estimate for each cafe with "baked in" varying effects. In other words, the "offset" that we see in equation 7 ($a_{\text{cafe}[i]}$ and $b_{\text{cafe}[i]}$) 

$$ W_i = (\alpha + a_{\text{cafe}[i]}) + (\beta + b_{\text{cafe}[i]}) \times A_i + \epsilon_{\text{cafe}} \tag{7} $$

$$ \mu_i = \alpha_{\text{cafe}[i]} + \beta_{\text{cafe}[i]} \times A_{i} \tag{9}$$

are already embedded in ($\alpha_{\text{cafe}[i]}$ and $\beta_{\text{cafe}[i]}$) in equation 9. We'll have to therefore subtract out the fixed effecs in the `pymc` output before we can compare with the `lmer` output. First, let's get fixed effects from `pymc`.



```python
df_summary_int_and_slope = az.summary(idata_m14_1, var_names=['a_bar', 'b_bar'])
df_summary_int_and_slope
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_2.5%</th>
      <th>hdi_97.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a_bar</th>
      <td>3.654</td>
      <td>0.223</td>
      <td>3.203</td>
      <td>4.074</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>4802.0</td>
      <td>3140.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b_bar</th>
      <td>-1.049</td>
      <td>0.109</td>
      <td>-1.265</td>
      <td>-0.844</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3446.0</td>
      <td>3200.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



These estimates and uncertainties compare well with the fixed estimates `lmer`.


```python
f, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,4))
# value to generate data
# a, average morning wait time was defined above
ax0.vlines(x=a, ymin=0.8, ymax=1.2, linestyle='dashed', color='red')
ax1.vlines(x=b, ymin=0.8, ymax=1.2, linestyle='dashed', color='red', label='simulated value')

# pymc fixed effects value
ax0.scatter(df_summary_int_and_slope.loc['a_bar', 'mean'], 1.1, color='navy')
ax0.hlines(xmin=df_summary_int_and_slope.loc['a_bar', 'hdi_2.5%'], xmax=df_summary_int_and_slope.loc['a_bar', 'hdi_97.5%'], y=1.1, color='navy')
ax1.scatter(df_summary_int_and_slope.loc['b_bar', 'mean'], 1.1, color='navy')
ax1.hlines(xmin=df_summary_int_and_slope.loc['b_bar', 'hdi_2.5%'], xmax=df_summary_int_and_slope.loc['b_bar', 'hdi_97.5%'], y=1.1, color='navy', label='pymc estimate')

# lmer fixed effects estimate
ax0.scatter(df_fe_summary.loc['(Intercept)', 'Estimate'], 0.9, color='darkgreen')
ax0.hlines(xmin=df_fe_summary.loc['(Intercept)', 'X2.5..'], xmax=df_fe_summary.loc['(Intercept)', 'X97.5..'], y=0.9, color='darkgreen')
ax1.scatter(df_fe_summary.loc['afternoon', 'Estimate'], 0.9, color='darkgreen')
ax1.hlines(xmin=df_fe_summary.loc['afternoon', 'X2.5..'], xmax=df_fe_summary.loc['afternoon', 'X97.5..'], y=0.9, color='darkgreen', label='lmer estimate')

# plot formatting
f.suptitle('Fixed effect estimates')
ax0.set_yticks([0.9, 1.1])
ax0.set_yticklabels(['lmer', 'pymc'])

ax1.set_yticks([0.9, 1.1])
ax1.set_yticklabels(['', ''])

ax0.set(xlabel='intercept')
ax1.set(xlabel='slope')
ax1.legend(fontsize=10)
plt.tight_layout()

```

    /var/folders/tw/b9j0wcdj6_9cyljwt364lx7c0000gn/T/ipykernel_5516/1253574855.py:30: UserWarning: This figure was using constrained_layout, but that is incompatible with subplots_adjust and/or tight_layout; disabling constrained_layout.
      plt.tight_layout()



    
![png](/assets/2022-09-13-mixed_effects_freqvsbayes_cafes_files/2022-09-13-mixed_effects_freqvsbayes_cafes_46_1.png)
    


As promised, here is the meme that rewards you for paying attention this far!
![jpg](/assets/2022-09-13-mixed_effects_freqvsbayes_cafes_files/spideman_IMG_4672.JPG)



Now to get the varying effects from  `pymc` output, we'll take each sample's intercept and slope and subtract the fixed estimate.


```python
# Convert to pandas dataframe and take a glimpse at the first few rows
idata_m14_1_df = idata_m14_1.to_dataframe()
idata_m14_1_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chain</th>
      <th>draw</th>
      <th>(posterior, a_bar)</th>
      <th>(posterior, b_bar)</th>
      <th>(posterior, ab_subject[0,0], 0, 0)</th>
      <th>(posterior, ab_subject[0,1], 0, 1)</th>
      <th>(posterior, ab_subject[1,0], 1, 0)</th>
      <th>(posterior, ab_subject[1,1], 1, 1)</th>
      <th>(posterior, ab_subject[10,0], 10, 0)</th>
      <th>(posterior, ab_subject[10,1], 10, 1)</th>
      <th>(posterior, ab_subject[11,0], 11, 0)</th>
      <th>(posterior, ab_subject[11,1], 11, 1)</th>
      <th>(posterior, ab_subject[12,0], 12, 0)</th>
      <th>(posterior, ab_subject[12,1], 12, 1)</th>
      <th>(posterior, ab_subject[13,0], 13, 0)</th>
      <th>(posterior, ab_subject[13,1], 13, 1)</th>
      <th>(posterior, ab_subject[14,0], 14, 0)</th>
      <th>(posterior, ab_subject[14,1], 14, 1)</th>
      <th>(posterior, ab_subject[15,0], 15, 0)</th>
      <th>(posterior, ab_subject[15,1], 15, 1)</th>
      <th>...</th>
      <th>(log_likelihood, wait[97], 97)</th>
      <th>(log_likelihood, wait[98], 98)</th>
      <th>(log_likelihood, wait[99], 99)</th>
      <th>(log_likelihood, wait[9], 9)</th>
      <th>(sample_stats, tree_depth)</th>
      <th>(sample_stats, max_energy_error)</th>
      <th>(sample_stats, process_time_diff)</th>
      <th>(sample_stats, perf_counter_diff)</th>
      <th>(sample_stats, energy)</th>
      <th>(sample_stats, step_size_bar)</th>
      <th>(sample_stats, diverging)</th>
      <th>(sample_stats, energy_error)</th>
      <th>(sample_stats, lp)</th>
      <th>(sample_stats, acceptance_rate)</th>
      <th>(sample_stats, n_steps)</th>
      <th>(sample_stats, largest_eigval)</th>
      <th>(sample_stats, smallest_eigval)</th>
      <th>(sample_stats, index_in_trajectory)</th>
      <th>(sample_stats, step_size)</th>
      <th>(sample_stats, perf_counter_start)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>3.397744</td>
      <td>-0.993140</td>
      <td>2.353823</td>
      <td>-0.712216</td>
      <td>3.936642</td>
      <td>-1.328451</td>
      <td>2.497521</td>
      <td>-0.990675</td>
      <td>4.589760</td>
      <td>-1.271864</td>
      <td>2.272038</td>
      <td>-0.780358</td>
      <td>3.400074</td>
      <td>-1.307487</td>
      <td>4.660517</td>
      <td>-0.920542</td>
      <td>3.967868</td>
      <td>-1.339014</td>
      <td>...</td>
      <td>-0.592594</td>
      <td>-0.280869</td>
      <td>-1.783441</td>
      <td>-0.404212</td>
      <td>5</td>
      <td>-0.452539</td>
      <td>0.234946</td>
      <td>0.067260</td>
      <td>194.679539</td>
      <td>0.246795</td>
      <td>False</td>
      <td>-0.226605</td>
      <td>-167.432037</td>
      <td>0.975607</td>
      <td>31.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-17</td>
      <td>0.284311</td>
      <td>192.355518</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>3.227032</td>
      <td>-1.105823</td>
      <td>2.486742</td>
      <td>-0.657790</td>
      <td>3.890044</td>
      <td>-1.788579</td>
      <td>2.894867</td>
      <td>-0.741011</td>
      <td>4.346072</td>
      <td>-1.048541</td>
      <td>2.446301</td>
      <td>-0.678041</td>
      <td>3.564795</td>
      <td>-1.520221</td>
      <td>5.013627</td>
      <td>-1.128684</td>
      <td>3.793134</td>
      <td>-1.084814</td>
      <td>...</td>
      <td>-0.581570</td>
      <td>-0.708670</td>
      <td>-1.709776</td>
      <td>-0.741664</td>
      <td>4</td>
      <td>0.498338</td>
      <td>0.123327</td>
      <td>0.033713</td>
      <td>196.867266</td>
      <td>0.246795</td>
      <td>False</td>
      <td>0.273832</td>
      <td>-177.694232</td>
      <td>0.809115</td>
      <td>15.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-8</td>
      <td>0.284311</td>
      <td>192.423125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>3.393307</td>
      <td>-0.926431</td>
      <td>2.348434</td>
      <td>-0.604619</td>
      <td>3.905778</td>
      <td>-1.355137</td>
      <td>2.712834</td>
      <td>-1.124770</td>
      <td>4.409195</td>
      <td>-1.291088</td>
      <td>2.324233</td>
      <td>-0.754508</td>
      <td>3.586107</td>
      <td>-1.562165</td>
      <td>5.050191</td>
      <td>-1.556993</td>
      <td>4.122478</td>
      <td>-1.718417</td>
      <td>...</td>
      <td>-0.452885</td>
      <td>-0.109849</td>
      <td>-2.293094</td>
      <td>-0.559207</td>
      <td>5</td>
      <td>-0.382814</td>
      <td>0.236803</td>
      <td>0.063232</td>
      <td>207.926089</td>
      <td>0.246795</td>
      <td>False</td>
      <td>-0.347905</td>
      <td>-176.112370</td>
      <td>0.968229</td>
      <td>31.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6</td>
      <td>0.284311</td>
      <td>192.457135</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>3.750943</td>
      <td>-1.109148</td>
      <td>2.613325</td>
      <td>-0.667234</td>
      <td>3.682009</td>
      <td>-1.293790</td>
      <td>2.558511</td>
      <td>-0.362557</td>
      <td>4.548968</td>
      <td>-1.266139</td>
      <td>2.264383</td>
      <td>-0.445725</td>
      <td>3.102086</td>
      <td>-0.903726</td>
      <td>4.589499</td>
      <td>-0.409875</td>
      <td>4.063760</td>
      <td>-1.249921</td>
      <td>...</td>
      <td>-1.239451</td>
      <td>-0.574010</td>
      <td>-0.906557</td>
      <td>-1.015460</td>
      <td>4</td>
      <td>-0.530897</td>
      <td>0.116930</td>
      <td>0.037484</td>
      <td>198.279760</td>
      <td>0.246795</td>
      <td>False</td>
      <td>-0.024171</td>
      <td>-180.489888</td>
      <td>0.987683</td>
      <td>15.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-9</td>
      <td>0.284311</td>
      <td>192.520656</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>3.416951</td>
      <td>-1.152993</td>
      <td>2.478859</td>
      <td>-0.812085</td>
      <td>3.773041</td>
      <td>-1.423143</td>
      <td>2.136978</td>
      <td>-0.465100</td>
      <td>4.385045</td>
      <td>-1.180823</td>
      <td>2.160109</td>
      <td>-0.395771</td>
      <td>3.459758</td>
      <td>-1.300131</td>
      <td>5.527213</td>
      <td>-2.107117</td>
      <td>3.906480</td>
      <td>-1.388326</td>
      <td>...</td>
      <td>-0.400278</td>
      <td>-0.240346</td>
      <td>-2.188396</td>
      <td>-0.471960</td>
      <td>5</td>
      <td>-0.382498</td>
      <td>0.241781</td>
      <td>0.072736</td>
      <td>207.993298</td>
      <td>0.246795</td>
      <td>False</td>
      <td>-0.041904</td>
      <td>-183.942618</td>
      <td>0.999986</td>
      <td>31.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-24</td>
      <td>0.284311</td>
      <td>192.558443</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 270 columns</p>
</div>




```python
# Get the "unbaked in" varying intercept and slope
bayesian_int = list()
bayesian_slope = list()
for i in range(20):
    idata_m14_1_df[f'varying_int_{i}'] = idata_m14_1_df[ ('posterior', f'ab_subject[{i},0]', i, 0)] - idata_m14_1_df[('posterior', 'a_bar')]
    bayesian_int.append(idata_m14_1_df[f'varying_int_{i}'].mean())

    idata_m14_1_df[f'varying_slope_{i}'] = idata_m14_1_df[ ('posterior', f'ab_subject[{i},1]', i, 1)] - idata_m14_1_df[('posterior', 'b_bar')]
    bayesian_slope.append(idata_m14_1_df[f'varying_slope_{i}'].mean())
```

We can now make a direct comparison between the `lmer` and `pymc` outputs. I'll ignore the uncertainties for the sake of a cleaner plot.


```python
random_sims_int = random_sims.loc[random_sims['term']=='(Intercept)', 'mean'].copy()
random_sims_slope = random_sims.loc[random_sims['term']=='afternoon', 'mean'].copy()

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))

min_max_int = [min(list(random_sims_int) + bayesian_int), max(list(random_sims_int) + bayesian_int)]
min_max_slope = [min(list(random_sims_slope) + bayesian_slope), max(list(random_sims_slope) + bayesian_slope)]

# intercepts
ax0.scatter(random_sims_int, bayesian_int, facecolors='none', edgecolors='navy')
ax0.plot(min_max_int, min_max_int, linestyle='dashed', color='gray')
ax0.set(xlabel='lmer intercept estimates', ylabel='pymc intercept estimates', title='Comparison of varying intercepts')

# slopes
ax1.scatter(random_sims_slope, bayesian_slope, facecolors='none', edgecolors='navy')
ax1.plot(min_max_slope, min_max_slope, linestyle='dashed', color='gray')
ax1.set(xlabel='lmer slope estimates', ylabel='pymc slope estimates', title='Comparison of varying slopes')
```




    [Text(0.5, 0, 'lmer slope estimates'),
     Text(0, 0.5, 'pymc slope estimates'),
     Text(0.5, 1.0, 'Comparison of varying slopes')]




    
![png](/assets/2022-09-13-mixed_effects_freqvsbayes_cafes_files/2022-09-13-mixed_effects_freqvsbayes_cafes_53_1.png)
    


As you can see we get very similar intercepts and slopes for the cafe-specific estimates (varying effects) for the intercept and slope between the `lmer` and `pymc` approaches.

# Summary

Here in this post, I set out to compare different mixed model approaches. I looked at the equations and the programmatic implementations. I concluded by showing how the two methods can arrive at the same answer. It required a careful understanding of the differences in equations and coding language- and package-specific implementations. There were various points of writing this post that confused me but it provided opportunities for deeper understanding.

# Acknowledgements and references

Acknowledgements
- Special shoutout to Patrick Robotham (@probot) from the University of Bayes Discord channel for helping me work through *many* of my confusions.
- Eric J. Daza about some discussions about mixed effects modeling. It reminded me about improving my knowledge in this area.
- Members of the Glymour group at UCSF for checking some of my code.

References
- [UCLA introduction to linear mixed models](https://stats.oarc.ucla.edu/other/mult-pkg/introduction-to-linear-mixed-models/).
- Richard McElreath's Statistical Rethinking for my introduction to Bayesian multilevel modeling and the Statistical Rethinking repo.
- Andrzej Gałecki and Tomasz Burzykowski's [Linear Mixed-Effecsts Models Using R](https://link.springer.com/book/10.1007/978-1-4614-3900-4) which references the `lme4` package. Dr. McElreath referenced this package as a non-Bayesian alternative in his book.
- Andrew Gelman wrote about why he doesn't like using "fixed and random effects" (in a [blog](https://statmodeling.stat.columbia.edu/2005/01/25/why_i_dont_use/) and in a [paper](https://projecteuclid.org/journals/annals-of-statistics/volume-33/issue-1/Analysis-of-variancewhy-it-is-more-important-than-ever/10.1214/009053604000001048.full)).
- TJ Mahr's [partial pooling blog post](https://www.tjmahr.com/plotting-partial-pooling-in-mixed-effects-models/).


```python
%load_ext watermark
%watermark -n -u -v -iv -w -p aesara,aeppl
```

    The watermark extension is already loaded. To reload it, use:
      %reload_ext watermark
    Last updated: Tue Sep 13 2022
    
    Python implementation: CPython
    Python version       : 3.10.6
    IPython version      : 8.4.0
    
    aesara: 2.8.2
    aeppl : 0.0.35
    
    pymc      : 4.1.7
    xarray    : 2022.6.0
    pandas    : 1.4.3
    sys       : 3.10.6 | packaged by conda-forge | (main, Aug 22 2022, 20:43:44) [Clang 13.0.1 ]
    arviz     : 0.12.1
    matplotlib: 3.5.3
    aesara    : 2.8.2
    numpy     : 1.23.2
    
    Watermark: 2.3.1
    



```r
%%R
sessionInfo()
```

    R version 4.1.3 (2022-03-10)
    Platform: x86_64-apple-darwin13.4.0 (64-bit)
    Running under: macOS Monterey 12.5.1
    
    Matrix products: default
    LAPACK: /Users/blacar/opt/anaconda3/envs/pymc_env2/lib/libopenblasp-r0.3.21.dylib
    
    locale:
    [1] C/UTF-8/C/C/C/C
    
    attached base packages:
    [1] tools     stats     graphics  grDevices utils     datasets  methods  
    [8] base     
    
    other attached packages:
     [1] merTools_0.5.2  arm_1.13-1      MASS_7.3-58.1   lme4_1.1-30    
     [5] Matrix_1.4-1    forcats_0.5.2   stringr_1.4.1   dplyr_1.0.9    
     [9] purrr_0.3.4     readr_2.1.2     tidyr_1.2.0     tibble_3.1.8   
    [13] ggplot2_3.3.6   tidyverse_1.3.2
    
    loaded via a namespace (and not attached):
     [1] httr_1.4.4          jsonlite_1.8.0      splines_4.1.3      
     [4] foreach_1.5.2       modelr_0.1.9        shiny_1.7.2        
     [7] assertthat_0.2.1    broom.mixed_0.2.9.4 googlesheets4_1.0.1
    [10] cellranger_1.1.0    globals_0.16.1      pillar_1.8.1       
    [13] backports_1.4.1     lattice_0.20-45     glue_1.6.2         
    [16] digest_0.6.29       promises_1.2.0.1    rvest_1.0.3        
    [19] minqa_1.2.4         colorspace_2.0-3    httpuv_1.6.5       
    [22] htmltools_0.5.3     pkgconfig_2.0.3     broom_1.0.0        
    [25] listenv_0.8.0       haven_2.5.1         xtable_1.8-4       
    [28] mvtnorm_1.1-3       scales_1.2.1        later_1.3.0        
    [31] tzdb_0.3.0          googledrive_2.0.0   farver_2.1.1       
    [34] generics_0.1.3      ellipsis_0.3.2      withr_2.5.0        
    [37] furrr_0.3.1         cli_3.3.0           mime_0.12          
    [40] magrittr_2.0.3      crayon_1.5.1        readxl_1.4.1       
    [43] fs_1.5.2            future_1.27.0       fansi_1.0.3        
    [46] parallelly_1.32.1   nlme_3.1-159        xml2_1.3.3         
    [49] hms_1.1.2           gargle_1.2.0        lifecycle_1.0.1    
    [52] munsell_0.5.0       reprex_2.0.2        compiler_4.1.3     
    [55] rlang_1.0.4         blme_1.0-5          grid_4.1.3         
    [58] nloptr_2.0.3        iterators_1.0.14    labeling_0.4.2     
    [61] boot_1.3-28         gtable_0.3.0        codetools_0.2-18   
    [64] abind_1.4-5         DBI_1.1.3           R6_2.5.1           
    [67] lubridate_1.8.0     fastmap_1.1.0       utf8_1.2.2         
    [70] stringi_1.7.8       parallel_4.1.3      Rcpp_1.0.9         
    [73] vctrs_0.4.1         dbplyr_2.2.1        tidyselect_1.1.2   
    [76] coda_0.19-4        

