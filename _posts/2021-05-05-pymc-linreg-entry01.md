---
title: "PyMC linear regression part 1: PyMC objects"
mathjax: true
toc: true
toc_sticky: true
toc_label:  'Contents'
categories: [data science, statistics]
---

I previously [wrote](https://benslack19.github.io/prior-likelihood-posterior-predictive/) about my discovery of [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/). The book's title could not be more spot-on--it's helped me look at statistics in a different way. This approach will set me up (hopefully) for learning new methods that will prove useful in my work.

I'm doing the problems with the Python package [PyMC3](https://docs.pymc.io/). Fortunately, a repo for the book's code using this package has already been created [here](https://github.com/pymc-devs/resources/tree/master/Rethinking_2). This repo gave me a starting point to write my own code and compare to the book's, which is written in R. I've also never used the PyMC3 package before so I took this as an opportunity to dive deep into some of the package's objects. The [PyMC3 documentation](https://docs.pymc.io/notebooks/api_quickstart.html) of course was also helpful.

In a series of posts, I will address a linear regression problem using PyMC3. Here I aim to get a better understanding of the `pymc` objects.


```python
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymc3 as pm
import scipy.stats as stats
import seaborn as sns
```





```python
%load_ext nb_black
%config InlineBackend.figure_format = 'retina'
%load_ext watermark
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```



**Question**

Ths question come's from the [winter 2020, week 2 homework](https://github.com/rmcelreath/stat_rethinking_2020/blob/main/homework/week02/week02.pdf).

**The weights listed below were recorded in the !Kung census, but heights were not recorded for these individuals. Provide predicted heights and 89% compatibility intervals for each of these individuals. That is, fill in the table below, using model-based predictions.**

| Individual | weight | expected height | 89% interval |
| ------ | ------ | -------- | ---------- |
| 1|  45  |  |
| 2 | 40 |  |  |
| 3 | 65 | |  |
|4  | 31  |  |  |

Let's quickly take a look at the data to get a handle on what we're working with.


```python
DATA_DIR = '/Users/blacar/Documents/ds_projects/stats_rethinking/pymc3_ed_resources/resources/Rethinking_2/Data/'
d = pd.read_csv(DATA_DIR + "Howell1.csv", sep=";", header=0)
d2 = d[d.age >= 18]  # filter to get only adults

d2.head()
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
      <th>height</th>
      <th>weight</th>
      <th>age</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>151.765</td>
      <td>47.825606</td>
      <td>63.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139.700</td>
      <td>36.485807</td>
      <td>63.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>136.525</td>
      <td>31.864838</td>
      <td>65.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>156.845</td>
      <td>53.041914</td>
      <td>41.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>145.415</td>
      <td>41.276872</td>
      <td>51.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>







```python
f, ax1 = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=d2, x="weight", y="height", color="gray")
```




    <AxesSubplot:xlabel='weight', ylabel='height'>




    
![png](/assets/2021-05-05-pymc-linreg-entry01_files/2021-05-05-pymc-linreg-entry01_6_1.png)
    





We'd expect a linear relationship between `weight` and `height` in the adult population and that's what we see here.

# Setting up the variables, producing `model ` and `trace` objects

First, I'll need to generate the posterior distribution from the entire dataset. I'll define the variables:

$\text{height}_i$ ~ Normal($\mu_i, \sigma$)
<br>
$\mu_i = \alpha + \beta(x_i  - \bar{x})$
<br>
$\alpha$ ~ $\text{Normal}(178, 20)$
<br>
$\beta$ ~ $\text{Log-Normal}(0, 1)$
<br>
$\sigma$ ~ $\text{Uniform}(0, 50)$

Important things to note about these variables.
- Whether the subscript *i* is there for a given variable matters. It will represent every row in the set of parameters returned by the function. We'll come back to this later.
- The tilde versus the equals sign matters. The former represents a stochastic relationship while the latter is deterministic.
- I used a prior for beta that will have a sensical relationship, such as being all positive. One way to accomplish this is to use lognormal.




```python
# Get the average weight as part of the model definition
xbar = d2.weight.mean()
```





```python
with pm.Model() as heights_model:

    # Priors are variables a, b, sigma
    # using pm.Normal is a way to represent the stochastic relationship the left has to right side of equation
    a = pm.Normal("a", mu=178, sd=20)
    b = pm.Lognormal("b", mu=0, sd=1)
    sigma = pm.Uniform("sigma", 0, 50)

    # This is a linear model (not really a prior or likelihood?)
    # Data included here (d2.weight, which is observed)
    # Mu is deterministic, but a and b are stochastic
    mu = a + b * (d2.weight - xbar)

    # Likelihood is height variable, which is also observed (data included here, d2.height))
    # Height is dependent on deterministic and stochastic variables
    height = pm.Normal("height", mu=mu, sd=sigma, observed=d2.height)

    # The next lines is doing the fitting and sampling all at once.
    # When I ran this without the return_inferencedata parameter set, I got a
    # warning and suggestion to set this explicitly. I wanted to see the difference.
    # Per the documentation: "With PyMC3 version >=3.9 the return_inferencedata=True
    # kwarg makes the sample function return an arviz.InferenceData object instead
    # of a MultiTrace. InferenceData has many advantages, compared to a MultiTrace.
    trace_m1 = pm.sample(1000, tune=1000, return_inferencedata=True)  #
    trace_m2 = pm.sample(1000, tune=1000, return_inferencedata=False)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [sigma, b, a]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 00:07<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 28 seconds.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [sigma, b, a]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 00:07<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 23 seconds.





# Inspecting the `pymc3` objects

Before going through the rest of the exercise, it is helpful to dive deep and understand the objects in this call.

## Inspecting the `model` object


```python
heights_model
```




$$
                \begin{array}{rcl}
                \text{a} &\sim & \text{Normal}\\\text{b_log__} &\sim & \text{TransformedDistribution}\\\text{sigma_interval__} &\sim & \text{TransformedDistribution}\\\text{b} &\sim & \text{Lognormal}\\\text{sigma} &\sim & \text{Uniform}\\\text{height} &\sim & \text{Normal}
                \end{array}
                $$






It is comforting that this object is a simple reflection of the input.

Now let's take a look at the object's properties and methods, using the Python `dir` function. It returned a long list so let's limit using `filter` and a lambda function to those without the double underscore. (If you really want to understand what the attribues with underscores are, you can look [here](https://stackoverflow.com/questions/1301346/what-is-the-meaning-of-single-and-double-underscore-before-an-object-name).)


```python
# But this is a way to limit the attributes to inspect
heights_model_methods2check = list(filter(lambda x: "__" not in x, dir(heights_model)))
heights_model_methods2check
```




    ['RV_dims',
     'Var',
     '_cache',
     '_config_context',
     '_context_class',
     '_parent',
     '_repr_latex_',
     '_str_repr',
     '_theano_config',
     'a',
     'add_coords',
     'add_random_variable',
     'b',
     'basic_RVs',
     'bijection',
     'check_bounds',
     'check_test_point',
     'cont_vars',
     'contexts',
     'coords',
     'd2logp',
     'd2logp_nojac',
     'datalogpt',
     'deterministics',
     'dict_to_array',
     'disc_vars',
     'dlogp',
     'dlogp_array',
     'dlogp_nojac',
     'fastd2logp',
     'fastd2logp_nojac',
     'fastdlogp',
     'fastdlogp_nojac',
     'fastfn',
     'fastlogp',
     'fastlogp_nojac',
     'flatten',
     'fn',
     'free_RVs',
     'height',
     'isroot',
     'logp',
     'logp_array',
     'logp_dlogp_function',
     'logp_elemwise',
     'logp_nojac',
     'logp_nojact',
     'logpt',
     'makefn',
     'missing_values',
     'model',
     'name',
     'name_for',
     'name_of',
     'named_vars',
     'ndim',
     'observed_RVs',
     'parent',
     'potentials',
     'prefix',
     'profile',
     'root',
     'shape_from_dims',
     'sigma',
     'test_point',
     'unobserved_RVs',
     'varlogpt',
     'vars']






It is still a pretty long list but we can poke around at a few of the attributes.


```python
heights_model.basic_RVs
```




    [a ~ Normal,
     b_log__ ~ TransformedDistribution,
     sigma_interval__ ~ TransformedDistribution,
     height ~ Normal]







```python
heights_model.height
```




$\text{height} \sim \text{Normal}(\mathit{mu}=f(f(\text{a}),~f(f(\text{b}),~array)),~\mathit{sigma}=f(\text{sigma}))$






Most of the information related to our input. Let's look at the `trace` objects, which were a result of the `pm.sample` command.

## Inspecting the `trace` variables

The `trace` object contains the samples collected, in the order they were collected per the [getting started](https://docs.pymc.io/notebooks/getting_started.html) tutorial. This is an important object so I wanted to dive deeper into it.

Note that I used the variable name "trace" simply because the repo version produced a `MultiTrace` object, which is what happens when the `return_inferencedata` flag is not set. As you'll see, in the first example, it is probably not the best name, but I'll leave the names alone for now.

### `trace_m1` is a `arviz.InferenceData`

Let's start off by simply looking at the object.


```python
type(trace_m1)
```




    arviz.data.inference_data.InferenceData




We can examine the output of `trace_m1` but the output won't render cleanly on this site. You can inspect it within your juptyer notebook.



```python
trace_m1
```

<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */



This was good to see, but I could go down a whole 'nother rabbit hole with just this object's attributes and functions. I'll leave it to [this Arviz page](https://arviz-devs.github.io/arviz/getting_started/XarrayforArviZ.html) to explain this object in more detail. I'll focus on the `trace_m2` object but here is a way to connect it with the posterior data from `trace_m1`.


```python
trace_m1.posterior
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:  (chain: 4, draw: 1000)
Coordinates:
  * chain    (chain) int64 0 1 2 3
  * draw     (draw) int64 0 1 2 3 4 5 6 7 8 ... 992 993 994 995 996 997 998 999
Data variables:
    a        (chain, draw) float64 154.0 154.0 155.1 154.1 ... 154.9 154.3 154.9
    b        (chain, draw) float64 0.8538 0.8616 0.9578 ... 0.866 0.9321 0.8399
    sigma    (chain, draw) float64 4.989 5.129 5.068 4.961 ... 5.09 5.019 5.242
Attributes:
    created_at:                 2022-04-22T19:06:49.969514
    arviz_version:              0.11.1
    inference_library:          pymc3
    inference_library_version:  3.11.0
    sampling_time:              27.866599082946777
    tuning_steps:               1000</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-47111624-66b9-47fa-b405-df306a3364f0' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-47111624-66b9-47fa-b405-df306a3364f0' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 4</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-087bdb3f-7bb5-49b3-a43d-ff1b3b6b94f6' class='xr-section-summary-in' type='checkbox'  checked><label for='section-087bdb3f-7bb5-49b3-a43d-ff1b3b6b94f6' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3</div><input id='attrs-6d56d2bf-1385-4988-99c0-f0f8e79bf612' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6d56d2bf-1385-4988-99c0-f0f8e79bf612' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0408a6e5-a14a-4156-af21-810c1b532c1d' class='xr-var-data-in' type='checkbox'><label for='data-0408a6e5-a14a-4156-af21-810c1b532c1d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-0bb3c704-4456-40f4-a9f4-904a70b80983' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0bb3c704-4456-40f4-a9f4-904a70b80983' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d10434d8-7364-4ed4-8790-0f89bbc81aae' class='xr-var-data-in' type='checkbox'><label for='data-d10434d8-7364-4ed4-8790-0f89bbc81aae' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8bc3d2bb-bb23-4910-ac46-7644f0193b74' class='xr-section-summary-in' type='checkbox'  checked><label for='section-8bc3d2bb-bb23-4910-ac46-7644f0193b74' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>a</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>154.0 154.0 155.1 ... 154.3 154.9</div><input id='attrs-7123a5a7-6cae-45e4-b9bf-d799dd4ddbf6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7123a5a7-6cae-45e4-b9bf-d799dd4ddbf6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c802de05-8a28-428f-9da3-120d8e74d3ff' class='xr-var-data-in' type='checkbox'><label for='data-c802de05-8a28-428f-9da3-120d8e74d3ff' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[154.03058495, 153.96997324, 155.11106991, ..., 154.63468783,
        154.76213813, 154.29138176],
       [155.10811531, 153.96269524, 154.65322055, ..., 154.3681408 ,
        155.49068922, 154.87128055],
       [154.10306815, 154.75601267, 154.3996128 , ..., 154.5419651 ,
        154.59997877, 154.41866941],
       [154.57285029, 154.61464995, 154.66532063, ..., 154.86002136,
        154.28550752, 154.87498365]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>b</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8538 0.8616 ... 0.9321 0.8399</div><input id='attrs-de9b6d77-a78a-42a1-aae8-a1d8534e1296' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-de9b6d77-a78a-42a1-aae8-a1d8534e1296' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-80cc45dc-e313-4053-ad63-cd79ed45a4af' class='xr-var-data-in' type='checkbox'><label for='data-80cc45dc-e313-4053-ad63-cd79ed45a4af' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.85378765, 0.86161909, 0.95776919, ..., 0.94140608, 0.88971675,
        0.91204005],
       [0.99177163, 0.82261803, 0.91716301, ..., 0.92739364, 0.8581938 ,
        0.9073712 ],
       [0.87734415, 0.88182227, 0.90922174, ..., 0.89340121, 0.90167945,
        0.92521348],
       [0.89919471, 0.89494823, 0.91869996, ..., 0.86604634, 0.93210088,
        0.83992027]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>4.989 5.129 5.068 ... 5.019 5.242</div><input id='attrs-01934606-be36-4d6b-8bc5-2fdbe44b72b9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-01934606-be36-4d6b-8bc5-2fdbe44b72b9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9081ce6c-cfde-47fb-922e-1ba1f27161b0' class='xr-var-data-in' type='checkbox'><label for='data-9081ce6c-cfde-47fb-922e-1ba1f27161b0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[4.98944173, 5.12879049, 5.06756793, ..., 4.86557056, 5.27635179,
        5.27497752],
       [5.16986056, 4.87706806, 5.15322605, ..., 5.31173081, 5.22379536,
        4.91267767],
       [4.8309535 , 5.32888694, 4.88953936, ..., 5.21380319, 4.9855454 ,
        5.1213464 ],
       [5.35629738, 5.20270692, 5.00285615, ..., 5.0899661 , 5.01923403,
        5.2419566 ]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-55c56b0c-19a5-4d20-9432-d24d08c16102' class='xr-section-summary-in' type='checkbox'  checked><label for='section-55c56b0c-19a5-4d20-9432-d24d08c16102' class='xr-section-summary' >Attributes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2022-04-22T19:06:49.969514</dd><dt><span>arviz_version :</span></dt><dd>0.11.1</dd><dt><span>inference_library :</span></dt><dd>pymc3</dd><dt><span>inference_library_version :</span></dt><dd>3.11.0</dd><dt><span>sampling_time :</span></dt><dd>27.866599082946777</dd><dt><span>tuning_steps :</span></dt><dd>1000</dd></dl></div></li></ul></div></div>






### `trace_m2` is a `MultiTrace` object

The `MultiTrace` object that is outputted as a result of setting `return_inferencedata=False` in the `pm.sample()` call. (It's the same object that is outputted in the PyMC3 repo of the book's code which is why I decided to work with it here.) This is an important object so I wanted to dive deeper into it.


```python
type(trace_m2)
```




    pymc3.backends.base.MultiTrace






The trace object contains the samples collected, in the order they were collected per the PyMC3's. [getting started tutorial](https://docs.pymc.io/notebooks/getting_started.html).  It is much clearer to see this concretely when we use the `trace_to_dataframe` function.


```python
trace_m2_df = pm.trace_to_dataframe(trace_m2)
trace_m2_df
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
      <th>a</th>
      <th>b</th>
      <th>sigma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>154.589194</td>
      <td>0.889071</td>
      <td>4.885394</td>
    </tr>
    <tr>
      <th>1</th>
      <td>154.589194</td>
      <td>0.889071</td>
      <td>4.885394</td>
    </tr>
    <tr>
      <th>2</th>
      <td>154.818923</td>
      <td>0.838186</td>
      <td>5.271951</td>
    </tr>
    <tr>
      <th>3</th>
      <td>154.920790</td>
      <td>0.987778</td>
      <td>5.107954</td>
    </tr>
    <tr>
      <th>4</th>
      <td>154.920790</td>
      <td>0.987778</td>
      <td>5.107954</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3995</th>
      <td>154.393120</td>
      <td>0.915738</td>
      <td>5.082579</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>155.231970</td>
      <td>0.894059</td>
      <td>5.201840</td>
    </tr>
    <tr>
      <th>3997</th>
      <td>154.932506</td>
      <td>0.905143</td>
      <td>5.038751</td>
    </tr>
    <tr>
      <th>3998</th>
      <td>154.841507</td>
      <td>0.881522</td>
      <td>5.391670</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>154.404171</td>
      <td>0.881739</td>
      <td>4.925559</td>
    </tr>
  </tbody>
</table>
<p>4000 rows × 3 columns</p>
</div>






Why are there 4000 rows? It can be explained by looking at some of the earlier code and the output.

This was in the initial model fitting:
<br>
`trace_m = pm.sample(1000, tune=1000)`
<br>
`Multiprocess sampling (4 chains in 4 jobs)`

I don't know how to take into account this multiprocess sampling, so I'll treat the 4000 rows as simply being different samples drawn from the posterior distribution. More explanation is shown [here](https://stackoverflow.com/questions/61969968/understanding-the-parameters-of-pymc3-package).

As you can see, each row is a different instantiation of alpha, beta, and sigma values. Each row is a different "*i*th" set of values that is in these equations.

$\text{height}_i$ ~ Normal($\mu_i, \sigma$)
<br>
$\mu_i = \alpha + \beta(x_i  - \bar{x})$

We'll come back to this. We can also check out the object attributes.


```python
trace_m2_methods2check = list(filter(lambda x: "__" not in x, dir(trace_m2)))
trace_m2_methods2check
```




    ['_attrs',
     '_report',
     '_slice',
     '_straces',
     'add_values',
     'chains',
     'get_sampler_stats',
     'get_values',
     'nchains',
     'point',
     'points',
     'remove_values',
     'report',
     'stat_names',
     'varnames']






Let's inspect some of the object methods.


```python
# Trace object's variable names
trace_m2.varnames
```




    ['a', 'b_log__', 'sigma_interval__', 'b', 'sigma']







```python
# Inspect samples, which are parameters
print("a samples: ", trace_m2["a"][0:5])
print("b samples: ", trace_m2["b"][0:5])
print("sigma samples: ", trace_m2["sigma"][0:5])
# print("mu samples: ", trace_m["mu"][0:5])
# print("height samples: ", trace_m["height"][0:5])

print("mu and height are deterministic or dependent on deterministic variables")
```

    a samples:  [154.58919424 154.58919424 154.81892314 154.92078981 154.92078981]
    b samples:  [0.88907119 0.88907119 0.83818599 0.9877783  0.9877783 ]
    sigma samples:  [4.88539414 4.88539414 5.27195133 5.10795395 5.10795395]
    mu and height are deterministic or dependent on deterministic variables






```python
# Another way to inspect
trace_m2.get_values("a")[0:5]
```




    array([154.58919424, 154.58919424, 154.81892314, 154.92078981,
           154.92078981])






A chain is a single run of Markov Chain Monte Carlo. I haven't learned MCMC yet, but chains in `pymc3` are explained [here](https://stackoverflow.com/questions/49825216/what-is-a-chain-in-pymc3).


```python
trace_m2.chains
```




    [0, 1, 2, 3]






## Code explanation

With regards to the sampling in the code above. This is taken from the PyMC example notebook.

> We could use a quadratic approximation like McElreath does in his book and we did in code 2.6. But Using PyMC3 is really simple to just sample from the model using a "sampler method". Most common sampler methods are members of the Markov Chain Monte Carlo Method (MCMC) family (for details read Section 2.4.3 and Chapter 8 of Statistical Rethinking).

> PyMC3 comes with various samplers. Some samplers are more suited than others for certain type of variable (and/or problems). For now we are going to let PyMC3 choose the sampler for us. PyMC3 also tries to provide a reasonable starting point for the simulation. By default PyMC3 uses the same adaptive procedure as in STAN `'jitter+adapt_diag'`, which starts with a identity mass matrix and then adapts a diagonal based on the variance of the tuning samples.

> You can read more details of PyMC3 [here](https://docs.pymc.io/notebooks/getting_started.html).

This is taken from the link which helps explain the code.

> **Gradient-based sampling methods**

> PyMC3 has the standard sampling algorithms like adaptive Metropolis-Hastings and adaptive slice sampling, but PyMC3’s most capable step method is the No-U-Turn Sampler. NUTS is especially useful on models that have many continuous parameters, a situation where other MCMC algorithms work very slowly. It takes advantage of information about where regions of higher probability are, based on the gradient of the log posterior-density. This helps it achieve dramatically faster convergence on large problems than traditional sampling methods achieve. PyMC3 relies on Theano to analytically compute model gradients via automatic differentiation of the posterior density. NUTS also has several self-tuning strategies for adaptively setting the tunable parameters of Hamiltonian Monte Carlo. For random variables that are undifferentiable (namely, discrete variables) NUTS cannot be used, but it may still be used on the differentiable variables in a model that contains undifferentiable variables.

> NUTS requires a scaling matrix parameter, which is analogous to the variance parameter for the jump proposal distribution in Metropolis-Hastings, although NUTS uses it somewhat differently. The matrix gives the rough shape of the distribution so that NUTS does not make jumps that are too large in some directions and too small in other directions. It is important to set this scaling parameter to a reasonable value to facilitate efficient sampling. This is especially true for models that have many unobserved stochastic random variables or models with highly non-normal posterior distributions. Poor scaling parameters will slow down NUTS significantly, sometimes almost stopping it completely. A reasonable starting point for sampling can also be important for efficient sampling, but not as often.

> PyMC3 automatically initializes NUTS to reasonable values based on the variance of the samples obtained during a tuning phase. A little bit of noise is added to ensure different, parallel, chains start from different points. Also, PyMC3 will automatically assign an appropriate sampler if we don’t supply it via the step keyword argument...

# Summary

In this post, I wanted to get my feet wet with using `pymc`. The main objects are created with the model definition (the code block starting with `with pm.Model() as heights_model` in this case). We see that we can get the posterior distribution in our `trace_m1` and `trace_m2` objects, the difference here is what we get back using the `return_inferencedata` flag. In the next post, we'll look closer at the posterior distribution and interpret it.


```python
%watermark -n -u -v -iv -w
```

    Last updated: Fri Apr 22 2022
    
    Python implementation: CPython
    Python version       : 3.8.6
    IPython version      : 7.20.0
    
    matplotlib: 3.3.4
    pandas    : 1.2.1
    pymc3     : 3.11.0
    seaborn   : 0.11.1
    sys       : 3.8.6 | packaged by conda-forge | (default, Jan 25 2021, 23:22:12) 
    [Clang 11.0.1 ]
    arviz     : 0.11.1
    scipy     : 1.6.0
    numpy     : 1.20.1
    
    Watermark: 2.1.0
    




