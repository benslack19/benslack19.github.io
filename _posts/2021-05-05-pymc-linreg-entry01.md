---
title: "PyMC objects with linear regression (part 1)"
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


    <IPython.core.display.Javascript object>


Of course, it's good to be cognizant of package versions so I am making that clear here.


```python
%watermark -n -u -v -iv -w
```

    Last updated: Wed May 05 2021
    
    Python implementation: CPython
    Python version       : 3.8.6
    IPython version      : 7.20.0
    
    matplotlib: 3.3.4
    seaborn   : 0.11.1
    numpy     : 1.20.1
    pymc3     : 3.11.0
    pandas    : 1.2.1
    scipy     : 1.6.0
    json      : 2.0.9
    arviz     : 0.11.1
    
    Watermark: 2.1.0
    



    <IPython.core.display.Javascript object>


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
d = pd.read_csv("../data/a_input/Howell1.csv", sep=";", header=0)
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




    <IPython.core.display.Javascript object>



```python
f, ax1 = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=d2, x="weight", y="height", color="gray")
```




    <AxesSubplot:xlabel='weight', ylabel='height'>




![png](/assets/2021-05-05-pymc-linreg-entry01_files/2021-05-05-pymc-linreg-entry01_8_1.png)



    <IPython.core.display.Javascript object>


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


    <IPython.core.display.Javascript object>



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
      100.00% [8000/8000 00:02<00:00 Sampling 4 chains, 0 divergences]
    </div>
    


    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 12 seconds.
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
      100.00% [8000/8000 00:02<00:00 Sampling 4 chains, 0 divergences]
    </div>
    


    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 12 seconds.



    <IPython.core.display.Javascript object>


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




    <IPython.core.display.Javascript object>


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




    <IPython.core.display.Javascript object>


It is still a pretty long list but we can poke around at a few of the attributes.


```python
heights_model.basic_RVs
```




    [a ~ Normal,
     b_log__ ~ TransformedDistribution,
     sigma_interval__ ~ TransformedDistribution,
     height ~ Normal]




    <IPython.core.display.Javascript object>



```python
heights_model.height
```




$\text{height} \sim \text{Normal}(\mathit{mu}=f(f(\text{a}),~f(f(\text{b}),~array)),~\mathit{sigma}=f(\text{sigma}))$




    <IPython.core.display.Javascript object>


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




    <IPython.core.display.Javascript object>



```python
trace_m1
```





            <div>
              <div class='xr-header'>
                <div class="xr-obj-type">arviz.InferenceData</div>
              </div>
              <ul class="xr-sections group-sections">
              
            <li class = "xr-section-item">
                  <input id="idata_posteriorc094c270-6fd7-404a-8f6d-6c7404a820c5" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_posteriorc094c270-6fd7-404a-8f6d-6c7404a820c5" class = "xr-section-summary">posterior</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
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
    created_at:                 2021-05-05T14:18:00.979266
    arviz_version:              0.11.1
    inference_library:          pymc3
    inference_library_version:  3.11.0
    sampling_time:              11.833754777908325
    tuning_steps:               1000</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-5dc514da-2c99-4cbc-8960-974f6503f43e' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-5dc514da-2c99-4cbc-8960-974f6503f43e' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 4</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-9f7d7962-3407-4721-bb27-d3fd0ed6bc3f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-9f7d7962-3407-4721-bb27-d3fd0ed6bc3f' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3</div><input id='attrs-bf86b1dc-feaf-40bb-8a52-b00970c68481' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bf86b1dc-feaf-40bb-8a52-b00970c68481' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3f51c02f-fe1a-4e5b-a74c-af0510edeee0' class='xr-var-data-in' type='checkbox'><label for='data-3f51c02f-fe1a-4e5b-a74c-af0510edeee0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-e153b273-c069-43e0-b2ae-6f2a42739cff' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e153b273-c069-43e0-b2ae-6f2a42739cff' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bdbd1d2a-c6e0-47c0-b987-a0b20a2067d8' class='xr-var-data-in' type='checkbox'><label for='data-bdbd1d2a-c6e0-47c0-b987-a0b20a2067d8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e7009a45-a058-4d55-aba8-ad9baed04d52' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e7009a45-a058-4d55-aba8-ad9baed04d52' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>a</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>154.0 154.0 155.1 ... 154.3 154.9</div><input id='attrs-36a65956-f7ed-40f2-b0a4-12d51911676e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-36a65956-f7ed-40f2-b0a4-12d51911676e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eb3f58cb-226e-481b-97b4-7afbb36c14ac' class='xr-var-data-in' type='checkbox'><label for='data-eb3f58cb-226e-481b-97b4-7afbb36c14ac' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[154.03058495, 153.96997324, 155.11106991, ..., 154.63468783,
        154.76213813, 154.29138176],
       [155.10811531, 153.96269524, 154.65322055, ..., 154.3681408 ,
        155.49068922, 154.87128055],
       [154.10306815, 154.75601267, 154.3996128 , ..., 154.5419651 ,
        154.59997877, 154.41866941],
       [154.57285029, 154.61464995, 154.66532063, ..., 154.86002136,
        154.28550752, 154.87498365]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>b</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8538 0.8616 ... 0.9321 0.8399</div><input id='attrs-617d3cc8-9d0b-412b-86f6-71d2d094ad5a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-617d3cc8-9d0b-412b-86f6-71d2d094ad5a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2dc1377e-25ce-4717-95f5-78ff64d3525e' class='xr-var-data-in' type='checkbox'><label for='data-2dc1377e-25ce-4717-95f5-78ff64d3525e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.85378765, 0.86161909, 0.95776919, ..., 0.94140608, 0.88971675,
        0.91204005],
       [0.99177163, 0.82261803, 0.91716301, ..., 0.92739364, 0.8581938 ,
        0.9073712 ],
       [0.87734415, 0.88182227, 0.90922174, ..., 0.89340121, 0.90167945,
        0.92521348],
       [0.89919471, 0.89494823, 0.91869996, ..., 0.86604634, 0.93210088,
        0.83992027]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>4.989 5.129 5.068 ... 5.019 5.242</div><input id='attrs-a5168d29-6fe0-429a-9984-1413787231ee' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a5168d29-6fe0-429a-9984-1413787231ee' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-418067e1-0466-447d-9bf8-6577d294b97f' class='xr-var-data-in' type='checkbox'><label for='data-418067e1-0466-447d-9bf8-6577d294b97f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[4.98944173, 5.12879049, 5.06756793, ..., 4.86557056, 5.27635179,
        5.27497752],
       [5.16986056, 4.87706806, 5.15322605, ..., 5.31173081, 5.22379536,
        4.91267767],
       [4.8309535 , 5.32888694, 4.88953936, ..., 5.21380319, 4.9855454 ,
        5.1213464 ],
       [5.35629738, 5.20270692, 5.00285615, ..., 5.0899661 , 5.01923403,
        5.2419566 ]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f93e807b-93e3-4d68-9bb9-83b4ad03de60' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f93e807b-93e3-4d68-9bb9-83b4ad03de60' class='xr-section-summary' >Attributes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2021-05-05T14:18:00.979266</dd><dt><span>arviz_version :</span></dt><dd>0.11.1</dd><dt><span>inference_library :</span></dt><dd>pymc3</dd><dt><span>inference_library_version :</span></dt><dd>3.11.0</dd><dt><span>sampling_time :</span></dt><dd>11.833754777908325</dd><dt><span>tuning_steps :</span></dt><dd>1000</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            
            <li class = "xr-section-item">
                  <input id="idata_log_likelihood7ad649af-8422-4525-a44e-b163a1e6e033" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_log_likelihood7ad649af-8422-4525-a44e-b163a1e6e033" class = "xr-section-summary">log_likelihood</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
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
Dimensions:       (chain: 4, draw: 1000, height_dim_0: 352)
Coordinates:
  * chain         (chain) int64 0 1 2 3
  * draw          (draw) int64 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
  * height_dim_0  (height_dim_0) int64 0 1 2 3 4 5 6 ... 346 347 348 349 350 351
Data variables:
    height        (chain, draw, height_dim_0) float64 -2.967 -3.53 ... -2.686
Attributes:
    created_at:                 2021-05-05T14:18:01.350474
    arviz_version:              0.11.1
    inference_library:          pymc3
    inference_library_version:  3.11.0</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-61452d13-c523-4cd3-aa7f-1e9dd1d7d2de' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-61452d13-c523-4cd3-aa7f-1e9dd1d7d2de' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 4</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>height_dim_0</span>: 352</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-9841662e-27eb-4aa2-8abb-54528d363ab5' class='xr-section-summary-in' type='checkbox'  checked><label for='section-9841662e-27eb-4aa2-8abb-54528d363ab5' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3</div><input id='attrs-35f387b3-dc8b-4b96-bc79-19c56276487a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-35f387b3-dc8b-4b96-bc79-19c56276487a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4e7c7154-3979-4eb0-a969-fcf05552b031' class='xr-var-data-in' type='checkbox'><label for='data-4e7c7154-3979-4eb0-a969-fcf05552b031' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-9585176e-3b00-4566-9e87-f7d18291640e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9585176e-3b00-4566-9e87-f7d18291640e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-92aed5d9-2dea-4f08-9266-5eebcf7280f4' class='xr-var-data-in' type='checkbox'><label for='data-92aed5d9-2dea-4f08-9266-5eebcf7280f4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>height_dim_0</span></div><div class='xr-var-dims'>(height_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 347 348 349 350 351</div><input id='attrs-3c96928d-fb72-40ce-8382-ca5f7abf98d1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3c96928d-fb72-40ce-8382-ca5f7abf98d1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f0022c4c-9adc-464c-aaf4-f1844b9e8a70' class='xr-var-data-in' type='checkbox'><label for='data-f0022c4c-9adc-464c-aaf4-f1844b9e8a70' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 349, 350, 351])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1f8214a8-53a6-4f1d-894d-88bdf1b17198' class='xr-section-summary-in' type='checkbox'  checked><label for='section-1f8214a8-53a6-4f1d-894d-88bdf1b17198' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>height</span></div><div class='xr-var-dims'>(chain, draw, height_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-2.967 -3.53 ... -3.294 -2.686</div><input id='attrs-22431cac-7a32-480d-8bca-56030b206f71' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-22431cac-7a32-480d-8bca-56030b206f71' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e030538f-aed8-49f3-9ad9-ca6ee50edc01' class='xr-var-data-in' type='checkbox'><label for='data-e030538f-aed8-49f3-9ad9-ca6ee50edc01' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-2.9673283 , -3.5300242 , -3.32319013, ..., -2.64248558,
         -3.14852964, -2.58562014],
        [-2.9644178 , -3.46988493, -3.26939686, ..., -2.66420808,
         -3.14493116, -2.60988281],
        [-3.257161  , -3.5695978 , -3.246173  , ..., -2.54833217,
         -3.66343082, -2.79185925],
        ...,
        [-3.14903486, -3.51493995, -3.20017511, ..., -2.53018212,
         -3.52573919, -2.68917918],
        [-3.12933628, -3.59117108, -3.35482149, ..., -2.6181984 ,
         -3.37012288, -2.71520537],
        [-3.05151698, -3.42132864, -3.18541043, ..., -2.63550195,
         -3.30771329, -2.68707836]],

       [[-3.27047287, -3.47149533, -3.14123445, ..., -2.56392536,
         -3.72797407, -2.83723355],
        [-2.93483578, -3.61345992, -3.43037213, ..., -2.65638507,
         -3.07528183, -2.54564115],
        [-3.12573657, -3.52193199, -3.25683655, ..., -2.59178386,
         -3.41992124, -2.70825713],
...
        [-3.08884247, -3.53541535, -3.29807898, ..., -2.61792729,
         -3.33235467, -2.68791013],
        [-3.11019005, -3.57744217, -3.30871844, ..., -2.57029959,
         -3.39379903, -2.66671153],
        [-3.08316291, -3.44686777, -3.18256333, ..., -2.59554369,
         -3.38331797, -2.68580825]],

       [[-3.09737555, -3.50707237, -3.27696656, ..., -2.63841384,
         -3.33815154, -2.71536766],
        [-3.10415485, -3.55340232, -3.31127972, ..., -2.61114668,
         -3.35423957, -2.69429518],
        [-3.13434432, -3.55082251, -3.26786619, ..., -2.56298046,
         -3.4499164 , -2.69045729],
        ...,
        [-3.1407533 , -3.71874285, -3.48313926, ..., -2.58895131,
         -3.36331039, -2.68081923],
        [-3.06129497, -3.41208961, -3.13829514, ..., -2.58232238,
         -3.37890872, -2.662754  ],
        [-3.12432337, -3.74945336, -3.55209698, ..., -2.62581188,
         -3.29435235, -2.68565508]]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-43df967c-7511-49ab-83a7-fe2147033851' class='xr-section-summary-in' type='checkbox'  checked><label for='section-43df967c-7511-49ab-83a7-fe2147033851' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2021-05-05T14:18:01.350474</dd><dt><span>arviz_version :</span></dt><dd>0.11.1</dd><dt><span>inference_library :</span></dt><dd>pymc3</dd><dt><span>inference_library_version :</span></dt><dd>3.11.0</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            
            <li class = "xr-section-item">
                  <input id="idata_sample_stats71a894a4-0be8-40d7-ac7c-7e730ed56892" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_sample_stats71a894a4-0be8-40d7-ac7c-7e730ed56892" class = "xr-section-summary">sample_stats</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
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
Dimensions:             (chain: 4, draw: 1000)
Coordinates:
  * chain               (chain) int64 0 1 2 3
  * draw                (draw) int64 0 1 2 3 4 5 6 ... 994 995 996 997 998 999
Data variables:
    process_time_diff   (chain, draw) float64 0.000545 0.000251 ... 0.000396
    diverging           (chain, draw) bool False False False ... False False
    mean_tree_accept    (chain, draw) float64 0.8152 1.0 1.0 ... 0.994 0.8108
    max_energy_error    (chain, draw) float64 0.9104 -0.09504 ... -0.1136 0.4776
    step_size_bar       (chain, draw) float64 1.093 1.093 1.093 ... 1.116 1.116
    energy_error        (chain, draw) float64 0.03642 -0.09504 ... 0.3852
    lp                  (chain, draw) float64 -1.082e+03 ... -1.081e+03
    perf_counter_diff   (chain, draw) float64 0.0005441 0.00025 ... 0.0003966
    perf_counter_start  (chain, draw) float64 9.599 9.6 9.6 ... 11.11 11.11
    depth               (chain, draw) int64 2 1 2 2 2 2 2 2 ... 2 1 2 2 2 2 2 2
    energy              (chain, draw) float64 1.084e+03 1.083e+03 ... 1.081e+03
    step_size           (chain, draw) float64 0.9807 0.9807 0.9807 ... 1.08 1.08
    tree_size           (chain, draw) float64 3.0 1.0 3.0 3.0 ... 3.0 3.0 3.0
Attributes:
    created_at:                 2021-05-05T14:18:00.983660
    arviz_version:              0.11.1
    inference_library:          pymc3
    inference_library_version:  3.11.0
    sampling_time:              11.833754777908325
    tuning_steps:               1000</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-bb155d26-deb1-45b7-895f-6c1b2d13b0bd' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-bb155d26-deb1-45b7-895f-6c1b2d13b0bd' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 4</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-bff4f07f-4b36-4e8d-8683-1209b526263d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-bff4f07f-4b36-4e8d-8683-1209b526263d' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3</div><input id='attrs-87f97724-30ae-4f8e-8f44-cf86118ccc09' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-87f97724-30ae-4f8e-8f44-cf86118ccc09' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-87b12922-c20b-4cc4-8185-507735ef7333' class='xr-var-data-in' type='checkbox'><label for='data-87b12922-c20b-4cc4-8185-507735ef7333' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-73cad399-cb87-42a1-a3f5-a4c1e1ba05ce' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-73cad399-cb87-42a1-a3f5-a4c1e1ba05ce' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-625998c3-935e-4b3a-95c2-3debfedf603d' class='xr-var-data-in' type='checkbox'><label for='data-625998c3-935e-4b3a-95c2-3debfedf603d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d7f06476-7f5a-47bf-af3e-0b98a28c8f84' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d7f06476-7f5a-47bf-af3e-0b98a28c8f84' class='xr-section-summary' >Data variables: <span>(13)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>process_time_diff</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.000545 0.000251 ... 0.000396</div><input id='attrs-7e329552-5761-493c-b65b-80808cb176b1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7e329552-5761-493c-b65b-80808cb176b1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f6bc7a65-952e-4326-bef2-880d5687e13e' class='xr-var-data-in' type='checkbox'><label for='data-f6bc7a65-952e-4326-bef2-880d5687e13e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.000545, 0.000251, 0.000597, ..., 0.000428, 0.000506, 0.000433],
       [0.000644, 0.000528, 0.000421, ..., 0.000587, 0.000459, 0.000478],
       [0.000806, 0.000528, 0.000468, ..., 0.000222, 0.000419, 0.000418],
       [0.000378, 0.000232, 0.000511, ..., 0.000522, 0.000396, 0.000396]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>diverging</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>False False False ... False False</div><input id='attrs-32aa24d1-c429-46b7-8794-7d1cab1026a3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-32aa24d1-c429-46b7-8794-7d1cab1026a3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-20b730cd-3a72-445b-a6fc-717e1145e835' class='xr-var-data-in' type='checkbox'><label for='data-20b730cd-3a72-445b-a6fc-717e1145e835' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>mean_tree_accept</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8152 1.0 1.0 ... 0.994 0.8108</div><input id='attrs-f8b86ea9-de52-429a-a6a7-47b362ffe27a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f8b86ea9-de52-429a-a6a7-47b362ffe27a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e3ca981a-e427-43b7-8548-15ae771d42e1' class='xr-var-data-in' type='checkbox'><label for='data-e3ca981a-e427-43b7-8548-15ae771d42e1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.81521779, 1.        , 1.        , ..., 0.62509995, 1.        ,
        0.81107892],
       [1.        , 0.87950998, 1.        , ..., 0.92492985, 0.89105261,
        0.98834575],
       [1.        , 0.94580583, 0.79279004, ..., 1.        , 0.97066098,
        0.90027375],
       [0.68892839, 1.        , 1.        , ..., 0.69562692, 0.99396752,
        0.81080662]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>max_energy_error</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.9104 -0.09504 ... -0.1136 0.4776</div><input id='attrs-8e6c049d-b353-4389-ae98-68daf9516f47' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8e6c049d-b353-4389-ae98-68daf9516f47' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8549cb0f-19f0-401e-93e7-5343af227b04' class='xr-var-data-in' type='checkbox'><label for='data-8549cb0f-19f0-401e-93e7-5343af227b04' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.91042764, -0.09503945, -0.83808615, ...,  0.7862915 ,
        -0.17763249,  0.50824189],
       [-0.37435743,  0.66563332, -1.8195728 , ...,  0.16537778,
         3.12022915, -1.69345875],
       [-0.57898891, -0.48840059,  0.9120955 , ..., -0.24756911,
         0.10225432,  0.13622737],
       [ 0.49131661, -0.23870978, -0.0459805 , ...,  0.4801973 ,
        -0.11361483,  0.47760448]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>step_size_bar</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.093 1.093 1.093 ... 1.116 1.116</div><input id='attrs-6ad3aa07-e43a-4a44-8e31-df3392a43e24' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6ad3aa07-e43a-4a44-8e31-df3392a43e24' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b026e11c-4006-480a-b74b-8c77084b44a1' class='xr-var-data-in' type='checkbox'><label for='data-b026e11c-4006-480a-b74b-8c77084b44a1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1.09295212, 1.09295212, 1.09295212, ..., 1.09295212, 1.09295212,
        1.09295212],
       [1.17824904, 1.17824904, 1.17824904, ..., 1.17824904, 1.17824904,
        1.17824904],
       [1.11544141, 1.11544141, 1.11544141, ..., 1.11544141, 1.11544141,
        1.11544141],
       [1.11569537, 1.11569537, 1.11569537, ..., 1.11569537, 1.11569537,
        1.11569537]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>energy_error</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.03642 -0.09504 ... 0.01965 0.3852</div><input id='attrs-c92e39e7-0de2-4195-9846-17e544778a8b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c92e39e7-0de2-4195-9846-17e544778a8b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-308842c8-acbb-4d6d-97d2-ad8aa9758835' class='xr-var-data-in' type='checkbox'><label for='data-308842c8-acbb-4d6d-97d2-ad8aa9758835' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.03642284, -0.09503945, -0.19363074, ...,  0.26058727,
        -0.1149865 ,  0.13879757],
       [-0.31444418,  0.13165454, -1.8195728 , ..., -0.16283821,
         1.61073217, -1.69345875],
       [-0.15159733, -0.48840059, -0.31776609, ..., -0.24756911,
        -0.06579727,  0.08097006],
       [ 0.2667921 , -0.23870978, -0.02529995, ...,  0.21950385,
         0.01965209,  0.3851733 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>lp</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.082e+03 ... -1.081e+03</div><input id='attrs-de880206-c396-43ac-bf13-a6921fe3978b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-de880206-c396-43ac-bf13-a6921fe3978b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f762676c-7b74-4cd7-a91e-f99bf39ab1fb' class='xr-var-data-in' type='checkbox'><label for='data-f762676c-7b74-4cd7-a91e-f99bf39ab1fb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1082.12210161, -1082.16278868, -1081.50053832, ...,
        -1080.00551779, -1079.65715579, -1080.04571624],
       [-1082.78549307, -1084.63820543, -1079.06137249, ...,
        -1080.0864814 , -1084.89866909, -1079.85809735],
       [-1081.94861736, -1080.00811114, -1079.7500572 , ...,
        -1079.22986325, -1079.05407988, -1079.29049144],
       [-1079.90738691, -1079.16194053, -1079.09117709, ...,
        -1079.82049429, -1079.88511239, -1080.89330197]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>perf_counter_diff</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0005441 0.00025 ... 0.0003966</div><input id='attrs-0acb39ca-75a5-44aa-ac5a-d80e3e39c6fc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0acb39ca-75a5-44aa-ac5a-d80e3e39c6fc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5a0f610c-a1fd-4a98-8ecc-9639a2b92f3f' class='xr-var-data-in' type='checkbox'><label for='data-5a0f610c-a1fd-4a98-8ecc-9639a2b92f3f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00054407, 0.00025   , 0.00059725, ..., 0.00042835, 0.0005051 ,
        0.000434  ],
       [0.00064395, 0.0005281 , 0.00042087, ..., 0.00058625, 0.00045914,
        0.0004774 ],
       [0.00082424, 0.00052749, 0.00046836, ..., 0.00022179, 0.00041825,
        0.0004171 ],
       [0.00037707, 0.00023211, 0.00051127, ..., 0.0005203 , 0.00039487,
        0.00039661]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>perf_counter_start</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>9.599 9.6 9.6 ... 11.11 11.11 11.11</div><input id='attrs-c09ca112-035e-4ee9-bd9a-c9b097ef535a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c09ca112-035e-4ee9-bd9a-c9b097ef535a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-84f3a79a-792f-4d77-a533-26498f9a6fe4' class='xr-var-data-in' type='checkbox'><label for='data-84f3a79a-792f-4d77-a533-26498f9a6fe4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 9.59901446,  9.59969387,  9.60007687, ..., 10.20728038,
        10.20782491, 10.20842682],
       [ 9.63946076,  9.64033849,  9.64097965, ..., 10.24716827,
        10.24785586, 10.24841378],
       [ 9.5905833 ,  9.59159761,  9.59226399, ..., 10.19245294,
        10.1927657 , 10.19327776],
       [10.63222614, 10.63269292, 10.63300685, ..., 11.10597952,
        11.10658778, 11.10707039]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>depth</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>2 1 2 2 2 2 2 2 ... 2 1 2 2 2 2 2 2</div><input id='attrs-503c6b53-e4ef-4a29-afca-c6f958beed77' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-503c6b53-e4ef-4a29-afca-c6f958beed77' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4decd672-3195-4e72-9403-5480cb6f8452' class='xr-var-data-in' type='checkbox'><label for='data-4decd672-3195-4e72-9403-5480cb6f8452' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2, 1, 2, ..., 2, 2, 2],
       [2, 2, 2, ..., 2, 2, 2],
       [2, 2, 2, ..., 1, 2, 2],
       [2, 1, 2, ..., 2, 2, 2]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>energy</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.084e+03 1.083e+03 ... 1.081e+03</div><input id='attrs-172059d5-6f12-4e56-ac56-60b14e1a87cb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-172059d5-6f12-4e56-ac56-60b14e1a87cb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-83e044eb-c2a8-40dc-94c7-e963471fb90e' class='xr-var-data-in' type='checkbox'><label for='data-83e044eb-c2a8-40dc-94c7-e963471fb90e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1084.37041449, 1083.1028563 , 1082.28201286, ..., 1081.53937098,
        1080.27140832, 1081.44229314],
       [1084.77414402, 1086.73611149, 1083.66176607, ..., 1081.52492819,
        1087.7836095 , 1084.09412778],
       [1083.86937627, 1082.83971769, 1083.38872058, ..., 1079.87520423,
        1079.51741058, 1079.47892824],
       [1080.54830525, 1079.69313736, 1079.21349331, ..., 1080.45478448,
        1080.19072416, 1081.34854447]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>step_size</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.9807 0.9807 0.9807 ... 1.08 1.08</div><input id='attrs-df103243-5517-4fff-b5af-ae4c6e26d854' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-df103243-5517-4fff-b5af-ae4c6e26d854' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ba2ad520-2f29-496d-9284-18a6aeee9453' class='xr-var-data-in' type='checkbox'><label for='data-ba2ad520-2f29-496d-9284-18a6aeee9453' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.98069767, 0.98069767, 0.98069767, ..., 0.98069767, 0.98069767,
        0.98069767],
       [1.11008493, 1.11008493, 1.11008493, ..., 1.11008493, 1.11008493,
        1.11008493],
       [0.78984578, 0.78984578, 0.78984578, ..., 0.78984578, 0.78984578,
        0.78984578],
       [1.08035127, 1.08035127, 1.08035127, ..., 1.08035127, 1.08035127,
        1.08035127]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>tree_size</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>3.0 1.0 3.0 3.0 ... 3.0 3.0 3.0 3.0</div><input id='attrs-316d204a-db4c-4c30-9eaa-5e8f40f9d070' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-316d204a-db4c-4c30-9eaa-5e8f40f9d070' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-19227e86-5b86-4535-a170-5f4522f7d206' class='xr-var-data-in' type='checkbox'><label for='data-19227e86-5b86-4535-a170-5f4522f7d206' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[3., 1., 3., ..., 3., 3., 3.],
       [3., 3., 3., ..., 3., 3., 3.],
       [3., 3., 3., ..., 1., 3., 3.],
       [3., 1., 3., ..., 3., 3., 3.]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c50a98c6-1fbe-4257-9357-0d4b5f4d5111' class='xr-section-summary-in' type='checkbox'  checked><label for='section-c50a98c6-1fbe-4257-9357-0d4b5f4d5111' class='xr-section-summary' >Attributes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2021-05-05T14:18:00.983660</dd><dt><span>arviz_version :</span></dt><dd>0.11.1</dd><dt><span>inference_library :</span></dt><dd>pymc3</dd><dt><span>inference_library_version :</span></dt><dd>3.11.0</dd><dt><span>sampling_time :</span></dt><dd>11.833754777908325</dd><dt><span>tuning_steps :</span></dt><dd>1000</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            
            <li class = "xr-section-item">
                  <input id="idata_observed_dataaae7326f-f62b-400e-b74d-6d501f8d51aa" class="xr-section-summary-in" type="checkbox">
                  <label for="idata_observed_dataaae7326f-f62b-400e-b74d-6d501f8d51aa" class = "xr-section-summary">observed_data</label>
                  <div class="xr-section-inline-details"></div>
                  <div class="xr-section-details">
                      <ul id="xr-dataset-coord-list" class="xr-var-list">
                          <div style="padding-left:2rem;"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
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
Dimensions:       (height_dim_0: 352)
Coordinates:
  * height_dim_0  (height_dim_0) int64 0 1 2 3 4 5 6 ... 346 347 348 349 350 351
Data variables:
    height        (height_dim_0) float64 151.8 139.7 136.5 ... 162.6 156.2 158.8
Attributes:
    created_at:                 2021-05-05T14:18:01.351357
    arviz_version:              0.11.1
    inference_library:          pymc3
    inference_library_version:  3.11.0</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-7a357aa5-a6b1-402c-9707-4aba816a2617' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-7a357aa5-a6b1-402c-9707-4aba816a2617' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>height_dim_0</span>: 352</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-dfd696f4-a1f0-4d42-9be3-05f2da8553ae' class='xr-section-summary-in' type='checkbox'  checked><label for='section-dfd696f4-a1f0-4d42-9be3-05f2da8553ae' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>height_dim_0</span></div><div class='xr-var-dims'>(height_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 347 348 349 350 351</div><input id='attrs-3febfd77-6e1b-4f58-9bd5-5bea356e9f2a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3febfd77-6e1b-4f58-9bd5-5bea356e9f2a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c5fc5abe-84af-47dc-99d3-818e449f386f' class='xr-var-data-in' type='checkbox'><label for='data-c5fc5abe-84af-47dc-99d3-818e449f386f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 349, 350, 351])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e2694d6e-612f-499f-b252-469078f339f9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e2694d6e-612f-499f-b252-469078f339f9' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>height</span></div><div class='xr-var-dims'>(height_dim_0)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>151.8 139.7 136.5 ... 156.2 158.8</div><input id='attrs-b6bb4946-5e70-41de-a53d-452d7a95bd9d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b6bb4946-5e70-41de-a53d-452d7a95bd9d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c1391295-22bf-4a7b-98b8-6625265f8454' class='xr-var-data-in' type='checkbox'><label for='data-c1391295-22bf-4a7b-98b8-6625265f8454' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([151.765 , 139.7   , 136.525 , 156.845 , 145.415 , 163.83  ,
       149.225 , 168.91  , 147.955 , 165.1   , 154.305 , 151.13  ,
       144.78  , 149.9   , 150.495 , 163.195 , 157.48  , 143.9418,
       161.29  , 156.21  , 146.4   , 148.59  , 147.32  , 147.955 ,
       161.925 , 146.05  , 146.05  , 152.7048, 142.875 , 142.875 ,
       147.955 , 160.655 , 151.765 , 162.8648, 171.45  , 147.32  ,
       147.955 , 154.305 , 143.51  , 146.7   , 157.48  , 165.735 ,
       152.4   , 141.605 , 158.8   , 155.575 , 164.465 , 151.765 ,
       161.29  , 154.305 , 145.415 , 145.415 , 152.4   , 163.83  ,
       144.145 , 153.67  , 142.875 , 167.005 , 158.4198, 165.735 ,
       149.86  , 154.94  , 160.9598, 161.925 , 147.955 , 159.385 ,
       148.59  , 136.525 , 158.115 , 144.78  , 156.845 , 179.07  ,
       170.18  , 146.05  , 147.32  , 162.56  , 152.4   , 160.02  ,
       149.86  , 142.875 , 167.005 , 159.385 , 154.94  , 162.56  ,
       152.4   , 170.18  , 146.05  , 159.385 , 151.13  , 160.655 ,
       169.545 , 158.75  , 149.86  , 153.035 , 161.925 , 162.56  ,
       149.225 , 163.195 , 161.925 , 145.415 , 163.195 , 151.13  ,
       150.495 , 170.815 , 157.48  , 152.4   , 147.32  , 145.415 ,
       157.48  , 154.305 , 167.005 , 142.875 , 152.4   , 160.    ,
       159.385 , 149.86  , 160.655 , 160.655 , 149.225 , 140.97  ,
...
       164.465 , 153.035 , 149.225 , 160.02  , 149.225 , 153.67  ,
       150.495 , 151.765 , 158.115 , 149.225 , 151.765 , 154.94  ,
       161.29  , 148.59  , 160.655 , 157.48  , 167.005 , 157.48  ,
       152.4   , 152.4   , 161.925 , 152.4   , 159.385 , 142.24  ,
       168.91  , 160.02  , 158.115 , 152.4   , 155.575 , 154.305 ,
       156.845 , 156.21  , 168.275 , 147.955 , 157.48  , 160.7   ,
       161.29  , 150.495 , 163.195 , 148.59  , 148.59  , 161.925 ,
       153.67  , 151.13  , 163.83  , 153.035 , 151.765 , 156.21  ,
       140.335 , 158.75  , 142.875 , 151.9428, 161.29  , 160.9852,
       144.78  , 160.02  , 160.9852, 165.989 , 157.988 , 154.94  ,
       160.655 , 147.32  , 146.7   , 147.32  , 172.9994, 158.115 ,
       147.32  , 165.989 , 149.86  , 161.925 , 163.83  , 160.02  ,
       154.94  , 152.4   , 146.05  , 151.9936, 151.765 , 144.78  ,
       160.655 , 151.13  , 153.67  , 147.32  , 139.7   , 157.48  ,
       154.94  , 143.51  , 158.115 , 147.32  , 160.02  , 165.1   ,
       154.94  , 153.67  , 141.605 , 163.83  , 161.29  , 154.9   ,
       161.3   , 170.18  , 149.86  , 160.655 , 154.94  , 166.37  ,
       148.2852, 151.765 , 148.59  , 153.67  , 146.685 , 154.94  ,
       156.21  , 160.655 , 146.05  , 156.21  , 152.4   , 162.56  ,
       142.875 , 162.56  , 156.21  , 158.75  ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-aa3da28a-5a24-482f-a896-43828e1d8d2d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-aa3da28a-5a24-482f-a896-43828e1d8d2d' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2021-05-05T14:18:01.351357</dd><dt><span>arviz_version :</span></dt><dd>0.11.1</dd><dt><span>inference_library :</span></dt><dd>pymc3</dd><dt><span>inference_library_version :</span></dt><dd>3.11.0</dd></dl></div></li></ul></div></div><br></div>
                      </ul>
                  </div>
            </li>
            
              </ul>
            </div>
            <style> /* CSS stylesheet for displaying InferenceData objects in jupyterlab.
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

.xr-sections.group-sections {
  grid-template-columns: auto;
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

.xr-attrs dt, dd {
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
.xr-wrap{width:700px!important;} </style>




    <IPython.core.display.Javascript object>


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
    created_at:                 2021-05-05T14:18:00.979266
    arviz_version:              0.11.1
    inference_library:          pymc3
    inference_library_version:  3.11.0
    sampling_time:              11.833754777908325
    tuning_steps:               1000</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-02edaf80-1edc-4299-8888-13b19a9e02ee' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-02edaf80-1edc-4299-8888-13b19a9e02ee' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 4</li><li><span class='xr-has-index'>draw</span>: 1000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-781d9ca7-32fd-4d34-a25e-0c009e88f61e' class='xr-section-summary-in' type='checkbox'  checked><label for='section-781d9ca7-32fd-4d34-a25e-0c009e88f61e' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3</div><input id='attrs-9e106f9f-c98c-4447-8dff-03505561c12f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9e106f9f-c98c-4447-8dff-03505561c12f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-864845d3-2cd0-4b95-8fc2-6c60e93e1d4e' class='xr-var-data-in' type='checkbox'><label for='data-864845d3-2cd0-4b95-8fc2-6c60e93e1d4e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-6fa1b4fc-cab0-4565-8e9a-d885dad58c42' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6fa1b4fc-cab0-4565-8e9a-d885dad58c42' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-772b5651-e969-487a-97d8-b4e5408c2b1b' class='xr-var-data-in' type='checkbox'><label for='data-772b5651-e969-487a-97d8-b4e5408c2b1b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-32f41a48-ad55-4e99-8363-b0ffb0a10065' class='xr-section-summary-in' type='checkbox'  checked><label for='section-32f41a48-ad55-4e99-8363-b0ffb0a10065' class='xr-section-summary' >Data variables: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>a</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>154.0 154.0 155.1 ... 154.3 154.9</div><input id='attrs-69c46ac0-0399-4e70-9d92-2e5fb839bb16' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-69c46ac0-0399-4e70-9d92-2e5fb839bb16' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1feae9b3-04aa-467c-a985-46ae57c1fb10' class='xr-var-data-in' type='checkbox'><label for='data-1feae9b3-04aa-467c-a985-46ae57c1fb10' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[154.03058495, 153.96997324, 155.11106991, ..., 154.63468783,
        154.76213813, 154.29138176],
       [155.10811531, 153.96269524, 154.65322055, ..., 154.3681408 ,
        155.49068922, 154.87128055],
       [154.10306815, 154.75601267, 154.3996128 , ..., 154.5419651 ,
        154.59997877, 154.41866941],
       [154.57285029, 154.61464995, 154.66532063, ..., 154.86002136,
        154.28550752, 154.87498365]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>b</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.8538 0.8616 ... 0.9321 0.8399</div><input id='attrs-f601f9ac-ec1a-4789-baad-8648f35ac4af' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f601f9ac-ec1a-4789-baad-8648f35ac4af' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5d6e1b7c-2bda-4462-819c-24244935e2f9' class='xr-var-data-in' type='checkbox'><label for='data-5d6e1b7c-2bda-4462-819c-24244935e2f9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.85378765, 0.86161909, 0.95776919, ..., 0.94140608, 0.88971675,
        0.91204005],
       [0.99177163, 0.82261803, 0.91716301, ..., 0.92739364, 0.8581938 ,
        0.9073712 ],
       [0.87734415, 0.88182227, 0.90922174, ..., 0.89340121, 0.90167945,
        0.92521348],
       [0.89919471, 0.89494823, 0.91869996, ..., 0.86604634, 0.93210088,
        0.83992027]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>4.989 5.129 5.068 ... 5.019 5.242</div><input id='attrs-db92dc88-f936-4fe1-8107-012baca6a128' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-db92dc88-f936-4fe1-8107-012baca6a128' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-efec83f0-bf50-42e7-b97e-d48f132f4874' class='xr-var-data-in' type='checkbox'><label for='data-efec83f0-bf50-42e7-b97e-d48f132f4874' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[4.98944173, 5.12879049, 5.06756793, ..., 4.86557056, 5.27635179,
        5.27497752],
       [5.16986056, 4.87706806, 5.15322605, ..., 5.31173081, 5.22379536,
        4.91267767],
       [4.8309535 , 5.32888694, 4.88953936, ..., 5.21380319, 4.9855454 ,
        5.1213464 ],
       [5.35629738, 5.20270692, 5.00285615, ..., 5.0899661 , 5.01923403,
        5.2419566 ]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-acfdfa54-9c8e-4666-82b8-49944ef8a455' class='xr-section-summary-in' type='checkbox'  checked><label for='section-acfdfa54-9c8e-4666-82b8-49944ef8a455' class='xr-section-summary' >Attributes: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2021-05-05T14:18:00.979266</dd><dt><span>arviz_version :</span></dt><dd>0.11.1</dd><dt><span>inference_library :</span></dt><dd>pymc3</dd><dt><span>inference_library_version :</span></dt><dd>3.11.0</dd><dt><span>sampling_time :</span></dt><dd>11.833754777908325</dd><dt><span>tuning_steps :</span></dt><dd>1000</dd></dl></div></li></ul></div></div>




    <IPython.core.display.Javascript object>


### `trace_m2` is a `MultiTrace` object

The `MultiTrace` object that is outputted as a result of setting `return_inferencedata=False` in the `pm.sample()` call. (It's the same object that is outputted in the PyMC3 repo of the book's code which is why I decided to work with it here.) This is an important object so I wanted to dive deeper into it.


```python
type(trace_m2)
```




    pymc3.backends.base.MultiTrace




    <IPython.core.display.Javascript object>


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




    <IPython.core.display.Javascript object>


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




    <IPython.core.display.Javascript object>


Let's inspect some of the object methods.


```python
# Trace object's variable names
trace_m2.varnames
```




    ['a', 'b_log__', 'sigma_interval__', 'b', 'sigma']




    <IPython.core.display.Javascript object>



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



    <IPython.core.display.Javascript object>



```python
# Another way to inspect
trace_m2.get_values("a")[0:5]
```




    array([154.58919424, 154.58919424, 154.81892314, 154.92078981,
           154.92078981])




    <IPython.core.display.Javascript object>


A chain is a single run of Markov Chain Monte Carlo. I haven't learned MCMC yet, but chains in `pymc3` are explained [here](https://stackoverflow.com/questions/49825216/what-is-a-chain-in-pymc3).


```python
trace_m2.chains
```




    [0, 1, 2, 3]




    <IPython.core.display.Javascript object>


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

```
