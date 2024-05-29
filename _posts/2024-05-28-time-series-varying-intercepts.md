---
title: "Time series with varying intercepts"
mathjax: true
toc: true
toc_sticky: true
categories: [data science, statistics]
---

I've done time-series data with time-to-event models and would like to explore modeling with mixed effects models. I'll take an interative approach, in the spirit of [Singer and Willet's Applied Longitudinal Data Analysis: Modeling Change and Event Occurrence](https://academic.oup.com/book/41753?login=false). I discovered this textbook when finding this [post by Nathaniel Forde](https://www.pymc.io/projects/examples/en/latest/time_series/longitudinal_models.html) on the pymc website.

In this post, we'll focus on varying intercepts, first from a Bayesian approach using pymc, followed by an example with statsmodels. In later posts, we'll increase the complexity such as incorporation of varying slopes.



```python
import arviz as az
import graphviz as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
```


```python
sns.set_context("talk")
sns.set_palette("colorblind")
```


```python
def draw_causal_graph(
    edge_list, node_props=None, edge_props=None, graph_direction="UD"
):
    """Utility to draw a causal (directed) graph
    Taken from: https://github.com/dustinstansbury/statistical-rethinking-2023/blob/a0f4f2d15a06b33355cf3065597dcb43ef829991/utils.py#L52-L66

    """
    g = gr.Digraph(graph_attr={"rankdir": graph_direction})

    edge_props = {} if edge_props is None else edge_props
    for e in edge_list:
        props = edge_props[e] if e in edge_props else {}
        g.edge(e[0], e[1], **props)

    if node_props is not None:
        for name, props in node_props.items():
            g.node(name=name, **props)
    return g


def standardize(x):
    x = (x - np.mean(x)) / np.std(x)
    return x
```

# Data generation


```python
# Generate synthetic data
n_patients = 30
n_timepoints = 5

# Create patient IDs
patient_ids = np.repeat(np.arange(n_patients), n_timepoints)

# Create time points
time = np.tile(np.arange(n_timepoints), n_patients)

# Create patient-specific attributes (age and treatment)
age = np.random.randint(40, 70, n_patients)
treatment = np.random.binomial(1, 0.5, n_patients)

# Repeat age and treatment to match the longitudinal measurements
age_repeated = np.repeat(age, n_timepoints)
treatment_repeated = np.repeat(treatment, n_timepoints)

# Combine into a DataFrame
df_data = pd.DataFrame(
    {
        "patient_id": patient_ids,
        "time": time,
        "age": age_repeated,
        "treatment": treatment_repeated,
    }
)

df_data.head(10)
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
      <th>patient_id</th>
      <th>time</th>
      <th>age</th>
      <th>treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>66</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>66</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>66</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>66</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>66</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>2</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>3</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>4</td>
      <td>53</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Here's the fun part. We'll simulate the outcome variable tumor size, using some mild assumptions and domain knowledge. First, we'll assume that all participants have been identified as having a solid tumor cancer. Therefore:

- Time is also a positive association risk factor.
- Age is likely a risk factor, so create a positive association between age and tumor size. (Note that we'll keep age as a constant for each patient, such that we're looking at a time window of a few months.)
- Whether one has received treatment should decrease the tumor size, so that will be a negative association.

This will be a simple linear model that we'll use to create data:

$$ s_i \sim \text{Normal}(\mu_i, \sigma) $$

$$ \mu_i = \alpha + \beta_T T_i + \beta_A A_i + \beta_R R_i $$

However, to be clear, we're using this model to *generate* our data but in this post, we'll focus on varying intercepts and ignore predictors time, age, and treatment.


```python
# Use a generative model to create tumor size with some randomness

alpha_tumor_size = 50  # intercept term
bT = 1  # positive association for time
bA = 0.25  # positive association for age
bR = -5  # negative association for treatment

mu_tumor_size = (
    alpha_tumor_size
    + bT * df_data["time"]
    + bA * df_data["age"]
    + bR * df_data["treatment"]
)

sigma_tumor_size = 2

df_data["tumor_size"] = np.random.normal(mu_tumor_size, sigma_tumor_size)

df_data.head()
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
      <th>patient_id</th>
      <th>time</th>
      <th>age</th>
      <th>treatment</th>
      <th>tumor_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>66</td>
      <td>1</td>
      <td>62.813967</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>66</td>
      <td>1</td>
      <td>61.505909</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>66</td>
      <td>1</td>
      <td>64.283770</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>66</td>
      <td>1</td>
      <td>65.343314</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>66</td>
      <td>1</td>
      <td>64.127617</td>
    </tr>
  </tbody>
</table>
</div>



# Data transformation

Before doing any modeling, we'll transform the data since in theory we shouldn't peek. But we can't pick priors unless we have some idea of what the data is like. An easy thing to do is standardize the data and therefore we can use a 0 mean, 2 SD prior to capture most of the data.


```python
df_data["tumor_size_std"] = standardize(df_data["tumor_size"])
```


```python
# how to represent patient_specific random effect?
draw_causal_graph(
    edge_list=[("age", "tumor"), ("treatment", "tumor"), ("time", "tumor")],
    graph_direction="LR",
)
```




    
![svg](/assets/2024-05-28-time-series-varying-intercepts_files/2024-05-28-time-series-varying-intercepts_10_0.svg)
    




```python
df_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 6 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   patient_id      150 non-null    int64  
     1   time            150 non-null    int64  
     2   age             150 non-null    int64  
     3   treatment       150 non-null    int64  
     4   tumor_size      150 non-null    float64
     5   tumor_size_std  150 non-null    float64
    dtypes: float64(2), int64(4)
    memory usage: 7.2 KB



```python
sns.relplot(
    data=df_data,
    x="time",
    y="tumor_size",
    col="patient_id",
    col_wrap=6,
    hue="treatment",
    kind="line",
)
```




    <seaborn.axisgrid.FacetGrid at 0x16d7634a0>




    
![png](/assets/2024-05-28-time-series-varying-intercepts_files/2024-05-28-time-series-varying-intercepts_12_1.png)
    



```python
ax = sns.boxplot(
    data=df_data,
    x="treatment",
    y="tumor_size",
)
ax.set_title("Effect of Treatment on tumor size")
```




    Text(0.5, 1.0, 'Effect of Treatment on tumor size')




    
![png](/assets/2024-05-28-time-series-varying-intercepts_files/2024-05-28-time-series-varying-intercepts_13_1.png)
    


# Varying intercepts using pymc

Let's define the equation. We're going to assume the tumor size is Gaussian distributed.

It will be a linear combination of independent variables for time, age, and treatment. How will we represent the `patient_id`? 

There will be a term for average tumor size and the patient-specific tumor size will be the "random effect".

```
s = tumor size
t = time
a = age
r = treatment
```

After reading McElreath, for now, I will ignore time, age, and treatment and just think of patient as a cluster and just do varying intercepts.

$$ \mu_i = \alpha_{\text{pt[i]}} $$


Let's do this step-by-step and work our way from the most naive, simplest models to more complex and informative.

0. **Complete pooling, intercepts only.** Ignore patients as clusters.
1. **No pooling, intercepts only**. Keep intercepts separate for each patient. Ignore information across patients.
2. **Partial pooling, intercepts only.** Share information across patients.

## Model 0: complete pooling

$$ s_i \sim \text{Normal}(\mu_i, \sigma) $$

$$ \mu_i = \alpha  $$

$$ \alpha \sim \text{Normal}(0, 1) $$

$$ \sigma \sim \text{Exponential}(1) $$

The `patient_id` variable is completely ignored. A subscript to denote the patient is not relevant here?


```python
# complete pooling, intercepts only
with pm.Model() as m0:

    # priors
    a = pm.Normal("a_bar", 0.0, 1)
    sigma = pm.Exponential("sigma", 1.0)

    # linear model
    mu = a

    # likelihood
    s = pm.Normal("s", mu=mu, sigma=sigma, observed=df_data["tumor_size_std"])

    trace_m0 = pm.sample(
        draws=1000, random_seed=19, return_inferencedata=True, progressbar=True
    )
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Sampling 4 chains, 0 divergences <span style="color: #1764f4; text-decoration-color: #1764f4">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span><span style="color: #3a3a3a; text-decoration-color: #3a3a3a">╺━━━</span> <span style="color: #800080; text-decoration-color: #800080"> 90%</span> <span style="color: #008080; text-decoration-color: #008080">0:00:02</span> / <span style="color: #808000; text-decoration-color: #808000">0:00:09</span>
</pre>




```python
az.summary(trace_m0)
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
      <th>hdi_3%</th>
      <th>hdi_97%</th>
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
      <td>-0.002</td>
      <td>0.081</td>
      <td>-0.161</td>
      <td>0.141</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4201.0</td>
      <td>2889.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>1.009</td>
      <td>0.059</td>
      <td>0.892</td>
      <td>1.114</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4028.0</td>
      <td>2539.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
f, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))  # do add subplots
df_data["tumor_size"].hist(ax=ax0)
df_data["tumor_size_std"].hist(ax=ax1)
```




    <Axes: >




    
![png](/assets/2024-05-28-time-series-varying-intercepts_files/2024-05-28-time-series-varying-intercepts_18_1.png)
    


## Model 2: no pooling
Acknowledge that there are patient clusters but do not share any information across them. In other words have a prior but no adaptive regularization.

$$ s_i \sim \text{Normal}(\mu_i, \sigma) $$

$$ \mu_i = \alpha_{\text{pt[i]}}  $$

$$ \alpha_j \sim \text{Normal}(0, 1) $$

$$ \sigma \sim \text{Exponential}(1) $$


```python
# no pooling, intercepts only
with pm.Model() as m1:

    # priors
    a = pm.Normal("a", 0.0, 1, shape=df_data["patient_id"].nunique())
    sigma = pm.Exponential("sigma", 1.0)

    # linear model... # initialize with pymc data?... represent patient as its own cluster
    mu = a[df_data["patient_id"]]

    # likelihood
    s = pm.Normal("s", mu=mu, sigma=sigma, observed=df_data["tumor_size_std"])

    trace_m1 = pm.sample(
        draws=1000, random_seed=19, return_inferencedata=True, progressbar=True
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [a, sigma]



    Output()


    IOPub message rate exceeded.
    The Jupyter server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--ServerApp.iopub_msg_rate_limit`.
    
    Current values:
    ServerApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    ServerApp.rate_limit_window=3.0 (secs)
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Sampling 4 chains, 0 divergences <span style="color: #1764f4; text-decoration-color: #1764f4">━━━━━━━━━━━━━━━━━━━━━━━╸</span><span style="color: #3a3a3a; text-decoration-color: #3a3a3a">━━━━━━━━━━━━━━━━</span> <span style="color: #800080; text-decoration-color: #800080"> 60%</span> <span style="color: #008080; text-decoration-color: #008080">0:00:05</span> / <span style="color: #808000; text-decoration-color: #808000">0:00:06</span>
</pre>



    IOPub message rate exceeded.
    The Jupyter server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--ServerApp.iopub_msg_rate_limit`.
    
    Current values:
    ServerApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    ServerApp.rate_limit_window=3.0 (secs)
    



```python
az.summary(trace_m1).head()
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
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a[0]</th>
      <td>0.249</td>
      <td>0.296</td>
      <td>-0.324</td>
      <td>0.794</td>
      <td>0.003</td>
      <td>0.004</td>
      <td>7399.0</td>
      <td>2915.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[1]</th>
      <td>-0.644</td>
      <td>0.296</td>
      <td>-1.198</td>
      <td>-0.087</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>8740.0</td>
      <td>3331.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[2]</th>
      <td>0.381</td>
      <td>0.300</td>
      <td>-0.178</td>
      <td>0.942</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>9641.0</td>
      <td>2892.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[3]</th>
      <td>0.831</td>
      <td>0.297</td>
      <td>0.259</td>
      <td>1.383</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>7541.0</td>
      <td>2610.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[4]</th>
      <td>0.426</td>
      <td>0.295</td>
      <td>-0.128</td>
      <td>0.968</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>9279.0</td>
      <td>2689.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
f, ax = plt.subplots()
ax.scatter(
    az.summary(trace_m1, var_names=["a"])["mean"],
    standardize(df_data.groupby("patient_id")["tumor_size"].mean()),
)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="dashed", color="gray")
ax.set(xlabel="raw data", ylabel="model parameter of intercepts");
```


    
![png](/assets/2024-05-28-time-series-varying-intercepts_files/2024-05-28-time-series-varying-intercepts_22_0.png)
    


## Model 2: partial pooling (varying intercepts)

$$ s_i \sim \text{Normal}(\mu_i, \sigma) $$

$$ \mu_i = \alpha_{\text{pt[i]}}  $$

$$ \alpha_j \sim \text{Normal}(\bar{\alpha}, \sigma_{\text{pt}}) $$

$$ \bar{\alpha} \sim \text{Normal}(0, 1) $$



$$ \sigma_{\text{pt}} \sim \text{Exponential}(1) $$

$$ \sigma \sim \text{Exponential}(1) $$

**Question**
- can sigma parameter be partially pooled?


```python
# multilevel model, random intercepts
with pm.Model() as m2:

    # prior for average patient
    a_bar = pm.Normal("a_bar", 0.0, 1)
    sigma = pm.Exponential("sigma", 1.0)

    # prior for SD of patients
    sigma_pt = pm.Exponential("sigma_pt", 1.0)

    # alpha priors for each patient
    a = pm.Normal("a", a_bar, sigma_pt, shape=len(df_data["patient_id"].unique()))

    # linear model
    mu = a[df_data["patient_id"]]

    # likelihood
    s = pm.Normal("s", mu=mu, sigma=sigma, observed=df_data["tumor_size_std"])

    trace_m2 = pm.sample(
        draws=1000, random_seed=19, return_inferencedata=True, progressbar=True
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [a_bar, sigma, sigma_pt, a]



    Output()



```python
az.summary(trace_m2, var_names=["a"]).head()
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
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a[0]</th>
      <td>0.236</td>
      <td>0.287</td>
      <td>-0.290</td>
      <td>0.773</td>
      <td>0.003</td>
      <td>0.004</td>
      <td>9775.0</td>
      <td>3129.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[1]</th>
      <td>-0.603</td>
      <td>0.293</td>
      <td>-1.152</td>
      <td>-0.035</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>10061.0</td>
      <td>3014.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[2]</th>
      <td>0.359</td>
      <td>0.287</td>
      <td>-0.179</td>
      <td>0.897</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>7390.0</td>
      <td>2768.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[3]</th>
      <td>0.773</td>
      <td>0.288</td>
      <td>0.254</td>
      <td>1.322</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>8147.0</td>
      <td>2850.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[4]</th>
      <td>0.398</td>
      <td>0.292</td>
      <td>-0.171</td>
      <td>0.928</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>8075.0</td>
      <td>2811.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.summary(trace_m2, var_names=["a_bar", "sigma"]).head()
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
      <th>hdi_3%</th>
      <th>hdi_97%</th>
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
      <td>0.003</td>
      <td>0.150</td>
      <td>-0.286</td>
      <td>0.284</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>6844.0</td>
      <td>3590.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>0.695</td>
      <td>0.045</td>
      <td>0.611</td>
      <td>0.778</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>4631.0</td>
      <td>2655.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Comparison of estimates with no pooling, partial pooling


While there isn't an appreciable difference, the multilevel model has a lower standard deviation for each cluster. This is the partial pooling effect.


```python
f, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

# means
ax0.scatter(
    az.summary(trace_m1, var_names=["a"])["mean"],
    az.summary(trace_m2, var_names=["a"])["mean"],
    facecolors="none",
    edgecolors="k",
)
ax0.plot([0, 1], [0, 1], transform=ax0.transAxes, linestyle="dashed", color="gray")
ax0.set(
    xlabel="no pooling",
    ylabel="partial pooling",
    title="Intercepts\n(mean)",
)

# SD
ax1.scatter(
    az.summary(trace_m1, var_names=["a"])["sd"],
    az.summary(trace_m2, var_names=["a"])["sd"],
    facecolors="none",
    edgecolors="k",
)
ax1.plot([0, 1], [0, 1], transform=ax0.transAxes, linestyle="dashed", color="gray")
ax1.set(
    xlabel="no pooling",
    ylabel="partial pooling",
    title="intercepts\n(standard deviation)",
)
ax1.plot([0, 1], [0, 1], transform=ax1.transAxes, linestyle="dashed", color="gray")

# Calculate the minimum and maximum of both x and y data
data_min = min(
    min(az.summary(trace_m1, var_names=["a"])["sd"]),
    min(az.summary(trace_m2, var_names=["a"])["sd"]),
)
data_max = max(
    max(az.summary(trace_m1, var_names=["a"])["sd"]),
    max(az.summary(trace_m2, var_names=["a"])["sd"]),
)

# Set the limits to be the same for both axes
ax1.set_xlim(data_min * 0.95, data_max * 1.05)
ax1.set_ylim(data_min * 0.95, data_max * 1.05)

f.tight_layout()
```


    
![png](/assets/2024-05-28-time-series-varying-intercepts_files/2024-05-28-time-series-varying-intercepts_29_0.png)
    


You can see that partial pooling decreases the standard error of the intercept parameter in most cases, even though the mean estimate does not really change. Let's see how to implement this in statsmodels.

# Using statsmodels

Using probabilistic programming provides a nice framework to get the random intercepts with probability distributions. But it may not scale as well. Let's explore varying intercepts using statsmodels.

## Varying intercept model


```python
# Define the mixed-effects model formula with only varying intercepts
model = smf.mixedlm("tumor_size_std ~ 1", df_data, groups=df_data["patient_id"])

# Fit the model
result = model.fit()

# Print the summary of the model
print(result.summary())
```

               Mixed Linear Model Regression Results
    ============================================================
    Model:            MixedLM Dependent Variable: tumor_size_std
    No. Observations: 150     Method:             REML          
    No. Groups:       30      Scale:              0.4754        
    Min. group size:  5       Log-Likelihood:     -186.1971     
    Max. group size:  5       Converged:          Yes           
    Mean group size:  5.0                                       
    -------------------------------------------------------------
                  Coef.   Std.Err.    z     P>|z|  [0.025  0.975]
    -------------------------------------------------------------
    Intercept     -0.000     0.146  -0.000  1.000  -0.287   0.287
    Group Var      0.546     0.272                               
    ============================================================
    


The main thing we want to look at is the bottom of the table. `Intercept` refers to the population's average ($\bar{\alpha}$ while `Group Var` refers to the variance of the random intercepts associated with the grouping variable (`patient_id`) which is `sigma_pt` in the `pymc` model. We can see that these are largely in alignment with the `pymc` results even if `Group Var`/`sigma_pt` differ in their means.


```python
az.summary(trace_m2, var_names=["a_bar", "sigma_pt"])
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
      <th>hdi_3%</th>
      <th>hdi_97%</th>
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
      <td>0.003</td>
      <td>0.150</td>
      <td>-0.286</td>
      <td>0.284</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>6844.0</td>
      <td>3590.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_pt</th>
      <td>0.759</td>
      <td>0.121</td>
      <td>0.538</td>
      <td>0.981</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>5584.0</td>
      <td>3368.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



# Comparing `pymc` and `statsmodels` output

Now let's see how each individual patient's estimates look between `pymc` and `statsmodels`. Statsmodels doesn't provide the SD directly. It may be derived by bootstrapping but we'll ignore this for now.


```python
# Extract the random effects
df_smf_random_effects = (
    pd.DataFrame(result.random_effects)
    .T.reset_index()
    .rename(columns={"index": "Patient", "Group": "random_effect_mean"})
)

df_smf_random_effects.head()
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
      <th>Patient</th>
      <th>random_effect_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.237706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.603279</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.358260</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.775414</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.398535</td>
    </tr>
  </tbody>
</table>
</div>




```python
f = plt.figure(figsize=(12, 5))
ax = f.add_subplot(1, 2, 1)

ax.scatter(
    az.summary(trace_m2, var_names=["a"])["mean"],
    df_smf_random_effects['random_effect_mean'],
    facecolors="none",
    edgecolors="k",
)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="dashed", color="gray");
ax.set(
    xlabel="pymc",
    ylabel="statsmodels",
    title="Varying intercepts estimate\nby package",
);
```


    
![png](/assets/2024-05-28-time-series-varying-intercepts_files/2024-05-28-time-series-varying-intercepts_39_0.png)
    


As we can see, the two packages give essentially the same results for varying intercepts.


```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Tue May 28 2024
    
    Python implementation: CPython
    Python version       : 3.12.3
    IPython version      : 8.24.0
    
    pymc       : 5.15.0
    graphviz   : 0.20.3
    seaborn    : 0.13.2
    matplotlib : 3.8.4
    statsmodels: 0.14.2
    scipy      : 1.13.0
    numpy      : 1.26.4
    pandas     : 2.2.2
    arviz      : 0.18.0
    
    Watermark: 2.4.3
    

