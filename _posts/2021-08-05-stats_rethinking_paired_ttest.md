---
title: "(DRAFT) Connecting a paired t-test with a Bayesian multilevel model"
toc: true
toc_sticky: true
mathjax: true
---

I wondered how to get the same numerical answer for the t-statistic when calculating by hand, using a `scipy.stats` function, and using a Bayesian multilevel model.


```python
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as stats
import seaborn as sns
import daft
from causalgraphicalmodels import CausalGraphicalModel

from scipy.optimize import curve_fit
```





```python
%load_ext nb_black
%config InlineBackend.figure_format = 'retina'
%load_ext watermark
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
az.rcParams["stats.hdi_prob"] = 0.89  # sets default credible interval used by arviz
```

    The nb_black extension is already loaded. To reload it, use:
      %reload_ext nb_black
    The watermark extension is already loaded. To reload it, use:
      %reload_ext watermark






```python
sns.set_context("talk")
```





```python
def standardize(x):
    x = (x - np.mean(x)) / np.std(x)
    return x
```




# A simple dataset taken from Wikipedia

The data I'll use is taken from [Wikipedia's entry on the paired t-test](https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples). It's a simple set with four subjects taking a test twice, an example of repeated measures.  Interestingly, in the Chapter 1 introduction and in Chapter 13, Dr. McElreath highlights multi-level models as a way to get "improved estimates for repeat sampling".


```python
df_tests = pd.DataFrame(
    {
        "Name": ["Mike", "Melanie", "Melissa", "Mitchell"],
        "Test_1": [35, 50, 90, 78],
        "Test_2": [67, 46, 86, 91],
    }
)
```





```python
df_tests
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
      <th>Name</th>
      <th>Test_1</th>
      <th>Test_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mike</td>
      <td>35</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Melanie</td>
      <td>50</td>
      <td>46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Melissa</td>
      <td>90</td>
      <td>86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mitchell</td>
      <td>78</td>
      <td>91</td>
    </tr>
  </tbody>
</table>
</div>






# Calculate the t-statistic

The paired t-test is calculating whether the average of the difference is significantly different. We can get this quickly with a Python function.

## T-statistic using scipy.stats


```python
stats.ttest_rel(df_tests["Test_1"], df_tests["Test_2"])
```




    Ttest_relResult(statistic=-1.0784834690588145, pvalue=0.3598054860063756)






## T-statistic calculated manually

The formula for the t-statistic is given on the same Wikipedia page.

$$t = \frac{\bar{X}_D - \mu_0}{s_D / \sqrt{n}}$$

where $\bar{X}_D$ is the average of the differences between all pairs and $s_D$ is the standard deviation of the differences. We can set $mu_0$ to 0 if we want to test whether the average is significantly different. This is straightforward to calculate.


```python
diff = df_tests["Test_1"] - df_tests["Test_2"]
```





```python
t_stat = diff.mean() / (diff.std() / np.sqrt(len(df_tests)))
print("Manual calculated t-statistic: ", t_stat)
```

    Manual calculated t-statistic:  -1.0784834690588145





We get exactly the same answer as `scipy.stats` so that is comforting.

# Calculate the t-statistic with a Bayesian approach

Now the fun part. We can set this up as a Bayesian multi-level linear model. A non-Bayesian approach for connecting the paired t-test to a linear mixed model was described [here](https://vasishth.github.io/Freq_CogSci/from-the-paired-t-test-to-the-linear-mixed-model.html).

First, let's do some table reformatting, so that we can use it in our model. We'll represent name and times into numerical codes and standardize the test scores.


```python
df_tests2 = pd.melt(df_tests, id_vars="Name")
df_tests2.columns = ["Name", "Time", "Score"]
df_tests2["Name_code"] = pd.Categorical(df_tests2["Name"]).codes
df_tests2["Time_code"] = pd.Categorical(df_tests2["Time"]).codes
df_tests2["Score_std"] = standardize(df_tests2["Score"])
df_tests2
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
      <th>Name</th>
      <th>Time</th>
      <th>Score</th>
      <th>Name_code</th>
      <th>Time_code</th>
      <th>Score_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mike</td>
      <td>Test_1</td>
      <td>35</td>
      <td>2</td>
      <td>0</td>
      <td>-1.610167</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Melanie</td>
      <td>Test_1</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>-0.875490</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Melissa</td>
      <td>Test_1</td>
      <td>90</td>
      <td>1</td>
      <td>0</td>
      <td>1.083649</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mitchell</td>
      <td>Test_1</td>
      <td>78</td>
      <td>3</td>
      <td>0</td>
      <td>0.495907</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mike</td>
      <td>Test_2</td>
      <td>67</td>
      <td>2</td>
      <td>1</td>
      <td>-0.042856</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Melanie</td>
      <td>Test_2</td>
      <td>46</td>
      <td>0</td>
      <td>1</td>
      <td>-1.071404</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Melissa</td>
      <td>Test_2</td>
      <td>86</td>
      <td>1</td>
      <td>1</td>
      <td>0.887735</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mitchell</td>
      <td>Test_2</td>
      <td>91</td>
      <td>3</td>
      <td>1</td>
      <td>1.132627</td>
    </tr>
  </tbody>
</table>
</div>






There are different "clusters" here. On one level, is time, whether it was the first or second test. On another level are the individual subjects.

## Non multi-level model with time as an indexed, categorical variable

Let's start without a multi-level model. Since the main question is whether there are differences between the first and second tests, we'll use only time as a cluster for now and ignore the subject cluster. This is an intercept only model, with time represented as an indexed categorical variable.

<span style="color:red">Not sure if I set this up right. I think we'd still use a Normal here but should I use a Student t distribution instead. Also I used a flat prior since non-Bayesians wouldn't regularize </span>.

$$\text{score}_i \text{ ~ Normal}(\mu_i, \sigma) $$
$$\mu_i = \alpha_{\text{TIME}}$$
$$\alpha_j \text{ ∼ Normal}(0, 10)$$
$$\sigma \text{ ~ Exp}(1)$$


```python
with pm.Model() as m1:

    # prior for SD of testers
    sigma = pm.Exponential("sigma", 1.0)

    # regularizing prior
    a = pm.Normal("a", 0, 10, shape=2)  # two time points

    # mu is deterministic, equivalent to alpha indexed by time
    mu = a[df_tests2["Time_code"]]

    # likelihood
    score = pm.Normal("score", mu=mu, sd=sigma, observed=df_tests2["Score_std"])

    trace_m1 = pm.sample(
        draws=1000, random_seed=19, return_inferencedata=True, progressbar=False
    )
```

    Auto-assigning NUTS sampler...
    INFO:pymc3:Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    INFO:pymc3:Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    INFO:pymc3:Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [a, sigma]
    INFO:pymc3:NUTS: [a, sigma]
    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 15 seconds.
    INFO:pymc3:Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 15 seconds.






```python
az.summary(trace_m1)
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a[0]</th>
      <td>-0.235</td>
      <td>0.638</td>
      <td>-1.242</td>
      <td>0.741</td>
      <td>0.013</td>
      <td>0.011</td>
      <td>2353.0</td>
      <td>1753.0</td>
      <td>2439.0</td>
      <td>2051.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[1]</th>
      <td>0.239</td>
      <td>0.644</td>
      <td>-0.723</td>
      <td>1.308</td>
      <td>0.012</td>
      <td>0.011</td>
      <td>2907.0</td>
      <td>1690.0</td>
      <td>2995.0</td>
      <td>2179.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>1.238</td>
      <td>0.381</td>
      <td>0.703</td>
      <td>1.730</td>
      <td>0.009</td>
      <td>0.006</td>
      <td>1965.0</td>
      <td>1917.0</td>
      <td>2084.0</td>
      <td>1943.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>






<span style="color:red">Not sure if I set this up right and even if I did, I'm not sure what to do next</span>.

Now let's calculate the contrast between the alpha parameters representing the two timepoints (??)


```python
# Put the trace object in a dataframe so we can pull out the posteriors
trace_m1_df = trace_m1.to_dataframe()

# calculate contrast
post_diff = (
    trace_m1_df[("posterior", "a[0]", 0)] - trace_m1_df[("posterior", "a[1]", 1)]
)
```




<span style="color:red">I'm just using the t-statistic equation here</span>.


```python
post_diff.mean() / (post_diff.std() / np.sqrt(4))
```




    -1.0553808889820766






<span style="color:red">With a pretty flat prior (what non-Bayesians would do), I got pretty close to the answer, but not sure if I just lucky</span>.

## Multi-level model

We can now recognize both clusters and let the model do adaptive regularization.

<span style="color:red">Not sure if I set this up right. I'm unsure what the top line sigma would be for example.</span>

$$\text{score}_i \text{ ~ Normal}(\mu_i, \sigma) $$
<br>
$$\mu_i = \alpha_{\text{TIME}} + \gamma_{\text{SUBJECT}} $$
<br>
$$\alpha_j \text{ ∼ Normal}(\bar{\alpha}, \sigma_\alpha)$$
<br>
$$\gamma_j \text{ ∼ Normal}(0, \sigma_\gamma) $$
<br>
$$\bar{\alpha} \text{ ∼ Normal}(0, 10)$$
<br>
$$\sigma_\alpha \text{ ~ Exp}(1)$$
<br>
$$\sigma_\gamma \text{ ~ Exp}(1)$$


Alpha is using an adaptive prior for time (j=1,2) and gamma is using adaptive prior for subject (j = 1..4). We'll use one global mean parameter ("bar_alpha") since each varying intercept type will be added to the same linear prediction. This global mean parameter is essentially the average across all tests and subjects. Like in the previous model, it is pretty wide to represent the non-Bayesian perspective. Each cluster also gets its own sigma.


```python
with pm.Model() as m2:
    # Top line sigma (not sure if this is right)
    sigma = pm.Exponential("sigma", 1.0)

    # prior for average person and timepoint
    a_bar = pm.Normal("a_bar", 0.0, 10)

    # prior for SD of testers and time
    sigma_a = pm.Exponential("sigma_a", 1.0)
    sigma_g = pm.Exponential("sigma_g", 1.0)

    # adaptive priors?
    a = pm.Normal("a", a_bar, sigma_a, shape=2)  # 2 time points
    g = pm.Normal("g", 0, sigma_g, shape=4)  # 4 subjects

    mu = a[df_tests2["Time_code"]] + g[df_tests2["Name_code"]]

    # likelihood
    score = pm.Normal("score", mu=mu, sd=sigma, observed=df_tests2["Score_std"])

    trace_m2 = pm.sample(
        draws=1000, random_seed=19, return_inferencedata=True, progressbar=False
    )
```

    Auto-assigning NUTS sampler...
    INFO:pymc3:Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    INFO:pymc3:Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    INFO:pymc3:Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [g, a, sigma_g, sigma_a, a_bar, sigma]
    INFO:pymc3:NUTS: [g, a, sigma_g, sigma_a, a_bar, sigma]
    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 14 seconds.
    INFO:pymc3:Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 14 seconds.
    There were 88 divergences after tuning. Increase `target_accept` or reparameterize.
    ERROR:pymc3:There were 88 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 118 divergences after tuning. Increase `target_accept` or reparameterize.
    ERROR:pymc3:There were 118 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 96 divergences after tuning. Increase `target_accept` or reparameterize.
    ERROR:pymc3:There were 96 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 125 divergences after tuning. Increase `target_accept` or reparameterize.
    ERROR:pymc3:There were 125 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.7129996194220574, but should be close to 0.8. Try to increase the number of tuning steps.
    WARNING:pymc3:The acceptance probability does not match the target. It is 0.7129996194220574, but should be close to 0.8. Try to increase the number of tuning steps.
    The number of effective samples is smaller than 10% for some parameters.
    WARNING:pymc3:The number of effective samples is smaller than 10% for some parameters.





<span style="color:red">The divergences indicate I need to re-paramaterize, but let me know if my overall approach is right</span>


```python
az.summary(trace_m2)
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a_bar</th>
      <td>-0.038</td>
      <td>0.926</td>
      <td>-1.355</td>
      <td>1.272</td>
      <td>0.040</td>
      <td>0.028</td>
      <td>549.0</td>
      <td>549.0</td>
      <td>529.0</td>
      <td>924.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>a[0]</th>
      <td>-0.162</td>
      <td>0.630</td>
      <td>-1.103</td>
      <td>0.822</td>
      <td>0.026</td>
      <td>0.019</td>
      <td>575.0</td>
      <td>575.0</td>
      <td>565.0</td>
      <td>820.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>a[1]</th>
      <td>0.126</td>
      <td>0.634</td>
      <td>-0.873</td>
      <td>1.082</td>
      <td>0.027</td>
      <td>0.019</td>
      <td>544.0</td>
      <td>544.0</td>
      <td>532.0</td>
      <td>846.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>g[0]</th>
      <td>-0.617</td>
      <td>0.682</td>
      <td>-1.603</td>
      <td>0.413</td>
      <td>0.028</td>
      <td>0.020</td>
      <td>589.0</td>
      <td>589.0</td>
      <td>601.0</td>
      <td>836.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>g[1]</th>
      <td>0.675</td>
      <td>0.683</td>
      <td>-0.322</td>
      <td>1.739</td>
      <td>0.029</td>
      <td>0.021</td>
      <td>548.0</td>
      <td>528.0</td>
      <td>556.0</td>
      <td>888.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>g[2]</th>
      <td>-0.531</td>
      <td>0.670</td>
      <td>-1.614</td>
      <td>0.395</td>
      <td>0.027</td>
      <td>0.020</td>
      <td>612.0</td>
      <td>568.0</td>
      <td>647.0</td>
      <td>751.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>g[3]</th>
      <td>0.535</td>
      <td>0.659</td>
      <td>-0.388</td>
      <td>1.611</td>
      <td>0.029</td>
      <td>0.021</td>
      <td>529.0</td>
      <td>486.0</td>
      <td>533.0</td>
      <td>506.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>0.784</td>
      <td>0.338</td>
      <td>0.324</td>
      <td>1.210</td>
      <td>0.013</td>
      <td>0.009</td>
      <td>682.0</td>
      <td>682.0</td>
      <td>620.0</td>
      <td>809.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sigma_a</th>
      <td>0.749</td>
      <td>0.692</td>
      <td>0.107</td>
      <td>1.527</td>
      <td>0.030</td>
      <td>0.021</td>
      <td>527.0</td>
      <td>527.0</td>
      <td>246.0</td>
      <td>169.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>sigma_g</th>
      <td>0.943</td>
      <td>0.548</td>
      <td>0.127</td>
      <td>1.631</td>
      <td>0.020</td>
      <td>0.014</td>
      <td>751.0</td>
      <td>751.0</td>
      <td>574.0</td>
      <td>598.0</td>
      <td>1.01</td>
    </tr>
  </tbody>
</table>
</div>







```python
# Put the trace object in a dataframe so we can pull out the posteriors
trace_m2_df = trace_m2.to_dataframe()

# calculate contrast
post_diff2 = (
    trace_m2_df[("posterior", "a[0]", 0)] - trace_m2_df[("posterior", "a[1]", 1)]
)
```




<span style="color:red">Again, I'm just using the t-statistic equation here</span>.


```python
post_diff2.mean() / (post_diff2.std() / np.sqrt(4))
```




    -1.2088091266973582






<span style="color:red">I get an answer further than the previous model but in the right direction. Again, not sure if I had done something incorrect</span>



Appendix: Environment and system parameters


```python
%watermark -n -u -v -iv -w
```

    Last updated: Wed Aug 04 2021
    
    Python implementation: CPython
    Python version       : 3.8.6
    IPython version      : 7.20.0
    
    seaborn   : 0.11.1
    scipy     : 1.6.0
    matplotlib: 3.3.4
    arviz     : 0.11.1
    daft      : 0.1.0
    pandas    : 1.2.1
    pymc3     : 3.11.0
    json      : 2.0.9
    numpy     : 1.20.1
    
    Watermark: 2.1.0
    




