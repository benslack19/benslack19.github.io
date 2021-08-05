---
title: "Connecting a paired t-test with a Bayesian multilevel model (DRAFT)"
toc: true
toc_sticky: true
mathjax: true
---


```python
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymc3 as pm
import scipy.stats as stats
import seaborn as sns
from scipy.special import expit as logistic
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





```python
sns.set_context("talk")
```





```python
def standardize(x):
    x = (x - np.mean(x)) / np.std(x)
    return x
```




# Dataset, extrapolated from Wikipedia

Example taken [here](https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples).


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
      <th>diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mike</td>
      <td>35</td>
      <td>67</td>
      <td>-32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Melanie</td>
      <td>50</td>
      <td>46</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Melissa</td>
      <td>90</td>
      <td>86</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mitchell</td>
      <td>78</td>
      <td>91</td>
      <td>-13</td>
    </tr>
  </tbody>
</table>
</div>






Interestingly, the table in this example is labeled as an "example of repeated measures." Interestingly, in the Chapter 1 introduction and in Chapter 13, McElreath highlights multi-level models as a way to get "improved estimates for repeat sampling".

# T-statistic using scipy.stats


```python
stats.ttest_rel(df_tests["Test_1"], df_tests["Test_2"])
```




    Ttest_relResult(statistic=-1.0784834690588145, pvalue=0.3598054860063756)






# Calculate t-statistic by hand

Why? Cause we're weird like that.

$$t = \frac{\bar{X}_D - \mu_0}{s_D / \sqrt{n}}$$

where \(\bar{X}_D\) is the average of the differences between all pairs and \(s_D\) is the standard deviation of the differences. We can set \(mu_0\) to 0 if we want to test whether the average is significantly different. This is straightforward to calculate.


```python
df_tests["diff"] = df_tests["Test_1"] - df_tests["Test_2"]
```





```python
t_stat = df_tests["diff"].mean() / (df_tests["diff"].std() / np.sqrt(len(df_tests)))
print("Manual calculated t-statistic: ", t_stat)
```

    Manual calculated t-statistic:  -1.0784834690588145





We get exactly the same answer as `scipy.stats` so that is comforting.

# Calculated with a Bayesian approach

Now the fun part. We can set this up as a multi-level linear model but let's start with a non multi-level model.

I'm going to treat each person as their own "cluster". The test is the repeated measure, with time as a categorical variable as indicated [here](https://www.researchgate.net/post/Paired-t-test-or-liner-mixed-model).

First, let's do some table reformatting so that we can use it in our model.


```python
df_tests2 = pd.melt(
    df_tests.drop("diff", axis=1),
    id_vars="Name",
)
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






## Non-multi-level model

$$\text{score}_i \text{ ~ Normal}(\mu_i, \sigma) $$
$$\mu_i = \alpha_{\text{SUBJECT}} + \gamma_{\text{TIME}} $$
$$\alpha_j \text{ ∼ Normal}(0, 1.5) \tag{regularizing prior}$$
$$\gamma_j \text{ ∼ Normal}(0, 1.5) \tag{regularizing prior}$$
$$\sigma \text{ ~ Exp}(1)$$


```python
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







```python
with pm.Model() as m1:

    # prior for SD of testers
    sigma = pm.Exponential("sigma", 1.0)

    # regularizing prior
    a = pm.Normal("a", 0, 1.5, shape=4)  # one for each subject
    g = pm.Normal("g", 0, 1.5, shape=2)  # one for each time point

    # mu is deterministic, equivalent to alpha indexed by time
    mu = a[df_tests2["Name_code"]] + g[df_tests2["Time_code"]]

    # likelihood
    score = pm.Normal("score", mu=mu, sd=sigma, observed=df_tests2["Score_std"])

    trace_m1 = pm.sample(
        draws=1000, random_seed=19, return_inferencedata=True, progressbar=False
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [g, a, sigma]
    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 22 seconds.
    There was 1 divergence after tuning. Increase `target_accept` or reparameterize.
    There were 4 divergences after tuning. Increase `target_accept` or reparameterize.
    The number of effective samples is smaller than 10% for some parameters.






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
      <td>-0.918</td>
      <td>0.738</td>
      <td>-2.022</td>
      <td>0.244</td>
      <td>0.030</td>
      <td>0.021</td>
      <td>615.0</td>
      <td>615.0</td>
      <td>619.0</td>
      <td>1418.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>a[1]</th>
      <td>0.838</td>
      <td>0.727</td>
      <td>-0.323</td>
      <td>1.989</td>
      <td>0.029</td>
      <td>0.020</td>
      <td>649.0</td>
      <td>649.0</td>
      <td>650.0</td>
      <td>1547.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>a[2]</th>
      <td>-0.782</td>
      <td>0.744</td>
      <td>-1.950</td>
      <td>0.379</td>
      <td>0.029</td>
      <td>0.021</td>
      <td>645.0</td>
      <td>645.0</td>
      <td>648.0</td>
      <td>1662.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>a[3]</th>
      <td>0.689</td>
      <td>0.728</td>
      <td>-0.465</td>
      <td>1.837</td>
      <td>0.028</td>
      <td>0.020</td>
      <td>688.0</td>
      <td>688.0</td>
      <td>687.0</td>
      <td>1464.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>g[0]</th>
      <td>-0.165</td>
      <td>0.674</td>
      <td>-1.175</td>
      <td>0.967</td>
      <td>0.028</td>
      <td>0.020</td>
      <td>560.0</td>
      <td>560.0</td>
      <td>566.0</td>
      <td>1216.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>g[1]</th>
      <td>0.258</td>
      <td>0.670</td>
      <td>-0.838</td>
      <td>1.266</td>
      <td>0.029</td>
      <td>0.021</td>
      <td>532.0</td>
      <td>532.0</td>
      <td>538.0</td>
      <td>1256.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>0.711</td>
      <td>0.310</td>
      <td>0.288</td>
      <td>1.074</td>
      <td>0.012</td>
      <td>0.008</td>
      <td>707.0</td>
      <td>707.0</td>
      <td>350.0</td>
      <td>106.0</td>
      <td>1.01</td>
    </tr>
  </tbody>
</table>
</div>







```python
trace_m1_df = trace_m1.to_dataframe()
trace_m1_df.head()
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
      <th>(posterior, a[0], 0)</th>
      <th>(posterior, a[1], 1)</th>
      <th>(posterior, a[2], 2)</th>
      <th>(posterior, a[3], 3)</th>
      <th>(posterior, g[0], 0)</th>
      <th>(posterior, g[1], 1)</th>
      <th>(posterior, sigma)</th>
      <th>(log_likelihood, score[0], 0)</th>
      <th>...</th>
      <th>(sample_stats, mean_tree_accept)</th>
      <th>(sample_stats, perf_counter_start)</th>
      <th>(sample_stats, energy_error)</th>
      <th>(sample_stats, step_size)</th>
      <th>(sample_stats, max_energy_error)</th>
      <th>(sample_stats, depth)</th>
      <th>(sample_stats, tree_size)</th>
      <th>(sample_stats, diverging)</th>
      <th>(sample_stats, energy)</th>
      <th>(sample_stats, process_time_diff)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>-1.130323</td>
      <td>0.428683</td>
      <td>-1.801091</td>
      <td>-0.103095</td>
      <td>0.236200</td>
      <td>0.720884</td>
      <td>0.425864</td>
      <td>-0.070954</td>
      <td>...</td>
      <td>0.871617</td>
      <td>1169.238632</td>
      <td>0.401664</td>
      <td>0.397527</td>
      <td>0.820186</td>
      <td>4</td>
      <td>15.0</td>
      <td>False</td>
      <td>20.229142</td>
      <td>0.001596</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>-1.664056</td>
      <td>0.340213</td>
      <td>-0.834904</td>
      <td>0.478362</td>
      <td>0.340148</td>
      <td>0.723387</td>
      <td>0.385726</td>
      <td>-4.147343</td>
      <td>...</td>
      <td>0.956333</td>
      <td>1169.240341</td>
      <td>-0.479124</td>
      <td>0.397527</td>
      <td>-1.047678</td>
      <td>2</td>
      <td>3.0</td>
      <td>False</td>
      <td>17.178882</td>
      <td>0.000467</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>-2.008341</td>
      <td>-0.181530</td>
      <td>-0.620523</td>
      <td>-0.158988</td>
      <td>0.735810</td>
      <td>0.925338</td>
      <td>0.786464</td>
      <td>-3.085416</td>
      <td>...</td>
      <td>0.495141</td>
      <td>1169.240921</td>
      <td>0.846038</td>
      <td>0.397527</td>
      <td>1.266118</td>
      <td>4</td>
      <td>15.0</td>
      <td>False</td>
      <td>21.314292</td>
      <td>0.001638</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>-1.065443</td>
      <td>1.599116</td>
      <td>-1.334224</td>
      <td>0.305367</td>
      <td>-0.219125</td>
      <td>0.344952</td>
      <td>0.559128</td>
      <td>-0.342726</td>
      <td>...</td>
      <td>0.987480</td>
      <td>1169.242676</td>
      <td>-0.056466</td>
      <td>0.397527</td>
      <td>-0.426190</td>
      <td>4</td>
      <td>15.0</td>
      <td>False</td>
      <td>22.029038</td>
      <td>0.001713</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>-1.466140</td>
      <td>1.068115</td>
      <td>-0.920037</td>
      <td>0.881574</td>
      <td>0.442463</td>
      <td>-0.066000</td>
      <td>0.654789</td>
      <td>-1.991440</td>
      <td>...</td>
      <td>0.758809</td>
      <td>1169.244531</td>
      <td>0.272440</td>
      <td>0.397527</td>
      <td>0.511085</td>
      <td>2</td>
      <td>3.0</td>
      <td>False</td>
      <td>21.150677</td>
      <td>0.000515</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>







```python
post_diff = (
    trace_m1_df[("posterior", "g[0]", 0)] - trace_m1_df[("posterior", "g[1]", 1)]
)
```





```python
post_diff.mean()
```




    -0.4223415186249017







```python
post_diff.std()
```




    0.5166181274773156






**Not sure if I set this up right and even if I did, I'm not sure what to do next.**

## Multi-level model

**Is this model right? I'm confused for what to put for the top line sigma for example**

$$\text{score}_i \text{ ~ Normal}(\mu_i, \sigma) $$
$$\mu_i = \alpha_{\text{SUBJECT}} + \gamma_{\text{TIME}} $$
$$\alpha_j \text{ ∼ Normal}(\bar{\alpha}, \sigma_\alpha) \tag{adaptive prior for subject j = 1..4}$$
$$\gamma_j \text{ ∼ Normal}(0, \sigma_\gamma) \tag{adaptive prior for time j = 1, 2}$$
$$\bar{\alpha} \text{ ∼ Normal}(0, 1.5) \tag{prior for average person}$$
$$\sigma_\alpha \text{ ~ Exp}(1) \tag{sigma for alpha}$$
$$\sigma_\gamma \text{ ~ Exp}(1) \tag{sigma for gamma}$$


```python
with pm.Model() as m2:
    # Top line sigma (not sure if this is right)
    sigma = pm.Exponential("sigma", 1.0)

    # prior for average person and timepoint
    a_bar = pm.Normal("a_bar", 0.0, 1.5)

    # prior for SD of testers and time
    sigma_a = pm.Exponential("sigma_a", 1.0)
    sigma_g = pm.Exponential("sigma_g", 1.0)

    # adaptive priors?
    a = pm.Normal("a", a_bar, sigma_a, shape=4)  # 4 unique subjects
    g = pm.Normal("g", 0, sigma_g, shape=2)  # 2 time points

    # mu is deterministic, equivalent to alpha indexed by time
    mu = a[df_tests2["Name_code"]] + g[df_tests2["Time_code"]]

    # likelihood
    score = pm.Normal("score", mu=mu, sd=sigma, observed=df_tests2["Score_std"])

    trace_m2 = pm.sample(
        draws=1000, random_seed=19, return_inferencedata=True, progressbar=False
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [g, a, sigma_g, sigma_a, a_bar, sigma]
    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 16 seconds.
    There were 54 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 636 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.17202140484289688, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 64 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 99 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.632080573851144, but should be close to 0.8. Try to increase the number of tuning steps.
    The rhat statistic is larger than 1.2 for some parameters.
    The estimated number of effective samples is smaller than 200 for some parameters.






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
      <td>-0.042</td>
      <td>0.618</td>
      <td>-0.997</td>
      <td>0.882</td>
      <td>0.046</td>
      <td>0.033</td>
      <td>177.0</td>
      <td>177.0</td>
      <td>122.0</td>
      <td>738.0</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>a[0]</th>
      <td>-0.537</td>
      <td>0.654</td>
      <td>-1.590</td>
      <td>0.357</td>
      <td>0.048</td>
      <td>0.050</td>
      <td>187.0</td>
      <td>85.0</td>
      <td>115.0</td>
      <td>409.0</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>a[1]</th>
      <td>0.474</td>
      <td>0.714</td>
      <td>-0.535</td>
      <td>1.463</td>
      <td>0.159</td>
      <td>0.114</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>21.0</td>
      <td>289.0</td>
      <td>1.14</td>
    </tr>
    <tr>
      <th>a[2]</th>
      <td>-0.471</td>
      <td>0.632</td>
      <td>-1.395</td>
      <td>0.473</td>
      <td>0.038</td>
      <td>0.036</td>
      <td>281.0</td>
      <td>152.0</td>
      <td>236.0</td>
      <td>500.0</td>
      <td>1.30</td>
    </tr>
    <tr>
      <th>a[3]</th>
      <td>0.366</td>
      <td>0.708</td>
      <td>-0.567</td>
      <td>1.341</td>
      <td>0.173</td>
      <td>0.125</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>640.0</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th>g[0]</th>
      <td>-0.063</td>
      <td>0.517</td>
      <td>-0.701</td>
      <td>0.660</td>
      <td>0.074</td>
      <td>0.053</td>
      <td>49.0</td>
      <td>49.0</td>
      <td>26.0</td>
      <td>332.0</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>g[1]</th>
      <td>-0.010</td>
      <td>0.561</td>
      <td>-0.644</td>
      <td>0.797</td>
      <td>0.124</td>
      <td>0.089</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>17.0</td>
      <td>6.0</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>0.951</td>
      <td>0.439</td>
      <td>0.445</td>
      <td>1.594</td>
      <td>0.139</td>
      <td>0.101</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>134.0</td>
      <td>1.24</td>
    </tr>
    <tr>
      <th>sigma_a</th>
      <td>0.758</td>
      <td>0.570</td>
      <td>0.097</td>
      <td>1.426</td>
      <td>0.155</td>
      <td>0.112</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>1.39</td>
    </tr>
    <tr>
      <th>sigma_g</th>
      <td>0.566</td>
      <td>0.497</td>
      <td>0.036</td>
      <td>1.082</td>
      <td>0.024</td>
      <td>0.017</td>
      <td>424.0</td>
      <td>424.0</td>
      <td>164.0</td>
      <td>71.0</td>
      <td>1.26</td>
    </tr>
  </tbody>
</table>
</div>







```python

```


```python

```

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
    




