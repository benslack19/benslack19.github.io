---
title: "Experimental Survey Designs"
mathjax: true
toc: true
toc_sticky: true
categories: [data science, statistics]
---

Notes for Chapter 2 of [Causal Inference with Survey Data](https://www.linkedin.com/learning/causal-inference-with-survey-data/observables-vs-unobservables-causes?u=185169545) on LinkedIn Learning, given by Franz Buscha.

# Experimental survey designs

- Randomized controlled trials are the gold standard for causal inference.
- Key steps in setting up an RCT:
    - Starting with your study sample, randomize assignment to control and treatment arms. The treatment group receives the intervention then the outcome of both control and treatment are studied. Randomization helps account for confounds.
    - The conrol group does not receive the intervention, but you can have multiple treatment groups, each with different interventions. In practice, though, most RCTs have only one treatment group.

<p style="text-align: center;font-weight:bold;">Randomization Methods</p>

| Method | Description | Key Points | 
| ----- | ----- | ------- |
| Simple randomization  | Assigns equal probability to treatment and control groups |- Easy to implement<br>- Can lead to unequal group sizes in small trials |
| Block randomization  | Get similar group sizes by dividing subjects into predetermined number of subjects (often a multiple of the number of groups, like 4, 8, etc. for two groups). Within each block, participants are then randomly assigned to treatment groups.  |- Requires choosing the total number of subjects in each block <br>- Can often avoid imbalance in small trials seen with simple randomization |
| Stratified randomization  | Balance based on characteristics/covariates (like age, sex, etc.) before randomizing within these strata |<ul><li>Ensures balance in important covariates between groups  |
| Cluster randomization  | Randomizes entire groups (like schools or hospitals) |- Suitable for group-level interventions or when individual assignment is impractical   |
| Covariate adaptive randomization  | Increases the probability of being assigned to a group to address a deficit of a particular characteristic   within the group |<ul><li>Effective in trials with small sample sizes or multiple important covariates |

- Before randomization, the number of observations is typically calculated ahead of time based on a specific effect size (power analysis). You need to consider measure of variability (standard deviation), significance level, and power.
- You can make a graph of sample size on y-axis and effect size on x-axis to see the relationship.

With this approach, causality can be assessed.


# Analyzing a randomized controlled trial

## Inside an RCT dataset
- Participant ID.
- **Demographic variables**. Age, sex, ethnicity, etc. Helps check that randomization occurred successfully.
- Baseline characteristics. Information collected at the beginning of the trial, before any intervention, including health status.
- **Group assignment**, e.g. control/treatment and sometimes randomization details.
- Intervention details. Specifics such as frequency, dosage, duration.
- **Outcome measures**. Primary and secondary outcomes throughout and at the end of the trial.
- **Time points**. Time points collected during the trial, for example baseline, midpoint, end of treatment, etc.
- Adverse events.
- Compliance data. How well participants adhered to the intervention protocol.
- Covariates. Variables that might be used to adjust theoutcomes.

Most critical data in bold.

## RCT analysis: descriptive
- Check descriptive statistics and examine effectiveness of randomization, e.g. checking demographic data. Look for imbalances.
- If there's an imbalance, then consider a covariate balancing method like regression.

## RCT analysis: comparison

| Analysis Type | Description |
| ----- | ------- | 
| Intention-to-Treat (ITT)  | Analyze groups as they were assigned, regardless of whether they completed the intervention. Maintains the benefits of randomization |
| Per-Protocol  | Includes only participants who completed the study. Useful in medical trials but may introduce bias due to loss of randomization (e.g. treatment failure may not be random).  |
| As-Treated  | Analyzed according to treatment they actualy received, regardless of original assignment. This can happen when there can be emerging issues for someone in a control group. |

## RCT analysis: statistics

| Analysis Type | Applicability | Statistical test/technique |
| ----- | ------- | ------- | 
| Continuous outcomes  | For normally distributed continuous outcomes | T-tests or ANOVA; linear regression if adjustment is needed |
| Non-normally distributed outcomes  | Non-normal or ordinal outcomes | Wilcoxon rank-sum, Mann-Whitney |
| Categorical outcomes  | Compare proportions between groups | Chi-square test; logistic regression if adjustment is needed |
| Time-to-event  | Time until event occurs | Kaplan-Meier curves, log-rank tests; Cox proportional hazard if adjustment is needed |

Regression techniques can be used when randomization doesn't go as expected.


```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Fri May 17 2024
    
    Python implementation: CPython
    Python version       : 3.11.7
    IPython version      : 8.21.0
    
    Watermark: 2.4.3
    

