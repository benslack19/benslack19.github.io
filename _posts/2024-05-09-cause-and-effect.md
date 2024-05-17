---
title: "Cause and effect"
mathjax: true
toc: true
toc_sticky: true
categories: [data science, statistics]
---

I'm basically a fan-boy of Richard McElreath's Statistical Rethinking. That's no secret. But I thought it would be prudent to learn more about causal inference from other sources. I found [Causal Inference with Survey Data](https://www.linkedin.com/learning/causal-inference-with-survey-data/) on LinkedIn Learning, given by Franz Buscha. I'm using this series of posts to take some notes. In addition, the lectures show some example code in another language (STATA, I believe) and coding them in Python could reinforce the learning lessons for me.


```python
import graphviz as gr
from scipy import stats
import statsmodels.api as sm
```

# Introduction of cause and effect
- Causal inference matters because it helps distinguish variables that are associations from those that can be causal. It's very common to misunderstand data causality unfortunately.
- Experiments are the gold standard because it can be used to control confounding. But experiments can be unethical or impractical in many cases. For example, researchers can't simply force smoking on groups of people. Establishing a link between smoking and lung cancer was done with survey data.- Survey data is often self-reported so it's ripe with biases. American Community Survey is one example of survey data. When using survey data, it's recommended to:
    - use probability sampling when possible
    - correct biases with weights
    - apply techniques for missing data
    - ensure survey's topical relevance
    - invest time in understanding survey data complexity

# Observables vs. unobservable causes
- Variables can be observed in the data but some variables are unboserved, due to lack of measurement or the ability to measure. Personal motivation is something hard to measure for example. But unobserved variables can bias results so they're crucial in causal inference.
- Endogeneity: occurs when an explanatory variable is correlated with the error term in a model. It can happen when a variable is left out of a regression model. Thinking about unobserved variables can guide researchers to using better causal techniques.
- DAGs help one can better understand causal flow of models. We have to be careful about the presence of backdoor paths between the predictor of interest and the outcome variable.



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
```


```python
draw_causal_graph(
    edge_list=[("X1", "Y"), ("X2", "Y"), ("X2", "X1")],
    edge_props={
        ("X2", "Y"): {"label": "backdoor", "style": "dashed"},
        ("X2", "Y"): {"style": "solid"},
    },
    graph_direction="LR",
)
```




    
![svg](/assets/2024-05-09-cause-and-effect_files/2024-05-09-cause-and-effect_4_0.svg)
    



- Not controlling for the backdoor path will lead to bias between X1 and Y
- X2 often called a confounder


```python
# create a dataframe that holds the number and probability for each group
n_obs = 200
x2 = stats.norm.rvs(loc=0, scale=1, size=200)
x1 = stats.norm.rvs(loc=0, scale=1, size=200) + x2
y = x1 + x2

model0 = sm.OLS(y, x1)
model1 = sm.OLS(y, x1 + x2)

results0 = model0.fit()
results1 = model1.fit()
```


```python
# with x1 only gives a biased estimate
results0.params
```




    array([1.46420051])




```python
# including x2 recovers the right parameter given the data generating process
results1.params
```




    array([1.])



But what if x2 is unobserved? Unfortunately the backdoor path cannot be closed. This is where basic regression analysis fails us. But this is where more advanced methods can help us in the case of unobserved confounds.

<p style="text-align: center;font-weight:bold;">DAGs and Methods</p>

| Methodology | Controls for Unobserved Confounds | 
| ----- | ------- |
| Regression analysis | No (controls for observed confounds only) | 
| Propensity score matching | No (controls for observed confounds only) | 
| Instrumental variables | Yes (if a valid instrument is used) | 
| Regression discontinuity designs | Partially (near the cutoff point) |
| Regression with time effects  | Partially (for time-varying confounders) |
| Fixed effects regression  | Yes (for time-invariant confounders within entities) |
| Difference-in-Differences Models | Yes (if trends are parallel without  treatment) |
| Synthetic control methods | Partially (for observed and unobserved pre-treatment confounders) |


# What are treatment effects?

- The causal literature has acronyms like ATE, ATT, LATE, CATE, etc.
- They're called potential outcome notation
- It can define the methodological approach and narrative of results
- Essentially this is defining counterfactuals (parallel universes): a comparison between two states of the world.
    - For example, imagine someone takes a pill and the outcome is measured. In a parallel world, they don't take it and the outcome is measured. The difference in outcomes is the treatment effect (the causal effect of the pill).
- Of course, we can't have both states. But we can have different states of causality. This results in different kinds of treatment effects.

**Average Treatment Effect (ATE)**

The average effect of the treatment across the entire population.
$$ATE = E[Y^1_i - Y^0_i]$$
where $Y^1$ is the outcome when treatment is given, $Y^0$ is the outcome when treatment is not given, $i$ is for individual, and $E$ is the expected, which means the effect is averaged out over all individuals.

*Interpretation*: If you impose a treatment on everyone, then this is the change the average individual will see. But it literally means everyone and so if a drug is sex-specific, using the ATE wouldn't make sense.

**Average Treatment Effect on the Treated (ATT)**

The average effect of the treatment for those treated.
$$ ATT = E[Y^1_i - Y^0_i | \text{Treated}=1 ] $$

*Interpretation*: Shows effect of intervention only on those that received the intervention (treatment). ATT is usually different from ATE due to selection. (Unsure about $Y^0$ since by definition it shouldn't exist here?)

Non-random treatment will likely lead to ATT and not ATE since people often self-select expecting benefits.

**Average Treatment Effect on the Untreated (ATU)**

The average effect of the treatment for those in the control group.
$$ ATU = E[Y^1_i - Y^0_i | \text{Treated}=0 ] $$

But this can't be estimated. It's still useful to think about what would have happened to those who were not reached by an intervention.

**Local Average Treatment Effect (LATE)**

The average effect of the treatment for those who complied
$$ LATE = E[Y^1_i - Y^0_i | \text{Compliers}=1 ] $$
Treatment conditions only received under certain conditons; conditions influenced by another "instrumental variable".
Example: Study the impact of receiving a scholarship. The instrumental variable might be living in a particular region, LATE would measure the effect of the scholarship on just those students who received it due to their location. Many compliers means LATE approaches ATE. But few compliers limits external validity.

Conclusion
- Understanding true effects in non-randomized settings (experiments approach ATE and ATT).
- Advanced methods lead to different effects
- Model selection matters!

# An applied example: The Lalonde debate

How different are nonexperimental methods compared to experimental methods? Focus of seminal study from Robert Lalonde in 1986.

- National supported work program was temporary job training program in the mid-1970s.
- It was designed to help people with temporary work experience.
- But there was a randomized component! 50/50 chance of getting help or no help.

Lalonde compared the randomzied experiment data with CPS and PSID surveys. The experimental data recovered the ATE. When looking at experimental data, there was a net positive in income suggesting the program worked. He got similar values using causal inference methods. However, survey data varied wildly. Depending on which one you choose can lead to very different policies.

Conclusion
- A wake-up call for economists!
- Methods matter, and careful design must be given to causal studies.
- This led to significant advances in methods and how to use surveys.




```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    The watermark extension is already loaded. To reload it, use:
      %reload_ext watermark
    Last updated: Thu May 09 2024
    
    Python implementation: CPython
    Python version       : 3.11.7
    IPython version      : 8.21.0
    
    seaborn    : 0.13.2
    statsmodels: 0.14.1
    scipy      : 1.12.0
    matplotlib : 3.8.2
    pandas     : 2.2.0
    numpy      : 1.25.2
    
    Watermark: 2.4.3
    

