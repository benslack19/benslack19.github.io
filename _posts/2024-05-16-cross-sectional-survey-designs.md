---
title: "Cross-Sectional Survey Designs"
mathjax: true
toc: true
toc_sticky: true
categories: [data science, statistics]
---

Notes for Chapter 3 of [Causal Inference with Survey Data](https://www.linkedin.com/learning/causal-inference-with-survey-data/surveys-with-cross-sectional-data?autoSkip=true&resume=false&u=185169545) on LinkedIn Learning, given by Franz Buscha. I'm using this series of posts to take some notes.


```python
import graphviz as gr
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
```

# Cross-sectional survey designs

- It's a snapshot in time, capturing information from many subjects.
- Most common type of survey.

**Examples**
1. Census surveys. Provides a snapshot of a country's population (e.g. US Census done every 10 years).
1. Expenditure surveys. Information on buying habits (e.g. annual Consumer Expenditure Survey).
1. Labor force surveys. Collect data on employment (e.g. UK Labour Force Survey, conducted quarterly).

**Advantages**
- Availability
- Cheap to conduct
- Versatility in topics

**Disadvantages**
- Lack temporal data
- Sampling, selection, and response bias
- Lack of depth (limited data on complex issues)

**Statistical Framework**
- A key to working with cross-sectional data is the $i$ subscript, such as in the form:

$$ Y_i = \beta_0 + \beta_1X1_i + \beta_2X2_i + ... \epsilon_i$$

- The $i$ denotes different observations in the data (e.g. subjects or entities at a single point in time)

**Conclusion**
- Broad application and more themes.
- Explanatory variables must be used in innovative ways for cause-and-effect analysis.

# Regression analysis

- A fundamental statistical method
- A powerful tool for controlling observable factors
- Mainstay of causal analysis

**DAG: Controlling for Observable Factors**

- A regression model can answer this question: What is the causal effect of X on Y?




```python
draw_causal_graph(
    edge_list=[("X1i", "Yi"), ("&#x03B5;", "Yi")],
    edge_props={("&#x03B5;", "Yi"): {"style": "dashed", "label": "&beta;"}},
    graph_direction="LR",
)
```




    
![svg](/assets/2024-05-16-cross-sectional-survey-designs_files/2024-05-16-cross-sectional-survey-designs_6_0.svg)
    



Other factors that are not seen in the survey data are summed up in the hidden error term.

$ Y_i = \beta_0 + \beta_1X1_i + \epsilon_i $

- Regression can control for many observable factors
- Effects estimated in a regression model are independent of other effects in the model
- Causal infrence relies on there being no confounders (exogeneity assumption)
- Variables that don't gice a choice are often exogenous (sex, age, parents birthplace, etc.). These are variables that are "hard to influence".
- Assumption of exogeneity can be difficult. There can be many factors that drive both Y and X1. This creates a backdoor pathway.



```python
# `&#x03B5;` is unicode for epsilon since `&epsilon;` fails to render
draw_causal_graph(
    edge_list=[("X1i", "Yi"), ("&#x03B5;", "Yi"), ("&#x03B5;", "X1i")],
    edge_props={
        ("X1i", "Yi"): {"label": "&beta;1"},
        ("&#x03B5;", "Yi"): {"style": "dashed"},
        ("&#x03B5;", "X1i"): {"style": "dashed", "label": "backdoor"},
    },
    graph_direction="LR",
)
```




    
![svg](/assets/2024-05-16-cross-sectional-survey-designs_files/2024-05-16-cross-sectional-survey-designs_8_0.svg)
    



If the backdoor is present, then the estimate of $\beta_1$ will not be correct.

But imagine that $X2i$ in the error term can be observed. A new DAG might look like this.


```python
draw_causal_graph(
    edge_list=[("X1i", "Yi"), ("&#x03B5;", "Yi"), ("X2i", "X1i"), ("X2i", "Yi")],
    edge_props={
        ("X1i", "Yi"): {"label": "&beta;1"},
        ("X2i", "Yi"): {"label": "&beta;2"},
        ("&#x03B5;", "Yi"): {"style": "dashed"},
        ("&#x03B5;", "X1i"): {"style": "dashed", "label": "backdoor"},
    },
    graph_direction="LR",
)
```




    
![svg](/assets/2024-05-16-cross-sectional-survey-designs_files/2024-05-16-cross-sectional-survey-designs_10_0.svg)
    



$ Y_i = \beta_0 + \beta_1X1_i + \beta_2X2_i + \epsilon_i $

$X2$ is now specifically controlled for.

Triangular Tables:
A way to observe the effect on a regression model of incrementally adding more variables but be careful of overfitting. Knowing what variables to include requires some domain knowledge.

**Advantages**
- Flexibility in variables
- Many different forms for different data
- Easy to understand

**Disadvantages**
- Often too simple
- Cannot control for unobserved confounders

**Conclusion**
- Don't dismiss basic regression
- Underpins more complex models
- Works well with large surveys and many variables


```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Sun May 19 2024
    
    Python implementation: CPython
    Python version       : 3.11.7
    IPython version      : 8.21.0
    
    graphviz: 0.20.1
    
    Watermark: 2.4.3
    

