---
title: "Follow-up after getting causal estimates"
mathjax: true
toc: true
toc_sticky: true
categories: [data science, statistics]
---

Notes for Chapter 5 of [Causal Inference with Survey Data](https://www.linkedin.com/learning/causal-inference-with-survey-data/how-to-evaluate-causal-robustness?autoSkip=true&resume=false&u=185169545) on LinkedIn Learning, given by Franz Buscha. I'm using this series of posts to take some notes.

# How to evaluate causal robustness

- Once you have your first causal estimate, you can't stop.
- Others may not believe you.

**Robust analysis**
- What is the resilience of your causal estimate? Ensure it's not an artifact of your analysis.
- Need to show extra analysis to validate.

**Specification robustness**
<br>
Examine the stability of model estimates when changing model specifications

![png](/assets/2024-05-28-other-models-and-follow-up_files/specification_robustness.png)

**Data robustness**
<br>
Examine the consistency of estimates across different datasets or subsamples

![png](/assets/2024-05-28-other-models-and-follow-up_files/data_robustness.png)

**Method robustness**
- Examine whether it holds across a range of analytical methods
- But generally not as common, but might be able to do regression and propensity score matching

**How to present robustness analysis**
Tables are common.

![png](/assets/2024-05-28-other-models-and-follow-up_files/present_robust_analysis_table.png)

Another advanced visualization example

![png](/assets/2024-05-28-other-models-and-follow-up_files/present_robust_analysis_viz.png)

## Summary

**Conclusion**
- Other forms of robustness analysis exist
- Specification, data, methods are the primary forms
- Few people believe a single number
- Always provide a range of results

# How to present causal statistics

- How do you get this information across?

- Clarity and simplicity
    - Clear research statement
    - Use simple language
    - Avoid complex terminology and jargon
- Set context
    - Present prior work (what people have done and what are the gaps), data (how does data look), methods (why the methodology, what are you concerned about)
- Be mindful of clear presentation in tables or visualization (like with coefficient plots) to reduce complexity.
- Interpret results carefully, avoid overstating, refer back to prior work and contextualize findings within larger set of results.
- Be transparent about assumptions, limitations, and replicability; share code, data and methods for replication

## Summary
- It takes a long time to get to the final stage.
- Your job is not to show off, but to communicate important insights and information.
- Summarize findings in a credible, convincing, and enjoyable way.


```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    The watermark extension is already loaded. To reload it, use:
      %reload_ext watermark
    Last updated: Tue May 28 2024
    
    Python implementation: CPython
    Python version       : 3.12.3
    IPython version      : 8.24.0
    
    Watermark: 2.4.3
    

