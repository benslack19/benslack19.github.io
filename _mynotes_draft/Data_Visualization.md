--- 
title: 'Data Visualization'
toc: true
toc_label:  'contents'
---

Visualizating data with Matplotlib, Seaborn, and custom functions.


```python
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, integrate
%matplotlib inline
```

## Create dataframe


```python
m = 1000    # Number of samples (training examples)
uniform_dist = np.arange(0, m)
```
