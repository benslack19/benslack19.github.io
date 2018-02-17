--- 
title: 'Python and Jupyter setup'
---

These notes describe some basic navigation around Python in the Jupyter environment.


```python
# Identify pandas version
import pandas as pd
pd.__version__
```




    '0.20.1'




```python
# How to install pip packages for Jupyter notebook
# Super helpful link: https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/
import sys
!{sys.executable} -m pip install ggplot
```


```python
# Get help on a function. For example:
pd.DataFrame.sort_values?
```


```python
# Select directory
import os
path = '/Users/lacar/benslack19.github.io/_mynotes'
os.chdir(path) 
cwd = os.getcwd()
print(cwd)
```

    /Users/lacar/benslack19.github.io/_mynotes



```python
## Seeing and revealing variables
%who
```

    NamespaceMagics	 cwd	 df	 df2	 df3	 get_ipython	 getsizeof	 json	 os	 
    path	 pd	 var_dic_list	 xSSF	 



```python
# Deleting variables

# You can delete individual names with del:
del x

# Or you can remove them from the globals() object:
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
```


```python
# Measuring time of execution with iPython magic

# There's a distinction between running in line mode versus cell mode (%time vs. %%time).
# For the latter to run, it has to be the very first line of the cell.

%%time
x = 5
y = 6
x + y
```

    Wall time: 0 ns


Stopping a busy kernel

[StackOverflow entry: 'How should I stop a busy cell in an iPython notebook?'](https://stackoverflow.com/questions/36205356/how-should-i-stop-a-busy-cell-in-an-ipython-notebook)

"Click on 'Interrupt' under 'Kernel' in the toolbar. Pressing I twice will also do the trick."
