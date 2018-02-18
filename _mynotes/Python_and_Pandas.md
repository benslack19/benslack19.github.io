--- 
title: 'Python and Pandas'
toc: true
toc_label:  'contents'
---

- Navigating Python in the Jupyter environment.
- Dataframe creation, indexing, and manipulation.

## Python and Jupyter

### Identify pandas version
<a href="#top">^</a>

```python
import pandas as pd
pd.__version__
```




    '0.21.0'



### Install pip packages
<a href="#top">^</a>

```python
# Super helpful link: https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/

import sys
!{sys.executable} -m pip install <package>
```

### Get help on a function
<a href="#top">^</a>

```python
# Example
pd.DataFrame.sort_values?
```


```python
## Change directory to defined path.
```


```python
import os
path = '/Users/lacar/benslack19.github.io/_mynotes'
os.chdir(path)   
os.getcwd()   # Get current working directory
```




    '/Users/lacar/benslack19.github.io/_mynotes'



### Identify variables
<a href="#top">^</a>

```python
%who
```

    NamespaceMagics	 get_ipython	 getsizeof	 json	 os	 path	 pd	 staff_df	 student_df	 
    sys	 var_dic_list	 x	 y	 



```python
# Deleting individual variables
```


```python
x = 5
del x
```

### Measure execution time
<a href="#top">^</a>

```python
%%time
# Line mode is with %time. Cell mode is run with %%time on the first line of the cell.
x = 5
y = 6
x + y
```

    CPU times: user 5 µs, sys: 0 ns, total: 5 µs
    Wall time: 29.1 µs


## Pandas

### Create dataframe
<a href="#top">^</a>

```python
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
student_df = student_df.set_index('Name')
student_df
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
      <th>School</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>James</th>
      <td>Business</td>
    </tr>
    <tr>
      <th>Mike</th>
      <td>Law</td>
    </tr>
    <tr>
      <th>Sally</th>
      <td>Engineering</td>
    </tr>
  </tbody>
</table>
</div>



### Create dataframe through column additions
<a href="#top">^</a>

```python
staff_df = pd.DataFrame()
staff_df['Name'] = ['Kelly', 'Sally', 'James']
staff_df['Role'] = ['Director of HR', 'Course liasion', 'Grader']
staff_df = staff_df.set_index('Name')
staff_df
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
      <th>Role</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Kelly</th>
      <td>Director of HR</td>
    </tr>
    <tr>
      <th>Sally</th>
      <td>Course liasion</td>
    </tr>
    <tr>
      <th>James</th>
      <td>Grader</td>
    </tr>
  </tbody>
</table>
</div>


