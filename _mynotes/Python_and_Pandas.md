--- 
title: 'Python and Pandas'
toc: true
toc_label:  'contents'
---

- Navigating Python in the Jupyter environment.
- Dataframe creation, indexing, and manipulation.

## Python and Jupyter

### Identify pandas version


```python
import pandas as pd
pd.__version__
```




    '0.21.0'



### Install pip packages


```python
# Super helpful link: https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/
import sys
!{sys.executable} -m pip install ggplot
```

    Requirement already satisfied (use --upgrade to upgrade): ggplot in /Users/lacar/anaconda/lib/python3.5/site-packages
    Requirement already satisfied (use --upgrade to upgrade): numpy in /Users/lacar/anaconda/lib/python3.5/site-packages (from ggplot)
    Requirement already satisfied (use --upgrade to upgrade): matplotlib in /Users/lacar/anaconda/lib/python3.5/site-packages (from ggplot)
    Requirement already satisfied (use --upgrade to upgrade): scipy in /Users/lacar/anaconda/lib/python3.5/site-packages (from ggplot)
    Requirement already satisfied (use --upgrade to upgrade): patsy>=0.4 in /Users/lacar/anaconda/lib/python3.5/site-packages (from ggplot)
    Requirement already satisfied (use --upgrade to upgrade): cycler in /Users/lacar/anaconda/lib/python3.5/site-packages (from ggplot)
    Requirement already satisfied (use --upgrade to upgrade): pandas in /Users/lacar/anaconda/lib/python3.5/site-packages (from ggplot)
    Requirement already satisfied (use --upgrade to upgrade): six in /Users/lacar/anaconda/lib/python3.5/site-packages (from ggplot)
    Requirement already satisfied (use --upgrade to upgrade): brewer2mpl in /Users/lacar/anaconda/lib/python3.5/site-packages (from ggplot)
    Requirement already satisfied (use --upgrade to upgrade): statsmodels in /Users/lacar/anaconda/lib/python3.5/site-packages (from ggplot)
    Requirement already satisfied (use --upgrade to upgrade): python-dateutil in /Users/lacar/anaconda/lib/python3.5/site-packages (from matplotlib->ggplot)
    Requirement already satisfied (use --upgrade to upgrade): pytz in /Users/lacar/anaconda/lib/python3.5/site-packages (from matplotlib->ggplot)
    Requirement already satisfied (use --upgrade to upgrade): pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=1.5.6 in /Users/lacar/anaconda/lib/python3.5/site-packages (from matplotlib->ggplot)
    [33mYou are using pip version 8.1.1, however version 9.0.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m


### Get help on a function


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



### Identify variables with iPython magic


```python
%who
```

    NamespaceMagics	 get_ipython	 getsizeof	 json	 os	 path	 pd	 sys	 var_dic_list	 
    x	 y	 



```python
# Deleting individual variables
```


```python
x = 5
del x
```

### Measure execution time


```python
%%time
# Line mode is with %time. Cell mode is run with %%time on the first line of the cell.
x = 5
y = 6
x + y
```

    CPU times: user 4 µs, sys: 0 ns, total: 4 µs
    Wall time: 10 µs


## Pandas

### Create dataframe


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


