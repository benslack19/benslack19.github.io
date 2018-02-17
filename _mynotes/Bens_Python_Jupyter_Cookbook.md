--- 
title: 'Ben's Python Jupyter Cookbook'
---
Stuff that I forget.

# Python and Jupyter setup

## Finding version


```python
import pandas as pd
pd.__version__
```




    '0.20.1'



## Getting help on a function


```python
# Example
pd.DataFrame.sort_values?
```

## Choosing the directory


```python
import os

path = 'C:\\Users\\Benjamin.Lacar\\Documents\\Box\\Box Sync\\BL (Benjamin.Lacar@fluidigm.com)\\Python\\Coursera_dataScience\\ZillowHomeValues'
os.chdir(path) 
cwd = os.getcwd()
print(cwd)
```

    C:\Users\Benjamin.Lacar\Documents\Box\Box Sync\BL (Benjamin.Lacar@fluidigm.com)\Python\Coursera_dataScience\ZillowHomeValues


## Seeing and revealing variables


```python
# Check for variables
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

## Difference between matplotlib.inline vs. matplotlib.notebook


```python
% matplotlib inline
% matplotlib notebook
# the former is the older version creates new plotes, the latter is interactive
```

## Stopping a busy kernel

[StackOverflow entry: 'How should I stop a busy cell in an iPython notebook?'](https://stackoverflow.com/questions/36205356/how-should-i-stop-a-busy-cell-in-an-ipython-notebook)

"Click on 'Interrupt' under 'Kernel' in the toolbar. Pressing I twice will also do the trick."

## Measuring time of execution with iPython magic

- There's a distinction between running in line mode versus cell mode (%time vs. %%time). For the latter to run, it has to be the very first line of the cell.


```python
%%time
x = 5
y = 6
x + y
```

    Wall time: 0 ns


# Pandas dataframe navigation and munging

## Data frame indexing and navigation


```python
%time 
# Creating a datframe from scratch
staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}])
staff_df = staff_df.set_index('Name')
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
student_df = student_df.set_index('Name')
```

    Wall time: 0 ns



```python
# Reading a csv file into a dataframe (can also be pd.read_excel() or pd.read_table() as appropriate )
df = pd.read_csv('City_Zhvi_AllHomes.csv')
```


```python
# Viewing the first 5 rows of a dataframe
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionID</th>
      <th>RegionName</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>1996-07</th>
      <th>...</th>
      <th>2016-10</th>
      <th>2016-11</th>
      <th>2016-12</th>
      <th>2017-01</th>
      <th>2017-02</th>
      <th>2017-03</th>
      <th>2017-04</th>
      <th>2017-05</th>
      <th>2017-06</th>
      <th>2017-07</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6181</td>
      <td>New York</td>
      <td>NY</td>
      <td>New York</td>
      <td>Queens</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>616100</td>
      <td>622100</td>
      <td>626700.0</td>
      <td>630300</td>
      <td>636800</td>
      <td>646200</td>
      <td>657400</td>
      <td>670800</td>
      <td>681000</td>
      <td>686400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12447</td>
      <td>Los Angeles</td>
      <td>CA</td>
      <td>Los Angeles-Long Beach-Anaheim</td>
      <td>Los Angeles</td>
      <td>2</td>
      <td>155000.0</td>
      <td>154600.0</td>
      <td>154400.0</td>
      <td>154200.0</td>
      <td>...</td>
      <td>598300</td>
      <td>604900</td>
      <td>609700.0</td>
      <td>612400</td>
      <td>616400</td>
      <td>621800</td>
      <td>626000</td>
      <td>628900</td>
      <td>630900</td>
      <td>632000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17426</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>Chicago</td>
      <td>Cook</td>
      <td>3</td>
      <td>109700.0</td>
      <td>109400.0</td>
      <td>109300.0</td>
      <td>109300.0</td>
      <td>...</td>
      <td>210900</td>
      <td>212800</td>
      <td>215300.0</td>
      <td>218200</td>
      <td>220400</td>
      <td>221100</td>
      <td>221800</td>
      <td>222400</td>
      <td>222900</td>
      <td>223400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13271</td>
      <td>Philadelphia</td>
      <td>PA</td>
      <td>Philadelphia</td>
      <td>Philadelphia</td>
      <td>4</td>
      <td>50000.0</td>
      <td>49900.0</td>
      <td>49600.0</td>
      <td>49400.0</td>
      <td>...</td>
      <td>132100</td>
      <td>132500</td>
      <td>133500.0</td>
      <td>134700</td>
      <td>135800</td>
      <td>136500</td>
      <td>136900</td>
      <td>137700</td>
      <td>138500</td>
      <td>138900</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40326</td>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>Phoenix</td>
      <td>Maricopa</td>
      <td>5</td>
      <td>87200.0</td>
      <td>87700.0</td>
      <td>88200.0</td>
      <td>88400.0</td>
      <td>...</td>
      <td>201200</td>
      <td>203200</td>
      <td>205100.0</td>
      <td>206600</td>
      <td>207900</td>
      <td>209100</td>
      <td>210000</td>
      <td>211800</td>
      <td>214100</td>
      <td>215800</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 262 columns</p>
</div>




```python
# Viewing the last 5 rows of a dataframe
df.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionID</th>
      <th>RegionName</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>1996-07</th>
      <th>...</th>
      <th>2016-10</th>
      <th>2016-11</th>
      <th>2016-12</th>
      <th>2017-01</th>
      <th>2017-02</th>
      <th>2017-03</th>
      <th>2017-04</th>
      <th>2017-05</th>
      <th>2017-06</th>
      <th>2017-07</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11251</th>
      <td>398292</td>
      <td>Town of Wrightstown</td>
      <td>WI</td>
      <td>Green Bay</td>
      <td>Brown</td>
      <td>11252</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>167500</td>
      <td>169500</td>
      <td>171400.0</td>
      <td>173600</td>
      <td>176100</td>
      <td>177300</td>
      <td>177100</td>
      <td>177800</td>
      <td>178800</td>
      <td>179600</td>
    </tr>
    <tr>
      <th>11252</th>
      <td>398343</td>
      <td>Urbana</td>
      <td>NY</td>
      <td>Corning</td>
      <td>Steuben</td>
      <td>11253</td>
      <td>66900.0</td>
      <td>65800.0</td>
      <td>65500.0</td>
      <td>65100.0</td>
      <td>...</td>
      <td>152700</td>
      <td>154100</td>
      <td>153300.0</td>
      <td>155100</td>
      <td>156500</td>
      <td>153100</td>
      <td>148100</td>
      <td>146600</td>
      <td>144200</td>
      <td>140900</td>
    </tr>
    <tr>
      <th>11253</th>
      <td>398496</td>
      <td>New Denmark</td>
      <td>WI</td>
      <td>Green Bay</td>
      <td>Brown</td>
      <td>11254</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>200400</td>
      <td>202400</td>
      <td>204900.0</td>
      <td>207100</td>
      <td>209200</td>
      <td>211000</td>
      <td>212400</td>
      <td>213700</td>
      <td>215300</td>
      <td>216600</td>
    </tr>
    <tr>
      <th>11254</th>
      <td>398839</td>
      <td>Angels</td>
      <td>CA</td>
      <td>NaN</td>
      <td>Calaveras</td>
      <td>11255</td>
      <td>115600.0</td>
      <td>116400.0</td>
      <td>118000.0</td>
      <td>119000.0</td>
      <td>...</td>
      <td>269100</td>
      <td>273100</td>
      <td>275500.0</td>
      <td>276700</td>
      <td>278100</td>
      <td>280900</td>
      <td>285200</td>
      <td>287500</td>
      <td>287100</td>
      <td>286200</td>
    </tr>
    <tr>
      <th>11255</th>
      <td>737788</td>
      <td>Lebanon Borough</td>
      <td>NJ</td>
      <td>New York</td>
      <td>Hunterdon</td>
      <td>11256</td>
      <td>143500.0</td>
      <td>143200.0</td>
      <td>141700.0</td>
      <td>140700.0</td>
      <td>...</td>
      <td>239900</td>
      <td>238800</td>
      <td>239700.0</td>
      <td>241700</td>
      <td>241200</td>
      <td>238600</td>
      <td>235100</td>
      <td>232500</td>
      <td>235900</td>
      <td>241400</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 262 columns</p>
</div>




```python
# Get a list of columns
df.columns
```




    Index(['RegionID', 'RegionName', 'State', 'Metro', 'CountyName', 'SizeRank',
           '1996-04', '1996-05', '1996-06', '1996-07',
           ...
           '2016-10', '2016-11', '2016-12', '2017-01', '2017-02', '2017-03',
           '2017-04', '2017-05', '2017-06', '2017-07'],
          dtype='object', length=262)




```python
# Accessing a dataframe column by number and creating a new dataframe
df2 = df.iloc[:,:6]
```

## Data frame sorting and filtering


```python
# Sorting dataframe by column(s)
df.sort_values(by=['State', 'RegionName'], ascending=True, inplace=True)
```


```python
# Filtering the dataframe by an element in a column and creating a new dataframe
df3 = df[df['Metro']=='San Francisco']
```


```python
# Accessing a dataframe by row index name and column index name(s)and creating a new series
xSSF = df3.loc[727, '1996-04':]
```


```python
# see if a pandas column contains a string
dfMetTP2['SampleID'].str.contains('NA12878')
```


```python
### May have to re-do the following
```


```python
# Accessing the date (re-do)
x['date'] = pd.to_datetime(x.index)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-21-1dafde5cf7fc> in <module>()
          1 # Accessing the date in a
    ----> 2 x['date'] = pd.to_datetime(x.index)
    

    NameError: name 'x' is not defined



```python
x2 = x[:-1]
x2SSF = xSSF[:-1]
```

# Advanced Pandas


```python
# cool pivot table example

df = pd.DataFrame({'Account_number':[1,1,2,2,2,3,3], 'Product':['A', 'A', 'A', 'B', 'B','A', 'B']})

df.pivot_table(index='Account_number', columns='Product', aggfunc=len, fill_value=0)
```

## Merging


```python
pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)
pd.merge(staff_df, student_df, how='inner', left_index=True, right_index=True)
# left regardless if they're in the overlap
pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)
```

## Method chaining


```python

(df.where(df['SUMLEV']==50)
    .dropna()
    .set_index(['STNAME','CTYNAME'])
    .rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'}))
```

## Lambda


```python
df.apply(lambda x: np.sum(x[rows]), axis=1)
```

## Vector functions

other:
df.iterrows
.iterritems
zip
enumerate

## Apply function


```python
import numpy as np
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min': np.min(data), 'max': np.max(data)})

df.apply(min_max, axis=1)
```

## Group by

Common workflow for groupby: split data, apply function, then combine results (split, apply, combine function).
Groupby object has agg method (aggregate). This method applies a function to the column or columns of data in the group, and returns the results.


```python
# Need to update table
import pandas as pd
import numpy as np
df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]
df
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-28-9096edd9935d> in <module>()
          1 import pandas as pd
          2 import numpy as np
    ----> 3 df = pd.read_csv('census.csv')
          4 df = df[df['SUMLEV']==50]
          5 df


    C:\Users\Benjamin.Lacar\AppData\Local\Continuum\Anaconda3_v440\lib\site-packages\pandas\io\parsers.py in parser_f(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, escapechar, comment, encoding, dialect, tupleize_cols, error_bad_lines, warn_bad_lines, skipfooter, skip_footer, doublequote, delim_whitespace, as_recarray, compact_ints, use_unsigned, low_memory, buffer_lines, memory_map, float_precision)
        653                     skip_blank_lines=skip_blank_lines)
        654 
    --> 655         return _read(filepath_or_buffer, kwds)
        656 
        657     parser_f.__name__ = name


    C:\Users\Benjamin.Lacar\AppData\Local\Continuum\Anaconda3_v440\lib\site-packages\pandas\io\parsers.py in _read(filepath_or_buffer, kwds)
        403 
        404     # Create the parser.
    --> 405     parser = TextFileReader(filepath_or_buffer, **kwds)
        406 
        407     if chunksize or iterator:


    C:\Users\Benjamin.Lacar\AppData\Local\Continuum\Anaconda3_v440\lib\site-packages\pandas\io\parsers.py in __init__(self, f, engine, **kwds)
        760             self.options['has_index_names'] = kwds['has_index_names']
        761 
    --> 762         self._make_engine(self.engine)
        763 
        764     def close(self):


    C:\Users\Benjamin.Lacar\AppData\Local\Continuum\Anaconda3_v440\lib\site-packages\pandas\io\parsers.py in _make_engine(self, engine)
        964     def _make_engine(self, engine='c'):
        965         if engine == 'c':
    --> 966             self._engine = CParserWrapper(self.f, **self.options)
        967         else:
        968             if engine == 'python':


    C:\Users\Benjamin.Lacar\AppData\Local\Continuum\Anaconda3_v440\lib\site-packages\pandas\io\parsers.py in __init__(self, src, **kwds)
       1580         kwds['allow_leading_cols'] = self.index_col is not False
       1581 
    -> 1582         self._reader = parsers.TextReader(src, **kwds)
       1583 
       1584         # XXX


    pandas\_libs\parsers.pyx in pandas._libs.parsers.TextReader.__cinit__ (pandas\_libs\parsers.c:4209)()


    pandas\_libs\parsers.pyx in pandas._libs.parsers.TextReader._setup_parser_source (pandas\_libs\parsers.c:8873)()


    FileNotFoundError: File b'census.csv' does not exist



```python
# Example to get multiple values in a groupby
(df.set_index('STNAME').groupby(level=0)['CENSUS2010POP']
    .agg({'avg': np.average, 'sum': np.sum}))
```

## Link for  tips on Pandas data manipulation
This [article](https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/) covers the following:
- Boolean indexing
- apply
- imputing missing files
- pivot table
- multi-indexing
- crosstab
- merging data frames
- sorting
- plotting
- cut function for binning
- coding nominal data
- iterating over rows

# Numerical series creations with numpy


```python
import numpy as np
ind = np.arange(1, 5)
```


```python
ind
```




    array([1, 2, 3, 4])



# Plot visualization in Python

## Plotting with matplotlib

You can show matplotlib figures directly in the notebook by using the %matplotlib notebook and %matplotlib inline magic commands.
%matplotlib notebook provides an interactive environment.



```python
%matplotlib notebook
# know how to get the backend and be aware of the backend layer

# Configures matplotlib to work in the browswer. It's a backend that's operating with the browser.
import matplotlib as mpl
mpl.get_backend()

# make sure we're using the nbagg backend
# access get/set layers
```

### Basic template and parts of a figure
http://matplotlib.org/faq/usage_faq.html#parts-of-a-figure


```python
import matplotlib.pyplot as plt
%matplotlib inline
y=x['date']
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(y, x2)
#ax1.title('Chula Vista home values')
ax2.plot(y, xSSF)
```




    [<matplotlib.lines.Line2D at 0x2406bac5b70>]




![png](Bens_Python_Jupyter_Cookbook_55_1.png)



```python
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 10, 0.2)
y = np.sin(x)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
plt.show()
```


![png](Bens_Python_Jupyter_Cookbook_56_0.png)


### Making a discontinuous axis
- [StackOverflow entry: "Python/Matplotlib - Is there a way to make a discontinuous axis?"](https://stackoverflow.com/questions/5656798/python-matplotlib-is-there-a-way-to-make-a-discontinuous-axis)
- [Matplotlib documentation](https://matplotlib.org/examples/pylab_examples/broken_axis.html)

### Iterating subplot axis array
- [StackOverflow entry: "matplotlib iterate subplot axis array through single list"](https://stackoverflow.com/questions/20288842/matplotlib-iterate-subplot-axis-array-through-single-list)


### Different size subplots with gridspec

[StackOverflow entry: "Matplotlib different size subplots"](https://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots)

### Adding a custom line to a plot


```python
# plot the  cutoff line
ax = plt.plot([0, 8], [0.90, 0.90], linewidth=1, linestyle='dashed', color='black', alpha=0.5)
```

## Plotting with Seaborn

### Seaborn factor plots


```python
sns_plot = sns.factorplot(x='IFC_no', y='PCT_READS_MAPPED_TO_GENOME', data=dfMetTP3, order=xorder, kind="box", color='#d3d3d3')   # hex value for light gray
```


```python

```


```python
xorder = sorted(dfMetTP3['IFC_no'].unique().tolist())
sns_plot = sns.factorplot(x='IFC_no', y='PCT_READS_MAPPED_TO_GENOME', data=dfMetTP3, order=xorder, kind="box", color='#d3d3d3')   # hex value for light gray
sns_plot = sns.swarmplot(x='IFC_no', y='PCT_READS_MAPPED_TO_GENOME', data=dfMetTP3, order=xorder, alpha=1, color=F_violet, size=4) #, palette=[F_teal, F_magenta, F_purple])

# plot the  cutoff line
ax = plt.plot([0, 8], [0.90, 0.90], linewidth=1, linestyle='dashed', color='black', alpha=0.5)
sns_plot.set_xticklabels(labels = xorder, rotation=45)
plt.ylim(0.87, 1)   # matplotlib function

vals = sns_plot.axes.get_yticks()
sns_plot.set_yticklabels(['{:2.0f}%'.format(x*100) for x in vals])
plt.ylabel('')
plt.xlabel('')
# plt.ylabel('% Reads Mapped to Genome')
plt.title('% Reads Mapped to Genome (NA12878)')

ax = plt.gca()
ax.text(4.1, 0.894, 'spec', fontsize=10, color='black', ha='center', va='bottom')

plt.subplots_adjust(bottom=0.2, top=0.91, left=0.12, right=0.97)
sns.set(font='Proxima Nova')
sns.set_style(style='white')

# for tick in ax.get_xticklabels():
#     tick.set_rotation(45)

# save the figure
plt.savefig('pctReadsMappedToGenome_boxplot_byIFC_wScatter.png', dpi=400)
```


```python
# note the workaround when x is omitted

sns_plot = sns.swarmplot(x=[""]*len(dfMetTP3), y='PCT_READS_MAPPED_TO_GENOME', hue='IFC_no', data=dfMetTP3, alpha=1, size=4, palette=F_colors) #, palette=[F_teal, F_magenta, F_purple])


```


```python
# plt.figure()

# XaxisData = 'TOTAL_READ'
# YaxisData ='PCT_READS_MAPPED_TO_GENOME'
# ax = dfMetTP2.plot.scatter(x=XaxisData, y=YaxisData, color=F_purple)
# fontTitle = {'fontname' : 'Montserrat'}
# font1 = {'fontname':'Proxima Nova'}
# plt.title(XaxisData.replace('_', ' ') + ' vs. ' + YaxisData.replace('_', ' '), **fontTitle)
# plt.xlabel(XaxisData.replace('_', ' '), **font1)
# plt.ylabel(YaxisData.replace('_', ' '), **font1)

# fig = ax.get_figure()
# fig.savefig(XaxisData + '_vs_' + YaxisData + '.png')

# may need to edit for now
# Plotting scatter plot with colors
# plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
# colors = [F_purple, F_magenta, F_teal] # list( F_colors[i] for i in [0, 2, 4] ) # F_colors # pd.tools.plotting._get_standard_colors(len(groups), color_type='random')

# fig, ax = plt.subplots()
# # ax.set_color_cycle(colors)
# ax.set_prop_cycle(c=colors)
# ax.margins(0.05)
# for name, group in IFCgroup:
#     ax.plot(group.TOTAL_READ, group.PCT_READS_MAPPED_TO_GENOME, marker='o', linestyle='', ms=6, label=name)
# ax.legend(numpoints=1, loc='lower right')

# XaxisData = 'TOTAL_READ'
# YaxisData ='PCT_READS_MAPPED_TO_GENOME'

# fontTitle = {'fontname' : 'Montserrat'}
# font1 = {'fontname':'Proxima Nova'}
# plt.title(XaxisData.replace('_', ' ') + ' vs. ' + YaxisData.replace('_', ' '), **fontTitle)
# plt.xlabel(XaxisData.replace('_', ' '), **font1)
# plt.ylabel(YaxisData.replace('_', ' '), **font1)

# plt.show()
```


```python
### making one figure with different subplots, each with different properties from each other

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(212)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(12,4))   #  sharey, sharex determines control of sharing axes properties

#####   properties for first figure   ##########
xorder = sorted(dfMetTP3['IFCtype, no'].unique().tolist())
sns.boxplot(x='IFCtype, no', y='PCT_READS_MAPPED_TO_GENOME', data=dfMetTP3,
                          order=xorder, color='#d3d3d3', ax=ax1)   # hex value for light gray
sns.swarmplot(x='IFCtype, no', y='PCT_READS_MAPPED_TO_GENOME', data=dfMetTP3,
                         order=xorder, alpha=1, color=F_violet, size=3, ax=ax1) #, palette=[F_teal, F_magenta, F_purple])

# plot the cutoff line
ax1.plot([0, 18], [0.90, 0.90], linewidth=1, linestyle='dashed', color='black', alpha=0.5)  # works with ax1
ax1.set_xticklabels(labels = xorder, rotation=90)   # works with ax1
ax1.set_ylim(0.87, 1)  # works with ax1

# plt.ylim(0.87, 1)   # matplotlib function, doesn't work with ax1

vals = ax1.axes.get_yticks()
ax1.set_yticklabels(['{:2.0f}%'.format(x*100) for x in vals])   # works with ax1
ax1.set_ylabel('')  # works with ax1
#plt.ylabel('')  # matplotlib function, doesn't work with ax1
ax1.set_xlabel('')  # works with ax1
ax1.set_title('% Reads Mapped to Genome (NA12878)')  # works with ax1

# ax = plt.gca()  # this line not needed
xvals = ax1.axes.get_xticks()
ax1.text(len(xvals)/2-0.5, 0.892, 'spec', fontsize=10, color='black', ha='center', va='bottom')

sns.boxplot(x='IFCtype, no', y='TOTAL_READ', data=dfMetTP3,
                          order=xorder, color='#d3d3d3', ax=ax2)   # hex value for light gray
sns.swarmplot(x='IFCtype, no', y='TOTAL_READ', data=dfMetTP3,
                         order=xorder, alpha=1, color=F_violet, size=3, ax=ax2) #, palette=[F_teal, F_magenta, F_purple])

ax2.set_xticklabels(labels = xorder, rotation=90)   # works with ax2
# ax2.set_ylim(0.87, 1)  # works with ax1
ax2.set_title('Total Reads (NA12878)')  # works with ax2

# plt.subplots_adjust(bottom=0.2, top=0.91, left=0.12, right=0.97)
sns.set(font='Proxima Nova')
sns.set_style(style='white')


```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, integrate

% matplotlib inline
```


```python
# set the right directory
import os
path = 'C:\\Users\\Benjamin.Lacar\\Desktop'
os.chdir(path) 
cwd = os.getcwd()
print(cwd)
```

    C:\Users\Benjamin.Lacar\Desktop



```python
y=np.linspace(0,100,100)
```


```python
y
```




    array([   0.        ,    1.01010101,    2.02020202,    3.03030303,
              4.04040404,    5.05050505,    6.06060606,    7.07070707,
              8.08080808,    9.09090909,   10.1010101 ,   11.11111111,
             12.12121212,   13.13131313,   14.14141414,   15.15151515,
             16.16161616,   17.17171717,   18.18181818,   19.19191919,
             20.2020202 ,   21.21212121,   22.22222222,   23.23232323,
             24.24242424,   25.25252525,   26.26262626,   27.27272727,
             28.28282828,   29.29292929,   30.3030303 ,   31.31313131,
             32.32323232,   33.33333333,   34.34343434,   35.35353535,
             36.36363636,   37.37373737,   38.38383838,   39.39393939,
             40.4040404 ,   41.41414141,   42.42424242,   43.43434343,
             44.44444444,   45.45454545,   46.46464646,   47.47474747,
             48.48484848,   49.49494949,   50.50505051,   51.51515152,
             52.52525253,   53.53535354,   54.54545455,   55.55555556,
             56.56565657,   57.57575758,   58.58585859,   59.5959596 ,
             60.60606061,   61.61616162,   62.62626263,   63.63636364,
             64.64646465,   65.65656566,   66.66666667,   67.67676768,
             68.68686869,   69.6969697 ,   70.70707071,   71.71717172,
             72.72727273,   73.73737374,   74.74747475,   75.75757576,
             76.76767677,   77.77777778,   78.78787879,   79.7979798 ,
             80.80808081,   81.81818182,   82.82828283,   83.83838384,
             84.84848485,   85.85858586,   86.86868687,   87.87878788,
             88.88888889,   89.8989899 ,   90.90909091,   91.91919192,
             92.92929293,   93.93939394,   94.94949495,   95.95959596,
             96.96969697,   97.97979798,   98.98989899,  100.        ])




```python
x = np.random.uniform(40,1000,100)

```


```python
# generate a set of numbers that can replicate a hypothetical bioanalyzer data

#k1 = np.random.normal(loc=35.0, scale=5, size=35)   # marker 1
k2 = np.random.normal(loc=50.0, scale=10, size=0)   # primers
k3 = np.random.normal(loc=100.0, scale=10, size=0)    # primer dimers
k4 = np.random.normal(loc=300.0, scale=15, size=70)    # desired peak 1
k5 = np.random.normal(loc=325.0, scale=15, size=80)    # desired peak 2
k6 = np.random.normal(loc=600.0, scale=200, size=10)    # genomic
k7 = np.random.normal(loc=800.0, scale=200, size=10)    # genomic
k8 = np.random.normal(loc=1000.0, scale=100, size=10)    # genomic
#k9 = np.random.normal(loc=10380, scale=5, size=50)   # marker 2
kC = np.concatenate([k2, k3, k4, k5, k6, k7, k8])
# plot as a histogram
# x = np.random.normal(size=100)
ax = sns.distplot(kC, hist=True, kde=False, bins=100, hist_kws={"alpha": 0.25, "color":F_purple});
sns.set(font='Proxima Nova')
sns.set_style(style='white', )

#ax.set_xlabel('No. of bases')

# save figure
plt.savefig('Plot5_new.png', dpi=400)
```


![png](Bens_Python_Jupyter_Cookbook_75_0.png)



```python
scipy.stats.skew(k2, )
```

### setting axis properties

see ax line down below
https://stackoverflow.com/questions/31632637/label-axes-on-seaborn-barplot



```python
ax = sns.factorplot(y='PCT_READS_MAPPED_TO_AMPLICONS_FROM_ALIGNED_READS', data=dfMetTP2, kind="box", palette=['gray'])
ax = sns.swarmplot(x=[""]*len(dfMetTP2), y='PCT_READS_MAPPED_TO_AMPLICONS_FROM_ALIGNED_READS', data=dfMetTP2, alpha=1, size=2,
              hue='IFC_no', palette=[F_teal, F_magenta, F_purple])

# plot our cutoff line
plt.plot([-.5, 0.5], [0.95, 0.95], linewidth=1, linestyle='dashed', color='black', alpha=0.5)

# for changing axes Seaborn's barplot returns an axis-object (not a figure). This means you can do the following: 
ax.set(title='% Reads Mapped to Target from Aligned Reads', ylabel='common y-label')  
plt.ylim(0.94, 1)   # matplotlib function


sns.set(font='Proxima Nova')
sns.set_style(style='white')
```


```python
# saving a figure
plt.savefig('pctReadsMappedToGenome_boxplot_byIFC_noScatter.png', dpi=400)
```


```python
# prevent saving from cutting off figure

figure.autolayout : True
    
    
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
```

### quickly making visuals
https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/


```python
import matplotlib.pyplot as plt
%matplotlib inline
# quick boxplot check
data.boxplot(column="ApplicantIncome",by="Loan_Status")
```


```python
# quick histogram check
dfMetTP2_b.hist(column='PCT_READS_MAPPED_TO_GENOME', by='IFC_no')
```


```python

```

# Statistics

### Making simulated histograms


```python
# generate a set of numbers that can replicate a hypothetical bioanalyzer data

#k1 = np.random.normal(loc=35.0, scale=5, size=35)   # marker 1
k2 = np.random.normal(loc=25.0, scale=5, size=800)   # primers
k3 = np.random.normal(loc=75.0, scale=10, size=300)    # primer dimers
k4 = np.random.normal(loc=150.0, scale=30, size=40)    # desired peak 1
k5 = np.random.normal(loc=300.0, scale=40, size=30)    # desired peak 2
k6 = np.random.normal(loc=400.0, scale=40, size=40)    # genomic
k7 = np.random.normal(loc=500.0, scale=40, size=50)    # genomic
k8 = np.random.normal(loc=600.0, scale=40, size=40)    # genomic
k8 = np.random.normal(loc=700.0, scale=40, size=40)    # genomic
k8 = np.random.normal(loc=800.0, scale=40, size=40)    # genomic
k8 = np.random.normal(loc=900.0, scale=40, size=40)    # genomic
k9 = np.random.normal(loc=1000, scale=40, size=30)   
k10 = np.random.normal(loc=1100, scale=40, size=30)   
k11 = np.random.normal(loc=1200, scale=40, size=30)   
k12 = np.random.normal(loc=1400, scale=40, size=30)   
kC = np.concatenate([k2, k3, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12])
# plot as a histogram
# x = np.random.normal(size=100)
ax = sns.distplot(kC, hist=True, kde=False, bins=100, hist_kws={"alpha": 0.25, "color":F_purple});
sns.set(font='Proxima Nova')
sns.set_style(style='white')

#ax.set_xlabel('No. of bases')
#plt.savefig('updatedPrimerDimer_bioA.png', dpi=400)
```


![png](Bens_Python_Jupyter_Cookbook_87_0.png)



```python
import scipy.stats as stats

statsTest = stats.f_oneway
df = dfMetTP2.copy()
colMetric = 'PCT_READS_MAPPED_TO_GENOME'
grpCol = 'IFC_no'
dfForStats = df.pivot(columns=grpCol, values=colMetric)
 
stats.f_oneway(dfForStats.iloc[:,0].dropna(), dfForStats.iloc[:,1].dropna(), dfForStats.iloc[:,2].dropna())
```


```python

```

### Python Coursera


```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure()

languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

plt.bar(pos, popularity, align='center')
plt.xticks(pos, languages)
plt.ylabel('% Popularity')
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

#TODO: remove all the ticks (both axes), and tick labels on the Y axis
# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='on', bottom='on', left='off', right='off', labelleft='on', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()

```


![png](Bens_Python_Jupyter_Cookbook_91_0.png)



```python
for spine in plt.gca().spines.values():
    print(spine)
```

    Spine
    Spine
    Spine
    Spine



![png](Bens_Python_Jupyter_Cookbook_92_1.png)



```python
# dejunkified plot

import matplotlib.pyplot as plt
import numpy as np

plt.figure()

languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

# change the bar color to be less bright blue
bars = plt.bar(pos, popularity, align='center', linewidth=0, color='lightslategrey')
# make one bar, the python bar, a contrasting color
bars[0].set_color('#1F77B4')

# soften all labels by turning grey
plt.xticks(pos, languages, alpha=0.8)
# remove the Y label since bars are directly labeled
#plt.ylabel('% Popularity', alpha=0.8)
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
# direct label each bar with Y axis values
for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())) + '%', 
                 ha='center', color='w', fontsize=11)
plt.show()
```


![png](Bens_Python_Jupyter_Cookbook_93_0.png)



```python

```


```python
### importing HTML table into pandas
import pandas as pd

url = 'https://en.wikipedia.org/wiki/List_of_San_Diego_Padres_seasons'
# but this can't find the table
data = pd.read_html(url, header=0)

# remove the footers too
df_winPerc = data[1].head(49)
```


```python
# get the salary info

url = 'http://www.baseballprospectus.com/compensation/?cyear=2017&team=SDN&pos='
# but this can't find the table
data = pd.read_html(url, header=0)

df = data[1].iloc[36:,:]
df.columns = df.iloc[0]
```


```python
df_Payroll = df.drop(36, axis=0).drop(['PR Sort', 'Diff', 'AvgPR Sort', 'Diff Sort'], axis=1).iloc[:, :3]
```


```python
df_Payroll.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>36</th>
      <th>Year</th>
      <th>Padres payroll</th>
      <th>Avg payroll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>1996</td>
      <td>$2,000,000</td>
      <td>$4,583,416</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1997</td>
      <td>$2,000,000</td>
      <td>$4,892,222</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2000</td>
      <td>$53,816,000</td>
      <td>$57,548,235</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2001</td>
      <td>$37,438,000</td>
      <td>$67,152,893</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2002</td>
      <td>$40,678,000</td>
      <td>$69,249,884</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_winPerc.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>Level</th>
      <th>League</th>
      <th>Division</th>
      <th>Finish</th>
      <th>Wins</th>
      <th>Losses</th>
      <th>Win%</th>
      <th>GB</th>
      <th>Postseason</th>
      <th>Awards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1969</td>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>6th</td>
      <td>52.0</td>
      <td>110.0</td>
      <td>0.321</td>
      <td>41</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970</td>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>6th</td>
      <td>63.0</td>
      <td>99.0</td>
      <td>0.389</td>
      <td>39</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1971</td>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>6th</td>
      <td>61.0</td>
      <td>100.0</td>
      <td>0.379</td>
      <td>28½</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1972</td>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>6th</td>
      <td>58.0</td>
      <td>95.0</td>
      <td>0.379</td>
      <td>36½</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1973</td>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>6th</td>
      <td>60.0</td>
      <td>102.0</td>
      <td>0.370</td>
      <td>39</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_all = pd.merge(df_winPerc, df_Payroll, how='inner', left_on='Season', right_on='Year')
```


```python
df_all.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>Level</th>
      <th>League</th>
      <th>Division</th>
      <th>Finish</th>
      <th>Wins</th>
      <th>Losses</th>
      <th>Win%</th>
      <th>GB</th>
      <th>Postseason</th>
      <th>Awards</th>
      <th>Year</th>
      <th>Padres payroll</th>
      <th>Avg payroll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1996</td>
      <td>MLB</td>
      <td>NL</td>
      <td>West ^</td>
      <td>1st</td>
      <td>91.0</td>
      <td>71.0</td>
      <td>0.562</td>
      <td>—</td>
      <td>Lost NLDS (Cardinals) 3–0</td>
      <td>Ken Caminiti (MVP) Bruce Bochy (MOY)</td>
      <td>1996</td>
      <td>$2,000,000</td>
      <td>$4,583,416</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997</td>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>4th</td>
      <td>76.0</td>
      <td>86.0</td>
      <td>0.469</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1997</td>
      <td>$2,000,000</td>
      <td>$4,892,222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>5th</td>
      <td>76.0</td>
      <td>86.0</td>
      <td>0.469</td>
      <td>21</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2000</td>
      <td>$53,816,000</td>
      <td>$57,548,235</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>4th</td>
      <td>79.0</td>
      <td>83.0</td>
      <td>0.488</td>
      <td>13</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2001</td>
      <td>$37,438,000</td>
      <td>$67,152,893</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2002</td>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>5th</td>
      <td>66.0</td>
      <td>96.0</td>
      <td>0.407</td>
      <td>32</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2002</td>
      <td>$40,678,000</td>
      <td>$69,249,884</td>
    </tr>
  </tbody>
</table>
</div>




```python

# convert currencies to float
df_all['Padres payroll'] = df_all['Padres payroll'].str.replace('$', '').str.replace(',','').astype(float)
df_all['Avg payroll'] = df_all['Avg payroll'].str.replace('$', '').str.replace(',','').astype(float)

```


```python
df_all['Padres Payroll % of MLB average'] = df_all['Padres payroll']/df_all['Avg payroll']
```


```python
# start at 2000s to since no data is available for 1998, 1999
df_all = df_all.iloc[2:,:]
```


```python
df_all['Season'] = pd.to_datetime(df_all['Season'])
```

    C:\Users\Benjamin.Lacar\AppData\Local\Continuum\Anaconda3_v440\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.



```python
df_all.set_index('Season', inplace=True)
```


```python
df_all.index
```




    DatetimeIndex(['2000-01-01', '2001-01-01', '2002-01-01', '2003-01-01',
                   '2004-01-01', '2005-01-01', '2006-01-01', '2007-01-01',
                   '2008-01-01', '2009-01-01', '2010-01-01', '2011-01-01',
                   '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01',
                   '2016-01-01'],
                  dtype='datetime64[ns]', name='Season', freq=None)




```python
df_all
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Level</th>
      <th>League</th>
      <th>Division</th>
      <th>Finish</th>
      <th>Wins</th>
      <th>Losses</th>
      <th>Win%</th>
      <th>GB</th>
      <th>Postseason</th>
      <th>Awards</th>
      <th>Year</th>
      <th>Padres payroll</th>
      <th>Avg payroll</th>
      <th>Padres Payroll % of MLB average</th>
    </tr>
    <tr>
      <th>Season</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>5th</td>
      <td>76.0</td>
      <td>86.0</td>
      <td>0.469</td>
      <td>21</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2000</td>
      <td>53816000.0</td>
      <td>57548235.0</td>
      <td>0.935146</td>
    </tr>
    <tr>
      <th>2001-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>4th</td>
      <td>79.0</td>
      <td>83.0</td>
      <td>0.488</td>
      <td>13</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2001</td>
      <td>37438000.0</td>
      <td>67152893.0</td>
      <td>0.557504</td>
    </tr>
    <tr>
      <th>2002-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>5th</td>
      <td>66.0</td>
      <td>96.0</td>
      <td>0.407</td>
      <td>32</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2002</td>
      <td>40678000.0</td>
      <td>69249884.0</td>
      <td>0.587409</td>
    </tr>
    <tr>
      <th>2003-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>5th</td>
      <td>64.0</td>
      <td>98.0</td>
      <td>0.395</td>
      <td>36½</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2003</td>
      <td>43565000.0</td>
      <td>72210211.0</td>
      <td>0.603308</td>
    </tr>
    <tr>
      <th>2004-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>3rd</td>
      <td>87.0</td>
      <td>75.0</td>
      <td>0.537</td>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2004</td>
      <td>59172333.0</td>
      <td>71437964.0</td>
      <td>0.828304</td>
    </tr>
    <tr>
      <th>2005-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West ^</td>
      <td>1st</td>
      <td>82.0</td>
      <td>80.0</td>
      <td>0.506</td>
      <td>—</td>
      <td>Lost NLDS (Cardinals) 3–0</td>
      <td>NaN</td>
      <td>2005</td>
      <td>62186333.0</td>
      <td>73700583.0</td>
      <td>0.843770</td>
    </tr>
    <tr>
      <th>2006-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West ^</td>
      <td>1st</td>
      <td>88.0</td>
      <td>74.0</td>
      <td>0.543</td>
      <td>—</td>
      <td>Lost NLDS (Cardinals) 3–1</td>
      <td>NaN</td>
      <td>2006</td>
      <td>69170167.0</td>
      <td>81320418.0</td>
      <td>0.850588</td>
    </tr>
    <tr>
      <th>2007-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>3rd</td>
      <td>89.0</td>
      <td>74.0</td>
      <td>0.546</td>
      <td>1½</td>
      <td>NaN</td>
      <td>Jake Peavy (CYA)</td>
      <td>2007</td>
      <td>58571067.0</td>
      <td>85813074.0</td>
      <td>0.682542</td>
    </tr>
    <tr>
      <th>2008-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>5th</td>
      <td>63.0</td>
      <td>99.0</td>
      <td>0.389</td>
      <td>21</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2008</td>
      <td>74010117.0</td>
      <td>93345041.0</td>
      <td>0.792866</td>
    </tr>
    <tr>
      <th>2009-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>4th</td>
      <td>75.0</td>
      <td>87.0</td>
      <td>0.463</td>
      <td>20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009</td>
      <td>44173200.0</td>
      <td>93808721.0</td>
      <td>0.470886</td>
    </tr>
    <tr>
      <th>2010-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>2nd</td>
      <td>90.0</td>
      <td>72.0</td>
      <td>0.555</td>
      <td>2</td>
      <td>NaN</td>
      <td>Heath Bell (ROY) Bud Black (MOY)</td>
      <td>2010</td>
      <td>37799300.0</td>
      <td>95698261.0</td>
      <td>0.394984</td>
    </tr>
    <tr>
      <th>2011-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>5th</td>
      <td>71.0</td>
      <td>91.0</td>
      <td>0.435</td>
      <td>23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2011</td>
      <td>45869140.0</td>
      <td>97450552.0</td>
      <td>0.470691</td>
    </tr>
    <tr>
      <th>2012-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>4th</td>
      <td>76.0</td>
      <td>86.0</td>
      <td>0.469</td>
      <td>18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012</td>
      <td>55621900.0</td>
      <td>100756166.0</td>
      <td>0.552045</td>
    </tr>
    <tr>
      <th>2013-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>T-3rd</td>
      <td>76.0</td>
      <td>86.0</td>
      <td>0.469</td>
      <td>16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2013</td>
      <td>68333600.0</td>
      <td>106658387.0</td>
      <td>0.640677</td>
    </tr>
    <tr>
      <th>2014-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>3rd</td>
      <td>77.0</td>
      <td>85.0</td>
      <td>0.475</td>
      <td>17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014</td>
      <td>90636600.0</td>
      <td>115428670.0</td>
      <td>0.785217</td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>4th</td>
      <td>74.0</td>
      <td>88.0</td>
      <td>0.457</td>
      <td>18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2015</td>
      <td>108387033.0</td>
      <td>125458486.0</td>
      <td>0.863927</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>MLB</td>
      <td>NL</td>
      <td>West</td>
      <td>5th</td>
      <td>68.0</td>
      <td>94.0</td>
      <td>0.420</td>
      <td>23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016</td>
      <td>100509500.0</td>
      <td>130290910.0</td>
      <td>0.771424</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
import os
path = 'C:\\Users\\Benjamin.Lacar\\Documents\\Box\\Box Sync\\BL (Benjamin.Lacar@fluidigm.com)\\Python\\Coursera_dataScience\\course2_downloads\\week4'
os.chdir(path) 
cwd = os.getcwd()
print(cwd)
```

    C:\Users\Benjamin.Lacar\Documents\Box\Box Sync\BL (Benjamin.Lacar@fluidigm.com)\Python\Coursera_dataScience\course2_downloads\week4



```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('white')
# use the 'seaborn-colorblind' style
plt.style.use('seaborn-colorblind')

fig, ax = plt.subplots()

ax.plot(df_all['Padres Payroll % of MLB average'], color='blue')
ax2 =ax.twinx()
ax2.plot(df_all['Win%'], color='orange')

# add correlation coefficient
cc = df_all[['Win%', 'Padres Payroll % of MLB average']].corr(method='pearson', min_periods=1).iloc[0,1];

ax.text(ax.get_xticks()[-1], ax.get_yticks()[-2], 'correlation='+str("%.1f" % (100*cc))+'%', ha='right');


# change left y-axis to percentage and make blue
vals = ax.get_yticks();
ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals], color='blue', size=13);
ax.set_ylabel('% Padres Payroll of league average', size=14)
ax.yaxis.label.set_color('blue')

# change right y-axis to percentage and make orange
vals = ax2.get_yticks();
ax2.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals], color='orange', size=12);
ax2.set_ylabel('Winning percentage', size=14)
ax2.yaxis.label.set_color('orange')

# edit title and xlabel
ax.set_xlabel('Season', size=14)
ax.set_title('Comparison of Padres Payroll and Winning Percentage in the 2000s', size=14)

# change years to show all and rotate
ax.set_xticks(df_all.index);
ax.set_xticklabels(df_all.index.year, size=12, rotation=90);


plt.savefig('pctPadresPayroll_pctWinning.png', dpi=400)
```


![png](Bens_Python_Jupyter_Cookbook_111_0.png)



```python
df_all[['Win%', 'Padres Payroll % of MLB average']].corr(method='pearson', min_periods=1).iloc[0,1]
```




    0.048123323911428026




```python
df_all.loc['Win%'].corr(df_all['Padres Payroll % of MLB average'])
```




    0.048123323911428012




```python
df_all.index
```




    DatetimeIndex(['2000-01-01', '2001-01-01', '2002-01-01', '2003-01-01',
                   '2004-01-01', '2005-01-01', '2006-01-01', '2007-01-01',
                   '2008-01-01', '2009-01-01', '2010-01-01', '2011-01-01',
                   '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01',
                   '2016-01-01'],
                  dtype='datetime64[ns]', name='Season', freq=None)




```python
df_all.loc[:'2007-01-01','Win%'].corr(df_all.loc[:'2007-01-01','Padres Payroll % of MLB average'])
```




    0.50195481219779436




```python
# trying with jsp

import pandas as pd
url = 'http://sandiego.padres.mlb.com/sd/history/year_by_year_results.jsp'
# but this can't find the table
pd.read_html(url)

# try beautiful soup and requests
from bs4 import BeautifulSoup
import re
import urllib.request
import requests

page = urllib.request.urlopen(url).read()
soup = BeautifulSoup(page)
```


```python
pd.read_html(requests.get(url).text)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-8-57b6eb4f0899> in <module>()
    ----> 1 pd.read_html(requests.get(url).text)
    

    C:\Users\Benjamin.Lacar\AppData\Local\Continuum\Anaconda3_v440\lib\site-packages\pandas\io\html.py in read_html(io, match, flavor, header, index_col, skiprows, attrs, parse_dates, tupleize_cols, thousands, encoding, decimal, converters, na_values, keep_default_na)
        904                   thousands=thousands, attrs=attrs, encoding=encoding,
        905                   decimal=decimal, converters=converters, na_values=na_values,
    --> 906                   keep_default_na=keep_default_na)
    

    C:\Users\Benjamin.Lacar\AppData\Local\Continuum\Anaconda3_v440\lib\site-packages\pandas\io\html.py in _parse(flavor, io, match, attrs, encoding, **kwargs)
        741             break
        742     else:
    --> 743         raise_with_traceback(retained)
        744 
        745     ret = []


    C:\Users\Benjamin.Lacar\AppData\Local\Continuum\Anaconda3_v440\lib\site-packages\pandas\compat\__init__.py in raise_with_traceback(exc, traceback)
        342         if traceback == Ellipsis:
        343             _, _, traceback = sys.exc_info()
    --> 344         raise exc.with_traceback(traceback)
        345 else:
        346     # this version of raise is a syntax error in Python 3


    ValueError: No tables found

