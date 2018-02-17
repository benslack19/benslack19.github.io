--- 
title: 'Pandas navigation and munging'
---

Basic pandas functions including data frame creation, importing, navigation and series selection. Advanced pandas functions including merging, apply, and group by.


```python
import pandas as pd
```

### Data frame indexing and navigation


```python
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

### Data frame sorting and filtering


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
# Accessing the date (re-do)
x['date'] = pd.to_datetime(x.index)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-21-1dafde5cf7fc> in <module>()
          1 # Accessing the date in a
    ----> 2 x['date'] = pd.to_datetime(x.index)
    

    NameError: name 'x' is not defined


### Pivot table


```python
# Cool pivot table example
df = pd.DataFrame({'Account_number':[1,1,2,2,2,3,3], 'Product':['A', 'A', 'A', 'B', 'B','A', 'B']})
df.pivot_table(index='Account_number', columns='Product', aggfunc=len, fill_value=0)
```

### Merging


```python
pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)
pd.merge(staff_df, student_df, how='inner', left_index=True, right_index=True)
# left regardless if they're in the overlap
pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)
```

### Method chaining


```python
(df.where(df['SUMLEV']==50)
    .dropna()
    .set_index(['STNAME','CTYNAME'])
    .rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'}))
```

### Lambda


```python
df.apply(lambda x: np.sum(x[rows]), axis=1)
```

### Other vector functions

- df.iterrows
- df.iterritems
- zip
- enumerate

### Apply function


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

### Group by

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
