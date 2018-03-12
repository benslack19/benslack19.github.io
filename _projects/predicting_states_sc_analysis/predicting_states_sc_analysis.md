---
title: Predicting phenotypic states from single cell analysis
excerpt: Using logistic regression to classify the activation state of a neuron from gene expression using Python and sci-kit learn.
header:
  image: /assets/cellstate/nerve-cell-2213009_1920.jpg
  teaser: /assets/cellstate/nerve-cell-2213009_1920.jpg 
---

[Single-cell analysis](https://en.wikipedia.org/wiki/Single-cell_analysis) has revolutionized biological studies by providing greater resolution to tissue profiling and allowing for rare cell characterization. By far, the most popular application of single-cell analysis is transcriptional profiling of cells; that is, determining what cellular [mRNA](https://en.wikipedia.org/wiki/Messenger_RNA) is being expressed. I've been fortunate to have the opportunity to contribute to this field from my [academic](https://www.nature.com/articles/ncomms11022) and [industry](https://www.fluidigm.com/applications/single-cell-analysis) work.

For some single-cell approaches, it is possible to link phenotypic information about the cell with the thousands of genes that a cell can express. For example, a certain phenotype could be indicated by expression of a known marker [protein](https://en.wikipedia.org/wiki/Protein). The protein's expression can be assessed by [flow sorting](https://en.wikipedia.org/wiki/Flow_cytometry) or cellular imaging with [fluorescence microscopy](https://en.wikipedia.org/wiki/Fluorescence_microscope) upstream of processing for single-cell transcriptome expression. However, it can be labor-intensive or not always possible to phenotype the cell, especially as current experiments scale to [thousands](https://www.nature.com/articles/ncomms14049) of single-cell samples. 

In this project, I am interested in exploring the possibility of using machine learning to train an algorithm towards identifying cellular phenotype based on gene expression alone. I'll use one set of data that we analyzed in my [academic work](https://www.nature.com/articles/ncomms11022). In our study, we looked in the mouse brain and characterized neurons activated by a novel experience. A commonly used phenotype of an activated neuron is whether it expresses the gene Fos (otherwise known as [c-Fos](https://en.wikipedia.org/wiki/C-Fos)).

This dataset begins with 96 samples of which 48 were Fos+ and 48 were Fos-, but as you'll see, filtering steps were applied to reduce the number of samples. The number of genes starts off with over 43,000(!) but as you see, this also gets reduced with filtering. I then use the unsupervised algorithm of PCA to see how whether activated and non-activated subpopulations separate  following dimensional reduction. Finally, I use logistic regression to train a subset of neurons for classification based on Fos state, then assess the accuracy of the model on a cross-validation subset.


```python
#Import packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Import data


```python
# Import gene expression data frame
parent_path = '/Users/lacar/Documents/Gage lab/Gage Lab Dropbox/CellIsolation_MethodsPaper/PaperAndFigs/FromOneDrive_finalDocuments/Lab/AnalysisOfSequencing'
child_path = '/SaraRSEM_output/update15_0225/'
os.chdir(parent_path + child_path)
df_tpm = pd.read_table('RSEM_geneSymbol_tpm_141204_allsamples.txt', sep=' ').transpose()
```


```python
# The genes are the columns (features) of the dataframe,
# while each single-cell sample are the row indexes.
# The values are the expression level of that gene for that sample.
df_tpm.head()
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
      <th>0610005C13Rik</th>
      <th>0610007P14Rik</th>
      <th>0610009B22Rik</th>
      <th>0610009E02Rik</th>
      <th>0610009L18Rik</th>
      <th>0610009O20Rik</th>
      <th>0610010F05Rik</th>
      <th>0610010K14Rik</th>
      <th>0610011F06Rik</th>
      <th>0610012D04Rik</th>
      <th>...</th>
      <th>Zwilch</th>
      <th>Zwint</th>
      <th>Zxda</th>
      <th>Zxdb</th>
      <th>Zxdc</th>
      <th>Zyg11a</th>
      <th>Zyg11b</th>
      <th>Zyx</th>
      <th>Zzef1</th>
      <th>Zzz3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nc_ui_ti_C10_141204</th>
      <td>6.54</td>
      <td>652.15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>22.39</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>22.03</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C11_141204</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>69.07</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C12_141204</th>
      <td>4.98</td>
      <td>29.90</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>173.27</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2905.22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>5.87</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C7_141204</th>
      <td>4.45</td>
      <td>56.36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.3</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>156.51</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.91</td>
      <td>0.0</td>
      <td>200.94</td>
      <td>0.0</td>
      <td>62.27</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C8_141204</th>
      <td>0.00</td>
      <td>86.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>2.99</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43309 columns</p>
</div>



## Data filtering

### Apply a gene filter


```python
# Filter the dataframe by removing genes that are not expressed in any cell
genes_retained_mask = df_tpm.sum(axis=0)>0
df_tpm2 = df_tpm[genes_retained_mask.index[genes_retained_mask]]
df_tpm2.head()
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
      <th>0610005C13Rik</th>
      <th>0610007P14Rik</th>
      <th>0610009B22Rik</th>
      <th>0610009E02Rik</th>
      <th>0610009L18Rik</th>
      <th>0610009O20Rik</th>
      <th>0610010F05Rik</th>
      <th>0610010K14Rik</th>
      <th>0610011F06Rik</th>
      <th>0610012G03Rik</th>
      <th>...</th>
      <th>Zw10</th>
      <th>Zwilch</th>
      <th>Zwint</th>
      <th>Zxda</th>
      <th>Zxdb</th>
      <th>Zxdc</th>
      <th>Zyg11b</th>
      <th>Zyx</th>
      <th>Zzef1</th>
      <th>Zzz3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nc_ui_ti_C10_141204</th>
      <td>6.54</td>
      <td>652.15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22.39</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>22.03</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C11_141204</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>69.07</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C12_141204</th>
      <td>4.98</td>
      <td>29.90</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>173.27</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2905.22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>5.87</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C7_141204</th>
      <td>4.45</td>
      <td>56.36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.3</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>156.51</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.91</td>
      <td>200.94</td>
      <td>0.0</td>
      <td>62.27</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C8_141204</th>
      <td>0.00</td>
      <td>86.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>2.99</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24345 columns</p>
</div>



### Apply a sample filter
Start by exploring the distribution of gene counts for each sample


```python
# Filter the dataframe by removing samples with low gene count
df_tpm2[df_tpm2 > 1].count(axis=1).hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x13eabd668>




![png](output_9_1.png)


Removing samples that have less than 3000 genes seems like a logical cutoff.


```python
# Filter the dataframe by removing samples with low gene count (3000 genes)
samples_retained_mask = df_tpm2[df_tpm2 > 1].count(axis=1) > 3000
```


```python
df_tpm3 = df_tpm2[samples_retained_mask]
```


```python
# Sanity check that appropriate filter is implemented
df_tpm3[df_tpm3 > 1].count(axis=1).hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x13db18278>




![png](output_13_1.png)



```python
# Create a vector containing the fos label
df_tpm3['SampleName'] = df_tpm3.index  # temporarily create a sample name vector
df_tpm3.loc[:,'fos_label'] = df_tpm3['SampleName'].str.split('_', expand=True)[2]
df_tpm3.drop('SampleName', axis=1, inplace=True)
```

    /Users/lacar/anaconda/lib/python3.5/site-packages/ipykernel/__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from ipykernel import kernelapp as app
    /Users/lacar/anaconda/lib/python3.5/site-packages/pandas/core/indexing.py:537: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s
    /Users/lacar/anaconda/lib/python3.5/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
df_tpm3.head()
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
      <th>0610005C13Rik</th>
      <th>0610007P14Rik</th>
      <th>0610009B22Rik</th>
      <th>0610009E02Rik</th>
      <th>0610009L18Rik</th>
      <th>0610009O20Rik</th>
      <th>0610010F05Rik</th>
      <th>0610010K14Rik</th>
      <th>0610011F06Rik</th>
      <th>0610012G03Rik</th>
      <th>...</th>
      <th>Zwilch</th>
      <th>Zwint</th>
      <th>Zxda</th>
      <th>Zxdb</th>
      <th>Zxdc</th>
      <th>Zyg11b</th>
      <th>Zyx</th>
      <th>Zzef1</th>
      <th>Zzz3</th>
      <th>fos_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nc_ui_ti_C10_141204</th>
      <td>6.54</td>
      <td>652.15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.00</td>
      <td>22.39</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>22.03</td>
      <td>0.00</td>
      <td>ti</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C12_141204</th>
      <td>4.98</td>
      <td>29.90</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>173.27</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.00</td>
      <td>2905.22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.87</td>
      <td>ti</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C7_141204</th>
      <td>4.45</td>
      <td>56.36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.3</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.00</td>
      <td>156.51</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.91</td>
      <td>200.94</td>
      <td>0.00</td>
      <td>62.27</td>
      <td>0.00</td>
      <td>ti</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C8_141204</th>
      <td>0.00</td>
      <td>86.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.99</td>
      <td>0.00</td>
      <td>ti</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C9_141204</th>
      <td>10.13</td>
      <td>5.22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.26</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>5.87</td>
      <td>1045.78</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.41</td>
      <td>421.23</td>
      <td>113.95</td>
      <td>0.00</td>
      <td>ti</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24346 columns</p>
</div>




```python
# Replace labels with 0 and 1 for classification
df_tpm3['fos_label'].replace(to_replace='ti', value=1, inplace=True)
df_tpm3['fos_label'].replace(to_replace='tn', value=0, inplace=True)
```

    /Users/lacar/anaconda/lib/python3.5/site-packages/pandas/core/generic.py:4619: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._update_inplace(new_data)



```python
# Show the number of examples for each group. Note that 1 = FOS+, 0 = FOS-.
df_tpm3.groupby('fos_label')['fos_label'].count()
```




    fos_label
    0    43
    1    39
    Name: fos_label, dtype: int64



## PCA visualization

Visualize how the groups would separate without using the label. 


```python
X = df_tpm3.loc[:,:'Zzz3']
```


```python
X
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
      <th>0610005C13Rik</th>
      <th>0610007P14Rik</th>
      <th>0610009B22Rik</th>
      <th>0610009E02Rik</th>
      <th>0610009L18Rik</th>
      <th>0610009O20Rik</th>
      <th>0610010F05Rik</th>
      <th>0610010K14Rik</th>
      <th>0610011F06Rik</th>
      <th>0610012G03Rik</th>
      <th>...</th>
      <th>Zw10</th>
      <th>Zwilch</th>
      <th>Zwint</th>
      <th>Zxda</th>
      <th>Zxdb</th>
      <th>Zxdc</th>
      <th>Zyg11b</th>
      <th>Zyx</th>
      <th>Zzef1</th>
      <th>Zzz3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nc_ui_ti_C10_141204</th>
      <td>6.54</td>
      <td>652.15</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>22.39</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>22.03</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C12_141204</th>
      <td>4.98</td>
      <td>29.90</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>173.27</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2905.22</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.87</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C7_141204</th>
      <td>4.45</td>
      <td>56.36</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.30</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>156.51</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.91</td>
      <td>200.94</td>
      <td>0.00</td>
      <td>62.27</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C8_141204</th>
      <td>0.00</td>
      <td>86.04</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.99</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C9_141204</th>
      <td>10.13</td>
      <td>5.22</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>5.87</td>
      <td>1045.78</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.41</td>
      <td>421.23</td>
      <td>113.95</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_D10_141204</th>
      <td>7.49</td>
      <td>226.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>141.49</td>
      <td>0.00</td>
      <td>71.68</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_D12_141204</th>
      <td>3.89</td>
      <td>128.93</td>
      <td>0.00</td>
      <td>97.92</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>8.36</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.45</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_D7_141204</th>
      <td>3.15</td>
      <td>11.40</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>183.85</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>396.24</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>30.89</td>
      <td>3.31</td>
      <td>0.00</td>
      <td>47.49</td>
      <td>27.62</td>
    </tr>
    <tr>
      <th>nc_ui_ti_D8_141204</th>
      <td>0.00</td>
      <td>0.73</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>7.26</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>235.04</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.53</td>
      <td>0.00</td>
      <td>21.60</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_ti_D9_141204</th>
      <td>0.00</td>
      <td>102.49</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>24.23</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>34.39</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_tn_C1_141204</th>
      <td>0.00</td>
      <td>842.40</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>12.05</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>134.88</td>
      <td>9.41</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_tn_C2_141204</th>
      <td>7.58</td>
      <td>96.30</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>18.26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>18.22</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>19.16</td>
      <td>0.00</td>
      <td>677.16</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_tn_C4_141204</th>
      <td>0.00</td>
      <td>117.85</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>924.68</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>9.06</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.22</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_tn_C5_141204</th>
      <td>0.00</td>
      <td>50.25</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.55</td>
      <td>0.00</td>
      <td>54.16</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_tn_C6_141204</th>
      <td>0.00</td>
      <td>38.55</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>12.10</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>2788.95</td>
      <td>7.97</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_tn_D1_141204</th>
      <td>13.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1582.86</td>
      <td>775.07</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>14.45</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>26.26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_tn_D2_141204</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>375.14</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>84.60</td>
      <td>0.00</td>
      <td>6.55</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_tn_D3_141204</th>
      <td>3.09</td>
      <td>79.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2401.01</td>
      <td>0.00</td>
      <td>20.86</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>2.47</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20.34</td>
      <td>0.00</td>
      <td>2.52</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_tn_D4_141204</th>
      <td>2.98</td>
      <td>6.23</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>8.01</td>
      <td>0.00</td>
      <td>...</td>
      <td>132.36</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>207.78</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.85</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ui_tn_D6_141204</th>
      <td>0.00</td>
      <td>75.68</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>774.78</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>22.19</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ux_ti_A10_141204</th>
      <td>5.77</td>
      <td>104.78</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.64</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>3.25</td>
      <td>397.42</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>9.02</td>
      <td>0.00</td>
      <td>63.11</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ux_ti_A11_141204</th>
      <td>0.62</td>
      <td>2.44</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>267.99</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>70.28</td>
      <td>0.66</td>
      <td>0.00</td>
      <td>423.78</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ux_ti_A12_141204</th>
      <td>2.82</td>
      <td>580.22</td>
      <td>0.00</td>
      <td>41.78</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>87.05</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>10.76</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.66</td>
      <td>0.00</td>
      <td>0.89</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ux_ti_A7_141204</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>556.46</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>402.40</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.37</td>
      <td>29.45</td>
      <td>25.56</td>
      <td>25.91</td>
    </tr>
    <tr>
      <th>nc_ux_ti_A8_141204</th>
      <td>4.19</td>
      <td>22.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>125.35</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.48</td>
      <td>0.00</td>
      <td>6.92</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ux_ti_A9_141204</th>
      <td>5.28</td>
      <td>28.03</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1479.09</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>297.28</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ux_ti_B10_141204</th>
      <td>0.00</td>
      <td>191.79</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>58.30</td>
      <td>1095.61</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.13</td>
      <td>0.00</td>
      <td>621.52</td>
      <td>7.44</td>
    </tr>
    <tr>
      <th>nc_ux_ti_B11_141204</th>
      <td>5.62</td>
      <td>40.65</td>
      <td>344.79</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>51.51</td>
      <td>0.00</td>
      <td>503.93</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>82.27</td>
    </tr>
    <tr>
      <th>nc_ux_ti_B12_141204</th>
      <td>16.47</td>
      <td>218.81</td>
      <td>59.00</td>
      <td>55.05</td>
      <td>0.00</td>
      <td>1.16</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>900.06</td>
      <td>428.88</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>11.64</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>43.36</td>
      <td>0.00</td>
      <td>0.89</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nc_ux_ti_B7_141204</th>
      <td>14.81</td>
      <td>8.73</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>46.01</td>
      <td>0.00</td>
      <td>39.22</td>
      <td>215.45</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>144.63</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>279.28</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>28.55</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>nm_ui_tn_G2_141204</th>
      <td>1.98</td>
      <td>11.61</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>288.72</td>
      <td>0.00</td>
      <td>554.45</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>170.54</td>
      <td>0.00</td>
      <td>1.43</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.39</td>
      <td>0.00</td>
      <td>518.41</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ui_tn_G3_141204</th>
      <td>6.35</td>
      <td>698.30</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>82.67</td>
      <td>0.00</td>
      <td>382.46</td>
      <td>679.54</td>
      <td>...</td>
      <td>0.00</td>
      <td>8.17</td>
      <td>498.64</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.16</td>
      <td>13.56</td>
      <td>0.00</td>
      <td>11.44</td>
      <td>9.50</td>
    </tr>
    <tr>
      <th>nm_ui_tn_G4_141204</th>
      <td>1.24</td>
      <td>133.41</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.81</td>
      <td>2101.75</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>3.24</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.74</td>
      <td>0.00</td>
      <td>5.42</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ui_tn_G5_141204</th>
      <td>0.81</td>
      <td>2.04</td>
      <td>0.00</td>
      <td>172.16</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>35.59</td>
      <td>117.70</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.74</td>
      <td>74.88</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>3.85</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>nm_ui_tn_H1_141204</th>
      <td>4.77</td>
      <td>74.99</td>
      <td>0.00</td>
      <td>5.31</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>6.60</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>40.41</td>
      <td>0.0</td>
      <td>25.42</td>
      <td>0.00</td>
      <td>12.27</td>
      <td>0.00</td>
      <td>283.11</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ui_tn_H3_141204</th>
      <td>3.14</td>
      <td>208.46</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2026.25</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1674.25</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>74.94</td>
      <td>243.38</td>
      <td>4.95</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ui_tn_H4_141204</th>
      <td>0.00</td>
      <td>23.87</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>5.84</td>
      <td>2.77</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>14.07</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ui_tn_H5_141204</th>
      <td>10.96</td>
      <td>717.60</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.20</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>945.34</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>99.41</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ui_tn_H6_141204</th>
      <td>4.30</td>
      <td>668.54</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3282.17</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>13.03</td>
      <td>0.00</td>
      <td>247.99</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ux_ti_E10_141204</th>
      <td>11.53</td>
      <td>3.07</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1415.95</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.75</td>
      <td>413.90</td>
      <td>0.00</td>
      <td>118.49</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ux_ti_E12_141204</th>
      <td>3.81</td>
      <td>30.29</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>19.97</td>
      <td>218.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>38.93</td>
      <td>1.70</td>
      <td>484.88</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.55</td>
      <td>2.11</td>
      <td>1.28</td>
      <td>90.38</td>
    </tr>
    <tr>
      <th>nm_ux_ti_E7_141204</th>
      <td>13.63</td>
      <td>306.36</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>212.22</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>182.53</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>327.69</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>39.21</td>
      <td>0.00</td>
      <td>126.18</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ux_ti_E8_141204</th>
      <td>1.87</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>38.64</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>279.38</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.24</td>
      <td>0.00</td>
      <td>39.04</td>
      <td>14.85</td>
    </tr>
    <tr>
      <th>nm_ux_ti_E9_141204</th>
      <td>9.41</td>
      <td>42.11</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>24.08</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1137.58</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>34.42</td>
      <td>15.59</td>
      <td>42.35</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>nm_ux_ti_F10_141204</th>
      <td>4.30</td>
      <td>2.69</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.12</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>3.58</td>
      <td>92.44</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>81.71</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>12.88</td>
    </tr>
    <tr>
      <th>nm_ux_ti_F11_141204</th>
      <td>3.52</td>
      <td>785.55</td>
      <td>0.00</td>
      <td>19.22</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.01</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1203.11</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>13.43</td>
      <td>0.00</td>
      <td>0.87</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ux_ti_F12_141204</th>
      <td>11.94</td>
      <td>17.62</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1060.78</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.60</td>
      <td>0.00</td>
      <td>9.16</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ux_ti_F8_141204</th>
      <td>2.02</td>
      <td>290.30</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.76</td>
      <td>0.00</td>
      <td>54.07</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>1.97</td>
      <td>206.42</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>32.69</td>
      <td>134.82</td>
      <td>8.41</td>
      <td>212.06</td>
    </tr>
    <tr>
      <th>nm_ux_ti_F9_141204</th>
      <td>1.25</td>
      <td>13.58</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>68.73</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>362.64</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>29.32</td>
      <td>0.99</td>
      <td>18.60</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ux_tn_E1_141204</th>
      <td>3.47</td>
      <td>42.26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20.51</td>
      <td>0.00</td>
      <td>8.45</td>
      <td>397.85</td>
      <td>...</td>
      <td>110.83</td>
      <td>0.00</td>
      <td>176.51</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>6.53</td>
      <td>0.00</td>
      <td>47.43</td>
      <td>187.16</td>
    </tr>
    <tr>
      <th>nm_ux_tn_E2_141204</th>
      <td>11.78</td>
      <td>187.09</td>
      <td>0.00</td>
      <td>194.90</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>170.15</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>431.72</td>
      <td>327.46</td>
    </tr>
    <tr>
      <th>nm_ux_tn_E3_141204</th>
      <td>0.00</td>
      <td>28.42</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>89.81</td>
      <td>0.00</td>
      <td>322.72</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>340.61</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.40</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ux_tn_E4_141204</th>
      <td>0.00</td>
      <td>29.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>24.10</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>96.62</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>81.64</td>
      <td>29.47</td>
      <td>0.00</td>
      <td>54.76</td>
      <td>1.54</td>
    </tr>
    <tr>
      <th>nm_ux_tn_E6_141204</th>
      <td>0.00</td>
      <td>4.25</td>
      <td>0.00</td>
      <td>215.23</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>6.90</td>
      <td>363.65</td>
      <td>...</td>
      <td>200.54</td>
      <td>4.99</td>
      <td>20.15</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>12.26</td>
      <td>0.00</td>
      <td>25.36</td>
      <td>43.66</td>
    </tr>
    <tr>
      <th>nm_ux_tn_F1_141204</th>
      <td>4.51</td>
      <td>185.16</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>66.81</td>
      <td>26.75</td>
      <td>121.31</td>
      <td>229.02</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>3.42</td>
      <td>306.02</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>42.24</td>
      <td>17.38</td>
      <td>0.00</td>
      <td>2.65</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ux_tn_F2_141204</th>
      <td>4.76</td>
      <td>11.22</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>280.52</td>
      <td>0.00</td>
      <td>238.84</td>
      <td>265.75</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>23.03</td>
      <td>0.00</td>
      <td>296.96</td>
      <td>0.00</td>
      <td>44.04</td>
      <td>74.04</td>
    </tr>
    <tr>
      <th>nm_ux_tn_F3_141204</th>
      <td>0.00</td>
      <td>58.08</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.91</td>
      <td>8.70</td>
      <td>0.00</td>
      <td>362.43</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>2.39</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20.08</td>
      <td>0.00</td>
      <td>141.28</td>
      <td>215.33</td>
    </tr>
    <tr>
      <th>nm_ux_tn_F4_141204</th>
      <td>8.54</td>
      <td>14.08</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>15.93</td>
      <td>93.82</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>479.99</td>
      <td>...</td>
      <td>58.78</td>
      <td>0.64</td>
      <td>20.62</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.21</td>
      <td>54.22</td>
      <td>60.36</td>
      <td>60.50</td>
    </tr>
    <tr>
      <th>nm_ux_tn_F5_141204</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>14.22</td>
      <td>0.00</td>
      <td>306.03</td>
      <td>0.00</td>
      <td>...</td>
      <td>71.38</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>30.03</td>
      <td>123.41</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>nm_ux_tn_F6_141204</th>
      <td>1.39</td>
      <td>152.23</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.46</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>1.76</td>
      <td>51.91</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.93</td>
      <td>0.00</td>
      <td>28.12</td>
      <td>89.01</td>
    </tr>
  </tbody>
</table>
<p>82 rows × 24345 columns</p>
</div>




```python
# Run PCA, ignoring the label
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
# X1 = pca.fit(X).transform(X)
```


```python
X
```




    array([[  5.52652470e+04,  -2.05759783e+03],
           [  2.53733115e+04,  -2.10047799e+03],
           [  6.66322621e+04,  -1.96155587e+03],
           [  8.32352373e+04,  -1.81724518e+03],
           [  5.45997386e+04,  -6.81069431e+02],
           [  9.19652024e+04,  -1.95559242e+03],
           [ -2.18024189e+04,  -2.43646338e+03],
           [ -1.49317140e+04,  -3.42942756e+03],
           [  1.16658116e+05,  -1.62510382e+03],
           [  7.62330066e+04,  -1.63814005e+03],
           [  1.76384422e+04,  -2.59021263e+03],
           [ -2.78924067e+04,  -2.62802324e+03],
           [ -9.67903136e+01,  -2.65966834e+03],
           [ -2.25236760e+04,  -3.06840477e+03],
           [ -8.30297113e+04,  -3.32312799e+03],
           [ -5.50922305e+04,  -2.78421433e+03],
           [ -3.26143895e+04,  -2.91828783e+03],
           [  2.96773157e+04,  -2.50948835e+03],
           [ -3.78271600e+04,  -2.38890803e+03],
           [  3.45556822e+04,  -2.51650901e+03],
           [  1.07190976e+05,  -2.02130611e+03],
           [ -4.04406675e+04,  -3.02800996e+03],
           [  3.29668464e+03,  -2.53665362e+03],
           [  7.89319253e+04,  -2.30082435e+03],
           [  1.13151957e+03,   1.70660953e+05],
           [  1.46227979e+04,  -2.73910236e+03],
           [ -3.73469643e+04,  -3.16308787e+03],
           [ -9.55144348e+03,  -2.78694815e+03],
           [ -6.37251499e+04,  -3.06765657e+03],
           [ -4.85620561e+04,  -3.19995956e+03],
           [ -3.81684655e+04,  -3.07065794e+03],
           [ -6.11499647e+04,  -3.44391468e+03],
           [ -6.34666457e+04,  -3.16313130e+03],
           [ -3.02012166e+04,  -3.12895227e+03],
           [ -3.05233805e+04,  -3.41155617e+03],
           [ -4.14047204e+04,  -2.89437785e+03],
           [ -2.09155743e+04,  -2.98046424e+03],
           [ -3.01887960e+04,  -3.07391953e+03],
           [ -2.38977999e+04,  -2.73036924e+03],
           [ -5.34614655e+04,  -3.72127594e+03],
           [ -6.94643160e+04,  -3.72380352e+03],
           [ -4.75487578e+03,  -2.92426860e+03],
           [  1.44653798e+04,  -3.10537653e+03],
           [  1.16201988e+05,  -1.37758927e+03],
           [  5.07283910e+04,   9.38544147e+02],
           [ -7.24458568e+02,  -2.21316161e+03],
           [ -7.14687686e+04,   2.54409156e+04],
           [  9.60515425e+04,  -1.82359884e+03],
           [  2.99789539e+04,  -2.33974065e+03],
           [ -3.02992325e+03,   4.67600252e+03],
           [  6.09754844e+04,  -2.09678187e+03],
           [ -5.81684923e+04,  -2.91267368e+03],
           [ -3.79679771e+04,  -2.97403838e+03],
           [ -3.32837225e+04,  -8.97011811e+02],
           [  2.81702806e+04,  -2.27795697e+03],
           [ -5.43883617e+04,  -3.07735894e+03],
           [ -1.23277541e+04,  -2.43273565e+03],
           [ -2.45422084e+04,  -2.87386041e+03],
           [  1.54557994e+03,  -2.57344706e+03],
           [ -1.12073946e+04,  -2.68386048e+03],
           [ -1.06424200e+04,  -2.60936475e+03],
           [  5.96346090e+04,  -2.12081300e+03],
           [  6.53168772e+03,  -2.57746009e+03],
           [  1.22125831e+04,  -2.41396217e+03],
           [  5.56211108e+04,  -2.13013075e+03],
           [ -2.73185358e+04,  -2.63550213e+03],
           [  6.84180013e+04,  -1.99954857e+03],
           [  2.84063697e+04,  -2.13275591e+03],
           [  4.31070774e+04,  -2.07251930e+03],
           [  3.30433283e+03,  -2.47232293e+03],
           [  6.41011030e+04,  -2.06422563e+03],
           [ -6.80262602e+04,  -3.11772879e+03],
           [ -8.78981847e+03,  -2.89641708e+03],
           [ -2.53930685e+04,  -2.97870227e+03],
           [ -1.82069482e+04,  -3.11640893e+03],
           [  4.32707178e+04,  -2.39207362e+03],
           [ -5.51193143e+04,  -3.26535018e+03],
           [ -3.18125538e+04,  -2.79223620e+03],
           [ -7.89211149e+03,  -2.57974637e+03],
           [ -7.60653760e+04,  -2.92967751e+03],
           [  4.45280049e+03,  -2.48434501e+03],
           [ -4.47759918e+04,  -2.07617386e+03]])




```python
# Add PCA vectors to dataframe
df_tpm3.loc[:,'PCAx'] = X[:,0]
df_tpm3.loc[:,'PCAy'] = X[:,1]
```

    /Users/lacar/anaconda/lib/python3.5/site-packages/pandas/core/indexing.py:537: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



```python
df_tpm3.head()
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
      <th>0610005C13Rik</th>
      <th>0610007P14Rik</th>
      <th>0610009B22Rik</th>
      <th>0610009E02Rik</th>
      <th>0610009L18Rik</th>
      <th>0610009O20Rik</th>
      <th>0610010F05Rik</th>
      <th>0610010K14Rik</th>
      <th>0610011F06Rik</th>
      <th>0610012G03Rik</th>
      <th>...</th>
      <th>Zxda</th>
      <th>Zxdb</th>
      <th>Zxdc</th>
      <th>Zyg11b</th>
      <th>Zyx</th>
      <th>Zzef1</th>
      <th>Zzz3</th>
      <th>fos_label</th>
      <th>PCAx</th>
      <th>PCAy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nc_ui_ti_C10_141204</th>
      <td>6.54</td>
      <td>652.15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>22.03</td>
      <td>0.00</td>
      <td>1</td>
      <td>55265.247009</td>
      <td>-2057.597830</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C12_141204</th>
      <td>4.98</td>
      <td>29.90</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>173.27</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.87</td>
      <td>1</td>
      <td>25373.311476</td>
      <td>-2100.477988</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C7_141204</th>
      <td>4.45</td>
      <td>56.36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.3</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.91</td>
      <td>200.94</td>
      <td>0.00</td>
      <td>62.27</td>
      <td>0.00</td>
      <td>1</td>
      <td>66632.262116</td>
      <td>-1961.555875</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C8_141204</th>
      <td>0.00</td>
      <td>86.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.99</td>
      <td>0.00</td>
      <td>1</td>
      <td>83235.237261</td>
      <td>-1817.245181</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C9_141204</th>
      <td>10.13</td>
      <td>5.22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.26</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.41</td>
      <td>421.23</td>
      <td>113.95</td>
      <td>0.00</td>
      <td>1</td>
      <td>54599.738613</td>
      <td>-681.069431</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24348 columns</p>
</div>




```python
# PCA, colored by ethnicities
f, ax1 = plt.subplots(1,1)
cmap = 'bwr'

points = ax1.scatter(df_tpm3['PCAx'], df_tpm3['PCAy'], c=df_tpm3['fos_label'], s=20, cmap=cmap)

# plot formatting
ax1.set(xlabel='')
ax1.set_ylabel('')
ax1.set_title('neurons clustered by FOS state', size=14)
#ax1.set_ylim(-4000,0)
#ax1.legend(loc=0, title='FOS state')
    
# figure properties
sns.set(font='Franklin Gothic Book')
ax1.tick_params(axis='both', which='both',length=0)   # remove tick marks
sns.set_style(style='white')
sns.despine(left=True, bottom=True, right=True)   #  remove frame

```


![png](output_25_0.png)


Yikes, this is not the prettiest plot. I spent time figuring out if I had done something wrong, even going back to Andrew Ng's lecture on PCA. I eventually figured out that I forgot to transform the feature values so that they're not spread out across so many different scales. Log2 normalization is common in the single-cell field. It's also clear that there are some outliers here. I'll take care of both of these modifications in the next bit of code.


```python
# Filter the samples that are obvious outliers
df_tpm4 = df_tpm3[df_tpm3['PCAy'] < 25000]
```


```python
# Use new dataframe and log transform so features are not spread across different scales
X = df_tpm4.loc[:,:'Zzz3']
X = np.log2(X+1)  # the +1 is added to deal with the log transformation
```


```python
# Run PCA again
pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
```


```python
# Add new PCA vectors to dataframe
df_tpm4.loc[:,'PCAx'] = X[:,0]
df_tpm4.loc[:,'PCAy'] = X[:,1]
```

    /Users/lacar/anaconda/lib/python3.5/site-packages/pandas/core/indexing.py:537: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s



```python
df_tpm4.shape
```




    (80, 24348)




```python
# PCA, colored by FOS state

f, ax1 = plt.subplots(1,1)
cmap = 'bwr'

points = ax1.scatter(df_tpm4['PCAx'], df_tpm4['PCAy'], c=df_tpm4['fos_label'], s=20, cmap=cmap)

# plot formatting
ax1.set(xlabel='')
ax1.set_ylabel('')
ax1.set_title('neurons, FOS+ (red), FOS- (blue)', size=14)

# figure properties
sns.set(font='Franklin Gothic Book')
ax1.tick_params(axis='both', which='both',length=0)   # remove tick marks
sns.set_style(style='white')
sns.despine(left=True, bottom=True, right=True)   #  remove frame

```


![png](output_32_0.png)


There we go. With the exception of a few FOS+ neurons, the groups separate quite nicely as we showed in the paper. The FOS+ neurons that are amongst the FOS- neurons are discussed further in our paper (we call them pseudo-FOS+ neurons), but for the purposes of this project, I will remove them from further analysis.


```python
# Filtered dataframe by removing pseudo-FOS+ neurons.
mask = (df_tpm4['fos_label']==1) & (df_tpm4['PCAy'] < 0)
df_tpm5 = df_tpm4[~mask]   # use ~ to get the inverse of what the mask is getting
```


```python
# Group sizes moving forward
df_tpm5.groupby('fos_label')['fos_label'].count()
```




    fos_label
    0    43
    1    30
    Name: fos_label, dtype: int64




```python
df_tpm5.shape
```




    (73, 24348)




```python
df_tpm5.head()
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
      <th>0610005C13Rik</th>
      <th>0610007P14Rik</th>
      <th>0610009B22Rik</th>
      <th>0610009E02Rik</th>
      <th>0610009L18Rik</th>
      <th>0610009O20Rik</th>
      <th>0610010F05Rik</th>
      <th>0610010K14Rik</th>
      <th>0610011F06Rik</th>
      <th>0610012G03Rik</th>
      <th>...</th>
      <th>Zxda</th>
      <th>Zxdb</th>
      <th>Zxdc</th>
      <th>Zyg11b</th>
      <th>Zyx</th>
      <th>Zzef1</th>
      <th>Zzz3</th>
      <th>fos_label</th>
      <th>PCAx</th>
      <th>PCAy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nc_ui_ti_C10_141204</th>
      <td>6.54</td>
      <td>652.15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>22.03</td>
      <td>0.00</td>
      <td>1</td>
      <td>-98.443511</td>
      <td>35.546565</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C12_141204</th>
      <td>4.98</td>
      <td>29.90</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>173.27</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.87</td>
      <td>1</td>
      <td>-89.695505</td>
      <td>58.960111</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C7_141204</th>
      <td>4.45</td>
      <td>56.36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.3</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.91</td>
      <td>200.94</td>
      <td>0.00</td>
      <td>62.27</td>
      <td>0.00</td>
      <td>1</td>
      <td>-73.874904</td>
      <td>53.031325</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C8_141204</th>
      <td>0.00</td>
      <td>86.04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.99</td>
      <td>0.00</td>
      <td>1</td>
      <td>-89.654770</td>
      <td>43.962223</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C9_141204</th>
      <td>10.13</td>
      <td>5.22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.26</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.41</td>
      <td>421.23</td>
      <td>113.95</td>
      <td>0.00</td>
      <td>1</td>
      <td>-77.416751</td>
      <td>33.628965</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24348 columns</p>
</div>



## Supervised machine learning

What machine learning algorithm should I apply? This is a classification problem (FOS+ vs. FOS-) but this is also a situation where the number of *m* examples (73) is much, much smaller than the number of *n* features?  << n features (24345)?

I reviewed my notes from the [Machine Learning course](https://www.coursera.org/learn/machine-learning) I completed and saw that logistic regression or support vector machines without a kernel ("linear") would be good options. Andrew Ng talks about this [here](https://youtu.be/FCUBwP-JTsA?t=867).


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
```


```python
# I didn't carry over the log2 normalization, so I'll do it again here.
# Using X and y to refer to my feature set and label set, respectively

X = np.log2(df_tpm5.loc[:,:'Zzz3']+1)   # a matrix
y = df_tpm5['fos_label']  # a vector
```

### Using all samples for training and testing

This is a first pass at trying the data on this model.


```python
# Followed DataSchool example found online
# (http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976)

# Instantiate a logistic regression model, and fit with X and y 
model = LogisticRegression()
model = model.fit(X, y)

# check the accuracy on the training set
model.score(X, y)
```




    1.0



A perfect accuracy, which is awesome, but not entirely surprising for two reasons. We used all of the samples for both training and testing and we saw that PCA split these two groups pretty cleanly. Nevertheless, let's see how the model coefficients look.


```python
# Examining the model coefficients
df_gene_model_coef = pd.DataFrame()
df_gene_model_coef['gene'] = X.columns
df_gene_model_coef['coef'] = np.transpose(model.coef_)
df_gene_model_coef.sort_values(by='coef', ascending=False).head(5)
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
      <th>gene</th>
      <th>coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020</th>
      <td>Arc</td>
      <td>0.012490</td>
    </tr>
    <tr>
      <th>17392</th>
      <td>Plk2</td>
      <td>0.011933</td>
    </tr>
    <tr>
      <th>1875</th>
      <td>Ankrd33b</td>
      <td>0.011126</td>
    </tr>
    <tr>
      <th>16864</th>
      <td>Pcdh8</td>
      <td>0.010876</td>
    </tr>
    <tr>
      <th>13538</th>
      <td>Ifrd1</td>
      <td>0.010867</td>
    </tr>
  </tbody>
</table>
</div>



While the coefficients are not very high, the top gene is Arc, a immediate early gene known for being involved in activity. This is encouraging for our model since it matches what we predict and also what we've shown through differential expression, which is shown in the paper.

### Evaluation by splitting into training and validation sets


```python
# Split into train and test sets. I'll use a ratio of 60/40 of train/test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



#### Sanity check by evaluating train and test samples and shape


```python
X_train.head(10)
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
      <th>0610005C13Rik</th>
      <th>0610007P14Rik</th>
      <th>0610009B22Rik</th>
      <th>0610009E02Rik</th>
      <th>0610009L18Rik</th>
      <th>0610009O20Rik</th>
      <th>0610010F05Rik</th>
      <th>0610010K14Rik</th>
      <th>0610011F06Rik</th>
      <th>0610012G03Rik</th>
      <th>...</th>
      <th>Zw10</th>
      <th>Zwilch</th>
      <th>Zwint</th>
      <th>Zxda</th>
      <th>Zxdb</th>
      <th>Zxdc</th>
      <th>Zyg11b</th>
      <th>Zyx</th>
      <th>Zzef1</th>
      <th>Zzz3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nm_ui_tn_H3_141204</th>
      <td>2.049631</td>
      <td>7.710531</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>10.985308</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.710161</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>6.246788</td>
      <td>7.932982</td>
      <td>2.572890</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>nm_ux_ti_E10_141204</th>
      <td>3.647315</td>
      <td>2.025029</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.468573</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.459432</td>
      <td>8.696620</td>
      <td>0.000000</td>
      <td>6.900746</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>nc_ux_ti_A7_141204</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>9.122724</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.656067</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2.424922</td>
      <td>4.928370</td>
      <td>4.731183</td>
      <td>4.750070</td>
    </tr>
    <tr>
      <th>nm_ui_tn_G1_141204</th>
      <td>2.969012</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>2.639232</td>
      <td>2.861955</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.169925</td>
      <td>3.231125</td>
      <td>0.000000</td>
      <td>5.099716</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>nc_ux_tn_A4_141204</th>
      <td>4.843481</td>
      <td>6.498251</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.907852</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>4.597531</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>4.794416</td>
      <td>0.000000</td>
      <td>5.205940</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>nc_ux_tn_A2_141204</th>
      <td>0.000000</td>
      <td>5.501121</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>6.785943</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.232085</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>6.188243</td>
      <td>0.000000</td>
      <td>6.928607</td>
      <td>7.432625</td>
    </tr>
    <tr>
      <th>nm_ux_tn_F4_141204</th>
      <td>3.253989</td>
      <td>3.914565</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.081510</td>
      <td>6.567119</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>8.909863</td>
      <td>...</td>
      <td>5.901591</td>
      <td>0.713696</td>
      <td>4.434295</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.144046</td>
      <td>5.787119</td>
      <td>5.939227</td>
      <td>5.942515</td>
    </tr>
    <tr>
      <th>nm_ui_tn_H4_141204</th>
      <td>0.000000</td>
      <td>4.636335</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>2.773996</td>
      <td>1.914565</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.913608</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>nc_ux_tn_B4_141204</th>
      <td>5.344828</td>
      <td>6.056367</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>7.488161</td>
      <td>0.000000</td>
      <td>9.641709</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.673556</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>nc_ui_ti_D7_141204</th>
      <td>2.053111</td>
      <td>3.632268</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>7.530211</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.633867</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.995032</td>
      <td>2.107688</td>
      <td>0.000000</td>
      <td>5.599615</td>
      <td>4.838952</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 24345 columns</p>
</div>




```python
print('training set shape: ', X_train.shape, 'test set shape: ', X_test.shape)
```

    training set shape:  (43, 24345) test set shape:  (30, 24345)


#### Run predictions!


```python
# Predict fos labels for the test set
predicted = model2.predict(X_test)
print(predicted)
```

    [0 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 0 0 1 1 0 1 0 0 0]



```python
# generate class probabilities
probs = model2.predict_proba(X_test)
probs
```




    array([[  9.79129870e-01,   2.08701297e-02],
           [  9.27710561e-04,   9.99072289e-01],
           [  2.52824449e-03,   9.97471756e-01],
           [  9.90136982e-01,   9.86301831e-03],
           [  9.94536631e-01,   5.46336891e-03],
           [  9.96892526e-01,   3.10747429e-03],
           [  9.96896054e-01,   3.10394595e-03],
           [  9.93028840e-01,   6.97116028e-03],
           [  9.98386314e-01,   1.61368561e-03],
           [  8.44309409e-01,   1.55690591e-01],
           [  4.00968687e-03,   9.95990313e-01],
           [  6.90008585e-03,   9.93099914e-01],
           [  7.28804895e-03,   9.92711951e-01],
           [  1.54732724e-03,   9.98452673e-01],
           [  9.90972984e-01,   9.02701592e-03],
           [  9.96487663e-01,   3.51233658e-03],
           [  9.97721457e-01,   2.27854258e-03],
           [  9.95794236e-01,   4.20576436e-03],
           [  7.64663538e-04,   9.99235336e-01],
           [  9.35144305e-01,   6.48556948e-02],
           [  3.57385183e-02,   9.64261482e-01],
           [  7.93948287e-01,   2.06051713e-01],
           [  9.79677545e-01,   2.03224554e-02],
           [  2.55274797e-02,   9.74472520e-01],
           [  6.76726234e-03,   9.93232738e-01],
           [  9.96743892e-01,   3.25610777e-03],
           [  7.75563327e-03,   9.92244367e-01],
           [  9.97680715e-01,   2.31928534e-03],
           [  9.92873225e-01,   7.12677514e-03],
           [  9.83361760e-01,   1.66382404e-02]])




```python
# generate evaluation metrics
print('Accuracy score: ', metrics.accuracy_score(y_test, predicted))
print('ROC AUC: ', metrics.roc_auc_score(y_test, probs[:, 1]))
```

    Accuracy score:  0.966666666667
    ROC AUC:  0.99537037037



```python
metrics.confusion_matrix(y_test, predicted, labels=[1, 0])
```




    array([[11,  1],
           [ 0, 18]])




```python
print('Confusion matrix: \n',
      'TP', 'FN\n', 'FP', 'TN\n',
      metrics.confusion_matrix(y_test, predicted, labels=[1, 0]))
```

    Confusion matrix: 
     TP FN
     FP TN
     [[11  1]
     [ 0 18]]



```python
print(metrics.classification_report(y_test, predicted))
```

                 precision    recall  f1-score   support
    
              0       0.95      1.00      0.97        18
              1       1.00      0.92      0.96        12
    
    avg / total       0.97      0.97      0.97        30
    


Therefore, all 18 FOS- neurons were predicted correctly while 11 of the 12 FOS+ neurons were predicted correctly. You can see the actual sample that was missed in the table below.


```python
pd.DataFrame({'actual': y_test, 'predicted': predicted})
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
      <th>actual</th>
      <th>predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>nm_ui_tn_H3_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nm_ux_ti_E10_141204</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nc_ux_ti_A7_141204</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nm_ui_tn_G1_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nc_ux_tn_A4_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nc_ux_tn_A2_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nm_ux_tn_F4_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nm_ui_tn_H4_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nc_ux_tn_B4_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nc_ui_ti_D7_141204</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nm_ui_ti_H7_141204</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nm_ux_ti_E7_141204</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nm_ui_ti_H10_141204</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nm_ux_ti_F8_141204</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nc_ux_tn_B3_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nm_ux_tn_E4_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nc_ux_tn_A3_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nm_ux_tn_F5_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nm_ux_ti_E8_141204</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nm_ui_tn_H6_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nc_ui_ti_D12_141204</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nm_ui_tn_H1_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nm_ux_tn_E6_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C9_141204</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nm_ux_ti_E12_141204</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nm_ux_tn_E2_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nc_ui_ti_C7_141204</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>nm_ux_tn_F6_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nm_ui_tn_G3_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>nc_ui_tn_C2_141204</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Summary

The goal of this classification project was to determine if we could train an algorithm to predict cellular phenotype based on gene expression. Using a previously published dataset based on activated neurons, I establish proof-of-principle that this is possible. Application of logistic regression leads to an accuracy score of 96.7%, which was just one actual FOS+ neuron falsely called as FOS-. This is pretty good when looking at two different cellular states for this sample size. Possible next steps are to try a different classification algorithm (like SVM) to try and get perfect accuracy, scale up and do more samples, and/or do more cellular phenotypes. Ultimately, this approach has the potential to enhance the value of single-cell applications, by providing phenotypic information in experiments where it would difficult or impossible to experimentally assess.
