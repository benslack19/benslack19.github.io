--- 
title: 'Creating scatter plots with pandas, matplotlib, and seaborn'
toc: true
toc_label:  'contents'
---

I wanted an easy way to plot scatter plots with marker size and marker color and the legend on the plot. Embarrassingly, I couldn't figure out how to install ggplot for Python so I wrote my own scatter plot. (I eventually figured out how to install it but after I already wrote this.)


```python
# importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from ggplot import *    # requires a pip install
%matplotlib inline
```

## Use a complex sample data frame for a variety of test cases

This is necessary to test the robustness of the visualization properties of the function. In addition to the variety of distributions, zeros, negatives, fractions, and a large range of numbers are represented.


```python
# Alter the number of samples (training examples).
# Use a small m (20) to verify plotting properties and large m (1000) as a test case

m = 30
```


```python
uniform_dist = np.arange(0, m)
gaussian_dist = np.random.normal(loc=0, scale=100, size=m)
bimodal_dist = (np.random.normal(loc=200, scale=100, size=int(m/2)).tolist()
                + np.random.normal(loc=800, scale=100, size=int(m/2)).tolist()) # I just mashed two normal distributions together
lognormal_dist = np.random.lognormal(5, 1, m)
poisson_dist = np.random.poisson(lam=0.3, size=m)
negbinomial_dist = np.random.negative_binomial(2, 0.1, size=m)
chisquare_dist = np.random.chisquare(df=4, size=m)
large_range = np.arange(0, m)**2       # Range of numbers across 6 log orders)
leftskew_dist = 100-np.random.negative_binomial(2, 0.1, size=m)

# Create dataframe with regression features
df = pd.DataFrame({'uniform': uniform_dist,
                  'gaussian': gaussian_dist,
                  'bimodal': bimodal_dist,
                  'lognormal': lognormal_dist,
                  'poisson': poisson_dist,
                   'negbinomial': negbinomial_dist,
                   'chisquare': chisquare_dist,
                   'large_range': large_range,
                   'leftskew_dist': leftskew_dist
                  })

# Create classification features and add to dataframe

k = 5     # number of groups
groupSize = round(m/k)
df['class_even'] = (['group 1']*groupSize + ['group 2']*groupSize
                     + ['group 3']*groupSize + ['group 4']*groupSize
                     + ['group 5']*(m-groupSize*(k-1)))

#                      + ['group 5']*groupSize + ['group 6']*groupSize
#                      + ['group 7']*groupSize + ['group 8']*groupSize
#                      + ['group 9']*groupSize
#                      + ['group 10']*(m-groupSize*(k-1)))

df['class_poisson'] = ['group ' + str(i+1) for i in poisson_dist.tolist()]

```

## g_scatter function


```python
def g_scatter(df, x, y, size, color, ax):
        
    markerSize=df[size]

    # Make a sizes vector and add it to the dataframe (removed at the end of the function)
    szMaxMarker = 200
    df['sizes_gScatter'] = 1+szMaxMarker*((markerSize-markerSize.min())/(markerSize.max()-markerSize.min()))
    

    # Create a plot as an underlayer, serving as a proxy for the legend for marker size
    noLegendPoints = 5  # number of points (default will be 5)

    # legend points, distributed through the size range
    legendPoints_ideal = np.linspace(markerSize.min(),
                                     markerSize.max(),
                                     noLegendPoints)  

    # get dataframe rows that are closest to the legendPoints_ideal
    # dense list comprehension to get the closest values in the dataset that are distributed
    legendPoints_data = [(df[size].iloc[(df[size]-x).abs().argsort().iloc[0]]) for x in legendPoints_ideal]
    
    # find the rows and make a new dataframe; 
    dfTemp = df[df[size].isin(legendPoints_data)]  
    dfTemp2 = dfTemp.sort_values(by='sizes_gScatter')   # sort by size

    # plot this as an underlayer (plotted here, before plotting all data)
    for i, data in dfTemp2.iterrows():
        # while this is an underlayer, set the plot points to white so it's the same as the background
        ax.scatter(data[x], data[y], s=(data['sizes_gScatter']), c='w', label= "%.2f" % data[size])

    # ax.scatter(data[x], data[y], s=markerSize)   #, c='w', label= "%.2f" % data['feature_sz']) #label=str(data['feature_sz']))

    # --- Place the legend which should only show for the underlaying plot ---
    first_legend = ax.legend(title=size, loc='upper left', bbox_to_anchor=(1,1))
    # Manually set the colors of the marker legends to black (the legend points would have been white or non-visible)
    for i in range(len(first_legend.legendHandles)):
         first_legend.legendHandles[i].set_color('black')

    # Add first legend manually to the current Axes.
    plt.gca().add_artist(first_legend)        

    # colors for categorical variables  
    #colors=['gray', 'blue', 'green', 'yellow', 'red']

    # ax.scatter(df['feature_x'], df['feature_y'], s=df['sizes'], color='k', label=None) #, color='k', 

    for i,data in enumerate(df[color].unique()):
        dfTemp = df[df[color]==data]
        ax.scatter(x=dfTemp[x], y=dfTemp[y], s=dfTemp['sizes_gScatter'], c=colors[i], label=None)

    # re-plotting for the purposes of the class legend

    # Using mpatches to add second legend with colors of the groups 
    import matplotlib.patches as mpatches
    groups = df[color].unique().tolist()
    group_patches = list()
    for i,data in enumerate(groups):
        #print(i, data, groups[i])
        group_patches.append(mpatches.Patch(color=colors[i], label=data))

    # changed the handlelength parameter to 0.7 to get square-shaped colored boxes in the legend    
    ax.legend(handles=group_patches, title='group', loc='upper left', bbox_to_anchor=(1,0.5), handlelength=0.7)

```


```python
colors = ['red', 'orange', 'yellow', 'greenyellow', 'green', 'cyan', 'blue', 'magenta', 'purple', 'black']
```

#### Test when zero is present


```python
# Plotting with g_scatter function
f, ax1 = plt.subplots(1,1);
g_scatter(df=df, x='uniform', y='uniform', size='uniform', color='class_even', ax=ax1);
ax1.set_title('g_scatter, uniform distribution');
ax1.set_xlabel('uniform');
ax1.set_ylabel('uniform');
```


![png](Python_Scatterplot_by_Ben_9_0.png)



```python
from ggplot import *
```

    /Users/lacar/anaconda/lib/python3.5/site-packages/ggplot/utils.py:81: FutureWarning: pandas.tslib is deprecated and will be removed in a future version.
    You can access Timestamp as pandas.Timestamp
      pd.tslib.Timestamp,
    /Users/lacar/anaconda/lib/python3.5/site-packages/ggplot/stats/smoothers.py:4: FutureWarning: The pandas.lib module is deprecated and will be removed in a future version. These are private functions and can be accessed from pandas._libs.lib instead
      from pandas.lib import Timestamp
    /Users/lacar/anaconda/lib/python3.5/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools



```python
# Plotting with Python ggplot
f, (ax1) = plt.subplots(1,1)
ggplot(df, aes(x='uniform', y='uniform', size='uniform', color='class_even')) + geom_point()
```


![png](Python_Scatterplot_by_Ben_11_0.png)





    <ggplot: (290789878)>



#### Test when negative numbers, fractions are present


```python
# Plotting with g_scatter function

colors = ['red', 'orange', 'yellow', 'greenyellow', 'green', 'cyan', 'blue', 'magenta', 'purple', 'black']

f, ax1 = plt.subplots(1,1);
g_scatter(df=df, x='uniform', y='gaussian', size='gaussian', color='class_even', ax=ax1);
ax1.set_title('g_scatter, marker size by gaussian feature');
ax1.set_xlabel('uniform')
ax1.set_ylabel('gaussian')
```




    <matplotlib.text.Text at 0x103f16780>




![png](Python_Scatterplot_by_Ben_13_1.png)



```python
# Plotting with Python ggplot
f, (ax1) = plt.subplots(1,1)
ggplot(df, aes(x='uniform', y='gaussian', size='gaussian', color='class_even')) + geom_point()
```


![png](Python_Scatterplot_by_Ben_14_0.png)





    <ggplot: (291414911)>



#### Test with negative binomial distribution and all values are positiive


```python
f, ax1 = plt.subplots(1,1);
g_scatter(df=df, x='uniform', y='negbinomial', size='negbinomial', color='class_even', ax=ax1);
ax1.set_title('g_scatter, negative binomial distribution');
```


![png](Python_Scatterplot_by_Ben_16_0.png)



```python
# Plotting with Python ggplot
f, (ax1) = plt.subplots(1,1)
ggplot(df, aes(x='uniform', y='uniform', size='negbinomial', color='class_even')) + geom_point()
```


![png](Python_Scatterplot_by_Ben_17_0.png)





    <ggplot: (294229466)>


