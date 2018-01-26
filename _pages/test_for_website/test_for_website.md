
## Test what this does


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
x = [0,1,2,3,4]
y = [3,4,5,6,7]
z = ['a','b','c','d','e']
df = pd.DataFrame(x,y)
```


```python
plt.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x11cdf4be0>]




![png](output_3_1.png)

