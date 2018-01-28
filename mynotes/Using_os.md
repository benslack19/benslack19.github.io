
**Using os.system and interpreting error messages**

[StackOverflow discussion](https://stackoverflow.com/questions/31262614/handle-result-of-os-system)


```python
import os
```


```python
os.getcwd()
```


```python
path = '/Users/mypath'
os.chdir(path)
```


```python
# If you can't find meaning of error messages, do this
from subprocess import check_output
out = check_output('system terminal command', shell=True, stderr=subprocess.STDOUT)
```
