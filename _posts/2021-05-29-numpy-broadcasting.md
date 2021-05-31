---
title: "Vectorization with `np.dot` and broadcasting"
mathjax: true
toc: true
toc_sticky: true
categories: [coding, linear algebra]
---

Vectorization and broadcasting are tricks I have used sparingly and absent-mindedly if at all. However, it is a critical skill for algorithmic code to run efficiently, particularly for deep learning networks. While the use of coding libraries have already implemented vectorization and broadcasting, it's good to know how this works. This post was inspired by a lesson in [Andrew Ng's deep learning course](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome).


```python
import numpy as np
```

# Multiplication of two vectors

We don't need neural network equations to understand the usefulness of vectorization and broadcasting. We'll just use some synthetic examples that mimic some of the calculation steps in neural network and logistic regression gradient descent. One step may require element-wise multiplication of two vectors and then summing as illustrated in this equation.

$$k = \sum_{i=1}^{m}y^{(i)}a^{(i)}$$

Both $y$ is a vector and $a$ are vectors of equal length and $m$ represents the length of either vector. Let's create some values to make some computations a little more concrete. The output $k$ will take some scalar value (a real number) as you will see. (Note that in the `np.random.randint` call, I'm stating a `size` parameter of 3 so that the vectors have three values each, but there is a wrinkle about this which I'll discuss in the broadcasting section.)


```python
y = np.random.randint(low=0, high=5, size=3)
a = np.random.randint(low=0, high=5, size=3)
m = len(a)   # this can also be len(y)
print("y: ", y)
print("a: ", a)
print("m: ", m)
```

    y:  [0 4 3]
    a:  [0 3 4]
    m:  3


A naive approach to calculating $k$ would be to run a for-loop like this. I'll show the element-wise multiplication at the different positions to make the step-by-step calculation clear.


```python
k = 0
for i in range(len(y)):
    print("Element-wise multiplication at position " + str(i) + ": ", y[i] * a[i])
    k += y[i] * a[i]

print("k (using a for-loop): ", k)    
```

    Element-wise multiplication at position 0:  0
    Element-wise multiplication at position 1:  12
    Element-wise multiplication at position 2:  12
    k (using a for-loop):  24


Now let's look at the vectorized implementation, which does not use an explicit for-loop. Instead we can use the [dot product](https://mathinsight.org/dot_product_examples) of the two vectors as a substitute for the for-loop. You can see what I mean by this by passing the two vectors into [numpy's dot product function](https://numpy.org/doc/stable/reference/generated/numpy.dot.html).


```python
print("k (using dot product): ", np.dot(y, a))
```

    k (using dot product):  24


With `np.dot` we get the same answer, and of course, it is much easier to code.

# Broadcasting

Numpy's [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) capabilities also allow for simpler coding. It essentially allows a variable to expand into the dimensions of another variable when a calculation is performed. Here is an example where we instantiate $b$ a scalar value and add that to $y$, the vector we created above.


```python
b_scalar = 3
```


```python
print("y.shape: ", y.shape)
```

    y.shape:  (3,)



```python
y + b_scalar
```




    array([3, 7, 6])



The addition automatically expands `b_scalar` into a vector without us having to explicitly write this out. It is automatically performing this:


```python
b_vector = np.array([3,3,3])
y + b_vector
```




    array([3, 7, 6])



The broadcasting property provide convenience in coding, but it can lead to some unexpected consequences if we are not careful as we will see later below. First, let's go back to the original problem with our two vectors.

# Respecting the dimensions of the vectors

I mentioned the `size` parameter of the `np.random.randint` call when I instantiated the $y$ and $a$ vectors. If we check the `shape` of the vector, we see that it's shown this way:


```python
print("y.shape: ", y.shape)
print("a.shape: ", a.shape)
```

    y.shape:  (3,)
    a.shape:  (3,)


But in a mathematical sense, the calculation of the dot product, requires that the vectors be in such a way that the number of columns in the first vector is equal to the number of rows in the second vector. It would be written out in equation form like this.

$$ k = (y_1\ y_2\ y_3)\left( \begin{array}{cc}
a_1 \\ a_2 \\ a_3 \end{array} \right)$$

However, when the `shape` of a vector is ambiguous (for example as shown as `(3,)`), `np.dot` is smart enough to understand the calculation. You can see that the call "breaks" when we explicitly state the wrong dimensions of the vector using `.reshape`.


```python
print("Both vectors as a single row: ", np.dot(y.reshape(1, 3), a.reshape(1, 3)))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-10-bfc5d7be5bc9> in <module>()
    ----> 1 print("Both vectors as a single row: ", np.dot(y.reshape(1, 3), a.reshape(1, 3)))
    

    ValueError: shapes (1,3) and (1,3) not aligned: 3 (dim 1) != 1 (dim 0)


It is good practice to use `.reshape` to state the desired dimensions of the vectors. (Even better, one can use the `size` parameter during instantiation of the vector.)


```python
print("Correct dimensions: ", np.dot(y.reshape(1, 3), a.reshape(3, 1)))
```

    Correct dimensions:  [[24]]


It is not enough to rely on the lack of error message to ensure that we are writing our code correctly. Because of broadcasting, one can write the first vector as a column and the second as a row. Because the number of columns of the first vector matches the number of rows of the second (both are 1), `np.dot` sees this as a legitimate calculation and a matrix is the output.


```python
print("First vector as a column, second vector as a row: \n", np.dot(y.reshape(3, 1), a.reshape(1, 3)))
```

    First vector as a column, second vector as a row: 
     [[ 0  0  0]
     [ 0 12 16]
     [ 0  9 12]]


Of course, there *are* cases where you may want to take advantage of broadcasting this way, such as in other calculations of logistic regression or neural network algorithms.

# Summary

Here, I've shown the usefulness of vectorization, as illustrated by `np.dot`, as well as the advantages and potential pitfalls of broadcasting. Paying attention to vector and matrix dimensions can go a long way towards resolving bugs and avoiding unintended results. These tricks can be used to optimize code and have it run efficiently in machine learning algorithms.
