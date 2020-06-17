---
title: Iterators in Python
toc: true
categories: [data science, coding]
---


One of the things about Python that I haven't fully appreciated are the use of iterators. I'll go over some iterators that are a part of base Python and then go over more sophisticated applications with the `itertools` package in another post. Like with many new concepts, I am grateful to be able to learn from other online sources which I acknowledge.



```python
# No packages necessary to import!
```

## Basic iterators

(I borrow shamelessly from [w3schools](https://www.w3schools.com/python/python_iterators.asp) since their page is so helpful.)

It is easy to confuse an **iterable** object versus its **iterator** object. Once I made this distinction, it made it easier to understand how some methods worked. Examples of *iterable objects* are lists, dictionaries, and tuples--objects you've likely already used many times. You can get an *iterator* from these objects using the `iter()` method.

Here's an example using a **tuple** as the iterable object.


```python
mytuple = ("apple", "banana", "cherry")
mytupleit = iter(mytuple)    # Generate it's iterator
```


```python
print(mytupleit)    # Note what the iterator object output when print() is called on it
```

    <tuple_iterator object at 0x103531630>



```python
# But the output of next() is a string
print(next(mytupleit))
```

    apple



```python
# ...and it advances to the next item
print(next(mytupleit))
```

    banana



```python
# ...and it advances to the next item
print(next(mytupleit))
```

    cherry



```python
# ...until the iterator is exhausted
print(next(mytupleit))
```


    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-144-979006de96a1> in <module>()
          1 # ...until the iterator is exhausted
    ----> 2 print(next(mytupleit))
    

    StopIteration: 


This is a simple example but it illustrates some interesting behavior. Note that there's no explicit loop which you could do on the *iterable object* (the tuple). Instead we're outputting each element by making a `next` call on the *iterator object* (`myit`). Each `next` call "remembers" where it is in the iterable object. (Interestingly, under the hood, the `for` loop is actually creating an iterator object and using `next` method.)

Here's an example using a **list** as the iterable object.


```python
mylist = ["Winfield", "Gwynn", "Hoffman"]
mylistit = iter(mylist)     # Generate it's iterator
```


```python
# Let's be lazy and print it on one line
print(next(mylistit), next(mylistit), next(mylistit))
```

    Winfield Gwynn Hoffman


If we want to be even more lazy, then we can also simply call `list` on the iterator to get back our original list (the iterable object). Remember the distinction between the iteratOR and the iterABLE objects!


```python
# Regenerate the iterator
mylistit = iter(mylist)
```


```python
# Get back the iterable object from the iterator
list(mylistit)
```




    ['Winfield', 'Gwynn', 'Hoffman']



Here's an example using a **string** as the iterable object.


```python
mystring = "SAN"
mystringit = iter(mystring)     # Generate it's iterator
```


```python
print(next(mystringit), next(mystringit), next(mystringit))
```

    S A N


Here's an example using the **`range`** as the iterable object. Note that the range object itself is a generator-like object.


```python
mynumbers = range(3)
mynumbersit = iter(mynumbers)
```


```python
print(next(mynumbersit), next(mynumbersit), next(mynumbersit))
```

    0 1 2


## Iterator operators `zip` and `map`

Both `zip` and `map` are iterator operators that are built-in Python functions.


```python
# Zip is commonly used to make tuples from two lists but the zip output itself is not a list
zip(mynumbers, mylist)
```




    <zip at 0x103524208>




```python
list(zip(mynumbers, mylist))
```




    [(0, 'Winfield'), (1, 'Gwynn'), (2, 'Hoffman')]



From [this link](https://realpython.com/python-itertools/#what-is-itertools-and-why-should-you-use-it):
"Under the hood, the zip() function works, in essence, by calling iter() on each of its arguments, then advancing each iterator returned by iter() with next() and aggregating the results into tuples. The iterator returned by zip() iterates over these tuples."

Let's look at an example of the `map()` function before discussing how it works.


```python
# Like when invoking zip, the map object itself is not a list
map(len, mylist)
```




    <map at 0x103548438>




```python
list(map(len, mylist))
```




    [8, 5, 7]



What is going on here when `map()` is called? Underneath, an `iter()` object is being first called on `mylist`, advancing with `next()` then applying the first argument (`len()`) to the value returned by `next()` at each step.

## Acknowledgements

Shout outs to the following:

[w3schools](https://www.w3schools.com/python/python_iterators.asp)

[RealPython](https://realpython.com/python-itertools/#what-is-itertools-and-why-should-you-use-it)

