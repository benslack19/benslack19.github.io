---
title: "Weird ways that covariance matrices are made"
mathjax: True
toc: true
toc_sticky: true
categories: [data science, statistics]
---

Covariance priors for multivariate normal models are an important tool for the implementation of varying effects. By representing more than one parameter with a covarying structure, even more partial pooling can result than if the parameters had their own separate distribution. Before talking more about varying effects, I thought I'd write about the weird ways that covariance matrixes are made.


```python
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as stats
import scipy.linalg as linalg
import seaborn as sns

%load_ext nb_black
%config InlineBackend.figure_format = 'retina'
%load_ext watermark
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
az.rcParams["stats.hdi_prob"] = 0.89  # sets default credible interval used by arviz

def standardize(x):
    x = (x - np.mean(x)) / np.std(x)
    return x
```




What is a a covariance matrix? One way to think of it is through an analogy: a standard deviation is to a univariate normal distribution as a covariate matrix is to a multivariate normal distribution.

In equation form, you could have variables that look like this:

$$x \sim \text{Normal}(\mu, \sigma) \tag{univariate normal distribution}$$

$$\begin{bmatrix}x_1 \\ x_2 \\ ... \\ x_n \end{bmatrix} \sim \text{MVNormal} \left( \begin{bmatrix} \mu_1 \\ \mu_2 \\ ... \\ \mu_n \end{bmatrix} , \Sigma \right) \tag{multivariate normal distribution}$$

In both cases, we have variables paramaterized by random distributions. In the univariate case, a single draw from the distribution will result in one value. In the multivariate case, a single draw will result in *n* values, one for each parameter. In the multivariate normal (MVN) case, we have a vector of means ($\mu$), but the interesting relationships will result from the covariance matrix $\Sigma$.. It will tell us about the variability of the parameters and also possible correlative relationships between them. This is seen in how we can construct covariance matrices.

Using numbers helps me understand things so let's use Dr. McElreath's example involving cafe waiting times. For the purposes of this post, you don't need to know the details of the problem, but it is described in [this lecture](https://www.youtube.com/watch?v=yfXpjmWgyXU&list=PLDcUM9US4XdNM4Edgs7weiyIguLSToZRI&index=17&t=484s).

The multivariate normal distribution for this cafe waiting times example is described here:

$$\begin{bmatrix}\alpha_{\text{cafe}} \\ \beta_{\text{cafe}} \end{bmatrix} \sim \text{MVNormal} \left( \begin{bmatrix}\alpha \\ \beta \end{bmatrix} , \textbf{S} \right)  \tag{population of varying effects}$$

We'll create a simple 2x2 covariance matrix but the lessons can be extended to larger sizes. To construct it, we'll need values for each parameter's standard deviation (what I'll call $\sigma$ below) and a correlation coefficient $\rho$. For a proper multivariate normal distribution, we'll also need values for the means (the $\mu$ vector described above), denoted as *a* and *b*.


```python
a = 3.5  # average morning wait time
b = -1.0  # average difference afternoon wait time
sigma_a = 1.0  # std dev in intercepts
sigma_b = 0.5  # std dev in slopes
rho = -0.7  # correlation between intercepts and slopes
```




While our focus is on the covariance matrix, let's get the first term of the MVN distribution out of the way. I'll generate the vector of the averages which is straightforward.


```python
Mu = [a, b]
print("Vector of means: ", Mu)
```

    Vector of means:  [3.5, -1.0]





# Intuitive construction

The first method can be made is the most intuitive for me.

$$ \textbf{S} = \begin{pmatrix} \sigma_{\alpha}^2 & \rho\sigma_{\alpha}\sigma_{\beta} \\ \rho\sigma_{\alpha}\sigma_{\beta} & \sigma_{\beta}^2 \end{pmatrix} $$

The diagonals show each individual parameter's variance (standard deviation squared) while the off-diagonal shows the co-variance, represented as the correlation coefficient $\rho$ multiplied by the parameters' standard deviations.

I'll use `Sigma1` with capital S to represent this covariance matrix with the `1` representing this first method of assembly but as you'll see, they will be equivalent. (In equations like the one shown above, the covariance matrix is represented by a bold, capital S.)


```python
cov_ab = rho * sigma_a * sigma_b
Sigma1 = np.array([[sigma_a**2, cov_ab], [cov_ab, sigma_b**2]])
Sigma1
```




    array([[ 1.  , -0.35],
           [-0.35,  0.25]])






The important parts are the off-diagonals, which shows a negative covariance between the $\alpha$ and $\beta$ terms. They are symmetric because the calculation is equivalent. Hopefully there's no confusion in how this covariance matrix resulted.

# Standard deviation diagonals

The second method for building the covariance matrix will be weirder:
- arrange the standard deviations along the diagonal and fill in zeros everywhere else
- matrix multiply by a *correlation* matrix
- matrix multiply by the same arrangement of standard deviations along the diagonal

Here's how it looks in equation form:

$$ \textbf{S} = \begin{pmatrix} \sigma_{\alpha} & 0 \\ 0 & \sigma_{\beta} \end{pmatrix} \textbf{R} \begin{pmatrix} \sigma_{\alpha} & 0 \\ 0 & \sigma_{\beta} \end{pmatrix}  $$



To create a matrix where the standard deviations are on the diagonal and zeros are everywhere, we can use a handy `numpy` function called diag that can be applied to the parameter standard deviations arranged in a vector:


```python
# put the sigmas in a vector first
sigmas = [sigma_a, sigma_b]

# represent on the diagonal
sigma_diag = np.diag(sigmas)
sigma_diag
```




    array([[1. , 0. ],
           [0. , 0.5]])






The $\textbf{R}$ matrix is where `rho` is arranged in the off-diagonals, where `rho` represents the correlation between the two parameters. The diagonals show values of 1 since each parameter will always be perfectly correlated with itself.


```python
Rmat = np.array([[1, rho], [rho, 1]])
Rmat
```




    array([[ 1. , -0.7],
           [-0.7,  1. ]])






Now the final step is the matrix multiplication. In `numpy`, you can do this with a small chain of matrix multiplication (taken from [this SO post](https://stackoverflow.com/questions/11838352/multiply-several-matrices-in-numpy)).


```python
Sigma2 = sigma_diag.dot(Rmat).dot(sigma_diag)
Sigma2
```




    array([[ 1.  , -0.35],
           [-0.35,  0.25]])






As expected, we get the same values of the covariance matrix as we did with the previous method.

# Cholesky factors

Ok, now we have the third method of creating a covariance matrix. As promised, it gets even more weird. It deserves its own exploration but I'll just show how it works now then explain later. The first thing we need to do is get the Cholesky factor which can be derived from the $\textbf{R}$ correlation matrix. There are other sources that explain Cholesky factors like [the Wikipedia page](https://en.wikipedia.org/wiki/Cholesky_decomposition).

The matrix $\textbf{R}$ can be derived from this Cholesky factor with the following equation:

$ \textbf{R} = \textbf{LL}^\intercal $

Accordingly, we can substitute for $\textbf{R}$ in the equation we saw above:

$$ \textbf{S} = \begin{pmatrix} \sigma_{\alpha} & 0 \\ 0 & \sigma_{\beta} \end{pmatrix} \textbf{LL}^\intercal  \begin{pmatrix} \sigma_{\alpha} & 0 \\ 0 & \sigma_{\beta} \end{pmatrix}  $$

$\textbf{L}$ is **not** simply the lower triangle simply of a correlation matrix.


```python
# WRONG - this is not how to get L
np.tril(Rmat)
```




    array([[ 1. ,  0. ],
           [-0.7,  1. ]])






There is a different `numpy` function that calculates the lower triangle properly. (Note that `scipy.linalg.cholesky` does the upper triangle. You'd modify the above equation by transposing L first then mutiplying by itself.)


```python
# numpy.linalg.cholesky does the lower triangle
L = np.linalg.cholesky(Rmat)
L
```




    array([[ 1.        ,  0.        ],
           [-0.7       ,  0.71414284]])






In code, we can get this third re-construction of $\textbf{S}$ like this:


```python
Sigma3 = sigma_diag.dot(L).dot(L.T).dot(sigma_diag)
Sigma3
```




    array([[ 1.  , -0.35],
           [-0.35,  0.25]])






As we would expect, all three ways to get a covariance matrix give equivalent results. Why would you even use this last, strange way? It will have to do with sampling in a varying effects problem. The Cholesky factors will allow us to generate non-centered paramaterizations. I'll cover this in a later post.


```python
%watermark -n -u -v -iv -w
```

    Last updated: Mon Mar 28 2022
    
    Python implementation: CPython
    Python version       : 3.8.6
    IPython version      : 7.20.0
    
    sys       : 3.8.6 | packaged by conda-forge | (default, Jan 25 2021, 23:22:12) 
    [Clang 11.0.1 ]
    matplotlib: 3.3.4
    pandas    : 1.2.1
    pymc3     : 3.11.0
    arviz     : 0.11.1
    scipy     : 1.6.0
    seaborn   : 0.11.1
    numpy     : 1.20.1
    
    Watermark: 2.1.0
    




