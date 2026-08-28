---
title: "Cross-entropy made easy with PyTorch"
mathjax: true
toc: true
toc_sticky: true
categories: [data science, statistics]
---

This is a side-bar post in a series about transformers. Cross-entropy is the loss function used in transformer models.



```python
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import torch
from torch.nn import functional as F

torch.manual_seed(1337)
```




    <torch._C.Generator at 0xfffe874450b0>



Let's look at how cross-entropy is calculated with `scipy` and with values in a simple numpy array before seeing how PyTorch expedites this calculation for us.

First, cross-entropy is a comparison of two probability distributions ($p$ and $q$), where $p$ serves as the target distribution and $q$ is the model's distribution at any point in training. (Note that this might be counter-intuitive as `p` does not line up with "predicted" distribution but this is the convention in scipy and pytorch so we'll stick with it.) The target distribution is what we hope the model will learn. Technically, cross-entropy can be applied to any probability distribution but in most machine learning instances, the probability distribution of the target is a 1 for correct class and 0 for all other classes. (We'll return to this point when discussing negative log likelihood.)

 The mathematical formula for cross-entropy is:

$ H(p,q) = - \sum_{i=1}^{n} p(x_i) \log q(x_i)$

The first value of $p$ will be the reference for the first value of $q$, the second value of $p$ will be reference for second value of $q$, etc. Let's see this equation play out in a dummy example. Imagine that there are two multiple choice questions, each has three choices (a, b, or c). There is only one right answer for each question. A model has already determined logits for both questions.


```python
logits_example = np.array(
    [
        [1.5, 0.2, 3.1],  # index 2 (or choice c) is highest at 3.1
        [2.0, 0.5, 0.1],  # index 0 (or choice a) is highest at 2.0
    ]
)
```

However, as mentioned above in the cross-entropy formula, we don't use logits directly. We get probabilities from logits using softmax. In the `scipy` function, the softmax function allows us to get probabilities where each row (each multiple choice question) will sum to 1.


```python
# need axis since each example is independent
prob_from_logits_example = softmax(logits_example, axis=1)
prob_from_logits_example
```




    array([[0.16062801, 0.04377624, 0.79559575],
           [0.72849194, 0.16254852, 0.10895953]])



We now have two examples of probability distributions $q$ that we want to assess. Now we need our target distributions $p$ for each. The correct answers are represented in a different array.  The correct answer for the first question is 'c' (index 2) while it is 'b' for the second question (index 1). We then assign a `1` at the index for the correct answers and 0 in the other two positions.


```python
prob_from_targets_example = np.array([[0, 0, 1], [0, 1, 0]])
```

# Cross-entropy in a loop

Let's translate the above mathematical formula into code, using the first example in a loop.


```python
# Extract the single example vectors first
p_vector = prob_from_targets_example[0]
q_vector = prob_from_logits_example[0]

cross_entropy = 0
for p, q in zip(p_vector, q_vector):
    cross_entropy += p * np.log(q)

# Apply the negative sign at the end to get the final cross-entropy loss
cross_entropy = -cross_entropy
print(
    f"Cross-entropy loss for the first example (manually calculated): {cross_entropy}"
)
```

    Cross-entropy loss for the first example (manually calculated): 0.22866407558146834


# Cross-entropy with scipy

Let's see what we get when we use the `entropy` function from `scipy` to calculate cross-entropy.


```python
cross_entropy_scipy = entropy(pk=prob_from_targets_example, axis=1) + entropy(
    pk=prob_from_targets_example, qk=prob_from_logits_example, axis=1
)  # This line is getting both examples
print(
    f"Cross-entropy loss for the first example (using scipy function): {cross_entropy_scipy[0]}"
)

```

    Cross-entropy loss for the first example (using scipy function): 0.22866407558146834



```python
assert cross_entropy == cross_entropy_scipy[0]
```

# Cross-entropy with PyTorch

Now let's look at how PyTorch makes this calculation much easier for us. First, we need to make our numpy array a tensor. Here is the [documentation](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.cross_entropy.html) for details.


```python
logits_example_tensor = torch.from_numpy(logits_example)
logits_example_tensor
```




    tensor([[1.5000, 0.2000, 3.1000],
            [2.0000, 0.5000, 0.1000]], dtype=torch.float64)



Then, we need to make our target array but here, only the *index* of the correct answer is needed. PyTorch one-hot encodes to a probability distribution underneath the hood for us. It has the capability to use probabilities as well if needed.


```python
targets_example = torch.tensor(
    [2, 1]
)  # meaning: correct answer is 'c', then correct answer is 'b'
targets_example
```




    tensor([2, 1])



Now, we can use PyTorch from the `torch.nn.functional` module to compute cross-entropy. One point to be aware of: the PyTorch function averages across each row by default yielding a single scalar value. To get the cross-entropy for each row, add `reduction='none'`.


```python
cross_entropy_pytorch = F.cross_entropy(
    torch.from_numpy(logits_example), targets_example, reduction="none"
)
cross_entropy_pytorch
```




    tensor([0.2287, 1.8168], dtype=torch.float64)




```python
# interestingly this fails `assert cross_entropy_scipy[0]==cross_entropy_pytorch[0]`
assert np.isclose(cross_entropy_scipy[0], cross_entropy_pytorch[0])
```

# Shape considerations when data has another dimension

One case we need to consider is if our dataset can be more complicated. Let's imagine that instead of just two questions, we have multiple (say n=4) students addressing those two questions. We'd like to capture the cross-entropy loss across all students and all questions.


```python
logits_example_with_students_tensor = torch.randn(
    2, 4, 3
)  # 2 questions, 4 students, 3 multiple choice answers
logits_example_with_students_tensor
```




    tensor([[[ 0.1808, -0.0700, -0.3596],
             [-0.9152,  0.6258,  0.0255],
             [ 0.9545,  0.0643, -0.0476],
             [-1.0996, -1.7524, -1.0971]],
    
            [[-1.1081, -1.8002, -0.4713],
             [ 0.0084,  0.1662,  1.2055],
             [ 0.1883, -2.1600, -0.1585],
             [-0.6300, -0.2221,  0.6924]]])



 To be clear, our tensor is arranged so that the first question is arranged first with all four responses for each question followed by the second question in the same arrangement. Our targets don't have to change because it is the same answers for each question and for each student.


```python
targets_example
```




    tensor([2, 1])



How would we calculate the loss in PyTorch? At first glance, you might think that our tensors are ready to pass in to `F.cross_entropy`...


```python
F.cross_entropy(logits_example_with_students_tensor, targets_example, reduction="none")
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[58], line 1
    ----> 1 F.cross_entropy(logits_example_with_students_tensor, targets_example, reduction="none")


    File /usr/local/lib/python3.13/site-packages/torch/nn/functional.py:3507, in cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing)
       3505 if size_average is not None or reduce is not None:
       3506     reduction = _Reduction.legacy_get_string(size_average, reduce)
    -> 3507 return torch._C._nn.cross_entropy_loss(
       3508     input,
       3509     target,
       3510     weight,
       3511     # pyrefly: ignore [bad-argument-type]
       3512     _Reduction.get_enum(reduction),
       3513     ignore_index,
       3514     label_smoothing,
       3515 )


    RuntimeError: Expected target size [2, 3], got [2]


...but we get an error. Per the [docs](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.cross_entropy.html), PyTorch expects us to shape the data in a particular way. You might have anticipated this since the shapes of the inputs and the targets are quite different.

How should we reshape it? It depends on what we want to do with the loss. In most cases, you just want to reduce the loss to a single value (across all students and questions.) But you may want to look at the loss in a more granular fashion. Perhaps you want to see how each student is performing or you may want to see how each *question* is performing. Let's see how we'd arrange the data in these scenarios.

## Cross-entropy loss for each student


```python
N_q, N_s, N_c = logits_example_with_students_tensor.shape
print(f"No. of questions: {N_q}, No. of students: {N_s}, No. of choices: {N_c}")

# 2 questions, 4 students, 3 multiple choice answers
display(logits_example_with_students_tensor)
```

    No. of questions: 2, No. of students: 4, No. of choices: 3



    tensor([[[ 0.1808, -0.0700, -0.3596],
             [-0.9152,  0.6258,  0.0255],
             [ 0.9545,  0.0643, -0.0476],
             [-1.0996, -1.7524, -1.0971]],
    
            [[-1.1081, -1.8002, -0.4713],
             [ 0.0084,  0.1662,  1.2055],
             [ 0.1883, -2.1600, -0.1585],
             [-0.6300, -0.2221,  0.6924]]])



```python
logits_example_each_student_tensor = logits_example_with_students_tensor.view(
    N_s * N_q, N_c
)  # shape: (N_s, N_q, N_c)
print(f"Combine questions. Date shape is: {logits_example_each_student_tensor.shape}")

display(logits_example_each_student_tensor)
```

    Combine questions. Date shape is: torch.Size([8, 3])



    tensor([[ 0.1808, -0.0700, -0.3596],
            [-0.9152,  0.6258,  0.0255],
            [ 0.9545,  0.0643, -0.0476],
            [-1.0996, -1.7524, -1.0971],
            [-1.1081, -1.8002, -0.4713],
            [ 0.0084,  0.1662,  1.2055],
            [ 0.1883, -2.1600, -0.1585],
            [-0.6300, -0.2221,  0.6924]])


Then arrange the targets so that cross-entropy can be calculated for each.


```python
targets_example_each_student_tensor = targets_example.repeat(N_s)
display(targets_example_each_student_tensor)
```


    tensor([2, 1, 2, 1, 2, 1, 2, 1])



```python
# most granular: for each student and question
cross_entropy_each_student_and_question = F.cross_entropy(
    logits_example_each_student_tensor,
    targets_example_each_student_tensor,
    reduction="none",
)

display(cross_entropy_each_student_and_question)
```


    tensor([1.3994, 0.5669, 1.5774, 1.5783, 0.5843, 1.5436, 0.9360, 1.4257])


You can then use this to get averages for each student or each question.

# Why is cross-entropy also known as negative log likelihood?

As hinted above, cross-entropy simplifies to negative log likelihood when the true distribution `p` is one-hot encoded, meaning there's a single correct answer and no probability of any other answer in a multi-class distribution.

Let's revisit the definition of cross-entropy.

$ H(p,q) = - \sum_{i=1}^{n} p(x_i) \log q(x_i)$

Let's use the first row of `logits_example_tensor` as our `q`, our hypothetical predicted distribution and see the corresponding true (target) distribution in `p` which is one-hot encoded. 


```python
print(f"q (predicted) distribution: {prob_from_logits_example[0, :]}")
print(f"p (true) distribution: {prob_from_targets_example[0, :]}")
```

    q (predicted) distribution: [0.16062801 0.04377624 0.79559575]
    p (true) distribution: [0 0 1]



```python
for i, (p_i, q_i) in enumerate(
    zip(prob_from_targets_example[0, :], prob_from_logits_example[0, :])
):
    print(f"Cross-entropy contribution of index {i}: {p_i * np.log(q_i)}")
```

    Cross-entropy contribution of index 0: -0.0
    Cross-entropy contribution of index 1: -0.0
    Cross-entropy contribution of index 2: -0.22866407558146834


From the cross-entropy formula, you can see that the one-hot encoding in the target distribution simplifies a lot of the work for us. In equation form, the cross-entropy simplifies to:

$H(p, q) = - \log q(x_{ci})$

where $ci$ indicates the index of the correct value, as indicated by the target vector. You might already recognize this as the negative log likelihood. How would this look if we arrive at this equation from the perspective of likelihood?

Likelihood $L$ in this scenario means looking at the product of the individual probabilities for each class that the model predicted. Those probabilities are given by $q$. We can ignore the true distribution $p$ for now:

$L = \Pi_{i=1}^{n} q(x_i)$

Log likelihood means taking the sum of the log of individual probabilities, such that the the output is more computationally stable:

$\log L = \Sigma_{i=1}^{n} \log q(x_i)$

Ultimately, our goal is to ensure the predicted data aligns with the truth data which is maximizing likelihood (or log likelihood). Then what's the rationale for making it negative? In machine learning frameworks, we are traditionally concerned with minimizing errors. Taking the log of probabilities (values less than $\leq$ 1) are inherently negative. When we take the whole log likelihood quantity and throw a negative sign in front of it, the net effect is turning it into a positive value. And therefore we can now use NLL in our model training since our objective is now to *minimize* it.

$ -\log L = NLL = -\Sigma_{i=1}^{n} \log q(x_i)$

The above equation is for all classes the model predicts, but if we just care about the correct class or index, which is our goal in model training, then we arrive at how cross-entropy and NLL align:

$ NLL_{ci} = - \log q(x_{ci}) = \text{cross entropy} $


```python
nll_value = -np.log(prob_from_logits_example[0, 2])
ce_value = entropy(
    pk=prob_from_targets_example[0, :], qk=prob_from_logits_example[0, :]
)
print(f"Negative log-likelihood value: {nll_value}")
print(f"Cross entropy value (via scipy): {ce_value}")
# use scipy function
assert nll_value == ce_value
```

    Negative log-likelihood value: 0.22866407558146834
    Cross entropy value (via scipy): 0.22866407558146834


Note: Around the time I was writing this, [3Blue1Brown came out with an amazing video](https://www.youtube.com/watch?v=GlYgs6v2YfU) connecting compression to cross-entropy.


```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Wed, 24 Jun 2026
    
    Python implementation: CPython
    Python version       : 3.13.14
    IPython version      : 9.14.1
    
    numpy: 2.4.6
    scipy: 1.17.1
    torch: 2.12.0
    
    Watermark: 2.6.0
    

