---
title: "Working with PyTorch's Dataset and Dataloader classes (part 1)"
mathjax: true
toc: true
toc_sticky: true
categories: [data science, statistics]
---

Recently, I built a simple NLP algorithm for a work project, following the template described in [this tutorial](https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#sphx-glr-beginner-nlp-deep-learning-tutorial-py). As I looked to increase my model's complexity, I started to come across references to [Dataset and Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) classes. I tried adapting my work-related code to use these objects, but I found myself running into [pesky bugs](https://media.giphy.com/media/xwEVCKetQWpeYyumJJ/giphy.gif). I thought I should take some time to figure out how to properly use `Dataset` and `Dataloader` objects. In this post, I adapt the PyTorch NLP tutorial to work with `Dataset` and `Dataloader` objects. Since my focus is primarily on using these objects, please refer to the [tutorial](https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#sphx-glr-beginner-nlp-deep-learning-tutorial-py) for details regarding the NLP model.


```python
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1)
```




    <torch._C.Generator at 0x7fef88a746f0>




```python
%load_ext nb_black
%config InlineBackend.figure_format = 'retina'
%load_ext watermark
```





```python
# Figure aesthetics
sns.set_theme()
sns.set_context("talk")
sns.set_style("white")
```




# First attempt

The tutorial generates a simple dataset to use for a logistic regression bag-of-words classifier. It takes a sentence and trains whether the sentence is in English or Spanish. The data was structured originally so each sample was a list.


```python
train_data = [
    ("me gusta comer en la cafeteria".split(), "SPANISH"),
    ("Give it to me".split(), "ENGLISH"),
    ("No creo que sea una buena idea".split(), "SPANISH"),
    ("No it is not a good idea to get lost at sea".split(), "ENGLISH"),
]

test_data = [
    ("Yo creo que si".split(), "SPANISH"),
    ("it is lost on me".split(), "ENGLISH"),
]
```




Before putting the data into the `Dataset` object, I'll organize it into a dataframe for easier input.


```python
# Combine so we have one data object
data = train_data + test_data

# Put into a dataframe
df_data = pd.DataFrame(data)
df_data.columns = ["words", "labels"]
df_data
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
      <th>words</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[me, gusta, comer, en, la, cafeteria]</td>
      <td>SPANISH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Give, it, to, me]</td>
      <td>ENGLISH</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[No, creo, que, sea, una, buena, idea]</td>
      <td>SPANISH</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[No, it, is, not, a, good, idea, to, get, lost...</td>
      <td>ENGLISH</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Yo, creo, que, si]</td>
      <td>SPANISH</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[it, is, lost, on, me]</td>
      <td>ENGLISH</td>
    </tr>
  </tbody>
</table>
</div>






## Putting the data in `Dataset` and output with `Dataloader`

Now it is time to put the data into a `Dataset` object. I referred to [PyTorch's tutorial on datasets and dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#) and [this helpful example specific to custom text](https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00), especially for making my own dataset class, which is shown here. 


```python
class TextDataset(Dataset):
    """
    Characterizes the pre-processed SRF custom dataset for PyTorch
    """

    def __init__(self, ids, text, labels):
        """
        Initialization. Ids can be useful after splitting the dataset.
        """
        self.ids = ids
        self.text = text
        self.labels = labels

    def __len__(self):
        """
        This is simply the number of labels in the dataseta.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Generate one sample of data
        """
        label = self.labels[idx]
        text = self.text[idx]
        sample = {"Text": text, "Label": label}
        return sample
```





```python
# Put train and test into dataset objects
train_ids = range(0, 4)
test_ids = range(4, 6)

train_DS1 = TextDataset(
    train_ids,
    df_data.loc[train_ids, "words"].tolist(),
    df_data.loc[train_ids, "labels"].tolist(),
)

test_DS1 = TextDataset(
    train_ids,
    df_data.loc[test_ids, "words"].tolist(),
    df_data.loc[test_ids, "labels"].tolist(),
)
```




When putting the data into their respective dataset objects, it is important to use the `.tolist()` method or else `DataLoader` will return an error when retrieving the data. Now let's use `DataLoader` and a simple for loop to return the values of the data. I'll use only the training data and a `batch_size` of 1 for this purpose.


```python
train_DL = DataLoader(train_DS1, batch_size=1, shuffle=False)

print("Batch size of 1")
for (idx, batch) in enumerate(train_DL):  # Print the 'text' data of the batch

    print(idx, "Text data: ", batch["Text"])  # Print the 'class' data of batch
    print(idx, "Label data: ", batch["Label"])
```

    Batch size of 1
    0 Text data:  [('me',), ('gusta',), ('comer',), ('en',), ('la',), ('cafeteria',)]
    0 Label data:  ['SPANISH']
    1 Text data:  [('Give',), ('it',), ('to',), ('me',)]
    1 Label data:  ['ENGLISH']
    2 Text data:  [('No',), ('creo',), ('que',), ('sea',), ('una',), ('buena',), ('idea',)]
    2 Label data:  ['SPANISH']
    3 Text data:  [('No',), ('it',), ('is',), ('not',), ('a',), ('good',), ('idea',), ('to',), ('get',), ('lost',), ('at',), ('sea',)]
    3 Label data:  ['ENGLISH']





At first glance, things might look okay but the eagle-eyed will notice that each element in our list is now wrapped as one element. If we increase `batch_size` to 2, we get an [ugly error](https://media.giphy.com/media/eGNtzon6aSbPA6qgU4/giphy.gif).


```python
train_DL2 = DataLoader(train_DS1, batch_size=2, shuffle=False)

print("Batch size of 2")
for (idx, batch) in enumerate(train_DL2):  # Print the 'text' data of the batch

    print(idx, "Text data: ", batch["Text"])  # Print the 'class' data of batch
    print(idx, "Label data: ", batch["Label"], "\n")
```

    Batch size of 2



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-9-b81921277760> in <module>
          2 
          3 print("Batch size of 2")
    ----> 4 for (idx, batch) in enumerate(train_DL2):  # Print the 'text' data of the batch
          5 
          6     print(idx, "Text data: ", batch["Text"])  # Print the 'class' data of batch


    ~/opt/anaconda3/envs/sdoh_text/lib/python3.8/site-packages/torch/utils/data/dataloader.py in __next__(self)
        515             if self._sampler_iter is None:
        516                 self._reset()
    --> 517             data = self._next_data()
        518             self._num_yielded += 1
        519             if self._dataset_kind == _DatasetKind.Iterable and \


    ~/opt/anaconda3/envs/sdoh_text/lib/python3.8/site-packages/torch/utils/data/dataloader.py in _next_data(self)
        555     def _next_data(self):
        556         index = self._next_index()  # may raise StopIteration
    --> 557         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        558         if self._pin_memory:
        559             data = _utils.pin_memory.pin_memory(data)


    ~/opt/anaconda3/envs/sdoh_text/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
         45         else:
         46             data = self.dataset[possibly_batched_index]
    ---> 47         return self.collate_fn(data)
    

    ~/opt/anaconda3/envs/sdoh_text/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py in default_collate(batch)
         71         return batch
         72     elif isinstance(elem, container_abcs.Mapping):
    ---> 73         return {key: default_collate([d[key] for d in batch]) for key in elem}
         74     elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
         75         return elem_type(*(default_collate(samples) for samples in zip(*batch)))


    ~/opt/anaconda3/envs/sdoh_text/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py in <dictcomp>(.0)
         71         return batch
         72     elif isinstance(elem, container_abcs.Mapping):
    ---> 73         return {key: default_collate([d[key] for d in batch]) for key in elem}
         74     elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
         75         return elem_type(*(default_collate(samples) for samples in zip(*batch)))


    ~/opt/anaconda3/envs/sdoh_text/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py in default_collate(batch)
         79         elem_size = len(next(it))
         80         if not all(len(elem) == elem_size for elem in it):
    ---> 81             raise RuntimeError('each element in list of batch should be of equal size')
         82         transposed = zip(*batch)
         83         return [default_collate(samples) for samples in transposed]


    RuntimeError: each element in list of batch should be of equal size





What's going on? With some investigation of which I'll spare you, it appears that having each sample data already as a list makes confuses `Dataloader`. Let's re-structure out data differently.

# Re-structuring data as a comma-separated string

Due to the structure of our model, we still need a way to vectorize each sentence sample, but we can't have each wrapped as a list. Here is a workaround even if the syntax is awkward. I'm rejoining the elements as a comma-separated string like this:


```python
", ".join("me gusta comer en la cafeteria".split())
```




    'me, gusta, comer, en, la, cafeteria'







```python
train_data2 = [
    (", ".join("me gusta comer en la cafeteria".split()), "SPANISH"),
    (", ".join("Give it to me".split()), "ENGLISH"),
    (", ".join("No creo que sea una buena idea".split()), "SPANISH"),
    (", ".join("No it is not a good idea to get lost at sea".split()), "ENGLISH"),
]

test_data2 = [
    (", ".join("Yo creo que si".split()), "SPANISH"),
    (", ".join("it is lost on me".split()), "ENGLISH"),
]
```





```python
data2 = train_data2 + test_data2
df_data2 = pd.DataFrame(data2)
df_data2.columns = ["words", "labels"]
```




Here's how the data looks.


```python
df_data2
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
      <th>words</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>me, gusta, comer, en, la, cafeteria</td>
      <td>SPANISH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Give, it, to, me</td>
      <td>ENGLISH</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No, creo, que, sea, una, buena, idea</td>
      <td>SPANISH</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No, it, is, not, a, good, idea, to, get, lost,...</td>
      <td>ENGLISH</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Yo, creo, que, si</td>
      <td>SPANISH</td>
    </tr>
    <tr>
      <th>5</th>
      <td>it, is, lost, on, me</td>
      <td>ENGLISH</td>
    </tr>
  </tbody>
</table>
</div>






## Putting the data in `Dataset` and output with `Dataloader`


```python
train_DS2 = TextDataset(
    train_ids,
    df_data2.loc[train_ids, "words"].tolist(),
    df_data2.loc[train_ids, "labels"].tolist(),
)
test_DS2 = TextDataset(
    test_ids,
    df_data2.loc[test_ids, "words"].tolist(),
    df_data2.loc[test_ids, "labels"].tolist(),
)
```





```python
train_DL2a = DataLoader(train_DS2, batch_size=1, shuffle=False)

print("batch size of 1")
for (idx, batch) in enumerate(train_DL2a):
    print(idx, "Text data: ", batch["Text"])
    print(idx, "Label data: ", batch["Label"], "\n")
```

    batch size of 1
    0 Text data:  ['me, gusta, comer, en, la, cafeteria']
    0 Label data:  ['SPANISH'] 
    
    1 Text data:  ['Give, it, to, me']
    1 Label data:  ['ENGLISH'] 
    
    2 Text data:  ['No, creo, que, sea, una, buena, idea']
    2 Label data:  ['SPANISH'] 
    
    3 Text data:  ['No, it, is, not, a, good, idea, to, get, lost, at, sea']
    3 Label data:  ['ENGLISH'] 
    





Great, we get closer to the expected output where we have one sample, represented as a string, in the list created by `DataLoader`. We still have to vectorize this before we input this into our model but we can worry about that later. Additionally, when we increase the `batch_size` we don't get an error anymore.


```python
train_DL2b = DataLoader(train_DS2, batch_size=2, shuffle=False)

print("batch size of 2")
for (idx, batch) in enumerate(train_DL2b):
    print(idx, "Text data: ", batch["Text"])
    print(idx, "Label data: ", batch["Label"], "\n")
```

    batch size of 2
    0 Text data:  ['me, gusta, comer, en, la, cafeteria', 'Give, it, to, me']
    0 Label data:  ['SPANISH', 'ENGLISH'] 
    
    1 Text data:  ['No, creo, que, sea, una, buena, idea', 'No, it, is, not, a, good, idea, to, get, lost, at, sea']
    1 Label data:  ['SPANISH', 'ENGLISH'] 
    





We can also verify that this works for our test set in its own `DataLoader` object.


```python
test_DL2b = DataLoader(test_DS2, batch_size=2, shuffle=False)

print("batch size of 2")
for (idx, batch) in enumerate(test_DL2b):
    print(idx, "Text data: ", batch["Text"])
    print(idx, "Label data: ", batch["Label"], "\n")
```

    batch size of 2
    0 Text data:  ['Yo, creo, que, si', 'it, is, lost, on, me']
    0 Label data:  ['SPANISH', 'ENGLISH'] 
    





# Train model using `DataLoader` objects


```python
# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_ix = {}
for sent, _ in data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2
```

    {'me': 0, 'gusta': 1, 'comer': 2, 'en': 3, 'la': 4, 'cafeteria': 5, 'Give': 6, 'it': 7, 'to': 8, 'No': 9, 'creo': 10, 'que': 11, 'sea': 12, 'una': 13, 'buena': 14, 'idea': 15, 'is': 16, 'not': 17, 'a': 18, 'good': 19, 'get': 20, 'lost': 21, 'at': 22, 'Yo': 23, 'si': 24, 'on': 25}






```python
sent = "me, gusta, comer"
sent.split(", ")
```




    ['me', 'gusta', 'comer']







```python
class BoWClassifier(nn.Module):  # inheriting from nn.Module!
    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec), dim=1)


def make_bow_vector(sentence, word_to_ix):
    """
    Edited from original to get words wrapped in a list back
    """
    sentence = sentence[0].split(", ")
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    """
    Altered to extract label from list
    """
    return torch.LongTensor([label_to_ix[label[0]]])
```




## Batch size of 1


```python
train_DL2a = DataLoader(train_DS2, batch_size=1, shuffle=False)
test_DL2a = DataLoader(test_DS2, batch_size=1, shuffle=False)
```





```python
model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
```





```python
for param in model.parameters():
    print(param)

```

    Parameter containing:
    tensor([[ 0.0544,  0.0097,  0.0716, -0.0764, -0.0143, -0.0177,  0.0284, -0.0008,
              0.1714,  0.0610, -0.0730, -0.1184, -0.0329, -0.0846, -0.0628,  0.0094,
              0.1169,  0.1066, -0.1917,  0.1216,  0.0548,  0.1860,  0.1294, -0.1787,
             -0.1865, -0.0946],
            [ 0.1722, -0.0327,  0.0839, -0.0911,  0.1924, -0.0830,  0.1471,  0.0023,
             -0.1033,  0.1008, -0.1041,  0.0577, -0.0566, -0.0215, -0.1885, -0.0935,
              0.1064, -0.0477,  0.1953,  0.1572, -0.0092, -0.1309,  0.1194,  0.0609,
             -0.1268,  0.1274]], requires_grad=True)
    Parameter containing:
    tensor([0.1191, 0.1739], requires_grad=True)





Note that model parameters are randomly initialized to very small, non-zero values so that gradient descent is not too slow. This point is explained more fully by Andrew Ng in [this video](https://www.youtube.com/watch?v=6by6Xas_Kho&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=35).


```python
label_to_ix = {"SPANISH": 0, "ENGLISH": 1}
```




### Run on test data before we train, just to see a before-and-after


```python
with torch.no_grad():
    for batch in test_DL2a:
        # Alter code from tutorial
        # for instance, label in test_data:
        instance, label = batch["Text"], batch["Label"]
        print(instance, label)

        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs, "\n")

# Print the matrix column corresponding to "creo"
print(
    "Tensor for 'creo' (before training): ",
    next(model.parameters())[:, word_to_ix["creo"]],
)
```

    ['Yo, creo, que, si'] ['SPANISH']
    tensor([[-0.9736, -0.4744]]) 
    
    ['it, is, lost, on, me'] ['ENGLISH']
    tensor([[-0.7289, -0.6586]]) 
    
    Tensor for 'creo' (before training):  tensor([-0.0730, -0.1041], grad_fn=<SelectBackward>)






```python
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    # for instance, label in data:

    for (idx, batch) in enumerate(train_DL2a):  # Print the 'text' data of the batch
        instance, label = batch["Text"], batch["Label"]

        # Step 1. Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Tensor as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

        if (idx % 4 == 0) & (epoch % 20 == 0):  # Edit when datasets are bigger
            print(f"epoch: {epoch}, training sample: {idx}, loss = {loss.item():0.04f}")
```

    epoch: 0, training sample: 0, loss = 0.8369
    epoch: 20, training sample: 0, loss = 0.0507
    epoch: 40, training sample: 0, loss = 0.0257
    epoch: 60, training sample: 0, loss = 0.0172
    epoch: 80, training sample: 0, loss = 0.0129





We see the loss decrease quickly and saturate by the end of the training epochs.

### Evaluation after training

Look at the test set again, after model training.


```python
with torch.no_grad():
    for batch in test_DL2a:
        # Alter code from tutorial
        # for instance, label in test_data:
        instance, label = batch["Text"], batch["Label"]
        print(instance, label)

        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs, "\n")
```

    ['Yo, creo, que, si'] ['SPANISH']
    tensor([[-0.2056, -1.6828]]) 
    
    ['it, is, lost, on, me'] ['ENGLISH']
    tensor([[-2.7960, -0.0630]]) 
    






```python
# Print the matrix column corresponding to "creo"
print(
    "Matrix for 'creo' (after training): ",
    next(model.parameters())[:, word_to_ix["creo"]],
)
```

    Matrix for 'creo' (after training):  tensor([ 0.3702, -0.5473], grad_fn=<SelectBackward>)





We see that the coefficients for the Spanish word "creo" separate quite nicely and relative to the initial values. [I believe](https://media.giphy.com/media/U8GLl0bUYFLZVquOfY/giphy.gif) that the model training was successful.

# Summary

In this post, I sought to better understand how to use `Dataset` and `Dataloader` objects, especially in the context of model training. Fleshing this out showed me where I had to re-structure my data to get my code to work properly. Here, I had a batch size of 1, to mimic the original PyTorch tutorial. In a later post, I'll write about how to take advantage of batching which is more relevant in larger datasets.

Appendix: Environment and system parameters


```python
%watermark -n -u -v -iv -w
```

    Last updated: Thu Jun 24 2021
    
    Python implementation: CPython
    Python version       : 3.8.6
    IPython version      : 7.22.0
    
    numpy  : 1.19.5
    torch  : 1.8.1
    re     : 2.2.1
    json   : 2.0.9
    seaborn: 0.11.1
    pandas : 1.2.1
    
    Watermark: 2.1.0
    




