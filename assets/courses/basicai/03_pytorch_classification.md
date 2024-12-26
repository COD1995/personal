---
layout: page
title: PyTorch Neural Network Classification
description: 
related_publications: false
toc:
    sidebar: left
back_link: '/teaching/aibasic'
back_text: 'AI Basics'
number_heading: true
enable_heading_styles: true
show_h1_number: true
start_h1_number: 2
---

A [classification problem](https://en.wikipedia.org/wiki/Statistical_classification) involves predicting whether something is one thing or another.

For example, you might want to:

<table class="styled-table">
    <thead>
    <tr>
    <th>Problem type</th>
    <th>What is it?</th>
    <th>Example</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><strong>Binary classification</strong></td>
    <td>Target can be one of two options, e.g. yes or no</td>
    <td>Predict whether or not someone has heart disease based on their health parameters.</td>
    </tr>
    <tr>
    <td><strong>Multi-class classification</strong></td>
    <td>Target can be one of more than two options</td>
    <td>Decide whether a photo is of food, a person or a dog.</td>
    </tr>
    <tr>
    <td><strong>Multi-label classification</strong></td>
    <td>Target can be assigned more than one option</td>
    <td>Predict what categories should be assigned to a Wikipedia article (e.g. mathematics, science & philosophy).</td>
    </tr>
    </tbody>
</table>

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager"
            path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/02-different-classification-problems.png"
            class="img-fluid rounded"
            caption="Various different classification problems in machine learning such as binary, multiclass, and multilabel classification."
            id="fig_different_classification_problems" %}
    </div>
</div>

Classification, along with regression (predicting a number, covered in <a href="{{ "/assets/courses/basicai/02_pytorch_workflow" | relative_url }}">pytorch workflow</a> is one of the most common types of machine learning problems.

In this session, we're going to work through a couple of different classification problems with PyTorch. 

In other words, taking a set of inputs and predicting what class those set of inputs belong to.

## What we're going to cover

In this notebook we're going to reiterate over the PyTorch workflow we covered in <a href="{{ "/assets/courses/basicai/02_pytorch_workflow" | relative_url }}">pytorch workflow</a>.

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager"
            path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/01_a_pytorch_workflow.png"
            class="img-fluid rounded"
            caption="A PyTorch workflow flowchart."
            id="fig_pytorch_workflow" %}
    </div>
</div>

Except instead of trying to predict a straight line (predicting a number, also called a regression problem), we'll be working on a **classification problem**.

Specifically, we're going to cover:

<table class="styled-table">
<thead>
<tr>
<th><strong>Topic</strong></th>
<th><strong>Contents</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>0. Architecture of a classification neural network</strong></td>
<td>Neural networks can come in almost any shape or size, but they typically follow a similar floor plan.</td>
</tr>
<tr>
<td><strong>1. Getting binary classification data ready</strong></td>
<td>Data can be almost anything but to get started we're going to create a simple binary classification dataset.</td>
</tr>
<tr>
<td><strong>2. Building a PyTorch classification model</strong></td>
<td>Here we'll create a model to learn patterns in the data, we'll also choose a <strong>loss function</strong>, <strong>optimizer</strong> and build a <strong>training loop</strong> specific to classification.</td>
</tr>
<tr>
<td><strong>3. Fitting the model to data (training)</strong></td>
<td>We've got data and a model, now let's let the model (try to) find patterns in the (<strong>training</strong>) data.</td>
</tr>
<tr>
<td><strong>4. Making predictions and evaluating a model (inference)</strong></td>
<td>Our model's found patterns in the data, let's compare its findings to the actual (<strong>testing</strong>) data.</td>
</tr>
<tr>
<td><strong>5. Improving a model (from a model perspective)</strong></td>
<td>We've trained and evaluated a model but it's not working, let's try a few things to improve it.</td>
</tr>
<tr>
<td><strong>6. Non-linearity</strong></td>
<td>So far our model has only had the ability to model straight lines, what about non-linear (non-straight) lines?</td>
</tr>
<tr>
<td><strong>7. Replicating non-linear functions</strong></td>
<td>We used <strong>non-linear functions</strong> to help model non-linear data, but what do these look like?</td>
</tr>
<tr>
<td><strong>8. Putting it all together with multi-class classification</strong></td>
<td>Let's put everything we've done so far for binary classification together with a multi-class classification problem.</td>
</tr>
</tbody>
</table>

## Architecture of a classification neural network

Before we get into writing code, let's look at the general architecture of a classification neural network.

<table class="styled-table">
    <thead>
    <tr>
    <th><strong>Hyperparameter</strong></th>
    <th><strong>Binary Classification</strong></th>
    <th><strong>Multiclass classification</strong></th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><strong>Input layer shape</strong> (<code>in_features</code>)</td>
    <td>Same as number of features (e.g. 5 for age, sex, height, weight, smoking status in heart disease prediction)</td>
    <td>Same as binary classification</td>
    </tr>
    <tr>
    <td><strong>Hidden layer(s)</strong></td>
    <td>Problem specific, minimum = 1, maximum = unlimited</td>
    <td>Same as binary classification</td>
    </tr>
    <tr>
    <td><strong>Neurons per hidden layer</strong></td>
    <td>Problem specific, generally 10 to 512</td>
    <td>Same as binary classification</td>
    </tr>
    <tr>
    <td><strong>Output layer shape</strong> (<code>out_features</code>)</td>
    <td>1 (one class or the other)</td>
    <td>1 per class (e.g. 3 for food, person or dog photo)</td>
    </tr>
    <tr>
    <td><strong>Hidden layer activation</strong></td>
    <td>Usually <a href="https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU">ReLU</a> (rectified linear unit) but <a href="https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions">can be many others</a></td>
    <td>Same as binary classification</td>
    </tr>
    <tr>
    <td><strong>Output activation</strong></td>
    <td><a href="https://en.wikipedia.org/wiki/Sigmoid_function">Sigmoid</a> (<a href="https://pytorch.org/docs/stable/generated/torch.sigmoid.html"><code>torch.sigmoid</code></a> in PyTorch)</td>
    <td><a href="https://en.wikipedia.org/wiki/Softmax_function">Softmax</a> (<a href="https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html"><code>torch.softmax</code></a> in PyTorch)</td>
    </tr>
    <tr>
    <td><strong>Loss function</strong></td>
    <td><a href="https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression">Binary crossentropy</a> (<a href="https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html"><code>torch.nn.BCELoss</code></a> in PyTorch)</td>
    <td>Cross entropy (<a href="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"><code>torch.nn.CrossEntropyLoss</code></a> in PyTorch)</td>
    </tr>
    <tr>
    <td><strong>Optimizer</strong></td>
    <td><a href="https://pytorch.org/docs/stable/generated/torch.optim.SGD.html">SGD</a> (stochastic gradient descent), <a href="https://pytorch.org/docs/stable/generated/torch.optim.Adam.html">Adam</a> (see <a href="https://pytorch.org/docs/stable/optim.html"><code>torch.optim</code></a> for more options)</td>
    <td>Same as binary classification</td>
    </tr>
    </tbody>
</table>

Of course, this ingredient list of classification neural network components will vary depending on the problem you're working on.

But it's more than enough to get started.

We're going to get hands-on with this setup throughout this notebook.

## Make classification data and get it ready

Let's begin by making some data.

We'll use the [`make_circles()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html) method from Scikit-Learn to generate two circles with different coloured dots. 


```python
from sklearn.datasets import make_circles


# Make 1000 samples 
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
<div class="bash-block">
<pre><code>noise=0.03, # a little bit of noise to the dots
random_state=42) # keep random state so we get the same values</code></pre>
</div>
```

Alright, now let's view the first 5 `X` and `y` values.


```python
print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")
```

<div class="bash-block">
<pre><code>First 5 X features:
[[ 0.75424625  0.23148074]
[-0.75615888  0.15325888]
[-0.81539193  0.17328203]
[-0.39373073  0.69288277]
[ 0.44220765 -0.89672343]]

First 5 y labels:
[1 1 1 1 0]</code></pre>
</div>


Looks like there's two `X` values per one `y` value. 

Let's keep following the data explorer's motto of *visualize, visualize, visualize* and put them into a pandas DataFrame.


```python
# Make DataFrame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
circles.head(10)
```

<div>
  <table class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>X1</th>
        <th>X2</th>
        <th>label</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>0.754246</td>
        <td>0.231481</td>
        <td>1</td>
      </tr>
      <tr>
        <th>1</th>
        <td>-0.756159</td>
        <td>0.153259</td>
        <td>1</td>
      </tr>
      <tr>
        <th>2</th>
        <td>-0.815392</td>
        <td>0.173282</td>
        <td>1</td>
      </tr>
      <tr>
        <th>3</th>
        <td>-0.393731</td>
        <td>0.692883</td>
        <td>1</td>
      </tr>
      <tr>
        <th>4</th>
        <td>0.442208</td>
        <td>-0.896723</td>
        <td>0</td>
      </tr>
      <tr>
        <th>5</th>
        <td>-0.479646</td>
        <td>0.676435</td>
        <td>1</td>
      </tr>
      <tr>
        <th>6</th>
        <td>-0.013648</td>
        <td>0.803349</td>
        <td>1</td>
      </tr>
      <tr>
        <th>7</th>
        <td>0.771513</td>
        <td>0.147760</td>
        <td>1</td>
      </tr>
      <tr>
        <th>8</th>
        <td>-0.169322</td>
        <td>-0.793456</td>
        <td>1</td>
      </tr>
      <tr>
        <th>9</th>
        <td>-0.121486</td>
        <td>1.021509</td>
        <td>0</td>
      </tr>
    </tbody>
  </table>
</div>


It looks like each pair of `X` features (`X1` and `X2`) has a label (`y`) value of either 0 or 1.

This tells us that our problem is **binary classification** since there's only two options (0 or 1).

How many values of each class are there?

```python
# Check different labels
circles.label.value_counts()
```
<div class="bash-block">
<pre><code>1    500
0    500
Name: label, dtype: int64</code></pre>
</div>

500 each, nice and balanced.

Let's plot them.


```python
# Visualize with a plot
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0], 
<div class="bash-block">
<pre><code>y=X[:, 1],
c=y,
cmap=plt.cm.RdYlBu);</code></pre>
```
<div style="text-align: left;">
  <img
    src="{{ "assets/img/02_pytorch_classification_files/02_pytorch_classification_14_0.png" | relative_url }}"
    alt="png"
    class="img-fluid"
    style="max-width: 80%; height: auto; display: block; margin-bottom: 1rem;"
  />
</div>

Alrighty, looks like we've got a problem to solve.

Let's find out how we could build a PyTorch neural network to classify dots into red (0) or blue (1).

<div class="note-box">
<strong>Note:</strong> This dataset is often what's considered a **toy problem** (a problem that's used to try and test things out on) in machine learning.  But it represents the major key of classification, you have some kind of data represented as numerical values and you'd like to build a model that's able to classify it, in our case, separate it into red or blue dots.
</div>

### Input and output shapes

One of the most common errors in deep learning is shape errors.

Mismatching the shapes of tensors and tensor operations will result in errors in your models.

We're going to see plenty of these throughout the course.

And there's no surefire way to make sure they won't happen, they will.

What you can do instead is continually familiarize yourself with the shape of the data you're working with.

I like referring to it as input and output shapes.

Ask yourself:

"What shapes are my inputs and what shapes are my outputs?"

Let's find out.


```python
# Check the shapes of our features and labels
X.shape, y.shape
```
<div class="bash-block">
<pre><code>((1000, 2), (1000,))</code></pre>
</div>

Looks like we've got a match on the first dimension of each.

There's 1000 `X` and 1000 `y`. 

But what's the second dimension on `X`?

It often helps to view the values and shapes of a single sample (features and labels).

Doing so will help you understand what input and output shapes you'd be expecting from your model.


```python
# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")
```

<div class="bash-block">
<pre><code>Values for one sample of X: [0.75424625 0.23148074] and the same for y: 1
Shapes for one sample of X: (2,) and the same for y: ()</code></pre>
</div>


This tells us the second dimension for `X` means it has two features (vector) where as `y` has a single feature (scalar).

We have two inputs for one output.

### Turn data into tensors and create train and test splits

We've investigated the input and output shapes of our data, now let's prepare it for being used with PyTorch and for modelling.

Specifically, we'll need to:
1. Turn our data into tensors (right now our data is in NumPy arrays and PyTorch prefers to work with PyTorch tensors).
2. Split our data into training and test sets (we'll train a model on the training set to learn the patterns between `X` and `y` and then evaluate those learned patterns on the test dataset).


```python
# Turn data into tensors
# Otherwise this causes issues with computations later on
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# View the first five samples
X[:5], y[:5]
```




<div class="bash-block">
<pre><code>(tensor([[ 0.7542,  0.2315],
[-0.7562,  0.1533],
[-0.8154,  0.1733],
[-0.3937,  0.6929],
[ 0.4422, -0.8967]]),
tensor([1., 1., 1., 1., 0.]))</code></pre>
</div>



Now our data is in tensor format, let's split it into training and test sets.

To do so, let's use the helpful function [`train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) from Scikit-Learn.

We'll use `test_size=0.2` (80% training, 20% testing) and because the split happens randomly across the data, let's use `random_state=42` so the split is reproducible.


```python
# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
<div class="bash-block">
<pre><code>y,
test_size=0.2, # 20% test, 80% train
random_state=42) # make the random split reproducible</code></pre>
</div>

len(X_train), len(X_test), len(y_train), len(y_test)
```




<div class="bash-block">
<pre><code>(800, 200, 800, 200)</code></pre>
</div>



Nice! Looks like we've now got 800 training samples and 200 testing samples.