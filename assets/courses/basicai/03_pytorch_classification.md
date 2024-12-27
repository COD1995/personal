---
layout: page
title: PyTorch Neural Network Classification
description: 
related_publications: false
toc:
    sidebar: right
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

## Building a model

We've got some data ready, now it's time to build a model.

We'll break it down into a few parts.

1. Setting up device agnostic code (so our model can run on CPU or GPU if it's available).
2. Constructing a model by subclassing `nn.Module`.
3. Defining a loss function and optimizer.
4. Creating a training loop (this'll be in the next section).

The good news is we've been through all of the above steps before in notebook 01.

Except now we'll be adjusting them so they work with a classification dataset.

Let's start by importing PyTorch and `torch.nn` as well as setting up device agnostic code.


```python
# Standard PyTorch imports
import torch
from torch import nn

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```




<div class="bash-block">
<pre><code>'cuda'</code></pre>
</div>



Excellent, now `device` is setup, we can use it for any data or models we create and PyTorch will handle it on the CPU (default) or GPU if it's available.

How about we create a model?

We'll want a model capable of handling our `X` data as inputs and producing something in the shape of our `y` data as outputs.

In other words, given `X` (features) we want our model to predict `y` (label).

This setup where you have features and labels is referred to as **supervised learning**. Because your data is telling your model what the outputs should be given a certain input.

To create such a model it'll need to handle the input and output shapes of `X` and `y`.

Remember how I said input and output shapes are important? Here we'll see why.

Let's create a model class that:
1. Subclasses `nn.Module` (almost all PyTorch models are subclasses of `nn.Module`).
2. Creates 2 `nn.Linear` layers in the constructor capable of handling the input and output shapes of `X` and `y`.
3. Defines a `forward()` method containing the forward pass computation of the model.
4. Instantiates the model class and sends it to the target `device`. 


```python
# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features, produces 1 feature (y)
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(self.layer_1(x)) # computation goes through layer_1 first then the output of layer_1 goes through layer_2

# 4. Create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)
model_0
```




<div class="bash-block">
<pre><code>CircleModelV0(
(layer_1): Linear(in_features=2, out_features=5, bias=True)
(layer_2): Linear(in_features=5, out_features=1, bias=True)
)</code></pre>
</div>



What's going on here?

We've seen a few of these steps before.

The only major change is what's happening between `self.layer_1` and `self.layer_2`.

`self.layer_1` takes 2 input features `in_features=2` and produces 5 output features `out_features=5`.

This is known as having 5 **hidden units** or **neurons**.

This layer turns the input data from having 2 features to 5 features.

Why do this?

This allows the model to learn patterns from 5 numbers rather than just 2 numbers, *potentially* leading to better outputs.

I say potentially because sometimes it doesn't work.

The number of hidden units you can use in neural network layers is a **hyperparameter** (a value you can set yourself) and there's no set in stone value you have to use.

Generally more is better but there's also such a thing as too much. The amount you choose will depend on your model type and dataset you're working with. 

Since our dataset is small and simple, we'll keep it small.

The only rule with hidden units is that the next layer, in our case, `self.layer_2` has to take the same `in_features` as the previous layer `out_features`.

That's why `self.layer_2` has `in_features=5`, it takes the `out_features=5` from `self.layer_1` and performs a linear computation on them, turning them into `out_features=1` (the same shape as `y`).


<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager"
            path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/02-tensorflow-playground-linear-activation.png"
            class="img-fluid rounded"
            caption="A visual example of what a classification neural network with linear activation looks like on the tensorflow playground"
            id="fig_a_visual_example_of_what_a_classification_neural_network_with_linear_activation_looks_like_on_the_tensorflow_playground" %}
    </div>
</div>

*A visual example of what a similar classification neural network to the one we've just built looks like. Try creating one of your own on the [TensorFlow Playground website](https://playground.tensorflow.org/).*

You can also do the same as above using [`nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html).

`nn.Sequential` performs a forward pass computation of the input data through the layers in the order they appear.


```python
# Replicate CircleModelV0 with nn.Sequential
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

model_0
```

<div class="bash-block">
<pre><code>Sequential(
(0): Linear(in_features=2, out_features=5, bias=True)
(1): Linear(in_features=5, out_features=1, bias=True)
)</code></pre>
</div>



Woah, that looks much simpler than subclassing `nn.Module`, why not just always use `nn.Sequential`?

`nn.Sequential` is fantastic for straight-forward computations, however, as the namespace says, it *always* runs in sequential order.

So if you'd like something else to happen (rather than just straight-forward sequential computation) you'll want to define your own custom `nn.Module` subclass.

Now we've got a model, let's see what happens when we pass some data through it.


```python
# Make predictions with the model
untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")
```

<div class="bash-block">
  <pre><code>tensor([[-0.4279],
        [-0.3417],
        [-0.5975],
        [-0.3801],
        [-0.5078],
        [-0.4559],
        [-0.2842],
        [-0.3107],
        [-0.6010],
        [-0.3350]], device='cuda:0', grad_fn=&lt;SliceBackward0&gt;)</code></pre>
</div>




Hmm, it seems there are the same amount of predictions as there are test labels but the predictions don't look like they're in the same form or shape as the test labels.

We've got a couple steps we can do to fix this, we'll see these later on.

### Setup loss function and optimizer

We've setup a loss (also called a criterion or cost function) and optimizer before in [notebook 01](https://www.learnpytorch.io/01_pytorch_workflow/#creating-a-loss-function-and-optimizer-in-pytorch).

But different problem types require different loss functions. 

For example, for a regression problem (predicting a number) you might use mean absolute error (MAE) loss.

And for a binary classification problem (like ours), you'll often use [binary cross entropy](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a) as the loss function.

However, the same optimizer function can often be used across different problem spaces.

For example, the stochastic gradient descent optimizer (SGD, `torch.optim.SGD()`) can be used for a range of problems, and the same applies to the Adam optimizer (`torch.optim.Adam()`). 

<table class="styled-table">
<thead>
<tr>
<th>Loss function/Optimizer</th>
<th>Problem type</th>
<th>PyTorch Code</th>
</tr>
</thead>
<tbody>
<tr>
<td>Stochastic Gradient Descent (SGD) optimizer</td>
<td>Classification, regression, many others.</td>
<td><a href="https://pytorch.org/docs/stable/generated/torch.optim.SGD.html"><code>torch.optim.SGD()</code></a></td>
</tr>
<tr>
<td>Adam Optimizer</td>
<td>Classification, regression, many others.</td>
<td><a href="https://pytorch.org/docs/stable/generated/torch.optim.Adam.html"><code>torch.optim.Adam()</code></a></td>
</tr>
<tr>
<td>Binary cross entropy loss</td>
<td>Binary classification</td>
<td><a href="https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html"><code>torch.nn.BCELossWithLogits</code></a> or <a href="https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html"><code>torch.nn.BCELoss</code></a></td>
</tr>
<tr>
<td>Cross entropy loss</td>
<td>Multi-class classification</td>
<td><a href="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"><code>torch.nn.CrossEntropyLoss</code></a></td>
</tr>
<tr>
<td>Mean absolute error (MAE) or L1 Loss</td>
<td>Regression</td>
<td><a href="https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html"><code>torch.nn.L1Loss</code></a></td>
</tr>
<tr>
<td>Mean squared error (MSE) or L2 Loss</td>
<td>Regression</td>
<td><a href="https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss"><code>torch.nn.MSELoss</code></a></td>
</tr>
</tbody>
</table>

*Table of various loss functions and optimizers, there are more but these are some common ones you'll see.*

Since we're working with a binary classification problem, let's use a binary cross entropy loss function.

<div class="note-box">
  <strong>Note:</strong>
  <p>
    A <em>loss function</em> measures how <em>wrong</em> your model predictions are. 
    The higher the loss, the worse your model is performing.
  </p>
  <p>
    In the PyTorch documentation, loss functions are sometimes called 
    "<em>loss criterion</em>" or just "<em>criterion</em>". These terms all describe 
    the same concept.
  </p>
</div>


PyTorch has two binary cross entropy implementations:
1. [`torch.nn.BCELoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) - Creates a loss function that measures the binary cross entropy between the target (label) and input (features).
2. [`torch.nn.BCEWithLogitsLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) - This is the same as above except it has a sigmoid layer ([`nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)) built-in (we'll see what this means soon).

Which one should you use? 

The [documentation for `torch.nn.BCEWithLogitsLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) states that it's more numerically stable than using `torch.nn.BCELoss()` after a `nn.Sigmoid` layer. 

So generally, implementation 2 is a better option. However for advanced usage, you may want to separate the combination of `nn.Sigmoid` and `torch.nn.BCELoss()` but that is beyond the scope of this notebook.

Knowing this, let's create a loss function and an optimizer. 

For the optimizer we'll use `torch.optim.SGD()` to optimize the model parameters with learning rate 0.1.

<div class="note-box">
  <strong>Note:</strong>
  <p>
    There's a 
    <a href="https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/4">
      discussion on the PyTorch forums
    </a> 
    about using <code>nn.BCELoss</code> vs. <code>nn.BCEWithLogitsLoss</code>. 
  </p>
  <p>
    It can be confusing at first, but—as with many things—practice helps solidify 
    the differences and best usage scenarios.
  </p>
</div>


```python
# Create a loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)
```

Now let's also create an **evaluation metric**.

An evaluation metric can be used to offer another perspective on how your model is going.

If a loss function measures how *wrong* your model is, I like to think of evaluation metrics as measuring how *right* it is.

Of course, you could argue both of these are doing the same thing but evaluation metrics offer a different perspective.

After all, when evaluating your models it's good to look at things from multiple points of view.

There are several evaluation metrics that can be used for classification problems but let's start out with **accuracy**.

Accuracy can be measured by dividing the total number of correct predictions over the total number of predictions.

For example, a model that makes 99 correct predictions out of 100 will have an accuracy of 99%.

Let's write a function to do so.

```python
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc
```

Excellent! We can now use this function whilst training our model to measure it's performance alongside the loss.

## Train model

Okay, now we've got a loss function and optimizer ready to go, let's train a model.

Do you remember the steps in a PyTorch training loop?

If not, here's a reminder.

Steps in training:

<details class="collapsible-callout">
  <summary>PyTorch training loop steps</summary>
  <ol>
    <li><strong>Forward pass</strong> - The model goes through all of the training data once, performing its 
      <code>forward()</code> function calculations (<code>model(x_train)</code>).
    </li>
    <li><strong>Calculate the loss</strong> - The model's outputs (predictions) are compared to the ground truth and 
      evaluated to see how wrong they are (<code>loss = loss_fn(y_pred, y_train)</code>).
    </li>
    <li><strong>Zero gradients</strong> - The optimizer's gradients are set to zero (they are accumulated by default) 
      so they can be recalculated for the specific training step (<code>optimizer.zero_grad()</code>).
    </li>
    <li><strong>Perform backpropagation on the loss</strong> - Computes the gradient of the loss with respect to every 
      model parameter to be updated (each parameter with <code>requires_grad=True</code>). This is known as 
      <strong>backpropagation</strong>, hence "backwards" (<code>loss.backward()</code>).
    </li>
    <li><strong>Step the optimizer (gradient descent)</strong> - Update the parameters with 
      <code>requires_grad=True</code> with respect to the loss gradients in order to improve them 
      (<code>optimizer.step()</code>).
    </li>
  </ol>
</details>

### Going from raw model outputs to predicted labels (logits -> prediction probabilities -> prediction labels)

Before the training loop steps, let's see what comes out of our model during the forward pass (the forward pass is defined by the `forward()` method).

To do so, let's pass the model some data.


```python
# View the frist 5 outputs of the forward pass on the test data
y_logits = model_0(X_test.to(device))[:5]
y_logits
```




<div class="bash-block">
  <pre><code>tensor([[-0.4279],
        [-0.3417],
        [-0.5975],
        [-0.3801],
        [-0.5078]], device='cuda:0', grad_fn=&lt;SliceBackward0&gt;)</code></pre>
</div>




Since our model hasn't been trained, these outputs are basically random.

But *what* are they?

They're the output of our `forward()` method.

Which implements two layers of `nn.Linear()` which internally calls the following equation:

$$
\mathbf{y} = x \cdot \mathbf{Weights}^T  + \mathbf{bias}
$$

The *raw outputs* (unmodified) of this equation ($\mathbf{y}$) and in turn, the raw outputs of our model are often referred to as [**logits**](https://datascience.stackexchange.com/a/31045).

That's what our model is outputing above when it takes in the input data ($x$ in the equation or `X_test` in the code), logits.

However, these numbers are hard to interpret.

We'd like some numbers that are comparable to our truth labels.

To get our model's raw outputs (logits) into such a form, we can use the [sigmoid activation function](https://pytorch.org/docs/stable/generated/torch.sigmoid.html).

Let's try it out.



```python
# Use sigmoid on model logits
y_pred_probs = torch.sigmoid(y_logits)
y_pred_probs
```

<div class="bash-block">
  <pre><code>tensor([[0.3946],
        [0.4154],
        [0.3549],
        [0.4061],
        [0.3757]], device='cuda:0', grad_fn=&lt;SigmoidBackward0&gt;)</code></pre>
</div>

Okay, it seems like the outputs now have some kind of consistency (even though they're still random).

They're now in the form of **prediction probabilities** (I usually refer to these as `y_pred_probs`), in other words, the values are now how much the model thinks the data point belongs to one class or another.

In our case, since we're dealing with binary classification, our ideal outputs are 0 or 1.

So these values can be viewed as a decision boundary.

The closer to 0, the more the model thinks the sample belongs to class 0, the closer to 1, the more the model thinks the sample belongs to class 1.

More specificially:
* If `y_pred_probs` >= 0.5, `y=1` (class 1)
* If `y_pred_probs` < 0.5, `y=0` (class 0)

To turn our prediction probabilities into prediction labels, we can round the outputs of the sigmoid activation function.


```python
# Find the predicted labels (round the prediction probabilities)
y_preds = torch.round(y_pred_probs)

# In full
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# Get rid of extra dimension
y_preds.squeeze()
```

<div class="bash-block">
    <pre><code>tensor([True, True, True, True, True], device='cuda:0')</code></pre>
</div>

<div class="bash-block">
  <pre><code>tensor([0., 0., 0., 0., 0.], device='cuda:0', grad_fn=&lt;SqueezeBackward0&gt;)</code></pre>
</div>




Excellent! Now it looks like our model's predictions are in the same form as our truth labels (`y_test`).


```python
y_test[:5]
```




<div class="bash-block">
<pre><code>tensor([1., 0., 1., 0., 1.])</code></pre>
</div>



This means we'll be able to compare our model's predictions to the test labels to see how well it's performing. 

To recap, we converted our model's raw outputs (logits) to prediction probabilities using a sigmoid activation function.

And then converted the prediction probabilities to prediction labels by rounding them.

<div class="note-box">
  <strong>Note:</strong>
  <p>
    The use of the sigmoid activation function is often only for binary classification logits. For multi-class classification, we'll be looking at using the <a href="https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html">softmax activation function</a> (this will come later on).
  </p>
  <p>
    Also, the sigmoid activation function is <em>not</em> required when passing our model's raw outputs to <code>nn.BCEWithLogitsLoss</code>. The "logits" in logits loss indicates it works on the model's raw logits output—because it has a sigmoid function built in.
  </p>
</div>

### Building a training and testing loop

Alright, we've discussed how to take our raw model outputs and convert them to prediction labels, now let's build a training loop.

Let's start by training for 100 epochs and outputing the model's progress every 10 epochs. 


```python
torch.manual_seed(42)

# Set the number of epochs
epochs = 100

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
  
    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train) 
    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   y_train) 
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
```

<div class="bash-block">
<pre><code>Epoch: 0 | Loss: 0.72090, Accuracy: 50.00% | Test loss: 0.72196, Test acc: 50.00%
Epoch: 10 | Loss: 0.70291, Accuracy: 50.00% | Test loss: 0.70542, Test acc: 50.00%
Epoch: 20 | Loss: 0.69659, Accuracy: 50.00% | Test loss: 0.69942, Test acc: 50.00%
Epoch: 30 | Loss: 0.69432, Accuracy: 43.25% | Test loss: 0.69714, Test acc: 41.00%
Epoch: 40 | Loss: 0.69349, Accuracy: 47.00% | Test loss: 0.69623, Test acc: 46.50%
Epoch: 50 | Loss: 0.69319, Accuracy: 49.00% | Test loss: 0.69583, Test acc: 46.00%
Epoch: 60 | Loss: 0.69308, Accuracy: 50.12% | Test loss: 0.69563, Test acc: 46.50%
Epoch: 70 | Loss: 0.69303, Accuracy: 50.38% | Test loss: 0.69551, Test acc: 46.00%
Epoch: 80 | Loss: 0.69302, Accuracy: 51.00% | Test loss: 0.69543, Test acc: 46.00%
Epoch: 90 | Loss: 0.69301, Accuracy: 51.00% | Test loss: 0.69537, Test acc: 46.00%</code></pre>
</div>

## Make predictions and evaluate the model

From the metrics it looks like our model is random guessing.

How could we investigate this further?

I've got an idea.

The data explorer's motto!

"Visualize, visualize, visualize!"

Let's make a plot of our model's predictions, the data it's trying to predict on and the decision boundary it's creating for whether something is class 0 or class 1.

To do so, we'll write some code to download and import the [`helper_functions.py` script](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py) from the [Learn PyTorch for Deep Learning repo](https://github.com/mrdbourke/pytorch-deep-learning).

It contains a helpful function called `plot_decision_boundary()` which creates a NumPy meshgrid to visually plot the different points where our model is predicting certain classes.

We'll also import `plot_predictions()` which we wrote in notebook 01 to use later.


```python
import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary
```

<div class="bash-block">
<pre><code>helper_functions.py already exists, skipping download</code></pre>
</div>


```python
# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
```

<div style="text-align: left;">
  <img
    src="{{ "assets/img/02_pytorch_classification_files/02_pytorch_classification_55_0.png" | relative_url }}"
    alt="png"
    class="img-fluid"
    style="max-width: 80%; height: auto; display: block; margin-bottom: 1rem;"
  />
</div>

Oh wow, it seems like we've found the cause of model's performance issue.

It's currently trying to split the red and blue dots using a straight line...

That explains the 50% accuracy. Since our data is circular, drawing a straight line can at best cut it down the middle.

In machine learning terms, our model is **underfitting**, meaning it's not learning predictive patterns from the data.

How could we improve this?

## Improving a model (from a model perspective) 

Let's try to fix our model's underfitting problem.

Focusing specifically on the model (not the data), there are a few ways we could do this.

<table class="styled-table">
<thead>
<tr>
<th>Model improvement technique*</th>
<th>What does it do?</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Add more layers</strong></td>
<td>Each layer <em>potentially</em> increases the learning capabilities of the model with each layer being able to learn some kind of new pattern in the data. More layers are often referred to as making your neural network <em>deeper</em>.</td>
</tr>
<tr>
<td><strong>Add more hidden units</strong></td>
<td>Similar to the above, more hidden units per layer means a <em>potential</em> increase in learning capabilities of the model. More hidden units are often referred to as making your neural network <em>wider</em>.</td>
</tr>
<tr>
<td><strong>Fitting for longer (more epochs)</strong></td>
<td>Your model might learn more if it had more opportunities to look at the data.</td>
</tr>
<tr>
<td><strong>Changing the activation functions</strong></td>
<td>Some data just can't be fit with only straight lines (like what we've seen), using non-linear activation functions can help with this (hint, hint).</td>
</tr>
<tr>
<td><strong>Change the learning rate</strong></td>
<td>Less model specific, but still related, the learning rate of the optimizer decides how much a model should change its parameters each step, too much and the model overcorrects, too little and it doesn't learn enough.</td>
</tr>
<tr>
<td><strong>Change the loss function</strong></td>
<td>Again, less model specific but still important, different problems require different loss functions. For example, a binary cross entropy loss function won't work with a multi-class classification problem.</td>
</tr>
<tr>
<td><strong>Use transfer learning</strong></td>
<td>Take a pretrained model from a problem domain similar to yours and adjust it to your own problem.</td>
</tr>
</tbody>
</table>

<div class="note-box">
  <strong>Note:</strong> 
  <p>
    Because you can adjust all of these by hand, they're referred to as <strong>hyperparameters</strong>. And this is also where machine learning's half art, half science aspect comes in—there's no real way to know upfront what the best combination of values is for your project. 
  </p>
  <p>
    Best to follow the data scientist's motto: <em>"experiment, experiment, experiment"</em>.
  </p>
</div>

Let's see what happens if we add an extra layer to our model, fit for longer (`epochs=1000` instead of `epochs=100`) and increase the number of hidden units from `5` to `10`.

We'll follow the same steps we did above but with a few changed hyperparameters.


```python
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10) # extra layer
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        
    def forward(self, x): # note: always make sure forward is spelt correctly!
        # Creating a model like this is the same as below, though below
        # generally benefits from speedups where possible.
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        # return z
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModelV1().to(device)
model_1
```

<div class="bash-block">
  <pre><code>CircleModelV1(
  (layer_1): Linear(in_features=2, out_features=10, bias=True)
  (layer_2): Linear(in_features=10, out_features=10, bias=True)
  (layer_3): Linear(in_features=10, out_features=1, bias=True)
)</code></pre>
</div>

Now we've got a model, we'll recreate a loss function and optimizer instance, using the same settings as before.


```python
# loss_fn = nn.BCELoss() # Requires sigmoid on input
loss_fn = nn.BCEWithLogitsLoss() # Does not require sigmoid on input
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)
```

Beautiful, model, optimizer and loss function ready, let's make a training loop.

This time we'll train for longer (`epochs=1000` vs `epochs=100`) and see if it improves our model. 


```python
torch.manual_seed(42)

epochs = 1000 # Train for longer

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    ### Training
    # 1. Forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels

    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_1.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_1(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

```

<div class="bash-block">
  <pre><code>Epoch: 0   | Loss: 0.69396, Accuracy: 50.88% | Test loss: 0.69261, Test acc: 51.00%
Epoch: 100 | Loss: 0.69305, Accuracy: 50.38% | Test loss: 0.69379, Test acc: 48.00%
Epoch: 200 | Loss: 0.69299, Accuracy: 51.12% | Test loss: 0.69437, Test acc: 46.00%
Epoch: 300 | Loss: 0.69298, Accuracy: 51.62% | Test loss: 0.69458, Test acc: 45.00%
Epoch: 400 | Loss: 0.69298, Accuracy: 51.12% | Test loss: 0.69465, Test acc: 46.00%
Epoch: 500 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.69467, Test acc: 46.00%
Epoch: 600 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.69468, Test acc: 46.00%
Epoch: 700 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.69468, Test acc: 46.00%
Epoch: 800 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.69468, Test acc: 46.00%
Epoch: 900 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.69468, Test acc: 46.00%</code></pre>
</div>

What? Our model trained for longer and with an extra layer but it still looks like it didn't learn any patterns better than random guessing.

Let's visualize.


```python
# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
```
<div style="text-align: left;">
  <img
    src="{{ "/assets/img/02_pytorch_classification_files/02_pytorch_classification_64_0.png" | relative_url }}"
    alt="A local PyTorch classification image"
    class="img-fluid"
    style="max-width: 80%; height: auto; display: block; margin-bottom: 1rem;"
  />
</div>

Hmmm.

Our model is still drawing a straight line between the red and blue dots.

If our model is drawing a straight line, could it model linear data? 

### Preparing data to see if our model can model a straight line
Let's create some linear data to see if our model's able to model it and we're not just using a model that can't learn anything.


```python
# Create some data (same as notebook 01)
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# Create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias # linear regression formula

# Check the data
print(len(X_regression))
X_regression[:5], y_regression[:5]
```

<div class="bash-block">
  <pre><code>100

(tensor([[0.0000],
         [0.0100],
         [0.0200],
         [0.0300],
         [0.0400]]),
 tensor([[0.3000],
         [0.3070],
         [0.3140],
         [0.3210],
         [0.3280]]))</code></pre>
</div>

Wonderful, now let's split our data into training and test sets.


```python
# Create train and test splits
train_split = int(0.8 * len(X_regression)) # 80% of data used for training set
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# Check the lengths of each split
print(len(X_train_regression), 
    len(y_train_regression), 
    len(X_test_regression), 
    len(y_test_regression))
```

<div class="bash-block">
  <pre><code>80 80 20 20</code></pre>
</div>

Beautiful, let's see how the data looks.

To do so, we'll use the `plot_predictions()` function we created in notebook 01. 

It's contained within the [`helper_functions.py` script](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py) on the Learn PyTorch for Deep Learning repo which we downloaded above.


```python
plot_predictions(train_data=X_train_regression,
    train_labels=y_train_regression,
    test_data=X_test_regression,
    test_labels=y_test_regression
);
```

<div style="text-align: left;">
  <img 
    src="{{ "/assets/img/02_pytorch_classification_files/02_pytorch_classification_71_0.png" | relative_url }}" 
    alt="png" 
    class="img-fluid" 
    style="max-width: 80%; height: auto; display: block; margin-bottom: 1rem;"
  />
</div>

### Adjusting `model_1` to fit a straight line

Now we've got some data, let's recreate `model_1` but with a loss function suited to our regression data.


```python
# Same architecture as model_1 (but using nn.Sequential)
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

model_2
```

<div class="bash-block">
  <pre><code>Sequential(
  (0): Linear(in_features=1, out_features=10, bias=True)
  (1): Linear(in_features=10, out_features=10, bias=True)
  (2): Linear(in_features=10, out_features=1, bias=True)
)</code></pre>
</div>

We'll setup the loss function to be `nn.L1Loss()` (the same as mean absolute error) and the optimizer to be `torch.optim.SGD()`. 


```python
# Loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)
```

Now let's train the model using the regular training loop steps for `epochs=1000` (just like `model_1`).

<div class="note-box">
  <strong>Note:</strong>
  <p>
    We've been writing similar training loop code over and over again. 
    I've made it that way on purpose though, to keep practicing.
  </p>
  <p>
    However, do you have ideas how we could functionize this? 
    That would save a fair bit of coding in the future. 
    Potentially there could be a function for training and a function for testing.
  </p>
</div>

```python
# Train the model
torch.manual_seed(42)

# Set the number of epochs
epochs = 1000

# Put data to target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

for epoch in range(epochs):
    ### Training 
    # 1. Forward pass
    y_pred = model_2(X_train_regression)
    
    # 2. Calculate loss (no accuracy since it's a regression problem, not classification)
    loss = loss_fn(y_pred, y_train_regression)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_2.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_pred = model_2(X_test_regression)
      # 2. Calculate the loss 
      test_loss = loss_fn(test_pred, y_test_regression)

    # Print out what's happening
    if epoch % 100 == 0: 
        print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")
```

<div class="bash-block">
  <pre><code>
Epoch: 0   | Train loss: 0.75986, Test loss: 0.54143
Epoch: 100 | Train loss: 0.09309, Test loss: 0.02901
Epoch: 200 | Train loss: 0.07376, Test loss: 0.02850
Epoch: 300 | Train loss: 0.06745, Test loss: 0.00615
Epoch: 400 | Train loss: 0.06107, Test loss: 0.02004
Epoch: 500 | Train loss: 0.05698, Test loss: 0.01061
Epoch: 600 | Train loss: 0.04857, Test loss: 0.01326
Epoch: 700 | Train loss: 0.06109, Test loss: 0.02127
Epoch: 800 | Train loss: 0.05599, Test loss: 0.01426
Epoch: 900 | Train loss: 0.05571, Test loss: 0.00603</code></pre>
</div>

Okay, unlike `model_1` on the classification data, it looks like `model_2`'s loss is actually going down.

Let's plot its predictions to see if that's so.

And remember, since our model and data are using the target `device`, and this device may be a GPU, however, our plotting function uses matplotlib and matplotlib can't handle data on the GPU.

To handle that, we'll send all of our data to the CPU using [`.cpu()`](https://pytorch.org/docs/stable/generated/torch.Tensor.cpu.html) when we pass it to `plot_predictions()`.

```python
# Turn on evaluation mode
model_2.eval()

# Make predictions (inference)
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

# Plot data and predictions with data on the CPU (matplotlib can't handle data on the GPU)
# (try removing .cpu() from one of the below and see what happens)
plot_predictions(train_data=X_train_regression.cpu(),
                 train_labels=y_train_regression.cpu(),
                 test_data=X_test_regression.cpu(),
                 test_labels=y_test_regression.cpu(),
                 predictions=y_preds.cpu());
```
<div style="text-align: left;">
  <img 
    src="{{ "/assets/img/02_pytorch_classification_files/02_pytorch_classification_79_0.png" | relative_url }}" 
    alt="png" 
    class="img-fluid" 
    style="max-width: 80%; height: auto; display: block; margin-bottom: 1rem;"
  />
</div>

Alright, it looks like our model is able to do far better than random guessing on straight lines.

This is a good thing.

It means our model at least has *some* capacity to learn.

