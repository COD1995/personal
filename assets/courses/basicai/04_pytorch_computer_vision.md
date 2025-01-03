---
layout: page
title: PyTorch Computer Vision
description: 
related_publications: false
toc:
    sidebar: right
back_link: '/teaching/aibasic'
back_text: 'AI Basics'
number_heading: true
enable_heading_styles: true
show_h1_number: true
start_h1_number: 3
---

[Computer vision](https://en.wikipedia.org/wiki/Computer_vision) is the art of teaching a computer to see.

For example, it could involve building a model to classify whether a photo is of a cat or a dog ([binary classification](https://developers.google.com/machine-learning/glossary#binary-classification)).

Or whether a photo is of a cat, dog or chicken ([multi-class classification](https://developers.google.com/machine-learning/glossary#multi-class-classification)).

Or identifying where a car appears in a video frame ([object detection](https://en.wikipedia.org/wiki/Object_detection)).

Or figuring out where different objects in an image can be separated ([panoptic segmentation](https://arxiv.org/abs/1801.00868)).

<div class="row mt-3">
  {% assign figure_counter = figure_counter | plus: 1 %}
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid
      figure_number=figure_counter
      loading="eager"
      path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-computer-vision-problems.png"
      class="img-fluid rounded"
      caption="Example computer vision problems for binary classification, multiclass classification, object detection and segmentation."
      id="example_computer_vision_problems"
    %}
  </div>
</div>

## Where does computer vision get used?

If you use a smartphone, you've already used computer vision.

Camera and photo apps use [computer vision to enhance](https://machinelearning.apple.com/research/panoptic-segmentation) and sort images.

Modern cars use [computer vision](https://youtu.be/j0z4FweCy4M?t=2989) to avoid other cars and stay within lane lines.

Manufacturers use computer vision to identify defects in various products.

Security cameras use computer vision to detect potential intruders.

In essence, anything that can be described in a visual sense can be a potential computer vision problem.

## What we're going to cover

We're going to apply the PyTorch Workflow we've been learning in the past couple of sections to computer vision.

<div class="row mt-3">
  {% assign figure_counter = figure_counter | plus: 1 %}
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid
      figure_number=figure_counter
      loading="eager"
      path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-pytorch-computer-vision-workflow.png"
      class="img-fluid rounded"
      caption="A PyTorch workflow with a computer vision focus."
      id="pytorch_workflow_computer_vision"
    %}
  </div>
</div>

Specifically, we're going to cover:

<div class="table-wrapper">
  <table class="styled-table">
    <thead>
      <tr>
        <th><strong>Topic</strong></th>
        <th><strong>Contents</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>0. Computer vision libraries in PyTorch</strong></td>
        <td>PyTorch has a bunch of built-in helpful computer vision libraries, let's check them out.</td>
      </tr>
      <tr>
        <td><strong>1. Load data</strong></td>
        <td>To practice computer vision, we'll start with some images of different pieces of clothing from <a href="https://github.com/zalandoresearch/fashion-mnist">FashionMNIST</a>.</td>
      </tr>
      <tr>
        <td><strong>2. Prepare data</strong></td>
        <td>We've got some images, let's load them in with a <a href="https://pytorch.org/docs/stable/data.html">PyTorch `DataLoader`</a> so we can use them with our training loop.</td>
      </tr>
      <tr>
        <td><strong>3. Model 0: Building a baseline model</strong></td>
        <td>Here we'll create a multi-class classification model to learn patterns in the data, we'll also choose a <strong>loss function</strong>, <strong>optimizer</strong>, and build a <strong>training loop</strong>.</td>
      </tr>
      <tr>
        <td><strong>4. Making predictions and evaluating model 0</strong></td>
        <td>Let's make some predictions with our baseline model and evaluate them.</td>
      </tr>
      <tr>
        <td><strong>5. Setup device agnostic code for future models</strong></td>
        <td>It's best practice to write device-agnostic code, so let's set it up.</td>
      </tr>
      <tr>
        <td><strong>6. Model 1: Adding non-linearity</strong></td>
        <td>Experimenting is a large part of machine learning, let's try and improve upon our baseline model by adding non-linear layers.</td>
      </tr>
      <tr>
        <td><strong>7. Model 2: Convolutional Neural Network (CNN)</strong></td>
        <td>Time to get computer vision specific and introduce the powerful convolutional neural network architecture.</td>
      </tr>
      <tr>
        <td><strong>8. Comparing our models</strong></td>
        <td>We've built three different models, let's compare them.</td>
      </tr>
      <tr>
        <td><strong>9. Evaluating our best model</strong></td>
        <td>Let's make some predictions on random images and evaluate our best model.</td>
      </tr>
      <tr>
        <td><strong>10. Making a confusion matrix</strong></td>
        <td>A confusion matrix is a great way to evaluate a classification model, let's see how we can make one.</td>
      </tr>
      <tr>
        <td><strong>11. Saving and loading the best performing model</strong></td>
        <td>Since we might want to use our model for later, let's save it and make sure it loads back in correctly.</td>
      </tr>
    </tbody>
  </table>
</div>

## Computer vision libraries in PyTorch

Before we get started writing code, let's talk about some PyTorch computer vision libraries you should be aware of.

<div class="table-wrapper">
  <table class="styled-table">
    <thead>
      <tr>
        <th><strong>PyTorch module</strong></th>
        <th><strong>What does it do?</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><a href="https://pytorch.org/vision/stable/index.html"><code>torchvision</code></a></td>
        <td>Contains datasets, model architectures, and image transformations often used for computer vision problems.</td>
      </tr>
      <tr>
        <td><a href="https://pytorch.org/vision/stable/datasets.html"><code>torchvision.datasets</code></a></td>
        <td>Here you'll find many example computer vision datasets for a range of problems from image classification, object detection, image captioning, video classification, and more. It also contains <a href="https://pytorch.org/vision/stable/datasets.html#base-classes-for-custom-datasets">a series of base classes for making custom datasets</a>.</td>
      </tr>
      <tr>
        <td><a href="https://pytorch.org/vision/stable/models.html"><code>torchvision.models</code></a></td>
        <td>This module contains well-performing and commonly used computer vision model architectures implemented in PyTorch. You can use these with your own problems.</td>
      </tr>
      <tr>
        <td><a href="https://pytorch.org/vision/stable/transforms.html"><code>torchvision.transforms</code></a></td>
        <td>Often images need to be transformed (turned into numbers/processed/augmented) before being used with a model. Common image transformations are found here.</td>
      </tr>
      <tr>
        <td><a href="https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset"><code>torch.utils.data.Dataset</code></a></td>
        <td>Base dataset class for PyTorch.</td>
      </tr>
      <tr>
        <td><a href="https://pytorch.org/docs/stable/data.html#module-torch.utils.data"><code>torch.utils.data.DataLoader</code></a></td>
        <td>Creates a Python iterable over a dataset (created with <code>torch.utils.data.Dataset</code>).</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="note-box">
  <strong>Note:</strong>
  <p>
    The <code>torch.utils.data.Dataset</code> and <code>torch.utils.data.DataLoader</code> classes aren't only for computer vision in PyTorch, they are capable of dealing with many different types of data.
  </p>
</div>

Now we've covered some of the most important PyTorch computer vision libraries, let's import the relevant dependencies.

```python
# Import PyTorch
import torch
from torch import nn

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Check versions
# Note: your PyTorch version shouldn't be lower than 1.10.0 and torchvision version shouldn't be lower than 0.11
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")
```

<div class="bash-block">
  <pre><code> PyTorch version: 2.0.1+cu118
torchvision version: 0.15.2+cu118
  </code></pre>
</div>
