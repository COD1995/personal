---
layout: page
title: PyTorch Custom Datasets
description: 
related_publications: false
toc:
    sidebar: right
back_link: '/teaching/aibasic'
back_text: 'AI Basics'
number_heading: true
enable_heading_styles: true
show_h1_number: true
start_h1_number: 4
---

In the last session, <a href="{{ '/assets/courses/basicai/04_pytorch_computer_vision' | relative_url }}">04 PyTorch Computer Vision</a>, we looked at how to build computer vision models on an in-built dataset in Pytorch (FashionMNIST).

The steps we took are similar across many different problems in machine learning. Find a dataset, turn the dataset into numbers, build a model (or find an existing model) to find patterns in those numbers that can be used for prediction. PyTorch has many built-in datasets used for a wide number of machine learning benchmarks, however, you'll often want to use your own 

## What is a custom dataset?

A **custom dataset** is a tailored collection of data specific to the problem you're solving.

It can include almost anything, such as:
- **Food images** for a classification app like [Nutrify](https://nutrify.app).
- **Customer reviews** with ratings for sentiment analysis.
- **Sound samples** with labels for a sound classification app.
- **Purchase histories** for building a recommendation system.

PyTorch has many built-in datasets used for a wide number of machine learning benchmarks, however, you'll often want to use your own **custom dataset**.
  
<div class="row mt-3">
  {% assign figure_counter = figure_counter | plus: 1 %}
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid
      figure_number=figure_counter
      loading="eager"
      path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pytorch-domain-libraries.png"
      class="img-fluid rounded"
      caption="PyTorch includes many existing functions to load in various custom datasets in the <a href='https://pytorch.org/vision/stable/index.html'>TorchVision</a>, <a href='https://pytorch.org/text/stable/index.html'>TorchText</a>, <a href='https://pytorch.org/audio/stable/index.html'>TorchAudio</a>, and <a href='https://pytorch.org/torchrec/'>TorchRec</a> domain libraries."
      id="pytorch_domain_libraries"
    %}
  </div>
</div>

While PyTorch provides a variety of built-in functions for loading datasets through libraries like `TorchVision`, `TorchText`, and `TorchAudio`, these predefined tools may not always meet the specific needs of your project.

In such cases, you can create a custom solution by subclassing `torch.utils.data.Dataset`. This approach allows you to define a dataset tailored to your unique requirements. By implementing the `__init__`, `__len__`, and `__getitem__` methods, you can handle specific data formats, apply custom preprocessing, and control how data samples are accessed and utilized during training.

Customizing a dataset gives you the flexibility to work with non-standard data types, formats, or use cases that aren't supported by default PyTorch utilities. This ensures your model has access to the data it needs in exactly the way you intend.

## What We're Going to Cover

In this section, we’ll apply the PyTorch Workflow introduced in 
<a href="{{ '/assets/courses/basicai/02_pytorch_workflow' | relative_url }}">pytorch workflow</a> and 
<a href="{{ 'assets/courses/basicai/03_pytorch_classification' | relative_url }}">pytorch classification</a> to solve a computer vision problem.

Unlike previous examples where we used pre-built datasets from PyTorch libraries, we’ll work with a custom dataset containing images of pizza, steak, and sushi. This provides an opportunity to explore how to handle unique datasets that aren't available out-of-the-box.

Our objective is to load these custom images, preprocess them appropriately, and then build a model capable of learning from this data. By the end, our model will be able to make predictions on unseen images, demonstrating its ability to classify different types of food accurately.

<div class="row mt-3">
  {% assign figure_counter = figure_counter | plus: 1 %}
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid
      figure_number=figure_counter
      loading="eager"
      path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pytorch-food-vision-layout.png"
      class="img-fluid rounded"
      caption="What we're going to build. We'll use <code>torchvision.datasets</code> as well as our own custom <code>Dataset</code> class to load in images of food and then we'll build a PyTorch computer vision model to hopefully be able to classify them."
      id="food_vision_pipeline"
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
        <td><strong>0. Importing PyTorch and setting up device-agnostic code</strong></td>
        <td>Let's get PyTorch loaded and then follow best practice to setup our code to be device-agnostic.</td>
      </tr>
      <tr>
        <td><strong>1. Get data</strong></td>
        <td>We're going to be using our own <strong>custom dataset</strong> of pizza, steak and sushi images.</td>
      </tr>
      <tr>
        <td><strong>2. Become one with the data (data preparation)</strong></td>
        <td>At the beginning of any new machine learning problem, it's paramount to understand the data you're working with. Here we'll take some steps to figure out what data we have.</td>
      </tr>
      <tr>
        <td><strong>3. Transforming data</strong></td>
        <td>Often, the data you get won't be 100% ready to use with a machine learning model, here we'll look at some steps we can take to <em>transform</em> our images so they're ready to be used with a model.</td>
      </tr>
      <tr>
        <td><strong>4. Loading data with <code>ImageFolder</code> (option 1)</strong></td>
        <td>PyTorch has many in-built data loading functions for common types of data. <code>ImageFolder</code> is helpful if our images are in standard image classification format.</td>
      </tr>
      <tr>
        <td><strong>5. Loading image data with a custom <code>Dataset</code></strong></td>
        <td>What if PyTorch didn't have an in-built function to load data with? This is where we can build our own custom subclass of <code>torch.utils.data.Dataset</code>.</td>
      </tr>
      <tr>
        <td><strong>6. Other forms of transforms (data augmentation)</strong></td>
        <td>Data augmentation is a common technique for expanding the diversity of your training data. Here we'll explore some of <code>torchvision</code>'s in-built data augmentation functions.</td>
      </tr>
      <tr>
        <td><strong>7. Model 0: TinyVGG without data augmentation</strong></td>
        <td>By this stage, we'll have our data ready, let's build a model capable of fitting it. We'll also create some training and testing functions for training and evaluating our model.</td>
      </tr>
      <tr>
        <td><strong>8. Exploring loss curves</strong></td>
        <td>Loss curves are a great way to see how your model is training/improving over time. They're also a good way to see if your model is <strong>underfitting</strong> or <strong>overfitting</strong>.</td>
      </tr>
      <tr>
        <td><strong>9. Model 1: TinyVGG with data augmentation</strong></td>
        <td>By now, we've tried a model <em>without</em>, how about we try one <em>with</em> data augmentation?</td>
      </tr>
      <tr>
        <td><strong>10. Compare model results</strong></td>
        <td>Let's compare our different models' loss curves and see which performed better and discuss some options for improving performance.</td>
      </tr>
      <tr>
        <td><strong>11. Making a prediction on a custom image</strong></td>
        <td>Our model is trained on a dataset of pizza, steak and sushi images. In this section we'll cover how to use our trained model to predict on an image <em>outside</em> of our existing dataset.</td>
      </tr>
    </tbody>
  </table>
</div>

## Importing PyTorch and setting up device-agnostic code

```python
import torch
from torch import nn

# Note: this notebook requires torch >= 1.10.0
torch.__version__
```