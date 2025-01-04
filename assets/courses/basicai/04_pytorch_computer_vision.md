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

## Getting a dataset

We're going to start with FashionMNIST.

MNIST stands for Modified National Institute of Standards and Technology.

The [original MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) contains thousands of examples of handwritten digits (from 0 to 9) and was used to build computer vision models to identify numbers for postal services.

[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), made by Zalando Research, is a similar setup.

Except it contains grayscale images of 10 different kinds of clothing.

<div class="row mt-3">
  {% assign figure_counter = figure_counter | plus: 1 %}
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid
      figure_number=figure_counter
      loading="eager"
      path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-fashion-mnist-slide.png"
      class="img-fluid rounded"
      caption="`torchvision.datasets` contains a lot of example datasets you can use to practice writing computer vision code on. FashionMNIST is one of those datasets. And since it has 10 different image classes (different types of clothing), it's a multi-class classification problem."
      id="fashionmnist_example"
    %}
  </div>
</div>

Later, we'll be building a computer vision neural network to identify the different styles of clothing in these images.

PyTorch has a bunch of common computer vision datasets stored in `torchvision.datasets`.

Including FashionMNIST in [`torchvision.datasets.FashionMNIST()`](https://pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html).

To download it, we provide the following parameters:
* `root: str` - which folder do you want to download the data to?
* `train: Bool` - do you want the training or test split?
* `download: Bool` - should the data be downloaded?
* `transform: torchvision.transforms` - what transformations would you like to do on the data?
* `target_transform` - you can transform the targets (labels) if you like too.
  
Many other datasets in `torchvision` have these parameter options.

```python
# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)
```

<div class="bash-block">
  <pre><code> Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

100%|██████████| 26421880/26421880 [00:01<00:00, 16189161.14it/s]

Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

100%|██████████| 29515/29515 [00:00<00:00, 269809.67it/s]

Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

100%|██████████| 4422102/4422102 [00:00<00:00, 4950701.58it/s]

Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

100%|██████████| 5148/5148 [00:00<00:00, 4744512.63it/s]

Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
  </code></pre>
</div>

Let's check out the first sample of the training data.

```python
# See first training sample
image, label = train_data[0]
image, label
```

<div class="bash-block">
  <pre><code> tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000],
         ...
          (remaining rows truncated for brevity)
         ...]]),
 9)
  </code></pre>
</div>

### Input and output shapes of a computer vision model

We've got a big tensor of values (the image) leading to a single value for the target (the label).

Let's see the image shape.


```python
# What's the shape of the image?
image.shape
```

<div class="bash-block">
  <pre><code> torch.Size([1, 28, 28])
  </code></pre>
</div>

The shape of the image tensor is `[1, 28, 28]` or more specifically:

```
[color_channels=1, height=28, width=28]
```

Having `color_channels=1` means the image is grayscale.

<div class="row mt-3">
  {% assign figure_counter = figure_counter | plus: 1 %}
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid
      figure_number=figure_counter
      loading="eager"
      path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-computer-vision-input-and-output-shapes.png"
      class="img-fluid rounded"
      caption="Various problems will have various input and output shapes. But the premise remains: encode data into numbers, build a model to find patterns in those numbers, convert those patterns into something meaningful."
      id="fashionmnist_input_output_shapes"
    %}
  </div>
</div>

If `color_channels=3`, the image comes in pixel values for red, green and blue (this is also known as the [RGB color model](https://en.wikipedia.org/wiki/RGB_color_model)).

The order of our current tensor is often referred to as `CHW` (Color Channels, Height, Width).

There's debate on whether images should be represented as `CHW` (color channels first) or `HWC` (color channels last).

<div class="note-box">
  <strong>Note:</strong>
  <p>
    You'll also see <code>NCHW</code> and <code>NHWC</code> formats where <code>N</code> stands for <em>number of images</em>. For example, if you have a <code>batch_size=32</code>, your tensor shape may be <code>[32, 1, 28, 28]</code>. We'll cover batch sizes later.
  </p>
</div>

PyTorch generally accepts `NCHW` (channels first) as the default for many operators.

However, PyTorch also explains that `NHWC` (channels last) performs better and is [considered best practice](https://pytorch.org/blog/tensor-memory-format-matters/#pytorch-best-practice).

For now, since our dataset and models are relatively small, this won't make too much of a difference.

But keep it in mind for when you're working on larger image datasets and using convolutional neural networks (we'll see these later).

Let's check out more shapes of our data.


```python
# How many samples are there?
len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets)
```

<div class="bash-block">
  <pre><code> (60000, 60000, 10000, 10000)
</code></pre>
</div>

So we've got 60,000 training samples and 10,000 testing samples.

What classes are there?

We can find these via the `.classes` attribute.


```python
# See classes
class_names = train_data.classes
class_names
```

<div class="bash-block">
  <pre><code> ['T-shirt/top',
 'Trouser',
 'Pullover',
 'Dress',
 'Coat',
 'Sandal',
 'Shirt',
 'Sneaker',
 'Bag',
 'Ankle boot']
  </code></pre>
</div>

Sweet! It looks like we're dealing with 10 different kinds of clothes.

Because we're working with 10 different classes, it means our problem is **multi-class classification**.

Let's get visual.

### Visualizing our data

```python
import matplotlib.pyplot as plt
image, label = train_data[0]
print(f"Image shape: {image.shape}")
plt.imshow(image.squeeze()) # image shape is [1, 28, 28] (colour channels, height, width)
plt.title(label);
```

<div class="bash-block">
  <pre><code> Image shape: torch.Size([1, 28, 28])
  </code></pre>
</div>
<div style="text-align: left;">
  <img 
    src="{{ "/assets/img/03_pytorch_computer_vision_files/03_pytorch_computer_vision_19_1.png" | relative_url }}" 
    alt="03 PyTorch Computer Vision Sample"
    class="img-fluid" 
    style="max-width: 80%; height: auto; display: block; margin-bottom: 1rem;"
  />
</div>

We can turn the image into grayscale using the `cmap` parameter of `plt.imshow()`.


```python
plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label]);
```

<div style="text-align: left;">
  <img 
    src="{{ "/assets/img/03_pytorch_computer_vision_files/03_pytorch_computer_vision_21_0.png" | relative_url }}" 
    alt="03 PyTorch Computer Vision Visualization"
    class="img-fluid" 
    style="max-width: 80%; height: auto; display: block; margin-bottom: 1rem;"
  />
</div>

Beautiful, well as beautiful as a pixelated grayscale ankle boot can get.

Let's view a few more.


```python
# Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False);
```
<div style="text-align: left;">
  <img 
    src="{{ "/assets/img/03_pytorch_computer_vision_files/03_pytorch_computer_vision_23_0.png" | relative_url }}" 
    alt="03 PyTorch Computer Vision Sample"
    class="img-fluid" 
    style="max-width: 80%; height: auto; display: block; margin-bottom: 1rem;"
  />
</div>

<div class="note-box">
  <strong>Question:</strong>
  <p>
    Do you think the above data can be modeled with only straight (linear) lines? Or do you think you'd also need non-straight (non-linear) lines?
  </p>
</div>

## Prepare DataLoader

Now we've got a dataset ready to go.

The next step is to prepare it with a [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) or `DataLoader` for short.

The `DataLoader` does what you think it might do.

It helps load data into a model.

For training and for inference.

It turns a large `Dataset` into a Python iterable of smaller chunks.

These smaller chunks are called **batches** or **mini-batches** and can be set by the `batch_size` parameter.

Why do this?

Because it's more computationally efficient.

In an ideal world you could do the forward pass and backward pass across all of your data at once.

But once you start using really large datasets, unless you've got infinite computing power, it's easier to break them up into batches.

It also gives your model more opportunities to improve.

With **mini-batches** (small portions of the data), gradient descent is performed more often per epoch (once per mini-batch rather than once per epoch).

What's a good batch size?

[32 is a good place to start](https://twitter.com/ylecun/status/989610208497360896?s=20&t=N96J_jotN--PYuJk2WcjMw) for a fair amount of problems.

But since this is a value you can set (a **hyperparameter**) you can try all different kinds of values, though generally powers of 2 are used most often (e.g. 32, 64, 128, 256, 512).

<div class="row mt-3">
  {% assign figure_counter = figure_counter | plus: 1 %}
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid
      figure_number=figure_counter
      loading="eager"
      path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-batching-fashionmnist.png"
      class="img-fluid rounded"
      caption="Batching FashionMNIST with a batch size of 32 and shuffle turned on. A similar batching process will occur for other datasets but will differ depending on the batch size."
      id="batching_fashionmnist"
    %}
  </div>
</div>

Let's create `DataLoader`'s for our training and test sets.


```python
from torch.utils.data import DataLoader

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch?
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Let's check out what we've created
print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
```

<div class="bash-block">
  <pre><code> Dataloaders: (<torch.utils.data.dataloader.DataLoader object at 0x7fc991463cd0>, <torch.utils.data.dataloader.DataLoader object at 0x7fc991475120>)
Length of train dataloader: 1875 batches of 32
Length of test dataloader: 313 batches of 32
  </code></pre>
</div>

```python
# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape
```

<div class="bash-block">
  <pre><code> (torch.Size([32, 1, 28, 28]), torch.Size([32]))
  </code></pre>
</div>

And we can see that the data remains unchanged by checking a single sample.


```python
# Show a sample
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis("Off");
print(f"Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")
```

<div class="bash-block">
  <pre><code> Image size: torch.Size([1, 28, 28])
Label: 6, label size: torch.Size([])
  </code></pre>
</div>

<div style="text-align: left;">
  <img 
    src="{{ "/assets/img/03_pytorch_computer_vision_files/03_pytorch_computer_vision_29_1.png" | relative_url }}" 
    alt="03 PyTorch Computer Vision Visualization"
    class="img-fluid" 
    style="max-width: 80%; height: auto; display: block; margin-bottom: 1rem;"
  />
</div>

## Model 0: Build a baseline model

Data loaded and prepared!

Time to build a **baseline model** by subclassing `nn.Module`.

A **baseline model** is one of the simplest models you can imagine.

You use the baseline as a starting point and try to improve upon it with subsequent, more complicated models.

Our baseline will consist of two [`nn.Linear()`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) layers.

We've done this in a previous section but there's going to be one slight difference.

Because we're working with image data, we're going to use a different layer to start things off.

And that's the [`nn.Flatten()`](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) layer.

`nn.Flatten()` compresses the dimensions of a tensor into a single vector.

This is easier to understand when you see it.


```python
# Create a flatten layer
flatten_model = nn.Flatten() # all nn modules function as a model (can do a forward pass)

# Get a single sample
x = train_features_batch[0]

# Flatten the sample
output = flatten_model(x) # perform forward pass

# Print out what happened
print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")

# Try uncommenting below and see what happens
#print(x)
#print(output)
```
<div class="bash-block">
  <pre><code> Shape before flattening: torch.Size([1, 28, 28]) -> [color_channels, height, width]
Shape after flattening: torch.Size([1, 784]) -> [color_channels, height*width]
  </code></pre>
</div>

The `nn.Flatten()` layer took our shape from `[color_channels, height, width]` to `[color_channels, height*width]`.

Why do this?

Because we've now turned our pixel data from height and width dimensions into one long **feature vector**.

And `nn.Linear()` layers like their inputs to be in the form of feature vectors.

Let's create our first model using `nn.Flatten()` as the first layer.


```python
from torch import nn
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)
```

Wonderful!

We've got a baseline model class we can use, now let's instantiate a model.

We'll need to set the following parameters:
* `input_shape=784` - this is how many features you've got going in the model, in our case, it's one for every pixel in the target image (28 pixels high by 28 pixels wide = 784 features).
* `hidden_units=10` - number of units/neurons in the hidden layer(s), this number could be whatever you want but to keep the model small we'll start with `10`.
* `output_shape=len(class_names)` - since we're working with a multi-class classification problem, we need an output neuron per class in our dataset.

Let's create an instance of our model and send to the CPU for now (we'll run a small test for running `model_0` on CPU vs. a similar model on GPU soon).

```python
torch.manual_seed(42)

# Need to setup model with input parameters
model_0 = FashionMNISTModelV0(input_shape=784, # one for every pixel (28x28)
    hidden_units=10, # how many units in the hidden layer
    output_shape=len(class_names) # one for every class
)
model_0.to("cpu") # keep model on CPU to begin with
```

<div class="bash-block">
  <pre><code> FashionMNISTModelV0(
  (layer_stack): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=10, bias=True)
    (2): Linear(in_features=10, out_features=10, bias=True)
  )
)
  </code></pre>
</div>

### Setup loss, optimizer and evaluation metrics

Since we're working on a classification problem, let's bring in our [`helper_functions.py` script](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py) and subsequently the `accuracy_fn()` we defined in [notebook 02](https://www.learnpytorch.io/02_pytorch_classification/).

<div class="note-box">
  <strong>Note:</strong>
  <p>
    Rather than importing and using our own accuracy function or evaluation metric(s), you could import various evaluation metrics from the <a href="https://torchmetrics.readthedocs.io/en/latest/">TorchMetrics package</a>.
  </p>
</div>

```python
import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)
```

<div class="bash-block">
  <pre><code> helper_functions.py already exists, skipping download
  </code></pre> 
</div>

```python
# Import accuracy metric
from helper_functions import accuracy_fn # Note: could also use torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # this is also called "criterion"/"cost function" in some places
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
```

### Creating a function to time our experiments

Loss function and optimizer ready!

It's time to start training a model.

But how about we do a little experiment while we train.

I mean, let's make a timing function to measure the time it takes our model to train on CPU versus using a GPU.

We'll train this model on the CPU but the next one on the GPU and see what happens.

Our timing function will import the [`timeit.default_timer()` function](https://docs.python.org/3/library/timeit.html#timeit.default_timer) from the Python [`timeit` module](https://docs.python.org/3/library/timeit.html).

```python
from timeit import default_timer as timer
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
```

### Creating a training loop and training a model on batches of data

Beautiful!

Looks like we've got all of the pieces of the puzzle ready to go, a timer, a loss function, an optimizer, a model and most importantly, some data.

Let's now create a training loop and a testing loop to train and evaluate our model.

We'll be using the same steps as the previous notebook(s), though since our data is now in batch form, we'll add another loop to loop through our data batches.

Our data batches are contained within our `DataLoader`s, `train_dataloader` and `test_dataloader` for the training and test data splits respectively.

A batch is `BATCH_SIZE` samples of `X` (features) and `y` (labels), since we're using `BATCH_SIZE=32`, our batches have 32 samples of images and targets.

And since we're computing on batches of data, our loss and evaluation metrics will be calculated **per batch** rather than across the whole dataset.

This means we'll have to divide our loss and accuracy values by the number of batches in each dataset's respective dataloader.

Let's step through it:
  1. Loop through epochs.
  2. Loop through training batches, perform training steps, calculate the train loss *per batch*.
  3. Loop through testing batches, perform testing steps, calculate the test loss *per batch*.
  4. Print out what's happening.
  5. Time it all (for fun).

A fair few steps but...

...if in doubt, code it out.

```python
# Import tqdm for progress bar
from tqdm.auto import tqdm

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 3

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    ### Training
    train_loss = 0
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        # 1. Forward pass
        y_pred = model_0(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively add up the loss per epoch

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X)

            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)

    ## Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# Calculate training time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_0.parameters()).device))
```

<div class="bash-block">
  <pre><code> 0%|          | 0/3 [00:00<?, ?it/s]

Epoch: 0
-------
Looked at 0/60000 samples
Looked at 12800/60000 samples
Looked at 25600/60000 samples
Looked at 38400/60000 samples
Looked at 51200/60000 samples

Train loss: 0.59039 | Test loss: 0.50954, Test acc: 82.04%

Epoch: 1
-------
Looked at 0/60000 samples
Looked at 12800/60000 samples
Looked at 25600/60000 samples
Looked at 38400/60000 samples
Looked at 51200/60000 samples

Train loss: 0.47633 | Test loss: 0.47989, Test acc: 83.20%

Epoch: 2
-------
Looked at 0/60000 samples
Looked at 12800/60000 samples
Looked at 25600/60000 samples
Looked at 38400/60000 samples
Looked at 51200/60000 samples

Train loss: 0.45503 | Test loss: 0.47664, Test acc: 83.43%

Train time on cpu: 32.349 seconds
  </code></pre>
</div>

Nice! Looks like our baseline model did fairly well.

It didn't take too long to train either, even just on the CPU, I wonder if it'll speed up on the GPU?

Let's write some code to evaluate our model.

## Make predictions and get Model 0 results

Since we're going to be building a few models, it's a good idea to write some code to evaluate them all in similar ways.

Namely, let's create a function that takes in a trained model, a `DataLoader`, a loss function and an accuracy function.

The function will use the model to make predictions on the data in the `DataLoader` and then we can evaluate those predictions using the loss function and accuracy function.


```python
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn
)
model_0_results
```