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

<div class="bash-block">
  <pre><code> {'model_name': 'FashionMNISTModelV0',
 'model_loss': 0.47663894295692444,
 'model_acc': 83.42651757188499}
  </code></pre>
</div>

Looking good!

We can use this dictionary to compare the baseline model results to other models later on.

## Setup device agnostic-code (for using a GPU if there is one)
We've seen how long it takes to train ma PyTorch model on 60,000 samples on CPU.

<div class="note-box">
  <strong>Note:</strong>
  <p>
    Model training time is dependent on hardware used. Generally, more processors mean faster training, and smaller models on smaller datasets will often train faster than large models and large datasets.
  </p>
</div>

Now let's setup some [device-agnostic code](https://pytorch.org/docs/stable/notes/cuda.html#best-practices) for our models and data to run on GPU if it's available.

If you're running this notebook on Google Colab, and you don't have a GPU turned on yet, it's now time to turn one on via `Runtime -> Change runtime type -> Hardware accelerator -> GPU`. If you do this, your runtime will likely reset and you'll have to run all of the cells above by going `Runtime -> Run before`.


```python
# Setup device agnostic code
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

<div class="bash-block">
  <pre><code> 'cuda'
  </code></pre>
</div>

Beautiful!

Let's build another model.

## Model 1: Building a better model with non-linearity

We learned about <a href="{{ '/assets/courses/basicai/03_pytorch_classification' | relative_url }}">the power of non-linearity</a> in previous session.


Seeing the data we've been working with, do you think it needs non-linear functions?

And remember, linear means straight and non-linear means non-straight.

Let's find out.

We'll do so by recreating a similar model to before, except this time we'll put non-linear functions (`nn.ReLU()`) in between each linear layer.


```python
# Create a model with non-linear and linear layers
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # flatten inputs into single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)
```

That looks good.

Now let's instantiate it with the same settings we used before.

We'll need `input_shape=784` (equal to the number of features of our image data), `hidden_units=10` (starting small and the same as our baseline model) and `output_shape=len(class_names)` (one output unit per class).

<div class="note-box">
  <strong>Note:</strong>
  <p>
    Notice how we kept most of the settings of our model the same except for one change: adding non-linear layers. This is a standard practice for running a series of machine learning experiments — change one thing and see what happens, then do it again, again, again.
  </p>
</div>

```python
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784, # number of input features
    hidden_units=10,
    output_shape=len(class_names) # number of output classes desired
).to(device) # send model to GPU if it's available
next(model_1.parameters()).device # check model device
```

<div class="bash-block">
  <pre><code>device(type='cuda', index=0)
  </code></pre>
</div>

### Setup loss, optimizer and evaluation metrics

As usual, we'll setup a loss function, an optimizer and an evaluation metric (we could do multiple evaluation metrics but we'll stick with accuracy for now).

```python
from helper_functions import accuracy_fn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)
```

### Functionizing training and test loops

So far we've been writing train and test loops over and over.

Let's write them again but this time we'll put them in functions so they can be called again and again.

And because we're using device-agnostic code now, we'll be sure to call `.to(device)` on our feature (`X`) and target (`y`) tensors.

For the training loop we'll create a function called `train_step()` which takes in a model, a `DataLoader` a loss function and an optimizer.

The testing loop will be similar but it'll be called `test_step()` and it'll take in a model, a `DataLoader`, a loss function and an evaluation function.

<div class="note-box">
  <strong>Note:</strong>
  <p>
    Since these are functions, you can customize them in any way you like. What we're making here can be considered barebones training and testing functions for our specific classification use case.
  </p>
</div>

```python
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
```

Woohoo!

Now we've got some functions for training and testing our model, let's run them.

We'll do so inside another loop for each epoch.

That way, for each epoch, we're going through a training step and a testing step.

<div class="note-box">
  <strong>Note:</strong>
  <p>
    You can customize how often you do a testing step. Sometimes people do them every five epochs or 10 epochs or, in our case, every epoch.
  </p>
</div>

Let's also time things to see how long our code takes to run on the GPU.


```python
torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
train_time_start_on_gpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader,
        model=model_1,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )
    test_step(data_loader=test_dataloader,
        model=model_1,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)
```

<div class="bash-block">
  <pre><code> 0%|          | 0/3 [00:00<?, ?it/s]

Epoch: 0
---------
Train loss: 1.09199 | Train accuracy: 61.34%
Test loss: 0.95636 | Test accuracy: 65.00%

Epoch: 1
---------
Train loss: 0.78101 | Train accuracy: 71.93%
Test loss: 0.72227 | Test accuracy: 73.91%

Epoch: 2
---------
Train loss: 0.67027 | Train accuracy: 75.94%
Test loss: 0.68500 | Test accuracy: 75.02%

Train time on cuda: 36.878 seconds
  </code></pre>
</div>

Excellent!

Our model trained but the training time took longer?

<div class="note-box">
  <strong>Note:</strong>
  <p>
    The training time on CUDA vs CPU will depend largely on the quality of the CPU/GPU you're using. Read on for a more explained answer.
  </p>
</div>

<div class="note-box">
  <strong>Question:</strong>
  <p>
    "I used a GPU but my model didn't train faster, why might that be?"
  </p>
  <strong>Answer:</strong>
  <p>
    Well, one reason could be because your dataset and model are both so small (like the dataset and model we're working with) the benefits of using a GPU are outweighed by the time it actually takes to transfer the data there.
  </p>
  <p>
    There's a small bottleneck between copying data from the CPU memory (default) to the GPU memory.
  </p>
  <p>
    So for smaller models and datasets, the CPU might actually be the optimal place to compute on.
  </p>
  <p>
    But for larger datasets and models, the speed of computing the GPU can offer usually far outweighs the cost of getting the data there.
  </p>
  <p>
    However, this is largely dependent on the hardware you're using. With practice, you will get used to where the best place to train your models is.
  </p>
</div>

Let's evaluate our trained `model_1` using our `eval_model()` function and see how it went.


```python
torch.manual_seed(42)

# Note: This will error due to `eval_model()` not using device agnostic code
model_1_results = eval_model(model=model_1,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn)
model_1_results
```

<div class="bash-block">
  <pre><code class="traceback">
---------------------------------------------------------------------------

RuntimeError                              Traceback (most recent call last)

&lt;cell line: 4&gt;
# Note: This will error due to `eval_model()` not using device agnostic code
----&gt; model_1_results = eval_model(model=model_1, 
                                     data_loader=test_dataloader,
                                     loss_fn=loss_fn)

&lt;ipython-input-20&gt; in eval_model(model, data_loader, loss_fn, accuracy_fn)
# Make predictions with the model
----&gt; y_pred = model(X)

&lt;ipython-input-22&gt; in forward(self, x)
----&gt; return self.layer_stack(x)

&lt;usr/local/lib/python3.10/dist-packages/torch/nn/modules/linear.py&gt; in forward(self, input)
----&gt; return F.linear(input, self.weight, self.bias)

<span class="error-msg">RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!</span>
  </code></pre>
</div>

Oh no!

It looks like our `eval_model()` function errors out with:

<div class="note-box">
  <strong>Error:</strong>
  <p>
    <code>RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat1 in method wrapper_addmm)</code>
  </p>
</div>

It's because we've setup our data and model to use device-agnostic code but not our evaluation function.

How about we fix that by passing a target `device` parameter to our `eval_model()` function?

Then we'll try calculating the results again.

```python
# Move values to device
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# Calculate model 1 results with device-agnostic code
model_1_results = eval_model(model=model_1, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn,
    device=device
)
model_1_results
```

<div class="bash-block">
  <pre><code> {'model_name': 'FashionMNISTModelV1',
 'model_loss': 0.6850008964538574,
 'model_acc': 75.01996805111821}
  </code></pre>
</div>

```python
# Check baseline results
model_0_results
```
<div class="bash-block">
  <pre><code> {'model_name': 'FashionMNISTModelV0',
 'model_loss': 0.47663894295692444,
 'model_acc': 83.42651757188499}
  </code></pre>
</div>

Woah, in this case, it looks like adding non-linearities to our model made it perform worse than the baseline.

That's a thing to note in machine learning, sometimes the thing you thought should work doesn't.

And then the thing you thought might not work does.

It's part science, part art.

From the looks of things, it seems like our model is **overfitting** on the training data.

Overfitting means our model is learning the training data well but those patterns aren't generalizing to the testing data.

Two of the main ways to fix overfitting include:
1. Using a smaller or different model (some models fit certain kinds of data better than others).
2. Using a larger dataset (the more data, the more chance a model has to learn generalizable patterns).

There are more, but I'm going to leave that as a challenge for you to explore.

Try searching online, "ways to prevent overfitting in machine learning" and see what comes up.

In the meantime, let's take a look at number 1: using a different model.

## Model 2: Building a Convolutional Neural Network (CNN)

Alright, time to step things up a notch.

It's time to create a [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNN or ConvNet).

CNN's are known for their capabilities to find patterns in visual data.

And since we're dealing with visual data, let's see if using a CNN model can improve upon our baseline.

The CNN model we're going to be using is known as TinyVGG from the [CNN Explainer](https://poloclub.github.io/cnn-explainer/) website.

It follows the typical structure of a convolutional neural network:

`Input layer -> [Convolutional layer -> activation layer -> pooling layer] -> Output layer`

Where the contents of `[Convolutional layer -> activation layer -> pooling layer]` can be upscaled and repeated multiple times, depending on requirements.

### What model should I use?

<div class="note-box">
  <strong>Question:</strong>
  <p>
    Wait, you say CNN's are good for images, are there any other model types I should be aware of?
  </p>
</div>

Good question.

This table is a good general guide for which model to use (though there are exceptions).

<div class="table-wrapper">
  <table class="styled-table">
    <thead>
      <tr>
        <th><strong>Problem type</strong></th>
        <th><strong>Model to use (generally)</strong></th>
        <th><strong>Code example</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Structured data (Excel spreadsheets, row and column data)</td>
        <td>Gradient boosted models, Random Forests, XGBoost</td>
        <td>
          <a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble"><code>sklearn.ensemble</code></a>, 
          <a href="https://xgboost.readthedocs.io/en/stable/">XGBoost library</a>
        </td>
      </tr>
      <tr>
        <td>Unstructured data (images, audio, language)</td>
        <td>Convolutional Neural Networks, Transformers</td>
        <td>
          <a href="https://pytorch.org/vision/stable/models.html"><code>torchvision.models</code></a>, 
          <a href="https://huggingface.co/docs/transformers/index">HuggingFace Transformers</a>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<div class="note-box">
  <strong>Note:</strong>
  <p>
    The table above is only for reference. The model you end up using will be highly dependent on the problem you're working on and the constraints you have (amount of data, latency requirements).
  </p>
</div>

Enough talking about models, let's now build a CNN that replicates the model on the [CNN Explainer website](https://poloclub.github.io/cnn-explainer/).

<div class="row mt-3">
  {% assign figure_counter = figure_counter | plus: 1 %}
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid
      figure_number=figure_counter
      loading="eager"
      path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-cnn-explainer-model.png"
      class="img-fluid rounded"
      caption="TinyVGG architecture, as setup by CNN explainer website."
      id="tinyvgg_architecture"
    %}
  </div>
</div>

To do so, we'll leverage the [`nn.Conv2d()`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) and [`nn.MaxPool2d()`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) layers from `torch.nn`.

```python
# Create a convolutional neural network
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,
    hidden_units=10,
    output_shape=len(class_names)).to(device)
model_2
```

<div class="bash-block">
  <pre><code>FashionMNISTModelV2(
  (block_1): Sequential(
    (0): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (block_2): Sequential(
    (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=490, out_features=10, bias=True)
  )
)
  </code></pre>
</div>

Nice!

Our biggest model yet!

What we've done is a common practice in machine learning.

Find a model architecture somewhere and replicate it with code.

### Stepping through `nn.Conv2d()`

In this course we will not go through the theoretical details of convolutional neural networks. If you are interested please check 


We could start using our model above and see what happens but let's first step through the two new layers we've added:
* [`nn.Conv2d()`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), also known as a convolutional layer.
* [`nn.MaxPool2d()`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html), also known as a max pooling layer.

<div class="note-box">
  <strong>Question:</strong>
  <p>
    What does the "2d" in <code>nn.Conv2d()</code> stand for?
  </p>
  <strong>Answer:</strong>
  <p>
    The 2d is for 2-dimensional data. As in, our images have two dimensions: height and width. Yes, there's a color channel dimension, but each of the color channel dimensions has two dimensions too: height and width.
  </p>
  <p>
    For other dimensional data (such as 1D for text or 3D for 3D objects), there's also <code>nn.Conv1d()</code> and <code>nn.Conv3d()</code>.
  </p>
</div>

To test the layers out, let's create some toy data just like the data used on CNN Explainer.

```python
torch.manual_seed(42)

# Create sample batch of random numbers with same size as image batch
images = torch.randn(size=(32, 3, 64, 64)) # [batch_size, color_channels, height, width]
test_image = images[0] # get a single image for testing
print(f"Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]")
print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]")
print(f"Single image pixel values:\n{test_image}")
```

<div class="bash-block">
  <pre><code>Image batch shape: torch.Size([32, 3, 64, 64]) -> [batch_size, color_channels, height, width]
Single image shape: torch.Size([3, 64, 64]) -> [color_channels, height, width]
Single image pixel values:
tensor([[[ 1.9269,  1.4873,  0.9007,  ...,  1.8446, -1.1845,  1.3835],
         [ 1.4451,  0.8564,  2.2181,  ...,  0.3399,  0.7200,  0.4114],
         [ 1.9312,  1.0119, -1.4364,  ..., -0.5558,  0.7043,  0.7099],
         ...,
         [-0.5610, -0.4830,  0.4770,  ..., -0.2713, -0.9537, -0.6737],
         [ 0.3076, -0.1277,  0.0366,  ..., -2.0060,  0.2824, -0.8111],
         [-1.5486,  0.0485, -0.7712,  ..., -0.1403,  0.9416, -0.0118]],

        [[-0.5197,  1.8524,  1.8365,  ...,  0.8935, -1.5114, -0.8515],
         [ 2.0818,  1.0677, -1.4277,  ...,  1.6612, -2.6223, -0.4319],
         [-0.1010, -0.4388, -1.9775,  ...,  0.2106,  0.2536, -0.7318],
         ...,
         [ 0.2779,  0.7342, -0.3736,  ..., -0.4601,  0.1815,  0.1850],
         [ 0.7205, -0.2833,  0.0937,  ..., -0.1002, -2.3609,  2.2465],
         [-1.3242, -0.1973,  0.2920,  ...,  0.5409,  0.6940,  1.8563]],

        [[-0.7978,  1.0261,  1.1465,  ...,  1.2134,  0.9354, -0.0780],
         [-1.4647, -1.9571,  0.1017,  ..., -1.9986, -0.7409,  0.7011],
         [-1.3938,  0.8466, -1.7191,  ..., -1.1867,  0.1320,  0.3407],
         ...,
         [ 0.8206, -0.3745,  1.2499,  ..., -0.0676,  0.0385,  0.6335],
         [-0.5589, -0.3393,  0.2347,  ...,  2.1181,  2.4569,  1.3083],
         [-0.4092,  1.5199,  0.2401,  ..., -0.2558,  0.7870,  0.9924]]])
  </code></pre>
</div>

Let's create an example `nn.Conv2d()` with various parameters:
* `in_channels` (int) - Number of channels in the input image.
* `out_channels` (int) - Number of channels produced by the convolution.
* `kernel_size` (int or tuple) - Size of the convolving kernel/filter.
* `stride` (int or tuple, optional) - How big of a step the convolving kernel takes at a time. Default: 1.
* `padding` (int, tuple, str) - Padding added to all four sides of input. Default: 0.

<div class="row mt-3">
  {% assign figure_counter = figure_counter | plus: 1 %}
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid
      figure_number=figure_counter
      loading="eager"
      path="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-conv2d-layer.gif"
      class="img-fluid rounded"
      caption="Example of what happens when you change the hyperparameters of a <code>nn.Conv2d()</code> layer."
      id="conv2d_layer_example"
    %}
  </div>
</div>


```python
torch.manual_seed(42)

# Create a convolutional layer with same dimensions as TinyVGG
# (try changing any of the parameters and see what happens)
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0) # also try using "valid" or "same" here

# Pass the data through the convolutional layer
conv_layer(test_image) # Note: If running PyTorch <1.11.0, this will error because of shape issues (nn.Conv.2d() expects a 4d tensor as input)
```

<div class="bash-block">
  <pre><code>tensor([[[ 1.5396,  0.0516,  0.6454,  ..., -0.3673,  0.8711,  0.4256],
         [ 0.3662,  1.0114, -0.5997,  ...,  0.8983,  0.2809, -0.2741],
         [ 1.2664, -1.4054,  0.3727,  ..., -0.3409,  1.2191, -0.0463],
         ...,
         [-0.1541,  0.5132, -0.3624,  ..., -0.2360, -0.4609, -0.0035],
         [ 0.2981, -0.2432,  1.5012,  ..., -0.6289, -0.7283, -0.5767],
         [-0.0386, -0.0781, -0.0388,  ...,  0.2842,  0.4228, -0.1802]],

        [[-0.2840, -0.0319, -0.4455,  ..., -0.7956,  1.5599, -1.2449],
         [ 0.2753, -0.1262, -0.6541,  ..., -0.2211,  0.1999, -0.8856],
         [-0.5404, -1.5489,  0.0249,  ..., -0.5932, -1.0913, -0.3849],
         ...,
         [ 0.3870, -0.4064, -0.8236,  ...,  0.1734, -0.4330, -0.4951],
         [-0.1984, -0.6386,  1.0263,  ..., -0.9401, -0.0585, -0.7833],
         [-0.6306, -0.2052, -0.3694,  ..., -1.3248,  0.2456, -0.7134]],

        [[ 0.4414,  0.5100,  0.4846,  ..., -0.8484,  0.2638,  1.1258],
         [ 0.8117,  0.3191, -0.0157,  ...,  1.2686,  0.2319,  0.5003],
         [ 0.3212,  0.0485, -0.2581,  ...,  0.2258,  0.2587, -0.8804],
         ...,
         [-0.1144, -0.1869,  0.0160,  ..., -0.8346,  0.0974,  0.8421],
         [ 0.2941,  0.4417,  0.5866,  ..., -0.1224,  0.4814, -0.4799],
         [ 0.6059, -0.0415, -0.2028,  ...,  0.1170,  0.2521, -0.4372]],

        ...,

        [[-0.2560, -0.0477,  0.6380,  ...,  0.6436,  0.7553, -0.7055],
         [ 1.5595, -0.2209, -0.9486,  ..., -0.4876,  0.7754,  0.0750],
         [-0.0797,  0.2471,  1.1300,  ...,  0.1505,  0.2354,  0.9576],
         ...,
         [ 1.1065,  0.6839,  1.2183,  ...,  0.3015, -0.1910, -0.1902],
         [-0.3486, -0.7173, -0.3582,  ...,  0.4917,  0.7219,  0.1513],
         [ 0.0119,  0.1017,  0.7839,  ..., -0.3752, -0.8127, -0.1257]],

        [[ 0.3841,  1.1322,  0.1620,  ...,  0.7010,  0.0109,  0.6058],
         [ 0.1664,  0.1873,  1.5924,  ...,  0.3733,  0.9096, -0.5399],
         [ 0.4094, -0.0861, -0.7935,  ..., -0.1285, -0.9932, -0.3013],
         ...,
         [ 0.2688, -0.5630, -1.1902,  ...,  0.4493,  0.5404, -0.0103],
         [ 0.0535,  0.4411,  0.5313,  ...,  0.0148, -1.0056,  0.3759],
         [ 0.3031, -0.1590, -0.1316,  ..., -0.5384, -0.4271, -0.4876]],

        [[-1.1865, -0.7280, -1.2331,  ..., -0.9013, -0.0542, -1.5949],
         [-0.6345, -0.5920,  0.5326,  ..., -1.0395, -0.7963, -0.0647],
         [-0.1132,  0.5166,  0.2569,  ...,  0.5595, -1.6881,  0.9485],
         ...,
         [-0.0254, -0.2669,  0.1927,  ..., -0.2917,  0.1088, -0.4807],
         [-0.2609, -0.2328,  0.1404,  ..., -0.1325, -0.8436, -0.7524],
         [-1.1399, -0.1751, -0.8705,  ...,  0.1589,  0.3377,  0.3493]]],
       grad_fn=&lt;SqueezeBackward1&gt;)
  </code></pre>
</div>

If we try to pass a single image in, we get a shape mismatch error:

<div class="note-box">
  <strong>Error:</strong>
  <p>
    <code>RuntimeError: Expected 4-dimensional input for 4-dimensional weight [10, 3, 3, 3], but got 3-dimensional input of size [3, 64, 64] instead</code>
  </p>
  <strong>Note:</strong>
  <p>
    If you're running PyTorch 1.11.0+, this error won't occur.
  </p>
</div>

This is because our `nn.Conv2d()` layer expects a 4-dimensional tensor as input with size `(N, C, H, W)` or `[batch_size, color_channels, height, width]`.

Right now our single image `test_image` only has a shape of `[color_channels, height, width]` or `[3, 64, 64]`.

We can fix this for a single image using `test_image.unsqueeze(dim=0)` to add an extra dimension for `N`.

```python
# Add extra dimension to test image
test_image.unsqueeze(dim=0).shape
```

<div class="bash-block">
  <pre><code>torch.Size([1, 3, 64, 64])</code></pre>
</div>

```python
# Pass test image with extra dimension through conv_layer
conv_layer(test_image.unsqueeze(dim=0)).shape
```

<div class="bash-block">
  <pre><code>torch.Size([1, 10, 62, 62])</code></pre>
</div>

Hmm, notice what happens to our shape (the same shape as the first layer of TinyVGG on [CNN Explainer](https://poloclub.github.io/cnn-explainer/)), we get different channel sizes as well as different pixel sizes.

What if we changed the values of `conv_layer`?