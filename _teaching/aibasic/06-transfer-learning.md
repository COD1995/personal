---
layout: lecture
module: "06"
title: Transfer Learning
description: Standing on the shoulders of a pretrained model — freezing, replacing the classifier head, and fine-tuning.
math: true
objectives:
  - Explain what transfer learning is and why features learned on ImageNet carry over to a new image problem.
  - Find pretrained models in torchvision, the Hugging Face Hub, timm, and PyTorch Hub, and compare them by size and accuracy.
  - Prepare images the way a pretrained model expects, with its automatic transforms or matching manual ones, and explain what normalization does.
  - Load EfficientNet-B0, freeze its backbone, replace its classifier head, and read trainable and non-trainable parameters in a torchinfo summary.
  - Train the new head with `engine.train` and use the model to predict on test images and on a photo of your own.
  - Decide between feature extraction and fine-tuning, and unfreeze part of a network with a lower learning rate.
---

* Contents
{:toc}

In [module 04]({{ '/teaching/aibasic/04-custom-datasets/' | relative_url }}) we trained TinyVGG from scratch to tell pizza, steak, and sushi apart, and it did not get far: with a couple of hundred training images, a small network starting from random numbers cannot learn much about what food looks like. In [module 05]({{ '/teaching/aibasic/05-going-modular/' | relative_url }}) we packed the data loading and training code into the `going_modular` package.

This module keeps the data and the training code and changes the model. Instead of starting from random numbers, we start from a network that has already learned to recognize a thousand kinds of objects in more than a million photographs, and we teach it only the last step: which of *our* three classes an image belongs to. This is **transfer learning**, and in practice it is usually the first thing to try on a new image problem, especially when data is scarce.

## What transfer learning is

**Transfer learning** means taking a model trained on one task and reusing what it learned for a different, related task. The model's learned parameters are called **pretrained weights**, and the large dataset they were learned on is the *source*. For images, the usual source is **ImageNet**: about 1.3 million photographs sorted into 1,000 categories, from goldfish to fire trucks. Models trained on it are published by the PyTorch team and others, free to download.

### Why it works

A convolutional network learns in layers. The early layers learn to respond to simple, general patterns — edges, color blobs, corners. The middle layers combine those into textures and shapes. The last layers combine shapes into parts of specific objects. Only the very last layer, the classifier, maps all of that to ImageNet's 1,000 labels.

The general patterns are useful for almost any photograph. The edges and textures that help recognize a fire truck also help recognize a steak. So we keep the layers that detect them and retrain only the part that turns them into a decision.

An engineering comparison: a structural engineer trained on steel who moves to timber does not relearn statics, load paths, and free-body diagrams. Those carry over. They learn only what is specific to the new material. A pretrained network's early layers are the statics; the classifier is the material-specific part.

### Backbone and head

It helps to split a network into two parts:

- The **backbone** (also called the *feature extractor*) is everything up to the last layer. It turns an image into a list of numbers — a *feature vector* — that summarizes what is in the picture.
- The **head** is the final classifier. It turns the feature vector into one score per class.

There are two ways to reuse a pretrained backbone:

- **Feature extraction.** Freeze the whole backbone so its weights never change, replace the head with a new one sized for your classes, and train only the head.
- **Fine-tuning.** Start as above, then *unfreeze* some of the later backbone layers and train them too, gently, with a small learning rate.

<figure class="figure figure-wide figure-plain">
  <img src="{{ '/assets/img/courses/aibasic/06-feature-extraction-vs-fine-tuning.svg' | relative_url }}" alt="Two diagrams of the same network: an image passes through six backbone blocks, a pooling layer, and a new three-class head. In feature extraction all six backbone blocks are frozen and only the new head trains. In fine-tuning the last two backbone blocks are also unfrozen and train with a small learning rate. The original 1,000-class ImageNet head is replaced." loading="lazy">
  <figcaption>Feature extraction trains only the new head; fine-tuning also lets the last few backbone blocks adjust. Early blocks detect general patterns and stay frozen in both. The original ImageNet head, with 1,000 outputs, is discarded.</figcaption>
</figure>

We start with feature extraction, which is fast and hard to get wrong, and come back to fine-tuning at the end.

## Where to find pretrained models

| Source | What it offers | How you load a model |
| --- | --- | --- |
| [`torchvision.models`](https://pytorch.org/vision/stable/models.html) | Image models from PyTorch itself: classification, detection, segmentation, video | `torchvision.models.efficientnet_b0(weights=...)` |
| [Hugging Face Hub](https://huggingface.co/models) | A very large public collection of models for images, text, and audio, uploaded by companies and researchers | the `transformers` or `timm` libraries |
| [timm](https://huggingface.co/docs/timm) (PyTorch Image Models) | A large collection of image classification models, often the newest architectures | `timm.create_model("efficientnet_b0", pretrained=True)` (timm's own argument; torchvision uses `weights=`) |
| [PyTorch Hub](https://pytorch.org/hub/) | Models published by research groups alongside their papers | `torch.hub.load("owner/repo", "model_name")` |

For this module, torchvision has everything we need. It can list its own catalog:

```python
import torch
import torchvision
from torchvision import models

classification_models = models.list_models(module=models)
print(f"torch {torch.__version__} | torchvision {torchvision.__version__}")
print(f"Image classification models in torchvision: {len(classification_models)}")
efficientnets = [name for name in classification_models if name.startswith("efficientnet")]
print("EfficientNet variants:", ", ".join(name.removeprefix("efficientnet_") for name in efficientnets))
```

```text
torch 2.14.0+cu130 | torchvision 0.29.0+cu130
Image classification models in torchvision: 80
EfficientNet variants: b0, b1, b2, b3, b4, b5, b6, b7, v2_l, v2_m, v2_s
```

You need torchvision 0.13 or later for the code in this module; Colab's version is well past that.

## Setting up

We need three things: the `torchinfo` library for model summaries, the `going_modular` package from module 05, and the data.

Colab does not come with `torchinfo`, so install it once per session:

```python
!pip install -q torchinfo
```

If you are working in the folder from module 05, `going_modular` is already there. On a fresh Colab runtime, download the reference copy of the package from the *Learn PyTorch for Deep Learning* repository on GitHub. The loop skips any file you already have:

```python
import requests
from pathlib import Path

Path("going_modular").mkdir(exist_ok=True)
base = "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/going_modular/going_modular/"
for name in ["data_setup.py", "engine.py", "model_builder.py", "utils.py"]:
    path = Path("going_modular") / name
    if not path.exists():
        path.write_text(requests.get(base + name).text)

from going_modular import data_setup, engine, utils
from torchinfo import summary
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

```text
'cpu'
```

On a Colab GPU runtime this prints `'cuda'`. Feature extraction runs on a CPU, but slowly; a GPU makes each epoch take seconds instead of minutes.

### The data

The same pizza, steak, and sushi images as in modules 04 and 05. This is the code from `get_data.py`, as a cell:

```python
import io
import zipfile

URL = ("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/"
       "main/data/pizza_steak_sushi.zip")
image_path = Path("data") / "pizza_steak_sushi"

if image_path.is_dir():
    print(f"{image_path} already exists, skipping download.")
else:
    print(f"Downloading and unzipping to {image_path}")
    response = requests.get(URL)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        zip_file.extractall(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"
```

```text
Downloading and unzipping to data/pizza_steak_sushi
```

## Preparing the data the way the model expects

A pretrained model has learned to work with images prepared in one particular way. Our images must be prepared the same way, or the learned weights see inputs unlike anything they were trained on. For torchvision's ImageNet models that means three things: images of about 224 × 224 pixels, pixel values converted to tensors between 0 and 1, and then **normalized** with ImageNet's statistics.

### What normalization does

`ToTensor()` turns each pixel into a number between 0 and 1 in each of the three color channels. **Normalization** then shifts and rescales each channel separately:

$$x_{\text{normalized}} = \frac{x - \text{mean}}{\text{std}}$$

For ImageNet models, `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]` for the red, green, and blue channels. These are the average and the spread of each channel over the ImageNet training photos. Subtracting the mean centers each channel near zero; dividing by the standard deviation makes its spread about one. After normalization, every pixel value lies between about −2.1 (black) and +2.6 (full brightness).

Networks train more smoothly on inputs centered near zero with similar scales, which is why the model was trained on normalized images — and why we must normalize ours with *the same* numbers. It is like a sensor amplifier calibrated with a particular zero offset and gain: feed it a signal scaled differently and every reading downstream is off, even though nothing is broken.

### Manual transforms

Before torchvision 0.13, you wrote the preparation yourself:

```python
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),                 # 1. resize every image to 224 x 224
    transforms.ToTensor(),                         # 2. pixels -> tensor with values in [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # 3. ImageNet normalization
                         std=[0.229, 0.224, 0.225]),
])
manual_transforms
```

```text
Compose(
    Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=True)
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
```

This works, but you have to know the right numbers for each model, and a typo fails silently: the model still runs, it just performs worse.

### Automatic transforms

Since torchvision 0.13, each set of pretrained weights carries its own preparation recipe. You pick the weights through an **enum** — a named list of options — and ask the weights for their transforms:

```python
weights = models.EfficientNet_B0_Weights.DEFAULT   # the best available weights for EfficientNet-B0
auto_transforms = weights.transforms()
auto_transforms
```

```text
ImageClassification(
    crop_size=[224]
    resize_size=[256]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    interpolation=InterpolationMode.BICUBIC
)
```

`DEFAULT` always points to the best weights torchvision has for that architecture; you could also ask for a specific version such as `IMAGENET1K_V1`. The recipe resizes the image so its shorter side is 256 pixels, cuts out the central 224 × 224 square, converts it to a tensor, and normalizes it with the same ImageNet numbers as above.

Let's see normalization at work on one training image:

```python
from PIL import Image

image_path_list = sorted(train_dir.glob("*/*.jpg"))
img = Image.open(image_path_list[0])

raw = transforms.ToTensor()(img)
prepared = auto_transforms(img)

print(f"Original image: {img.size[0]} x {img.size[1]} pixels")
print(f"ToTensor only:   shape {tuple(raw.shape)}, values {raw.min():.2f} to {raw.max():.2f}")
print(f"Auto transforms: shape {tuple(prepared.shape)}, values {prepared.min():.2f} to {prepared.max():.2f}")
print(f"Channel means after normalizing: {prepared.mean(dim=(1, 2))}")
```

```text
Original image: 512 x 512 pixels
ToTensor only:   shape (3, 512, 512), values 0.00 to 1.00
Auto transforms: shape (3, 224, 224), values -2.12 to 2.64
Channel means after normalizing: tensor([-0.1126, -0.1474, -0.4693])
```

The image shrank to 224 × 224, and its values now straddle zero instead of sitting between 0 and 1. The channel means are not exactly zero — this is one photo, not the average of ImageNet — but they are on the right scale.

> **Note.** Automatic transforms guarantee the match with the pretrained weights. Manual transforms give you more control, for example to add data augmentation from module 04. Either is fine as long as the resize and the normalization match the weights you load.
{: .callout}

### DataLoaders

`create_dataloaders` from module 05 takes the transform as an argument, so it works unchanged:

```python
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=auto_transforms,
    batch_size=32,
)
print(f"Classes: {class_names}")
print(f"Batches: {len(train_dataloader)} train, {len(test_dataloader)} test")
```

```text
Classes: ['pizza', 'steak', 'sushi']
Batches: 8 train, 3 test
```

## Loading a pretrained model

### Choosing one

There is no single best model; there is a trade-off between accuracy and size. A bigger model is usually more accurate but slower and harder to fit on a phone or a small embedded computer. Each weights enum records its own facts, so we can compare a few without guessing:

```python
candidates = [
    models.ResNet18_Weights.DEFAULT,
    models.EfficientNet_B0_Weights.DEFAULT,
    models.EfficientNet_B2_Weights.DEFAULT,
    models.ResNet50_Weights.DEFAULT,
    models.ViT_B_16_Weights.DEFAULT,
]
for w in candidates:
    meta = w.meta
    top1 = meta["_metrics"]["ImageNet-1K"]["acc@1"]
    print(f"{str(w):38} {meta['num_params']:>11,} params {meta['_file_size']:6.1f} MB  top-1 {top1:.1f}%")
```

```text
ResNet18_Weights.IMAGENET1K_V1          11,689,512 params   44.7 MB  top-1 69.8%
EfficientNet_B0_Weights.IMAGENET1K_V1    5,288,548 params   20.5 MB  top-1 77.7%
EfficientNet_B2_Weights.IMAGENET1K_V1    9,109,994 params   35.2 MB  top-1 80.6%
ResNet50_Weights.IMAGENET1K_V2          25,557,032 params   97.8 MB  top-1 80.9%
ViT_B_16_Weights.IMAGENET1K_V1          86,567,656 params  330.3 MB  top-1 81.1%
```

*Top-1 accuracy* (the last column) is how often the model's single best guess is right on ImageNet's 50,000 validation images; the file size is the download. EfficientNet-B0 is a good fit for us: small, fast, and accurate for its size. It comes from the paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). The larger `b1` to `b7` variants trade speed for accuracy; you will try one in the exercises.

### Loading EfficientNet-B0

One line builds the architecture and loads the weights:

```python
model = models.efficientnet_b0(weights=weights).to(device)
```

The first time you run this, torchvision downloads the weights file (about 20 MB) and keeps a copy on disk, so later loads are instant.

> **Watch out.** Older tutorials write `models.efficientnet_b0(pretrained=True)`. That argument has been deprecated since torchvision 0.13 and may be removed; use `weights=` with an enum as above. `weights=None` gives you the same architecture with random weights — useful for comparison, useless for transfer learning.
{: .callout-warn}

### Inside the model

EfficientNet-B0 has three parts, which you can reach as attributes:

- `model.features` — the backbone: a stack of convolutional blocks that turns a 3 × 224 × 224 image into 1,280 feature maps of size 7 × 7.
- `model.avgpool` — averages each 7 × 7 map to a single number, giving a feature vector of length 1,280.
- `model.classifier` — the head, which turns those 1,280 numbers into 1,000 ImageNet scores.

```python
model.classifier
```

```text
Sequential(
  (0): Dropout(p=0.2, inplace=True)
  (1): Linear(in_features=1280, out_features=1000, bias=True)
)
```

The head is small: a dropout layer (explained below) and a single `Linear` layer from 1,280 features to 1,000 classes. To see the whole network with shapes and parameter counts, use `torchinfo.summary`. It runs an example input through the model and records what happens at each layer. We pass the input size — a batch holding one 224 × 224 color image — and choose which columns to show:

```python
summary(model,
        input_size=(1, 3, 224, 224),     # [batch, color channels, height, width]
        col_names=["output_size", "num_params"],
        col_width=17,
        row_settings=["var_names"],      # show attribute names such as (features)
        depth=2,                         # how many levels of nesting to show
        verbose=0)                       # return the table rather than also printing it
```

```text
==============================================================================================
Layer (type (var_name))                                      Output Shape      Param #
==============================================================================================
EfficientNet (EfficientNet)                                  [1, 1000]         --
├─Sequential (features)                                      [1, 1280, 7, 7]   --
│    └─Conv2dNormActivation (0)                              [1, 32, 112, 112] 928
│    └─Sequential (1)                                        [1, 16, 112, 112] 1,448
│    └─Sequential (2)                                        [1, 24, 56, 56]   16,714
│    └─Sequential (3)                                        [1, 40, 28, 28]   46,640
│    └─Sequential (4)                                        [1, 80, 14, 14]   242,930
│    └─Sequential (5)                                        [1, 112, 14, 14]  543,148
│    └─Sequential (6)                                        [1, 192, 7, 7]    2,026,348
│    └─Sequential (7)                                        [1, 320, 7, 7]    717,232
│    └─Conv2dNormActivation (8)                              [1, 1280, 7, 7]   412,160
├─AdaptiveAvgPool2d (avgpool)                                [1, 1280, 1, 1]   --
├─Sequential (classifier)                                    [1, 1000]         --
│    └─Dropout (0)                                           [1, 1280]         --
│    └─Linear (1)                                            [1, 1000]         1,281,000
==============================================================================================
Total params: 5,288,548
Trainable params: 5,288,548
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 385.87
==============================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 107.89
Params size (MB): 21.15
Estimated Total Size (MB): 129.64
==============================================================================================
```

Read it from top to bottom, as the image flows through. Each block of `features` shrinks the picture (112, 56, 28, 14, 7 pixels across) while adding channels (up to 1,280): the network trades *where* for *what*. The pooling layer collapses each 7 × 7 map to one number, and the classifier maps 1,280 numbers to 1,000. The footer counts 5,288,548 parameters, all of them trainable for now. With a batch of 32 images the table would be the same, with 32 in place of 1 at the start of every shape.

## Feature extraction: freeze the backbone, replace the head

### Freezing

Every parameter in PyTorch has an attribute `requires_grad`. When it is `True`, `loss.backward()` computes a gradient for it and the optimizer updates it. Setting it to `False` **freezes** the parameter: it takes part in the forward pass but never changes. Freezing the backbone keeps everything it learned on ImageNet, and it also makes training much faster, because no gradients need to be computed for most of the network.

A small helper counts what will train:

```python
def count_parameters(model: torch.nn.Module):
    """Returns (trainable, total) parameter counts for a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

# Freeze every parameter in the backbone
for param in model.features.parameters():
    param.requires_grad = False

trainable, total = count_parameters(model)
print(f"Trainable: {trainable:,} of {total:,}")
```

```text
Trainable: 1,281,000 of 5,288,548
```

Only the old head, with its 1,281,000 parameters, is left trainable. We replace that next.

### A new head for three classes

The ImageNet head outputs 1,000 scores; we need 3. We build a new head with the same structure — dropout, then a `Linear` layer — but with `out_features=len(class_names)`. The input size, 1,280, must match what the backbone produces: that is the `[1, 1280, 1, 1]` output of `avgpool` in the summary.

**Dropout** randomly sets a fraction of its inputs to zero on every training step (here 20%, `p=0.2`), which stops the head from leaning too hard on any single feature and helps against overfitting. It is switched off automatically in `model.eval()` mode. `inplace=True` just saves a little memory; we copy it from the original head.

```python
torch.manual_seed(42)

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=len(class_names),   # one output per class: pizza, steak, sushi
                    bias=True),
).to(device)

model.classifier
```

```text
Sequential(
  (0): Dropout(p=0.2, inplace=True)
  (1): Linear(in_features=1280, out_features=3, bias=True)
)
```

New layers are created with `requires_grad=True` and random weights, and on the CPU — hence the `.to(device)`. The summary now tells the story of feature extraction:

```python
summary(model,
        input_size=(1, 3, 224, 224),
        col_names=["num_params", "trainable"],
        col_width=14,
        row_settings=["var_names"],
        depth=1,
        verbose=0)
```

```text
========================================================================================
Layer (type (var_name))                                      Param #        Trainable
========================================================================================
EfficientNet (EfficientNet)                                  --             Partial
├─Sequential (features)                                      (4,007,548)    False
├─AdaptiveAvgPool2d (avgpool)                                --             --
├─Sequential (classifier)                                    3,843          True
========================================================================================
Total params: 4,011,391
Trainable params: 3,843
Non-trainable params: 4,007,548
Total mult-adds (Units.MEGABYTES): 384.59
========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 107.88
Params size (MB): 16.05
Estimated Total Size (MB): 124.53
========================================================================================
```

Three things to notice:

- The total dropped from 5,288,548 to 4,011,391 parameters, because the new three-class head is much smaller than the 1,000-class one.
- The model is **partially** trainable. The backbone's 4,007,548 parameters are frozen (torchinfo puts their count in parentheses); only the head's 3,843 train — $$1280 \times 3$$ weights plus 3 biases.
- We will train less than 0.1% of the network. That is why transfer learning needs so little data: 225 images are far too few to set 4 million parameters, but plenty to set 3,843.

## Training the head

Everything else is the same as in modules 04 and 05: a cross-entropy loss for multi-class classification, the Adam optimizer, and `engine.train` from our package.

```python
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 5
```

Passing all of `model.parameters()` to the optimizer is fine: the frozen ones never receive a gradient, so the optimizer leaves them alone.

```python
from timeit import default_timer as timer

torch.manual_seed(42)
start_time = timer()

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=NUM_EPOCHS,
                       device=device)

end_time = timer()
print(f"Total training time: {end_time - start_time:.1f} seconds")
```

Run this in Colab and watch the five epoch lines. What you should expect, with the real pretrained weights:

- **Accuracy jumps early.** Test accuracy should be far above TinyVGG's within the first epoch or two and keep improving — the new model should beat the TinyVGG from module 04 by a wide margin within a few epochs. For reference, the companion chapter reports test accuracy in the mid-80s percent after five epochs; your numbers will differ somewhat from run to run and GPU to GPU.
- **Training is quick on a GPU,** typically well under a minute for all five epochs, because gradients are computed only for the head.
- **Train accuracy can sit below test accuracy.** That looks odd but is normal here: dropout is active during training and off during testing, and the test set is small.

> **Note.** These notes were run on a computer that cannot download the pretrained weights, so the code above was checked with randomly initialized weights and its output is not shown: numbers from a random backbone would say nothing about transfer learning. Everything else on this page — shapes, parameter counts, summaries — is real output.
{: .callout}

## Evaluating: loss curves

The `results` dictionary holds one value per epoch for each metric. The plotting function from module 04 draws them:

```python
import matplotlib.pyplot as plt

def plot_loss_curves(results):
    """Plots training and test loss and accuracy curves from a results dictionary."""
    epochs = range(1, len(results["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="train_loss")
    plt.plot(epochs, results["test_loss"], label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="train_accuracy")
    plt.plot(epochs, results["test_acc"], label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

plot_loss_curves(results)
```

What to look for, using the vocabulary of module 04:

- **Both loss curves should fall** and the accuracy curves rise. That means the head is learning something that generalizes to images it has not seen.
- **The test loss should stay close to the training loss.** A test loss that turns upward while the training loss keeps falling would be the signature of overfitting; with only 3,843 trainable parameters and dropout, that is unlikely over five epochs.
- **Curves still falling at epoch 5** suggest more epochs would help — you will test that in the exercises.

## Making predictions

### A prediction function

To predict on a single image we repeat, in order, what training did to every image, and then turn the output into a class name:

1. Open the image and apply **the same transform** used for training — here, `weights.transforms()`.
2. Add a batch dimension with `unsqueeze(dim=0)`, since the model expects `[batch, channels, height, width]`.
3. Put the model in evaluation mode and run it under `torch.inference_mode()`.
4. Turn the logits into probabilities with softmax, and pick the highest with argmax.

```python
from typing import List, Tuple

def pred_and_plot_image(model: torch.nn.Module,
                        class_names: List[str],
                        image_path: str,
                        image_size: Tuple[int, int] = (224, 224),
                        transform=None,  # any callable that turns a PIL image into a tensor
                        device: torch.device = device):
    """Predicts the class of one image with a model and plots the image with the prediction.

    Args:
        model: A trained PyTorch model.
        class_names: List of class names, in the order the model outputs them.
        image_path: Path to the image file.
        image_size: Size to resize to if no transform is given.
        transform: Transform to apply; defaults to resize + ImageNet normalization.
        device: Device to run the model on.
    """
    img = Image.open(image_path)

    # 1. Use the given transform, or build one that matches ImageNet models
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # 2-3. Evaluation mode, no gradients, batch dimension, right device
    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        logits = model(transformed_image.to(device))

    # 4. Logits -> probabilities -> label
    probs = torch.softmax(logits, dim=1)
    label = torch.argmax(probs, dim=1)

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[label]} | Prob: {probs.max():.3f}")
    plt.axis(False)
    plt.show()
```

### On test images

Pick three test images at random and look at each prediction:

```python
import random

test_image_paths = list(test_dir.glob("*/*.jpg"))
for image_path in random.sample(test_image_paths, k=3):
    pred_and_plot_image(model=model,
                        class_names=class_names,
                        image_path=image_path,
                        transform=weights.transforms())
```

The true class is the name of each image's folder; add `print(image_path)` inside the loop if you want to check each title against it. With the pretrained backbone, expect most predictions to be right and many to come with a high probability. When one is wrong, look at the photo: often the food is small in the frame, partly hidden, or the plate holds more than one class.

### On an image of your own

The real test of a model is data from outside its dataset. Here is a photo that was never in the training or test folders:

```python
custom_image_path = Path("data") / "04-pizza-dad.jpeg"

if not custom_image_path.is_file():
    url = "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg"
    custom_image_path.write_bytes(requests.get(url).content)
    print(f"Downloaded {custom_image_path}")

pred_and_plot_image(model=model,
                    class_names=class_names,
                    image_path=custom_image_path,
                    transform=weights.transforms())
```

```text
Downloaded data/04-pizza-dad.jpeg
```

It shows a man eating pizza. Expect the transferred model to call it pizza, and with far more confidence than TinyVGG managed on the same photo in module 04. Then upload a photo of your own (in Colab: the folder icon on the left, then the upload button) and change `custom_image_path` to it.

> **Watch out.** The model can only answer "pizza", "steak", or "sushi". Show it a photo of a bridge and it will still pick one of the three, sometimes with high probability. A classifier has no built-in way to say "none of these"; the probability is its confidence *among the classes it knows*, not a measure of whether the question makes sense.
{: .callout-warn}

## Fine-tuning: letting the backbone adjust

Feature extraction assumes the backbone's features already suit your images. When your images look quite different from everyday photos — thermal images, micrographs of a material, X-rays of a weld — or when you have more data, it can pay to let the later backbone layers adjust as well. That is **fine-tuning**.

| Your situation | A sensible start |
| --- | --- |
| Little data, images similar to everyday photos | Feature extraction |
| Little data, images unlike everyday photos | Feature extraction first; fine-tune the last blocks cautiously |
| Plenty of data, similar images | Fine-tune the last few blocks |
| Plenty of data, very different images | Fine-tune many or all blocks |

Two rules keep fine-tuning from undoing what the backbone learned:

- **Train the new head first.** A freshly initialized head produces large, random gradients. Let it settle with the backbone frozen — as we just did — before unfreezing anything.
- **Use a smaller learning rate,** often ten times smaller than for the head. The unfrozen layers should be nudged, not rewritten.

Unfreezing is the freezing loop in reverse, applied to the last few blocks. `model.features[-2:]` selects the last two entries of the backbone — blocks 7 and 8 in the summary above:

```python
for param in model.features[-2:].parameters():
    param.requires_grad = True

trainable, total = count_parameters(model)
print(f"Trainable: {trainable:,} of {total:,}")
```

```text
Trainable: 1,133,235 of 4,011,391
```

Now a little over a quarter of the network will train: the last two blocks, which hold the most specific features, plus the head. The earlier blocks, with their general edge and texture detectors, stay frozen. With only 225 training images we unfreeze cautiously; with more data you could go further back, for example `model.features[-3:]`. A new optimizer with a ten-times-smaller learning rate, given only the parameters that should change:

```python
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                             lr=0.0001)
FINE_TUNE_EPOCHS = 5
```

```python
fine_tune_results = engine.train(model=model,
                                 train_dataloader=train_dataloader,
                                 test_dataloader=test_dataloader,
                                 optimizer=optimizer,
                                 loss_fn=loss_fn,
                                 epochs=FINE_TUNE_EPOCHS,
                                 device=device)
```

Each epoch now takes longer, because gradients flow through two more blocks. On a dataset this small and this close to ImageNet, do not expect a large gain over feature extraction — the test set has only 75 images, so a difference of one or two images is noise. Fine-tuning earns its cost when there is more data or when the images differ from everyday photos; exercise 7 asks you to compare the two on a larger slice of the data.

> **Note.** When you do fine-tune, save the model with `utils.save_model` from module 05 and record which blocks were unfrozen and which learning rate you used. Module 07 turns that record-keeping into a habit.
{: .callout}

## Summary

| Task | Code |
| --- | --- |
| Pick pretrained weights | `weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT` |
| Matching transforms | `weights.transforms()`, or `Resize` + `ToTensor` + `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` |
| Load the model | `model = torchvision.models.efficientnet_b0(weights=weights).to(device)` |
| Inspect it | `summary(model, input_size=(1, 3, 224, 224), col_names=[...], row_settings=["var_names"])` |
| Freeze the backbone | `for p in model.features.parameters(): p.requires_grad = False` |
| Replace the head | `model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, len(class_names)))` |
| Train | `engine.train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device)` |
| Predict | same transform, `unsqueeze(0)`, `model.eval()`, `torch.inference_mode()`, `softmax`, `argmax` |
| Fine-tune | unfreeze the last blocks, e.g. `model.features[-2:]`, new optimizer with a ~10× smaller learning rate |

Three ideas to carry forward: before building a model from scratch, look for one that has already learned something close to your problem; prepare your data exactly the way the pretrained model's data was prepared; and train as little of the network as the problem allows — the head first, more only if the evidence says it helps.

## Exercises

{: .exercises}
1. Train the feature-extraction model in Colab with a GPU and record the five epoch lines. How does the final test accuracy compare with TinyVGG's from module 05?
2. **Confusion matrix.** Predict on every image in the test set and plot a confusion matrix (for example with `torchmetrics.ConfusionMatrix` and `mlxtend.plotting.plot_confusion_matrix`). Which pair of classes gets mixed up most?
3. **Most wrong.** From the same predictions, find the five wrong predictions made with the highest probability and plot them. Are they labeling mistakes, hard photos, or model errors?
4. Predict on a photo of your own pizza, steak, or sushi, and on a photo with none of them. What probability does the model give the non-food photo?
5. Train for 10 epochs instead of 5. Does the test loss keep improving, level off, or start rising?
6. **More data.** Download `pizza_steak_sushi_20_percent.zip` (same address, `_20_percent` added before `.zip`), which has about twice as many images, and retrain the feature extractor for 5 epochs. Compare with the 10% data.
7. On the 20% data, compare feature extraction alone with feature extraction followed by fine-tuning the last two blocks, then the last three. Keep everything else fixed. Is the difference larger than one or two test images?
8. Swap in `efficientnet_b2`. Its head has a different `in_features`: find it from `model.classifier` or the summary, build the new head accordingly, and compare training time and accuracy with B0.
9. Train the same head on top of `models.efficientnet_b0(weights=None)` — random weights, same architecture, backbone frozen. What happens, and what does that tell you about where the accuracy of transfer learning comes from?
10. **In your own words.** Pick an image problem from your field (for example, classifying corrosion levels on steel or defects in welds). Would ImageNet features transfer well? Would you start with feature extraction or fine-tuning, and why?

## Going further

- [torchvision models and pretrained weights](https://pytorch.org/vision/stable/models.html) — every architecture, its weights enum, and its ImageNet accuracy.
- PyTorch's own [transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — the same two approaches on a different dataset (ants and bees).
- Yosinski et al., [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792) — the experiments behind "early layers are general, late layers are specific".
- Tan and Le, [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) — the paper behind the model we used.
- The [timm documentation](https://huggingface.co/docs/timm) — a much larger catalog of pretrained image models, with the same freeze-and-replace workflow.
