---
layout: page
title: Introduction, Preliminaries & Linear Neural Network
description: 
related_publications: false
toc:
    sidebar: left
back_link: '/teaching/deeplearnig'
back_text: 'Deep Learning Course Page'
number_heading: true
---
Artificial Intelligence (AI) has become an integral part of our daily lives, revolutionizing industries and transforming how we interact with technology. From voice assistants to recommendation systems and autonomous vehicles, AI is everywhere, driving innovation and solving complex problems. In this section, we introduce foundational concepts and mathematical preliminaries that form the backbone of modern AI systems. We will begin by exploring linear neural networks, one of the simplest yet most important building blocks of deep learning models, setting the stage for more advanced architectures and techniques in the field.

## AI is Ubiquitous
You may find the applications of AI in every aspect of our lives these days.
<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            path="https://i.imgur.com/tamo2Hw.png"
            class="img-fluid rounded z-depth-1"
            caption="AI applications in our daily lives."
            id="ai-ubiquitous" %}
    </div>
</div>

## AI Paradox
<div style="display: flex; align-items: flex-start;">
  <!-- Text Content -->
  <div style="margin-right: 15px; max-width: 50%;">
    <p>
    <!-- First Bullet List -->
    <ul style="margin-right: 20px;">
        <li>Problems difficult for humans are easy for AI</li>
    </ul>
    
    <!-- Second Bullet List -->
    <ul>
        <li>Problems easy for humans are difficult for AI</li>
    </ul>
    </p>
  </div>

  <!-- Image -->
  <div style="flex-grow: 1; max-width: 50%;">
    <img
      src="https://i.imgur.com/HmExOOn.png"
      alt="A description of the image"
      style="width: 100%; height: auto;"
    >
  </div>
</div>

## What tasks require Intelligence?

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            path="https://i.imgur.com/2a3uuEK.png"
            class="img-fluid rounded z-depth-1"
            caption="Tasks that require intelligence."
            id="tasks-intelligence" %}
    </div>
</div>

## Knowledge-Based AI
<div class="row mt-3">
    <div class="col-12 text-center">
        <figure id="knowledge-based-ai">
            <div style="display: flex; justify-content: center; gap: 10px;">
                {% assign figure_counter = figure_counter | plus: 1 %}
                <div style="flex: 1; max-width: 45%; display: flex; justify-content: center;">
                    <img
                        src="https://i.imgur.com/XYxUdWj.png"
                        alt="Knowledge-based AI example 1"
                        class="img-fluid rounded"
                        style="width: 100%; height: auto; object-fit: contain;"
                        id="knowledge-ai-1">
                </div>
                <div style="flex: 1; max-width: 45%; display: flex; justify-content: center;">
                    <img
                        src="https://i.imgur.com/PyZ9SuN.png"
                        alt="Knowledge-based AI example 2"
                        class="img-fluid rounded"
                        style="width: 100%; height: auto; object-fit: contain;"
                        id="knowledge-ai-2">
                </div>
            </div>
            <figcaption class="caption mt-2">
                Figure {{ figure_counter }}: Knowledge-based AI examples demonstrating various concepts.
            </figcaption>
        </figure>
    </div>
</div>

- **Disadvantage**: Unwieldy Process
  - Time of human expert
  - People struggle to formalize rules with enough complexity to describe the world

## The Machine Learning Approach
 - Allow computers to learn from experience
 - Determine what features to use
 - Learn to map the features to output
<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            path="https://i.imgur.com/ljOLzhP.png"
            class="img-fluid rounded"
            caption="The machine learning approach."
            id="ml-approach" %}
    </div>
</div>

## Machine Learning Pipeline
<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            path="https://i.imgur.com/9sOnyhU.png"
            class="img-fluid rounded"
            caption="The machine learning pipeline."
            id="ml-pipeline" %}
    </div>
</div>

## Supervised Learning vs Unsupervised Learning 

**Supervised Learning**
 - Data-set: collection of **labeled examples** $$\left(\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}_{i=1}^{N}\right)$$
 - Goal: produce a model that takes $$\mathbf{x}$$ as input and predict $$\hat{y}$$

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            path="https://i.imgur.com/Ht73LSY.png"
            class="img-fluid rounded"
            caption="Supervised Learning."
            id="supervised" %}
    </div>
</div>

**Unsupervised Learning**
- Data-set: collection of **unlabeled example** $$\left(\left\{\mathbf{x}_{i}\right\}_{i=1}^{N}\right)$$
- Goal: create a model that take $$\mathbf{x}$$ as input and either transform it into another vector or into a value that can be used to solve a practical problem.

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            path="https://i.imgur.com/FFsupol.png"
            class="img-fluid rounded"
            caption="Unsupervised Learning."
            id="unsupervised" %}
    </div>
</div>

## Semi-Supervised Learning
- Data-set: collection of **labeled and unlabeled examples** $$\left(\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}_{i=1}^{N}, \left\{\mathbf{x}_{i}\right\}_{i=1}^{M}\right)$$
- Goal: The hope here is that using many unlabeled examples can help the learning algorithm to find (we might say “produce” or “compute”) a better model.

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            path="https://i.imgur.com/cAUzsQV.png"
            class="img-fluid rounded"
            caption="Semi-Supervised Learning."
            id="semi-supervised" %}
    </div>
</div>