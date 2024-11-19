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
# Introduction 
Artificial Intelligence (AI) has become an integral part of our daily lives, revolutionizing industries and transforming how we interact with technology. From voice assistants to recommendation systems and autonomous vehicles, AI is everywhere, driving innovation and solving complex problems. In this section, we introduce foundational concepts and mathematical preliminaries that form the backbone of modern AI systems. We will begin by exploring linear neural networks, one of the simplest yet most important building blocks of deep learning models, setting the stage for more advanced architectures and techniques in the field.

## AI is Ubiquitous
You may find the applications of AI in every aspect of our lives these days.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.imgur.com/tamo2Hw.png" class="img-fluid rounded z-depth-1" %}
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
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.imgur.com/2a3uuEK.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Knowledge-Based AI
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.imgur.com/XYxUdWj.png" class="img-fluid rounded" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.imgur.com/PyZ9SuN.png" class="img-fluid rounded" %}
    </div>
</div>

- Disadvantage: Unwiedy Process
  - Time of human expert
  - People struggle to formalize rules with enough complexity to describe the world

## The Machine Learning Approach
 - Allow computers to learn from experience
 - Determine what features to use
 - Learn to map the features to output
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.imgur.com/ljOLzhP.png" class="img-fluid rounded " %}
    </div>
</div>

## Machine Learning Pipeline
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://i.imgur.com/9sOnyhU.png" class="img-fluid rounded" %}
    </div>
</div>

## Supervised Learning vs Unsupervised Learning 

**Supervised Learning**