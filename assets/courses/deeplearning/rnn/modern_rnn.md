---
layout: page
title: Modern Recurrent Neural Networks
description: 
related_publications: false
toc:
    sidebar: left
back_link: '/teaching/deeplearnig'
back_text: 'Deep Learning Course Page'
number_heading: true
enable_heading_styles: true
show_h1_number: true
start_h1_number: 4
figure_counter: 0
---

We introduced the key ideas behind recurrent neural networks (RNNs). However, just as with convolutional neural networks, there has been a tremendous amount of innovation in RNN architectures, culminating in several complex designs that have proven successful in practice. In particular, the most popular designs feature mechanisms for mitigating the notorious numerical instability faced by RNNs, as typified by vanishing and exploding gradients. Recall that we dealt with exploding gradients by applying a blunt gradient clipping heuristic. Despite the efficacy of this hack, it leaves open the problem of vanishing gradients.

In this section, we introduce the key ideas behind the most successful RNN architectures for sequences, which stem from two landmark papers:

**Long Short-Term Memory (LSTM)** Long Short-Term Memory (LSTM) networks, introduced by Hochreiter and Schmidhuber in 1997, replace traditional nodes in the hidden layer of an RNN with *memory cells*. These memory cells overcome the training difficulties of earlier recurrent networks. Intuitively:
- LSTM memory cells avoid the vanishing gradient problem by maintaining values in the memory state over long time sequences. 
- A recurrent edge with weight 1 allows information to cascade across many successive time steps.
- **Multiplicative gates** control:
  1. Which inputs are allowed into the memory state,
  2. When the memory state influences the network's output.

This innovation enables LSTMs to handle long-range dependencies in sequences, which were previously infeasible with basic RNNs.

**Bidirectional Recurrent Neural Networks (BiRNNs)** Bidirectional Recurrent Neural Networks (Schuster and Paliwal, 1997) introduce the concept of processing sequences in both forward and backward directions. Unlike traditional RNNs, where only past inputs affect the output:
- **BiRNNs** use information from both preceding and subsequent time steps to determine the output at any point.
- This architecture is particularly effective for sequence labeling tasks, such as:
  - **Natural Language Processing**: Named Entity Recognition (NER), Part-of-Speech (POS) tagging,
  - **Speech Recognition**,
  - **Handwriting Recognition**.

**Combined Innovations** LSTM and BiRNN architectures are not mutually exclusive. They have been successfully combined for tasks such as:
- **Phoneme Classification** (Graves and Schmidhuber, 2005),
- **Handwriting Recognition** (Graves et al., 2008).

we will cover the following topics:
1. **LSTM Architecture**: A detailed breakdown of its structure and function,
2. **Gated Recurrent Units (GRUs)**: A lightweight variation of LSTMs,
3. **Bidirectional RNNs**: Leveraging future and past sequence information,
4. **Deep RNNs**: Stacking RNN layers for greater capacity,
5. **Sequence-to-Sequence Tasks**: Applying RNNs to machine translation with key ideas like encoder-decoder architectures and beam search.