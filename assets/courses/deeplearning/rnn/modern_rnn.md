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

## Long Short-Term Memory (LSTM)

Shortly after the first Elman-style RNNs were trained using backpropagation, the challenges of learning long-term dependencies (due to vanishing and exploding gradients) became evident. These problems were discussed by Bengio and Hochreiter. Hochreiter had identified these issues as early as 1991 in his Master's thesis, although the work was not widely recognized since it was written in German.

While **gradient clipping** provides a solution for exploding gradients, vanishing gradients require a more intricate approach. One of the first and most effective solutions came with the **long short-term memory (LSTM)** model proposed by Hochreiter and Schmidhuber. 
LSTMs are similar to standard recurrent neural networks, but they replace each ordinary recurrent node with a *memory cell*. Each memory cell contains an *internal state*, a node with a self-connected recurrent edge of fixed weight 1, allowing the gradient to flow across many time steps without vanishing or exploding.

### Gated Memory Cell 

Each memory cell is equipped with an *internal state*
and a number of multiplicative gates that determine whether:
1. A given input should impact the internal state (*input gate*),
2. The internal state should be flushed to $$0$$ (*forget gate*),
3. The internal state of a given neuron should influence the cell's output (*output gate*).

<strong style="color: red; font-weight: 900;">Gated Hidden States</strong>The key distinction between vanilla RNNs and LSTMs
is that the latter support gating of the hidden state.
This gating mechanism determines:
- When a hidden state should be *updated*,
- When it should be *reset*.

These mechanisms are learned, enabling the network to:
- Retain critical information from the first token by learning not to update the hidden state,
- Ignore irrelevant observations,
- Reset the latent state when necessary.

<strong style="color: red; font-weight: 900;">Input Gate, Forget Gate, and Output Gate</strong>
The data feeding into the LSTM gates are the input at the current time step and the hidden state of the previous time step, as illustrated in Figure {{ figure_counter }}.

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager" path="https://d2l.ai/_images/lstm-0.svg" class="img-fluid rounded"
            caption="Computing the input gate, forget gate, and output gate in an LSTM model."
            id="fig_lstm_0" %}
    </div>
</div>

Mathematically, given:
- $$\mathbf{X}_t \in \mathbb{R}^{n \times d}$$ (input),
- $$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$$ (previous hidden state),
the gates are computed as:

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xi}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hi}} + \mathbf{b}_\textrm{i}),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xf}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hf}} + \mathbf{b}_\textrm{f}),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xo}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{ho}} + \mathbf{b}_\textrm{o}),
\end{aligned}
$$

where:
- $$\mathbf{W}_{\textrm{xi}}, \mathbf{W}_{\textrm{xf}}, \mathbf{W}_{\textrm{xo}} \in \mathbb{R}^{d \times h}$$,
- $$\mathbf{W}_{\textrm{hi}}, \mathbf{W}_{\textrm{hf}}, \mathbf{W}_{\textrm{ho}} \in \mathbb{R}^{h \times h}$$,
- $$\mathbf{b}_\textrm{i}, \mathbf{b}_\textrm{f}, \mathbf{b}_\textrm{o} \in \mathbb{R}^{1 \times h}$$.

**Input Node** The *input node* $$\tilde{\mathbf{C}}_t$$ is calculated as:

$$
\tilde{\mathbf{C}}_t = \textrm{tanh}(\mathbf{X}_t \mathbf{W}_{\textrm{xc}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hc}} + \mathbf{b}_\textrm{c}),
$$

where:
- $$\mathbf{W}_{\textrm{xc}} \in \mathbb{R}^{d \times h}$$,
- $$\mathbf{W}_{\textrm{hc}} \in \mathbb{R}^{h \times h}$$,
- $$\mathbf{b}_\textrm{c} \in \mathbb{R}^{1 \times h}$$.

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager" path="https://d2l.ai/_images/lstm-1.svg" class="img-fluid rounded"
            caption="Computing the input node in an LSTM model."
            id="fig_lstm_1" %}
    </div>
</div>

<strong style="color: red; font-weight: 900;">Memory Cell Internal State</strong> The memory cell internal state $$\mathbf{C}_t$$ is updated using:

$$
\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.
$$

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager" path="https://d2l.ai/_images/lstm-2.svg" class="img-fluid rounded"
            caption="Computing the memory cell internal state in an LSTM model."
            id="fig_lstm_2" %}
    </div>
</div>

<strong style="color: red; font-weight: 900;">Hidden State</strong> The hidden state $$\mathbf{H}_t$$ is computed as:

$$
\mathbf{H}_t = \mathbf{O}_t \odot \textrm{tanh}(\mathbf{C}_t).
$$

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager" path="https://d2l.ai/_images/lstm-3.svg" class="img-fluid rounded"
            caption="Computing the hidden state in an LSTM model."
            id="fig_lstm_3" %}
    </div>
</div>

## Gated Recurrent Units (GRUs)

As RNNs and particularly the LSTM architecture gained popularity during the 2010s, researchers sought simplified architectures that retained the core concepts of internal state and gating mechanisms but with faster computation. The **gated recurrent unit (GRU)** proposed by Cho et al. is one such architecture, offering a streamlined version of the LSTM that achieves comparable performance but is computationally faster.

**Reset Gate and Update Gate** In GRUs, the LSTM's three gates are replaced by two:
1. **Reset Gate**: Controls how much of the previous state is retained.
2. **Update Gate**: Determines how much of the new state is derived from the old state.

Both gates use sigmoid activation functions, ensuring their values lie in the range $$(0, 1)$$. Fig. [5](#fig_gru_1) illustrates the inputs for both gates, which include the input of the current time step and the hidden state of the previous time step.

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager" path="https://d2l.ai/_images/gru-1.svg" class="img-fluid rounded"
            caption="Computing the reset gate and the update gate in a GRU model."
            id="fig_gru_1" %}
    </div>
</div>

**Mathematics**: For a given time step $$t$$, let:
- $$\mathbf{X}_t \in \mathbb{R}^{n \times d}$$ be the input (batch size $$n$$, input features $$d$$),
- $$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$$ be the hidden state (hidden units $$h$$).

The gates are computed as:

$$
\begin{aligned}
\mathbf{R}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xr}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hr}} + \mathbf{b}_\textrm{r}),\\
\mathbf{Z}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xz}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hz}} + \mathbf{b}_\textrm{z}),
\end{aligned}
$$

where:
- $$\mathbf{W}_{\textrm{xr}}, \mathbf{W}_{\textrm{xz}} \in \mathbb{R}^{d \times h}$$ and $$\mathbf{W}_{\textrm{hr}}, \mathbf{W}_{\textrm{hz}} \in \mathbb{R}^{h \times h}$$ are weights,
- $$\mathbf{b}_\textrm{r}, \mathbf{b}_\textrm{z} \in \mathbb{R}^{1 \times h}$$ are biases.

**Candidate Hidden State** The reset gate integrates with the computation of the *candidate hidden state* $$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$$:

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + (\mathbf{R}_t \odot \mathbf{H}_{t-1}) \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}),$$

where:
- $$\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$$ and $$\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$$ are weights,
- $$\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$$ is the bias,
- $$\odot$$ represents elementwise multiplication (Hadamard product).

Fig.[6](#fig_gru_2) shows the computational flow for the candidate hidden state.

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager" path="https://d2l.ai/_images/gru-2.svg" class="img-fluid rounded"
            caption="Computing the candidate hidden state in a GRU model."
            id="fig_gru_2" %}
    </div>
</div>

**Hidden State** Finally, the **update gate** $$\mathbf{Z}_t$$ determines the balance between the old hidden state $$\mathbf{H}_{t-1}$$ and the candidate hidden state $$\tilde{\mathbf{H}}_t$$:

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$

Fig.[7](#fig_gru_3) illustrates this process.

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager" path="https://d2l.ai/_images/gru-3.svg" class="img-fluid rounded"
            caption="Computing the hidden state in a GRU model."
            id="fig_gru_3" %}
    </div>
</div>

**Summary** GRUs provide two key mechanisms:
- **Reset Gate**: Helps capture short-term dependencies by resetting the hidden state.
- **Update Gate**: Helps capture long-term dependencies by balancing new and old information.