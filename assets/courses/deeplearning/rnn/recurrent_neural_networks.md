---
layout: page
title: Recurrent Neural Networks
description: 
related_publications: false
toc:
    sidebar: left
back_link: '/teaching/deeplearnig'
back_text: 'Deep Learning Course Page'
number_heading: true
---

Models like linear/ logistic regression, multilayer perceptrons (MLPs) and convolutional neural networks (CNNs) operate on fixed-length input (tabular or image data) without sequential structure. *<u>What is a sequential data?</u>* Tasks like video analysis, time-series prediction, image captioning, speech synthesis, and translation involve sequentially structured inputs and outputs, requiring specialized models.

## Introduction 
RNNs are designed to capture the dynamics of sequences through *recurrent connections* that pass information across adjacent time steps. Recurrent neural networks are *unrolled* across time steps (or sequence steps), with the *same* underlying parameters applied at each step. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://d2l.ai/_images/unfolded-rnn.svg" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    RNNs can be visualized as feedforward neural networks where parameters are shared across time steps. On the left recurrent connections are depicted via cyclic edges. On the right, we unfold the RNN over time steps. Here, recurrent edges span adjacent time steps, while conventional connections are computed synchronously.
</div>

RNNs have achieved breakthroughs in tasks like handwritting recognition, translation, and medical diagosis but have been partly replaced by Transformer models.

## Working with Sequences

**Shift in Perspective**  Traditional models work with a single feature vector $$\mathbf{x} \in \mathbb{R}^{d}$$. Sequence models handle ordered lists of feature vectors: $$\mathbf{x}_{1}, \ldots, \mathbf{x}_{T}$$, where each vector $$\mathbf{x}_{t} \in \mathbb{R}^{d}$$ is indexed by time step $$t \in \mathbb{Z}^{+}$$.

**Sequential Data**  Some datasets consist of one long sequence (e.g., climate sensor data), sampled into subsequences of length $$T$$.  
More commonly, data arrive as multiple sequences (e.g., documents, patient stays), where each sequence has its own length $$T_{i}$$.  
Unlike independent feature vectors, elements within a sequence are dependent:
- Later words in a document depend on earlier ones.
- A patient's medication depends on prior events.

**Sequential Dependence**  Sequences reflect patterns that make auto-fill features and predictions possible.  
Sequences are modeled as samples from a fixed underlying distribution over entire sequences, $$P(X_{1}, \dots, X_{T})$$, rather than assuming independence or stationarity.

**Examples of Sequential Tasks**  
1. **Fixed input to fixed target**: Predict a label $$y$$ from a sequence (e.g., sentiment classification).  
2. **Fixed input to sequential target**: Predict $$(y_{1}, \ldots, y_{T})$$ from an input (e.g., image captioning).  
3. **Sequential input to sequential target**:
   - *Aligned*: Input at each step aligns with the target (e.g., part-of-speech tagging).
   - *Unaligned*: No step-wise correspondence (e.g., machine translation).

**Sequence Modeling** The simplest task is **unsupervised density modeling**:  Estimate $$p(\mathbf{x}_{1}, \ldots, \mathbf{x}_{T})$$, the likelihood of a sequence, where the probability reflects the joint likelihood of the sequence elements based on their relationships. This is useful for understanding and generating sequences.

### Autoregressive Models

**Sequence Data**  Autoregressive models analyze sequentially structured data. Consider stock prices like those in the FTSE 100 index: $$x_t, x_{t-1}, \ldots, x_1,$$ where each $$x_t$$ represents the price at time $$t$$. The goal is to predict the next value $$x_t$$ based on its history.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://d2l.ai/_images/ftse100.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
FTSE 100 index over about 30 years.
</div>


**Modeling Sequential Data** Given a sequence, the trader wants to estimate the conditional distribution:  

$$P(x_t \mid x_{t-1}, \ldots, x_1),$$  

focusing on key statistics like the expected value:

$$\mathbb{E}[x_t \mid x_{t-1}, \ldots, x_1].$$  

Autoregressive models perform this task by regressing $$x_t$$ onto previous values $$x_{t-1}, \ldots, x_1$$. However, the challenge lies in the *variable input size*, as the number of inputs grows with $$t$$.

**Strategies for Fixed-Length Inputs**  
1. **Windowing**: Instead of using the full history, consider only a window of size $$\tau$$:  

   $$x_{t-1}, \ldots, x_{t-\tau}.$$  

   This reduces the number of inputs to a fixed size for $$t > \tau$$, enabling the use of models like linear regression or neural networks.

2. **Latent State Representations**: Maintain a latent summary $$h_t$$ of past observations.  

   - Predict $$x_t$$ using: $$\hat{x}_t = P(x_t \mid h_t).$$  

   - Update the latent state with: $$h_t = g(h_{t-1}, x_{t-1}).$$  

   This approach creates *latent autoregressive models*, as $$h_t$$ is not directly observed.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://d2l.ai/_images/sequence-model.svg" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
A latent autoregressive model.
</div>

**Training Data and Stationarity** Training data is often constructed by sampling fixed-length windows from historical data. Even though the specific values $$x_t$$ may change, the underlying generation process is often assumed to be stationary, meaning the dynamics of $$P(x_t \mid x_{t-1}, \ldots, x_1)$$ remain consistent over time.

### Sequence Models

**Joint Probability of Sequences** Sequence models estimate the joint probability of an entire sequence, typically for data composed of discrete *tokens* like words. These models are often referred to as *language models* when dealing with natural language data. Language models are particularly useful for:

- Evaluating the likelihood of sequences (e.g., comparing naturalness of sentences in machine translation or speech recognition).
- Sampling sequences and optimizing for the most likely outcomes.

The joint probability of a sequence $$P(x_1, \ldots, x_T)$$ can be decomposed using the chain rule of probability into conditional densities:

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

For discrete signals, the model must act as a probabilistic classifier, outputting a distribution over the vocabulary for the next word based on the leftward context.

#### Markov Models
:label:`subsec_markov-models`

Instead of conditioning on the entire history $$x_{t-1}, \ldots, x_1$$, we may limit the context to the previous $$\tau$$ time steps, i.e., $$x_{t-1}, \ldots, x_{t-\tau}$$. This is known as the *Markov condition*, where the future is conditionally independent of the past given the recent history. When:

- $$\tau = 1$$, it is called a *first-order Markov model*.
- $$\tau = k$$, it is called a *k\textsuperscript{th}-order Markov model*.

For a first-order Markov model, the factorization simplifies to:

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}).$$

Markov models are practical even if the condition is not strictly true. With real-world text, additional context improves predictions, but the marginal benefits diminish as the context length increases. Hence, many models rely on the Markov assumption for computational efficiency.

For discrete data like language, Markov models estimate $$P(x_t \mid x_{t-1})$$ via relative frequency counts and efficiently compute the most likely sequence using dynamic programming.


#### The Order of Decoding

The factorization of a sequence can follow any order (e.g., left-to-right or right-to-left):

1. **Left-to-right (natural reading order)**:
   $$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$
2. **Right-to-left**:
   $$P(x_1, \ldots, x_T) = P(x_T) \prod_{t=T-1}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$

While all orders are mathematically valid, left-to-right decoding is preferred for several reasons:
- **Naturalness**: Matches the human reading process.
- **Incrementality**: Probabilities over sequences can be extended by multiplying by the conditional probability of the next token:
  $$P(x_{t+1}, \ldots, x_1) = P(x_{t}, \ldots, x_1) \cdot P(x_{t+1} \mid x_{t}, \ldots, x_1).$$
- **Predictive Power**: Predicting adjacent tokens is often more feasible than predicting tokens at arbitrary positions.
- **Causal Relationships**: Forward predictions (e.g., $$P(x_{t+1} \mid x_t)$$) often align with causality, whereas reverse predictions ($$P(x_t \mid x_{t+1})$$) are generally infeasible.

For instance, in causal systems, forward predictions might follow:

$$x_{t+1} = f(x_t) + \epsilon,$$

where $$\epsilon$$ represents additive noise. The reverse relationship generally does not hold.

For a more detailed exploration of causality and sequence modeling, refer to :cite:`Peters.Janzing.Scholkopf.2017`.



