---
layout: page
title: Recurrent Neural Network
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

**Extending Beyond $$n$$-grams** In <a href="{{ 'assets/courses/deeplearning/rnn/markov' | relative_url }}">Markov Models and n-grams</a>, we introduced Markov models and $$n$$-grams for language modeling, where the conditional probability of a token $$x_t$$ at time step $$t$$ depends only on the previous $$n-1$$ tokens. To account for tokens occurring earlier than $$t-(n-1)$$, we could increase $$n$$. However, this approach comes at a $$\textcolor{red}{\text{significant cost}}$$
—model parameters grow exponentially with $$n$$, as we need to store $$
|\mathcal{V}|^{n}$$ parameters for a vocabulary set $$\mathcal{V}$$. Instead of modeling:

$$P(x_t \mid x_{t-1}, \ldots, x_{t-n+1}),$$

we can introduce a *latent variable model*:

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

where $$h_{t-1}$$ is a *hidden state* that captures all relevant sequence information up to time step $$t-1$$.

**Hidden State Representation** The hidden state $$h_t$$ at any time step $$t$$ is computed as a function of the current input $$x_t$$ and the previous hidden state $$h_{t-1}$$:

\begin{equation}
\label{eq:eq_ht_xt} h_t = f(x_t, h_{t-1}). 
\end{equation}

For a sufficiently expressive function $$f$$ in eq.\eqref{eq:eq_ht_xt}, the latent variable model can, in principle, capture all information observed so far, eliminating the need for approximation. However, storing all past information can make computation and storage infeasible.

**Distinction Between Hidden Layers and Hidden States** *Hidden layers* refer to intermediate layers in a neural network that are not directly exposed to input or output. In contrast, *hidden states* in the context of sequence modeling are technically inputs to the computation at any given time step. Hidden states encapsulate information from past time steps and are crucial for modeling sequences.

**Recurrent Neural Network** Recurrent Neural Networks (RNNs) are a class of neural networks designed specifically for sequential data, incorporating *hidden states* to capture dependencies over time. Before delving into the RNN architecture, let us revisit the foundational concept of Multilayer Perceptrons (MLPs). 

## Neural Networks without Hidden States
Let's take a look at an MLP with a single hidden layer.  Let the hidden layer's activation function be $$\phi$$. Given a minibatch of examples $$\mathbf{X} \in \mathbb{R}^{n \times d}$$ with batch size $$n$$ and $$d$$ inputs, the hidden layer output $$\mathbf{H} \in \mathbb{R}^{n \times h}$$ is calculated as:

\begin{equation}\label{eq:rnn_h_with_state}
\mathbf{H}=\phi\left(\mathbf{X} \mathbf{W}_{\mathrm{xh}}+\mathbf{b}\_\mathrm{h}\right)
\end{equation}

In eq.\eqref{eq:rnn_h_with_state}, we have:
- Weight parameter $$\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$$,
- Bias parameter $$\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$$,
- Number of hidden units $$h$$ for the hidden layer.

The hidden layer output $$\mathbf{H}$$ is then used as input to the output layer:

\begin{equation}
\mathbf{O} = \mathbf{H} \mathbf{W}_{\textrm{hq}} + \mathbf{b}\_\textrm{q},
\end{equation}

where:
- $$\mathbf{O} \in \mathbb{R}^{n \times q}$$ is the output variable,
- $$\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$$ is the weight parameter,
- $$\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$$ is the bias parameter of the output layer.

For classification problems, we can apply $$\mathrm{softmax}(\mathbf{O})$$ to compute the probability distribution of the output categories.

## Recurrent Neural Networks with Hidden States
RNNs differ from MLPs by incorporating *hidden states*. Assume we have:
- A minibatch of inputs $$\mathbf{X}_t \in \mathbb{R}^{n \times d}$$ at time step $$t$$,
- Hidden state $$\mathbf{H}_t \in \mathbb{R}^{n \times h}$$ for the same time step.

Unlike MLPs, the hidden state $$\mathbf{H}_{t-1}$$ from the previous time step is saved, and a new weight parameter $$\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$$ describes how to use it in the current time step. Specifically, the hidden state for time step $$t$$ is calculated as:

\begin{equation}
\mathbf{H}\_t = \phi(\mathbf{X}\_t \mathbf{W}\_{\textrm{xh}} + \mathbf{H}\_{t-1} \mathbf{W}\_{\textrm{hh}} + \mathbf{b}\_\textrm{h}).
\label{rnn_h_with_state}
\end{equation}

In \eqref{rnn_h_with_state}, the term $$\mathbf{H}_{t-1} \mathbf{W}\_{\textrm{hh}}$$ introduces recurrent computation. This allows $$\mathbf{H}_t$$ to capture historical sequence information, making it a *hidden state*. Layers performing this recurrent computation are called *recurrent layers*.

The output layer computes the output for time step $$t$$ as:

\begin{equation}
\mathbf{O}\_t = \mathbf{H}\_t \mathbf{W}\_{\textrm{hq}} + \mathbf{b}\_\textrm{q}.
\end{equation}

The parameters of an RNN include:
- Weights: $$\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}, \mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$$,
- Biases: $$\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}, \mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$$.

Unlike MLPs, RNN parameters are shared across all time steps, meaning the model's size does not increase with sequence length.

**RNN Computation at Ajacent Time Steps** Fig.[1](#rnn) illustrates an RNN across three adjacent time steps. At any time step $$t$$:
1. Concatenate $$\mathbf{X}_t$$ (current input) and $$\mathbf{H}_{t-1}$$ (previous hidden state),
2. Pass the concatenated result into a fully connected layer with activation $$\phi$$ to compute $$\mathbf{H}_t$$.

The hidden state $$\mathbf{H}_t$$ is used for:
- Computing $$\mathbf{H}_{t+1}$$ in the next time step,
- Producing the output $$\mathbf{O}_t$$.

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager" path="https://d2l.ai/_images/rnn.svg" class="img-fluid rounded" caption="An RNN with a hidden state." id="rnn" %}
    </div>
</div>

**Efficient Computation with Concatenation** We can compute $$\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$$ equivalently as a single matrix multiplication. Concatenate $$\mathbf{X}_t$$ and $$\mathbf{H}_{t-1}$$, and multiply it by the concatenated weights $$[\mathbf{W}_{\textrm{xh}}, \mathbf{W}_{\textrm{hh}}]$$. The following Python snippet demonstrates this:

```python
import torch
X = torch.randn((3, 1))  # Shape: (3, 1)
W_xh = torch.randn((1, 4))  # Shape: (1, 4)
H = torch.randn((3, 4))  # Shape: (3, 4)
W_hh = torch.randn((4, 4))  # Shape: (4, 4)

# Compute hidden state
output = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
print(output.shape)  # Output shape: (3, 4)
```
```bash
tensor([[ 1.2526,  0.0580, -3.3460, -0.2519],
        [-1.3064,  1.4132, -0.1435,  0.3482],
        [ 3.1495,  0.8172,  1.5167, -0.9038]])
```
*equivalent to*

```python
torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
```
```bash
tensor([[ 1.2526,  0.0580, -3.3460, -0.2519],
        [-1.3064,  1.4132, -0.1435,  0.3482],
        [ 3.1495,  0.8172,  1.5167, -0.9038]])
```
## RNN-Based Character-Level Language Models

For language modeling as discussed before, the goal is to predict the next token based on the current and previous tokens. To achieve this, the original sequence is shifted by one token to create the targets (labels). Neural networks for language modeling, as proposed by [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), can use RNNs to accomplish this task. Here's how an RNN processes the sequence:

**Key Idea** For simplicity, we tokenize text into characters rather than words, building a **character-level language model**. For instance, consider the text sequence `"machine"`:
- **Input Sequence**: `"machin"`
- **Target Sequence**: `"achine"`

**RNN Processing** Fig.[2](#fig_rnn_train) demonstrates how RNNs predict the next character based on the current and previous characters in the sequence.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% assign figure_counter = figure_counter | plus: 1 %}
        {% include figure.liquid 
            figure_number=figure_counter
            loading="eager" path="https://d2l.ai/_images/rnn-train.svg" class="img-fluid rounded" caption="A character-level language model based on the RNN. The input and target sequences are 'machin' and 'achine', respectively." id="fig_rnn_train" %}
    </div>
</div>

At each time step:
- The RNN processes the input token, updating its **hidden state**.
- The output at each time step is used to predict the next character.

For example:
- **Input at Time Step 3**: `"m", "a", "c"`
- **Target**: `"h"`
- **Output**: $$\mathbf{O}_3$$, determined by the sequence `"m", "a", "c"`. The cross-entropy loss is computed between the predicted distribution and the actual target (`"h"`).

**Practical Considerations** Each token is represented as a $$d$$-dimensional vector. With a batch size $$n > 1$$, the input at time step $$t$$, $$\mathbf{X}_t$$, is an $$n \times d$$ matrix, consistent with the description in [Recurrent Neural Networks with Hidden States](#recurrent-neural-networks-with-hidden-states). 