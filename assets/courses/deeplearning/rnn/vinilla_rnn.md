---
layout: page
title: Recurrent Neural Network
description: 
related_publications: false
toc:
    sidebar: right
back_link: '/teaching/deeplearning'
back_text: 'Deep Learning'
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

### Gradient Clipping
While you are already used to thinking of neural networks as “deep” in the sense that many layers separate the input and output even within a single time step, the length of the sequence introduces a *new notion of depth*. 
  - In addition to the passing through the network in the input-to-output direction, inputs at the first time step must pass through a chain of $$T$$ layers along the time steps in order to influence the output of the model at the final time step.
  - Taking the backwards view, in each iteration, we backpropagate gradient through time, resulting in a chain of matrix-products of length $$\mathcal{O}(T)$$. This can result in numerical instability, causing gradients either to explode or vanish, depending on the properties of the weight matrices.

Dealing with vanishing and exploding gradients is a fundamental problem when designing RNNs and has inspired some of the biggest advances in modern neural network architectures. Later, we will talk about specialized architectures that were designed in hopes of mitigating the vanishing gradient problem. However, even modern RNNs often suffer from exploding gradients. One inelegant but ubiquitous solution is to simply clip the gradients forcing the resulting *“clipped”* gradients to take smaller values.

**Gradient Descent and Objective Changes** In gradient descent, the parameter vector $$\mathbf{x}$$ is updated as:

$$
\mathbf{x} \leftarrow \mathbf{x} - \eta \mathbf{g},
$$

where:
- $$\eta > 0$$ is the learning rate, controlling the step size,
- $$\mathbf{g}$$ is the gradient of $$f$$ at $$\mathbf{x}$$, indicating the direction of steepest ascent.

$$
|f(\mathbf{x})-f(\mathbf{y})| \leq L\|\mathbf{x}-\mathbf{y}\|
$$

If the objective function $$f$$ is **Lipschitz continuous** with constant $$L$$, then:

$$
|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|,
$$

for any $$\mathbf{x}$$ and $$\mathbf{y}$$. Applying this to a gradient update:

$$
\mathbf{y} = \mathbf{x} - \eta \mathbf{g},
$$

the change in the objective function is bounded by:

$$
|f(\mathbf{x}) - f(\mathbf{x} - \eta \mathbf{g})| \leq L \eta \|\mathbf{g}\|.
$$

Thus, the change in the objective depends on the gradient norm $$\|\mathbf{g}\|$$, the learning rate $$\eta$$, and $$L$$. Large gradient norms can cause excessively large changes, leading to instability in training.

1. **Controlled Updates**: The Lipschitz constant $$L$$, the learning rate $$\eta$$, and the gradient norm $$\|\mathbf{g}\|$$ together determine the maximum change in the objective.

2. **Stability**: Large gradient norms $$\|\mathbf{g}\|$$ can cause excessively large changes in $$f(\mathbf{x})$$, potentially destabilizing training. This is why gradient clipping or smaller $$\eta$$ may be necessary in practice.

**Gradient Clipping** To prevent exploding gradients, the gradient clipping heuristic modifies the gradient $$\mathbf{g}$$ as follows:

$$
\mathbf{g} \leftarrow \min \left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}
$$

where $$\theta$$ is a predefined threshold. This ensures the gradient norm does not exceed $$\theta$$ while preserving the direction of $$\mathbf{g}$$.

**Benefits and Limitations** Gradient clipping:
- Limits the gradient norm, improving stability.
- Reduces the influence of individual minibatches or samples, enhancing robustness.
However, it introduces a bias since the true gradient is not always followed, making analytical
reasoning about side effects difficult. Despite being a heuristic, gradient clipping is widely
adopted in RNN implementations.

```python
import torch

def clip_gradients(model, theta):
    """Clip gradients of a model to a maximum norm theta."""
    params = [p for p in model.parameters() if p.grad is not None]
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```
## Backpropagation Through Time
We mentioned that gradient clipping is vital for preventing the occasional massive gradients from destabilizing training. We hinted that the exploding gradients stem from backpropagating across long sequences. Before introducing a slew of modern RNN architectures, let's take a closer look at how *backpropagation* works in sequence models in mathematical detail. Hopefully, this discussion will bring some precision to the notion of *vanishing* and *exploding* gradients.

Applying backpropagation in RNNs is called *backpropagation through time* (BPTT) . This procedure requires us to expand (or unroll) the computational graph of an RNN one time step at a time. The unrolled RNN is essentially a feedforward neural network with the special property that the same parameters are repeated throughout the unrolled network, appearing at each time step. Then, just as in any feedforward neural network, we can apply the chain rule, backpropagating gradients through the unrolled net.

### Analysis of Gradients in RNNs
We start with a simplified model of how an RNN works. In this simplified model, let $$h_t$$ be the hidden state, $$x_t$$ the input, and $$o_t$$ the output at time step $$t$$. Recall that the input and the hidden state can be concatenated before being multiplied by one weight variable in the hidden layer. Let $$w_\textrm{h}$$ and $$w_\textrm{o}$$ represent the weights of the hidden and output layers, respectively. Then, we can represent the hidden states and outputs as:

$$
\begin{aligned}
h_t &= f(x_t, h_{t-1}, w_\textrm{h}), \\
o_t &= g(h_t, w_\textrm{o}),
\end{aligned}
$$

where $$f$$ and $$g$$ are transformations of the hidden layer and output layer, respectively.

The objective function $$L$$ over $$T$$ time steps is defined as:

$$
L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_\textrm{h}, w_\textrm{o}) = \frac{1}{T} \sum_{t=1}^T l(y_t, o_t).
$$

**Gradient Computation** Using the chain rule, the gradient with respect to $$w_\textrm{h}$$ is:

$$
\frac{\partial L}{\partial w_\textrm{h}} = \frac{1}{T} \sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_\textrm{h}} = \frac{1}{T} \sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_\textrm{o})}{\partial h_t} \frac{\partial h_t}{\partial w_\textrm{h}}.
$$

The term $$\frac{\partial h_t}{\partial w_\textrm{h}}$$ depends on both $$h_{t-1}$$ and $$w_\textrm{h}$$, requiring recursive computation. Using the chain rule, this can be expressed as:

$$
\frac{\partial h_t}{\partial w_\textrm{h}} = \frac{\partial f(x_t, h_{t-1}, w_\textrm{h})}{\partial w_\textrm{h}} + \frac{\partial f(x_t, h_{t-1}, w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.
$$

This recursive structure leads to challenges when sequences are long, as gradients can either vanish or explode.

### Backpropagation Through Time in Detail

Let $$\mathbf{h}_t$$, $$\mathbf{x}_t$$, and $$y_t$$ represent the hidden state, input, and target at time step $$t$$. For simplicity, assume the activation function is an identity mapping ($$\phi(x) = x$$). The hidden state and output are computed as:

$$
\begin{aligned}
\mathbf{h}_t &= \mathbf{W}_\textrm{hx} \mathbf{x}_t + \mathbf{W}_\textrm{hh} \mathbf{h}_{t-1}, \\
\mathbf{o}_t &= \mathbf{W}_\textrm{qh} \mathbf{h}_t.
\end{aligned}
$$

The objective function over $$T$$ time steps is:

$$
L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).
$$

**Gradient Computation** For gradients with respect to the output layer weights $$\mathbf{W}_\textrm{qh}$$:

$$
\frac{\partial L}{\partial \mathbf{W}_\textrm{qh}} = \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

where $$\frac{\partial L}{\partial \mathbf{o}_t}$$ can be computed directly from the loss.

For the hidden layer at time step $$T$$:

$$
\frac{\partial L}{\partial \mathbf{h}_T} = \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.
$$

For earlier time steps $$t < T$$:

$$
\frac{\partial L}{\partial \mathbf{h}_t} = \mathbf{W}_\textrm{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.
$$

This recurrence highlights the issue of vanishing or exploding gradients due to repeated multiplication with $$\mathbf{W}_\textrm{hh}$$.

###  Visualization of Gradient Strategies

Below, Fig.[1](#fig_truncated_bptt) compares different gradient computation strategies:

<div class="row mt-3">
    {% assign figure_counter = figure_counter | plus: 1 %}
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
            figure_number=figure_counter
            loading="eager" path="https://d2l.ai/_images/truncated-bptt.svg" class="img-fluid rounded" caption="Comparing strategies for computing gradients in RNNs. From top to bottom: randomized truncation, regular truncation, and full computation." id="fig_truncated_bptt" %}
    </div>
</div>

1. Randomized truncation partitions text into segments of varying lengths.
2. Regular truncation breaks the text into fixed-length subsequences.
3. Full computation considers the entire sequence, which is computationally infeasible.

## Summary

Backpropagation through time applies backpropagation to sequence models. Key takeaways:
- Truncation methods improve computational feasibility and numerical stability.
- Long sequences amplify challenges with vanishing or exploding gradients.
- Efficient computation requires caching intermediate values.

