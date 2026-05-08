---
layout: page
title: "Ch 5 — Neural Networks"
description: Feed-forward networks, backpropagation, regularization, mixture density nets, Bayesian neural networks.
math: true
slides: true
back_link: '/teaching/introml'
back_text: 'Intro to ML'
---

Linear models are tractable but limited: their flexibility lives entirely in the basis functions \\(\boldsymbol{\phi}(\mathbf{x})\\), which we have to choose by hand. Neural networks remove that limitation by **learning the basis functions from data**. A network is a stack of linear-then-nonlinear transformations whose parameters get optimized end-to-end against the loss.

## Prerequisites

- **Linear algebra & calculus** for vector-valued functions, plus the **chain rule** in matrix form (we re-derive backprop).
- **Logistic regression and softmax** (Ch 4) — neural networks reduce to these in the single-layer case.

---

## 5.1 Feed-Forward Networks

A two-layer network with \\(M\\) hidden units, \\(K\\) outputs, and inputs \\(\mathbf{x} \in \mathbb{R}^{D}\\) computes

$$
a_j^{(1)} = \sum_{i=1}^{D} w_{ji}^{(1)} x_i + w_{j0}^{(1)},
\qquad
z_j = h(a_j^{(1)}),
\qquad
y_k(\mathbf{x}, \mathbf{w}) = g\\!\left(\sum_{j=1}^{M} w_{kj}^{(2)} z_j + w_{k0}^{(2)}\right).
$$

The activation \\(h(\cdot)\\) is typically tanh, sigmoid, or ReLU; the output activation \\(g(\cdot)\\) is identity (regression), sigmoid (binary classification), or softmax (multiclass).

### What does a single hidden layer buy?

The **universal approximation theorem** says that a two-layer network with enough hidden units can approximate *any* continuous function on a compact domain to any desired accuracy. The catch is "enough hidden units" — the bound is existence-only, with no guarantee on parameter count or learnability.

Empirically, **deeper** networks generalize better than equally-wide shallow ones: each layer composes features from the previous one, so deep nets express hierarchies cheaply. This is the empirical observation that drove the deep-learning revolution; the theory still has loose ends.

---

## 5.2 Network Training

Pick a loss matched to the task:

| Task | Output activation | Loss |
|---|---|---|
| Regression | identity | sum-of-squares |
| Binary classification | sigmoid | cross-entropy |
| Multiclass | softmax | cross-entropy |

For all three, the gradient w.r.t. the *pre-activation* of the output layer simplifies to the same elegant form: \\(\delta\_k = y\_k - t\_k\\). That's not a coincidence; it's because each loss is the canonical link function for its output activation, derived from the corresponding exponential-family likelihood.

### Backpropagation

The gradient of the loss with respect to a weight \\(w\_{ji}^{(\ell)}\\) is computed by **two passes**:

1. **Forward pass.** Compute pre-activations \\(a\_j^{(\ell)}\\) and activations \\(z\_j^{(\ell)}\\) layer by layer.
2. **Backward pass.** Compute "errors" \\(\delta\_j^{(\ell)} = \partial E / \partial a\_j^{(\ell)}\\) starting from the output layer (where \\(\delta\_k = y\_k - t\_k\\) for matched loss/activation) and propagating backward via

$$
\delta_j^{(\ell)} = h'(a_j^{(\ell)}) \sum_k w_{kj}^{(\ell+1)} \delta_k^{(\ell+1)}.
\tag{5.1}
$$

3. **Combine.** \\(\partial E / \partial w\_{ji}^{(\ell)} = \delta\_j^{(\ell)}\, z\_i^{(\ell-1)}\\).

This is just the chain rule, but written so that each weight gradient costs O(1) matrix work per training point — the total cost of a gradient is the same order as a forward pass.

### Optimization

Vanilla gradient descent is too slow for non-convex landscapes. Practical optimizers:

- **SGD with momentum** — a moving-average velocity term smooths over noisy gradients.
- **Adam / RMSProp** — per-parameter adaptive step sizes, scaled by an estimate of recent gradient magnitude.
- **Second-order** methods (L-BFGS, natural-gradient) work for smaller networks but rarely scale.

Modern deep learning lives almost entirely in the SGD-with-Adam regime.

---

## 5.3 Regularization

Neural nets can interpolate noisy data trivially. Common regularizers:

- **Weight decay** — \\(\lambda\\\|\mathbf{w}\\\|^2\\) penalty, equivalent to a Gaussian prior on \\(\mathbf{w}\\). Same as Ch 1's ridge term.
- **Early stopping** — track validation error during training; stop when it begins to rise. Equivalent in spirit to weight decay along the optimization trajectory.
- **Dropout** — randomly zero out hidden units during each forward pass. Acts like an ensemble of exponentially many sub-networks; surprisingly cheap and effective.
- **Data augmentation** — apply label-preserving transformations (rotations, crops, flips for images; noise injection for audio). Encodes domain invariances into the loss.
- **Batch normalization** — re-normalize each layer's activations during training. Reduces internal covariate shift; allows much higher learning rates.

The right regularizer is the one that encodes the *correct* invariances for your problem.

---

## 5.4 Mixture Density Networks

For some regression problems, the conditional distribution \\(p(t \mid \mathbf{x})\\) is **multimodal** — several plausible \\(t\\) for each \\(\mathbf{x}\\). A network with a single Gaussian output gets the conditional mean (averaging across modes) — usually a useless answer.

A **mixture density network** outputs the parameters of a *mixture* of Gaussians:

$$
p(t \mid \mathbf{x}) = \sum_{k=1}^{K} \pi_k(\mathbf{x})\, \mathcal{N}(t \mid \mu_k(\mathbf{x}), \sigma_k^{2}(\mathbf{x})),
$$

with all of \\(\pi\_k, \mu\_k, \sigma\_k\\) computed by the network itself. Trains by minimizing the negative log-likelihood. Useful for inverse problems (multiple right answers) and for any task where calibrated multi-modal predictions matter.

---

## 5.5 Bayesian Neural Networks

Putting a prior on the weights and computing a posterior is conceptually clean but **intractable** for any non-trivial network — millions of parameters, no conjugacy. Practical approximations:

- **Laplace approximation** at the MAP (Ch 4 idea, applied per layer).
- **Variational inference** (Ch 10) — approximate the posterior with a factorized Gaussian.
- **MC dropout** — interpret each dropout mask as a sample from an approximate posterior; predictive uncertainty = std-dev across forward passes with different masks.
- **Deep ensembles** — train several networks from different initializations; treat the spread of predictions as an uncertainty estimate. Embarrassingly parallel and competitive with the more elaborate methods.

Calibrated uncertainty matters for safety-critical applications (medicine, autonomous driving) and for active learning.

---

## Takeaways

- **Linear → nonlinear** by composing layers; the **universal approximation theorem** says one hidden layer is enough in principle, but depth wins in practice.
- **Backpropagation** is the chain rule applied carefully, costing one extra forward pass per gradient.
- **Cross-entropy + softmax** (or sum-of-squares + identity) gives the elegant \\(\delta = y - t\\) output gradient — a clean signal of "what was wrong."
- **Regularization is essential** — weight decay, dropout, augmentation, batch norm. The right one depends on the problem.
- **Bayesian deep learning** is hard, but approximations (MC dropout, deep ensembles) give actionable uncertainty for cheap.

Forward to <a href="{{ '/assets/courses/introml/06_kernel_methods/' | relative_url }}">Ch 6 — Kernel Methods</a>.

---

<small><em>Figures from Bishop, <em>Pattern Recognition and Machine Learning</em> (Springer, 2006), © Springer / C. M. Bishop. Reproduced for educational use.</em></small>
