---
layout: page
title: "Batch Normalization (Ioffe & Szegedy, 2015)"
description: "Reading note · Pattern Recognition · Neural Network Foundations"
back_link: '/teaching/pattern/'
back_text: "Back to Pattern Recognition"
---

## TL;DR

Insert a layer that, for each mini-batch and each feature, **subtracts the batch mean and divides by the batch standard deviation**, then applies a learnable affine \\(\gamma x + \beta\\). Doing this between linear layers and non-linearities makes deep networks dramatically easier to train — convergence is faster, learning rates can be 10× larger, and the network is much less sensitive to weight initialization.

The 2015 paper introduces this as **Batch Normalization (BN)**, motivates it via "internal covariate shift," and shows it cuts ImageNet training time by half.

---

## Problem &amp; Motivation

In a deep network, the input distribution to layer \\(\ell\\) keeps shifting as the parameters of layers \\(1, \ldots, \ell - 1\\) update. Layer \\(\ell\\) is constantly adapting to a moving target — what Ioffe and Szegedy call **internal covariate shift**. Two practical symptoms:

- **Slow convergence.** Each layer's optimal weights drift with each update; the optimizer has to chase them.
- **Sensitivity to learning rate / initialization.** Small changes blow up downstream activations or collapse them to zero, depending on whether the chain of layers amplifies or contracts variance.

Older fixes (careful initialization, lower learning rates) manage symptoms without addressing the underlying problem.

Batch Normalization's pitch: **stabilize the distribution of activations at every layer**, so each layer always sees inputs with roughly zero mean and unit variance. This lets the layer focus on learning a useful function rather than tracking distribution drift.

(The "internal covariate shift" framing has been disputed in subsequent work — Santurkar et al., 2018, argue BN's real benefit is that it smooths the loss landscape, making large gradient steps safe. Either way, the empirical effect is unambiguous.)

---

## The Layer

Given a mini-batch of activations \\(\mathcal{B} = \\{x\_1, \ldots, x\_m\\}\\) for a single feature, compute

$$
\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i, \qquad
\sigma_\mathcal{B}^{2} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^{2}.
$$

Normalize:

$$
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^{2} + \epsilon}}.
$$

Each \\(\hat{x}\_i\\) has zero mean and unit variance across the mini-batch.

Then apply a learnable affine:

$$
y_i = \gamma\, \hat{x}_i + \beta.
$$

\\(\gamma\\) and \\(\beta\\) are **per-feature trainable parameters**. They restore the network's ability to learn arbitrary mean and scale — if the optimal pre-activation distribution for a particular feature is *not* \\(\mathcal{N}(0,1)\\), the affine can recover it.

A subtle but important point: setting \\(\gamma = \sigma\_\mathcal{B}\\) and \\(\beta = \mu\_\mathcal{B}\\) recovers the unnormalized identity, so BN strictly increases the network's representational capacity.

### Where in the architecture?

The original paper places BN **before** the non-linearity:

$$
\ldots \to \text{Linear} \to \text{BN} \to \text{ReLU} \to \ldots
$$

Some later work (especially for residual networks) places it after. The pre-vs-post-activation choice is a minor empirical question — both work.

For convolutional layers, the per-feature normalization is per-channel: every channel has its own \\(\mu, \sigma, \gamma, \beta\\), and the statistics are taken over the spatial dimensions *and* the batch.

---

## Train vs Test

A wrinkle: at training time, \\(\mu\_\mathcal{B}\\) and \\(\sigma\_\mathcal{B}\\) are computed *over the mini-batch*. At test time, the mini-batch may not exist (single-image inference) or may be the wrong distribution (different domain).

The paper's solution: maintain **running estimates** of \\(\mu\\) and \\(\sigma^{2}\\) during training using exponential moving averages, then use these fixed statistics at inference. So:

- **Train:** normalize using current batch statistics; update the running mean/variance.
- **Test:** normalize using the running estimates, no randomness. The whole BN layer becomes a fixed affine \\(\gamma' x + \beta'\\) that can be folded into the preceding linear layer for free.

This is why BN at inference adds zero compute beyond the original linear layer — it's just a re-parameterization.

---

## Why It Works

A few mechanisms, all reinforcing:

### 1. Stable activation distributions

Each layer always sees zero-mean, unit-variance inputs (modulo the learnable affine). The chain of layers no longer amplifies or contracts variance pathologically; the network can be much deeper without collapsing or saturating.

### 2. Effective regularization

The mini-batch statistics are noisy: each mini-batch sees slightly different \\(\mu\\) and \\(\sigma\\). The per-example normalization is therefore a stochastic perturbation that varies between batches, acting like a regularizer (similar in spirit to dropout). This is part of why BN networks generalize as well as they do despite using larger learning rates.

### 3. Higher learning rates are safe

Pre-BN, a large learning rate would push some pre-activations to extreme values, where sigmoid/tanh saturate or ReLU dies. BN re-centers and rescales, so even after a big gradient step the next layer still sees sane inputs. The paper shows ImageNet training works at \\(\alpha = 0.1\\) with BN vs \\(\alpha \approx 0.01\\) without.

### 4. Initialization tolerance

Networks without BN need careful initialization (Xavier, He) to keep variance preserved through the layers. With BN inserted, the variance is enforced at every layer regardless of initialization — initial weights matter much less.

---

## Empirical Results From the Paper

- **MNIST.** A simple MLP trains 6× faster with BN to the same accuracy.
- **ImageNet (Inception).** Batch-normalized Inception reaches the same accuracy as the baseline in **half the steps**, and beats the baseline's final accuracy by a small margin. With 5× learning rate and BN, the network reaches the baseline's final accuracy in 14× fewer steps.
- **Single-network state-of-the-art.** A BN-Inception ensemble achieves 4.9 % top-5 error on ImageNet — the best single-network result at the time of publication.

The training-speed wins are the headline. Modern CNNs would not be feasible at their current depths without BN or one of its descendants.

---

## Practical Caveats

### 1. Small batch sizes break BN

The mini-batch statistics become noisy when the batch size is small (say, \\(\leq 8\\)). Per-image inference or memory-constrained training (large models, small per-GPU batches) suffers. Group Normalization (Wu &amp; He, 2018) is the standard fix: normalize over **groups of channels** within a single example, batch-size-independent.

### 2. Recurrent networks

BN's batch statistics don't transfer cleanly across time steps in an RNN — the input distribution at time \\(t\\) differs from time \\(t-1\\). **Layer Normalization** (Ba, Kiros &amp; Hinton, 2016) was developed for this: normalize across the *features* of a single example rather than across the batch. Layer Norm has become the default for transformers.

### 3. Train-test discrepancy

The running statistics estimated during training don't always match the deployed-input distribution. This is especially problematic when:

- Test-time batch sizes are small or single-image.
- Domain shifts between training and deployment.
- Augmentations differ between train and test.

Mitigation: re-estimate BN statistics on a representative held-out set after training. Some libraries do this automatically.

### 4. Combining with dropout

BN provides its own regularization. Adding heavy dropout on top often hurts. Modern practice: lighter dropout or no dropout when BN is in use.

---

## Modern Context

- **Layer Normalization** (Ba, Kiros &amp; Hinton, 2016): per-example normalization; default for transformers and RNNs.
- **Instance Normalization** (Ulyanov, Vedaldi &amp; Lempitsky, 2016): per-channel-per-example; default for style transfer.
- **Group Normalization** (Wu &amp; He, 2018): in-between BN and LN; works at any batch size.
- **RMS Norm** (Zhang &amp; Sennrich, 2019): drop the mean-centering, just rescale by RMS. Cheaper, often equally effective. Used in modern transformer variants (LLaMA, etc.).
- **Pre-activation vs post-activation residual connections** (He et al., 2016) interact non-trivially with normalization placement. Pre-norm is the default in modern transformer stacks.

The "normalize at every layer" idea is now table stakes. Which *kind* of normalization to use is the modern question.

---

## Reading the Paper

The paper is short but math-dense. Key sections:

- §2: motivation through internal covariate shift. Read once for the framing; the framing has been re-litigated in subsequent work.
- §3: the BN layer itself (Algorithm 1 and equations 5–8). Worth working through on paper.
- §3.2: training and inference modes; the running-statistics trick.
- §4: experimental results on MNIST and ImageNet/Inception.

Critical follow-ups:

- Santurkar et al., 2018, "How Does Batch Normalization Help Optimization?" — argues that BN's benefit is loss landscape smoothness, not covariate shift.
- Ba, Kiros &amp; Hinton, 2016, "Layer Normalization" — the per-example variant.
- Wu &amp; He, 2018, "Group Normalization" — the batch-size-independent variant.

---

## Takeaways

- **BN normalizes per feature, per mini-batch:** subtract \\(\mu\_\mathcal{B}\\), divide by \\(\sigma\_\mathcal{B}\\), then apply a learnable affine \\(\gamma x + \beta\\).
- **Running statistics** are saved during training and used at inference, so BN at test time is a fixed affine that folds into the preceding linear layer.
- **Why it works** is partly stable activations, partly regularization from batch noise, partly enabling high learning rates. The "internal covariate shift" story is the original motivation but probably not the deepest reason.
- **Modern descendants** — Layer Norm, Group Norm, RMS Norm — generalize BN to settings where batch statistics aren't appropriate. Some form of normalization is in essentially every modern deep network.
