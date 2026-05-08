---
layout: page
title: "Ch 7 — Sparse Kernel Machines"
description: Support vector machines, the maximum-margin principle, soft margins, multi-class extensions, the relevance vector machine.
math: true
slides: true
back_link: '/teaching/introml'
back_text: 'Intro to ML'
---

Kernel methods (Ch 6) are powerful but require evaluating the kernel at *every* training point at prediction time — expensive when \\(N\\) is large. **Sparse** kernel methods produce solutions where most of the kernel coefficients are zero, so prediction depends on only a handful of "important" training points. This chapter covers the two prototypes: the **support vector machine** (SVM) and the **relevance vector machine** (RVM).

## Prerequisites

- **Kernel methods** (Ch 6) — dual representations, Mercer kernels.
- **Constrained optimization & Lagrange multipliers** — SVM training is a quadratic program with linear constraints.
- **Logistic regression / probabilistic discriminative models** (Ch 4) — context for what SVM/RVM gain and lose.

---

## 7.1 Maximum Margin Classifiers

For two linearly-separable classes \\(\lbrace t\_n \in \pm 1\rbrace\\) and a linear classifier \\(y(\mathbf{x}) = \mathbf{w}^{\top}\boldsymbol{\phi}(\mathbf{x}) + b\\), the **margin** is the distance from the decision boundary to the closest training point. The maximum-margin solution maximizes that distance.

Among all separating hyperplanes, the maximum-margin one is provably the best in a generalization-bound sense (PAC-Bayes, VC-dimension), and empirically tends to generalize well.

After scaling \\((\mathbf{w}, b)\\) so that the closest points have \\(t\_n y(\mathbf{x}\_n) = 1\\), the problem becomes

$$
\min_{\mathbf{w}, b} \;\tfrac{1}{2} \\|\mathbf{w}\\|^{2}
\quad \text{subject to} \quad
t_n (\mathbf{w}^{\top}\boldsymbol{\phi}_n + b) \geq 1 \;\forall n.
\tag{7.1}
$$

Convex quadratic objective, linear constraints — a quadratic program with a unique global optimum.

### The dual form

Introducing Lagrange multipliers \\(a\_n \geq 0\\) for each constraint and eliminating \\(\mathbf{w}\\) gives the dual

$$
\max_{\mathbf{a}} \;\sum_{n=1}^{N} a_n - \tfrac{1}{2} \sum_{n,m} a_n a_m\, t_n t_m\, k(\mathbf{x}_n, \mathbf{x}_m),
\quad
0 \leq a_n,\; \sum_n a_n t_n = 0.
\tag{7.2}
$$

Two important consequences:

- The data appear only inside \\(k(\mathbf{x}\_n, \mathbf{x}\_m)\\) — kernel trick again.
- KKT conditions force \\(a\_n = 0\\) for any point with \\(t\_n y(\mathbf{x}\_n) > 1\\). Only points *exactly on the margin* (where \\(t\_n y(\mathbf{x}\_n) = 1\\)) get \\(a\_n > 0\\). These are the **support vectors**, and they are the only points that contribute to predictions.

This is **the** sparsity property: prediction cost scales with the number of support vectors, not the full \\(N\\).

<figure class="figure">
  <img src="{{ '/assets/img/prml/Figure7.1.png' | relative_url }}" alt="Maximum-margin classifier" loading="lazy">
  <figcaption><strong>Fig 7.1.</strong> Maximum-margin classifier. The solid line is the decision boundary; the dashed lines pass through the support vectors (highlighted). The margin width is \(2/\\|\mathbf{w}\\|\) and is maximized.</figcaption>
</figure>

---

## 7.2 Soft Margin SVM

Real data are rarely separable. The **soft-margin SVM** relaxes the hard constraints by introducing slack variables \\(\xi\_n \geq 0\\):

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \; \tfrac{1}{2}\\|\mathbf{w}\\|^{2} + C \sum_{n} \xi_n,
\quad
t_n y(\mathbf{x}_n) \geq 1 - \xi_n,\; \xi_n \geq 0.
\tag{7.3}
$$

\\(C > 0\\) is a hyperparameter trading margin width against mis-classification slack. \\(C \to \infty\\) recovers the hard-margin problem; \\(C \to 0\\) maximizes the margin while ignoring the data entirely.

The dual is identical to (7.2) except the constraints become \\(0 \leq a\_n \leq C\\) — points that violate the margin saturate at \\(a\_n = C\\). Three categories of training points:

- \\(a\_n = 0\\): on the correct side of the margin, ignored at prediction time.
- \\(0 < a\_n < C\\): exactly on the margin, *real* support vectors.
- \\(a\_n = C\\): inside the margin or misclassified.

### The hinge loss

Eliminating \\(\xi\_n\\) shows the soft-margin SVM is equivalent to minimizing

$$
\sum_n [1 - t_n y(\mathbf{x}_n)]_+ + \lambda \\|\mathbf{w}\\|^{2},
\qquad [u]_+ = \max(0, u),
$$

where \\([1 - t y]\_+\\) is the **hinge loss**. Compared to logistic regression's smooth log-loss, hinge is non-smooth and has a flat zero region — that flat region is precisely what gives sparsity (most points sit there with zero gradient, hence zero coefficient).

---

## 7.3 Multi-Class SVM

SVMs are intrinsically two-class; extending to \\(K\\) classes is awkward.

- **One-vs-rest:** train \\(K\\) binary classifiers, each separating one class from all others. Requires care with imbalanced classes; outputs aren't probabilities.
- **One-vs-one:** train \\(K(K-1)/2\\) pairwise classifiers; predict by majority vote. More classifiers but each on smaller data.
- **Crammer–Singer formulation:** a single optimization problem maximizing the margin across all classes simultaneously. Cleaner but harder to optimize.

For most applications: train \\(K\\) one-vs-rest models, calibrate with Platt scaling if you need probabilities.

---

## 7.4 SVM for Regression

A regression analog: the **\\(\epsilon\\)-insensitive loss** is zero for residuals smaller than \\(\epsilon\\) and linear outside. This produces sparse solutions where only points outside the \\(\epsilon\\)-tube are support vectors. Training is a quadratic program; the dual depends on the data only through kernels.

---

## 7.5 The Relevance Vector Machine

The SVM's flaws: outputs aren't probabilities; the hyperparameter \\(C\\) (and any kernel parameters) must be cross-validated; sparsity comes from a non-smooth loss (mathematically inelegant).

The **relevance vector machine** addresses all three. Place a Gaussian prior on each coefficient \\(w\_n\\), with **per-coefficient precision** \\(\alpha\_n\\):

$$
p(\mathbf{w} \mid \boldsymbol{\alpha}) = \prod_{n=1}^{N} \mathcal{N}(w_n \mid 0, \alpha_n^{-1}).
$$

Maximizing the marginal likelihood w.r.t. \\(\boldsymbol{\alpha}\\) — the same Ch 3 evidence procedure — drives many \\(\alpha\_n \to \infty\\). The corresponding \\(w\_n\\) collapse to zero, leaving only a few "**relevance vectors**" with active weights.

Compared to SVM:

| | SVM | RVM |
|---|---|---|
| Probabilistic outputs | No (not directly) | Yes |
| Hyperparameter tuning | Cross-validate \\(C\\), kernel | Marginal likelihood |
| Sparsity | Hinge loss | Automatic relevance determination |
| Kernel constraints | Mercer-positive | Any function |
| Training cost | Convex QP, fast | Non-convex evidence max, slower |

RVMs typically give *more* sparsity than SVMs and yield calibrated predictive distributions. They're slower to train and the optimization is non-convex, so the SVM remains the more popular choice in practice — but the RVM is the cleaner Bayesian story.

---

## Takeaways

- **Maximum margin** is the principled answer to "which separating hyperplane?". The dual depends only on kernel evaluations.
- **Support vectors** are points exactly on (or violating) the margin; everything else has zero coefficient. Sparsity follows from the hinge loss.
- **Soft margin** + the hyperparameter \\(C\\) handle non-separable data; equivalent to L2-regularized hinge-loss minimization.
- **RVM** trades convexity for Bayesian elegance: probabilistic outputs, automatic hyperparameter selection, and even sparser solutions.

Forward to <a href="{{ '/assets/courses/introml/08_graphical_models/' | relative_url }}">Ch 8 — Graphical Models</a>.

---

<small><em>Figures from Bishop, <em>Pattern Recognition and Machine Learning</em> (Springer, 2006), © Springer / C. M. Bishop. Reproduced for educational use.</em></small>
