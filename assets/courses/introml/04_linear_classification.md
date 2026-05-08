---
layout: page
title: "Ch 4 — Linear Models for Classification"
description: Discriminant functions, generative vs discriminative models, logistic regression, Laplace approximation.
math: true
slides: true
back_link: '/teaching/introml'
back_text: 'Intro to ML'
---

In regression, the target \\(t\\) is continuous; in classification, it's a label from \\(K\\) discrete classes. Bishop organizes classifiers along a spectrum of how directly they go from inputs to labels:

| Approach | Models | Pros | Cons |
|---|---|---|---|
| **Discriminant function** | \\(f: \mathbf{x} \to k\\) directly | Simple, often fast | No probabilities — no calibrated confidence |
| **Discriminative probabilistic** | \\(p(\mathcal{C}\_k \mid \mathbf{x})\\) directly | Calibrated probabilities | Doesn't model the input distribution |
| **Generative** | \\(p(\mathbf{x} \mid \mathcal{C}\_k)\\) and \\(p(\mathcal{C}\_k)\\) | Lets you sample inputs, handle missing data | Hardest to learn; bad in high dimensions |

This chapter covers all three for *linear* decision boundaries.

## Prerequisites

- **Linear algebra:** projections, eigenvectors (for Fisher's LDA).
- **Probability:** Gaussians, the multivariate Gaussian's log-density (Ch 2).
- **Calculus:** the gradient and Hessian; Newton's method.

---

## 4.1 Discriminant Functions

A linear discriminant for two classes is

$$
y(\mathbf{x}) = \mathbf{w}^{\top}\mathbf{x} + w_0,
$$

with the decision rule "predict \\(\mathcal{C}\_1\\) if \\(y(\mathbf{x}) \geq 0\\), else \\(\mathcal{C}\_2\\)." The decision boundary \\(y = 0\\) is a hyperplane.

For \\(K > 2\\) classes, the **one-vs-rest** trick (one binary classifier per class) and the **one-vs-one** trick (\\(K(K-1)/2\\) pairwise classifiers) both leave ambiguous regions. The clean fix is to use \\(K\\) linear discriminants \\(y\_k(\mathbf{x}) = \mathbf{w}\_k^{\top}\mathbf{x} + w\_{k0}\\) and assign \\(\mathbf{x}\\) to the class with the largest \\(y\_k\\).

### Least squares for classification

If we encode classes as one-hot targets and run least-squares regression on each output, we get a closed-form classifier. It's mathematically tidy but **highly sensitive to outliers** — a few extreme points can rotate the decision boundary in a way that misclassifies large clusters. Don't use it as your real classifier; use it as a baseline.

### Fisher's linear discriminant

A better approach for two classes: project onto a 1-D direction \\(\mathbf{w}\\) chosen to maximize between-class variance and minimize within-class variance. Defining \\(\mathbf{S}\_W\\) (within-class scatter) and \\(\mathbf{S}\_B\\) (between-class scatter), Fisher's criterion is

$$
J(\mathbf{w}) = \frac{\mathbf{w}^{\top}\mathbf{S}_B \mathbf{w}}{\mathbf{w}^{\top}\mathbf{S}_W \mathbf{w}},
\qquad
\mathbf{w}^{\star} \propto \mathbf{S}_W^{-1}(\mathbf{m}_2 - \mathbf{m}_1),
\tag{4.1}
$$

a generalized-eigenvalue problem with a closed-form solution. For \\(K\\) classes, the same idea generalizes to projecting onto a \\((K-1)\\)-dim subspace.

### The perceptron

Rosenblatt's classical algorithm. For \\(t\_n \in \lbrace -1, +1\rbrace\\), define the per-example loss

$$
E_P(\mathbf{w}) = - \sum_{n \in \mathcal{M}} t_n\, \mathbf{w}^{\top}\boldsymbol{\phi}(\mathbf{x}_n),
$$

summed only over **misclassified** examples \\(\mathcal{M}\\). Stochastic gradient descent gives the update \\(\mathbf{w} \leftarrow \mathbf{w} + \eta\, t\_n \boldsymbol{\phi}(\mathbf{x}\_n)\\) on each misclassified point. The **perceptron convergence theorem** says: if the data are linearly separable, the algorithm finds a separating hyperplane in finite steps. Otherwise it doesn't terminate.

---

## 4.2 Probabilistic Generative Models

Model each class-conditional density and the prior, then apply Bayes:

$$
p(\mathcal{C}_k \mid \mathbf{x})
  = \frac{p(\mathbf{x} \mid \mathcal{C}_k)\, p(\mathcal{C}_k)}{\sum_j p(\mathbf{x} \mid \mathcal{C}_j)\, p(\mathcal{C}_j)}.
$$

For two classes the posterior simplifies to

$$
p(\mathcal{C}_1 \mid \mathbf{x}) = \sigma(a),
\qquad
a = \ln \frac{p(\mathbf{x} \mid \mathcal{C}_1) p(\mathcal{C}_1)}{p(\mathbf{x} \mid \mathcal{C}_2) p(\mathcal{C}_2)},
\qquad
\sigma(a) = \frac{1}{1 + e^{-a}}.
$$

The **logistic sigmoid** appears naturally — not as a hand-picked nonlinearity but as a consequence of Bayes' rule. For \\(K > 2\\), the sigmoid generalizes to the **softmax** \\(p(\mathcal{C}\_k \mid \mathbf{x}) = \exp(a\_k)/\sum\_j \exp(a\_j)\\).

### Gaussian class-conditionals

If each \\(p(\mathbf{x} \mid \mathcal{C}\_k)\\) is Gaussian with a *shared* covariance \\(\boldsymbol{\Sigma}\\), the log-ratio is **linear** in \\(\mathbf{x}\\) and the decision boundary is a hyperplane (this is **Linear Discriminant Analysis**, LDA). With *class-specific* covariances, the log-ratio becomes quadratic — **QDA**, with curved decision boundaries.

### Naive Bayes

If you assume the features of \\(\mathbf{x}\\) are conditionally independent given the class, \\(p(\mathbf{x} \mid \mathcal{C}\_k) = \prod\_d p(x\_d \mid \mathcal{C}\_k)\\), the model decouples per dimension. Astonishingly often a strong baseline despite the obviously wrong independence assumption.

---

## 4.3 Probabilistic Discriminative Models

Skip modeling \\(p(\mathbf{x})\\) — directly parameterize \\(p(\mathcal{C}\_k \mid \mathbf{x})\\).

### Logistic regression

For two classes,

$$
p(\mathcal{C}_1 \mid \boldsymbol{\phi}) = \sigma(\mathbf{w}^{\top}\boldsymbol{\phi}),
\qquad
\boldsymbol{\phi} = \boldsymbol{\phi}(\mathbf{x}).
$$

The negative log-likelihood (a.k.a. **cross-entropy** loss) is

$$
E(\mathbf{w}) = - \sum_{n=1}^{N} \lbrace t_n \ln y_n + (1 - t_n) \ln(1 - y_n) \rbrace,
\qquad y_n = \sigma(\mathbf{w}^{\top}\boldsymbol{\phi}_n).
$$

The gradient has the same form as in linear regression:

$$
\nabla E(\mathbf{w}) = \sum_{n=1}^{N} (y_n - t_n)\, \boldsymbol{\phi}_n = \boldsymbol{\Phi}^{\top}(\mathbf{y} - \mathbf{t}).
\tag{4.2}
$$

No closed form, but the loss is **convex**, so any local minimum is global. Solve via **iteratively reweighted least squares (IRLS)** — Newton's method specialized to logistic regression.

For \\(K > 2\\), softmax regression generalizes the same way and again has convex log-loss.

---

## 4.4 The Laplace Approximation and Bayesian Logistic Regression

Logistic regression's posterior \\(p(\mathbf{w} \mid \mathcal{D})\\) is **not** Gaussian — there's no conjugate prior. The **Laplace approximation** fits a Gaussian to the posterior at its mode:

$$
q(\mathbf{w}) = \mathcal{N}(\mathbf{w} \mid \mathbf{w}_{\mathrm{MAP}}, \mathbf{H}^{-1}),
$$

where \\(\mathbf{H} = -\nabla\nabla \ln p(\mathbf{w}\mid \mathcal{D})\\big|\_{\mathbf{w}=\mathbf{w}\_{\mathrm{MAP}}}\\) is the negative Hessian of the log-posterior. It captures local curvature; the predictive distribution then comes from integrating \\(q\\).

The Laplace approximation appears throughout PRML — Bayesian logistic regression here, neural network MAP later, and as a baseline against the variational methods of Ch 10.

---

## Takeaways

- The **logistic sigmoid** isn't arbitrary — it falls out of applying Bayes to two Gaussian class-conditionals with shared covariance.
- **Generative** vs **discriminative**: generative models are richer but harder to fit; discriminative models trade away the input-distribution model for sharper decisions on labels.
- **Logistic regression** is the workhorse — convex log-loss, IRLS solver, easy to extend with basis functions, a clean Bayesian story via Laplace.
- **LDA / Fisher's discriminant / Naive Bayes** remain useful as fast baselines and dimensionality-reduction tools.

Forward to <a href="{{ '/assets/courses/introml/05_neural_networks/' | relative_url }}">Ch 5 — Neural Networks</a>.

---

<small><em>Figures from Bishop, <em>Pattern Recognition and Machine Learning</em> (Springer, 2006), © Springer / C. M. Bishop. Reproduced for educational use.</em></small>
