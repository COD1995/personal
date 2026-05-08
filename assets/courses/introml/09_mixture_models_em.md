---
layout: page
title: "Ch 9 — Mixture Models &amp; EM"
description: K-means clustering, Gaussian mixture models, the Expectation–Maximization algorithm.
math: true
slides: true
back_link: '/teaching/introml'
back_text: 'Intro to ML'
---

A single Gaussian fits unimodal data; many real datasets have multiple clusters. **Mixture models** describe data as drawn from one of several component distributions, with component identity an unobserved latent variable. The **Expectation–Maximization (EM) algorithm** fits them by alternating between (E) inferring soft assignments and (M) updating component parameters — provably non-decreasing in the log-likelihood at every step.

## Prerequisites

- **Gaussians** (Ch 2) — multivariate density, MLEs.
- **Maximum likelihood**, log-likelihood, Jensen's inequality.
- **Latent-variable thinking** — comfortable with marginalizing over unobserved variables.

---

## 9.1 K-Means Clustering

The simplest clustering algorithm. Given data \\(\lbrace \mathbf{x}\_n\rbrace\\) and a fixed number \\(K\\) of clusters, partition the data so each point belongs to one cluster, with cluster mean \\(\boldsymbol{\mu}\_k\\). Minimize

$$
J = \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk}\, \\|\mathbf{x}_n - \boldsymbol{\mu}_k\\|^{2},
\qquad r_{nk} \in \lbrace 0, 1\rbrace,\;\sum_k r_{nk} = 1.
\tag{9.1}
$$

Alternate between:

- **Assignment step.** With \\(\boldsymbol{\mu}\\) fixed, set \\(r\_{nk} = 1\\) for the closest mean, else 0.
- **Update step.** With \\(r\\) fixed, \\(\boldsymbol{\mu}\_k = \frac{\sum\_n r\_{nk} \mathbf{x}\_n}{\sum\_n r\_{nk}}\\).

Each step decreases (never increases) \\(J\\), so the algorithm converges in finite steps to a local minimum. Sensitive to initialization; common practice is to run several times with different seeds and pick the best, or use **k-means++** seeding.

K-means is a **hard** clustering: each point is fully assigned to one cluster. The natural softening is the GMM.

---

## 9.2 Gaussian Mixture Models

A GMM is

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),
\qquad \pi_k \geq 0,\; \sum_k \pi_k = 1.
\tag{9.2}
$$

Equivalently, introduce a one-hot latent \\(\mathbf{z}\\) with \\(p(z\_k = 1) = \pi\_k\\), and \\(p(\mathbf{x} \mid z\_k = 1) = \mathcal{N}(\boldsymbol{\mu}\_k, \boldsymbol{\Sigma}\_k)\\). Then

$$
p(\mathbf{x}) = \sum_{\mathbf{z}} p(\mathbf{z})\, p(\mathbf{x} \mid \mathbf{z}).
$$

The latent-variable formulation is what makes the GMM amenable to EM.

### Maximum likelihood is harder than it looks

The log-likelihood

$$
\ln p(\mathcal{D} \mid \boldsymbol{\theta}) = \sum_{n=1}^{N} \ln \\!\left( \sum_{k=1}^{K} \pi_k\, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
$$

has a sum *inside* the log, blocking the closed-form trick that works for a single Gaussian. Worse, the likelihood is **unbounded above** — drive a single component's covariance to zero centered on a data point and the likelihood diverges. Practical fixes: regularize the covariance, restart on degeneracy, or use a Bayesian prior.

---

## 9.3 EM for the Gaussian Mixture

Define the **responsibilities**

$$
\gamma(z_{nk}) = p(z_k = 1 \mid \mathbf{x}_n) = \frac{\pi_k\, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j\, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}.
\tag{9.3}
$$

Then alternate:

- **E-step.** Compute \\(\gamma(z\_{nk})\\) for every \\(n, k\\) using current parameters.
- **M-step.** Given the responsibilities, update parameters via weighted MLE:

$$
\begin{aligned}
\boldsymbol{\mu}_k &= \frac{1}{N_k} \sum_n \gamma(z_{nk})\, \mathbf{x}_n, \\
\boldsymbol{\Sigma}_k &= \frac{1}{N_k} \sum_n \gamma(z_{nk})\, (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^{\top}, \\
\pi_k &= \frac{N_k}{N}, \qquad N_k = \sum_n \gamma(z_{nk}).
\end{aligned}
$$

Each step monotonically increases the log-likelihood; iterate to convergence. The result is a *local* maximum.

K-means is the limit of GMM-EM as all covariances shrink to zero — responsibilities become hard 0/1 assignments.

<figure class="figure">
  <img src="{{ '/assets/img/prml/Figure9.1.png' | relative_url }}" alt="GMM clustering iteration" loading="lazy">
  <figcaption><strong>Fig 9.1.</strong> EM on the Old Faithful dataset. Top row: initial state, after several E/M iterations, and at convergence. The mixture components grow to cover the two natural clusters.</figcaption>
</figure>

---

## 9.4 EM in General

The recipe extends to any latent-variable model. Want to maximize \\(\ln p(\mathbf{X} \mid \boldsymbol{\theta})\\), where \\(\mathbf{Z}\\) is hidden. Key identity (Jensen's inequality):

$$
\ln p(\mathbf{X} \mid \boldsymbol{\theta})
  = \mathcal{L}(q, \boldsymbol{\theta}) + \mathrm{KL}(q \\,\|\, p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})),
$$

with

$$
\mathcal{L}(q, \boldsymbol{\theta}) = \sum_{\mathbf{Z}} q(\mathbf{Z}) \ln \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})}.
$$

KL is non-negative, so \\(\mathcal{L}\\) is a **lower bound** on the log-likelihood. EM alternates:

- **E-step:** maximize \\(\mathcal{L}\\) over \\(q\\) with \\(\boldsymbol{\theta}\\) fixed → \\(q^{\star}(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})\\), making KL \\(= 0\\).
- **M-step:** maximize \\(\mathcal{L}\\) over \\(\boldsymbol{\theta}\\) with \\(q\\) fixed.

After every full iteration, \\(\ln p\\) has not decreased. The lower-bound view explains why EM converges and points the way to **variational EM** when the true posterior \\(p(\mathbf{Z} \mid \mathbf{X})\\) is intractable: just constrain \\(q\\) to a tractable family (Ch 10).

### Examples beyond GMMs

- **Hidden Markov models** — EM = Baum–Welch.
- **Probabilistic PCA** — closed-form M-steps.
- **Latent Dirichlet Allocation** — mean-field variational EM.
- **k-means** — a degenerate EM with hard assignments and identity covariance.

---

## 9.5 Bayesian Mixture Models

Putting priors on \\(\pi, \boldsymbol{\mu}, \boldsymbol{\Sigma}\\) gives a **Bayesian GMM**. The Dirichlet prior on \\(\pi\\) (with concentration \\(\alpha\\)) lets the model effectively *prune* unused components — set \\(\alpha\\) small and EM-like fitting will drive most \\(\pi\_k\\) to zero, choosing \\(K\\) automatically. Variational inference (Ch 10) is the standard tool here, since the posterior is non-conjugate.

The fully nonparametric extension is the **Dirichlet process mixture**, which lets \\(K\\) grow with the data. Outside PRML's main scope but a natural endpoint of this thread.

---

## Takeaways

- **K-means** is hard-assignment clustering with \\(O(N K T)\\) cost; great as a fast baseline or initializer for GMMs.
- **GMMs** model data as a weighted sum of Gaussians; latent identities are inferred via responsibilities.
- **EM** is the workhorse algorithm: alternate between inferring latent posteriors (E) and updating parameters (M); guaranteed monotonic in log-likelihood.
- The general EM derivation via the **lower bound \\(\mathcal{L}\\)** is the bridge to variational inference (Ch 10) — same machinery, weakened E-step.

Forward to <a href="{{ '/assets/courses/introml/10_approximate_inference/' | relative_url }}">Ch 10 — Approximate Inference</a>.

---

<small><em>Figures from Bishop, <em>Pattern Recognition and Machine Learning</em> (Springer, 2006), © Springer / C. M. Bishop. Reproduced for educational use.</em></small>
