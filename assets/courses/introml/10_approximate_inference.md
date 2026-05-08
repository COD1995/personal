---
layout: page
title: "Ch 10 — Approximate Inference"
description: Variational inference, mean-field factorization, expectation propagation, the variational view of Bayesian learning.
math: true
slides: true
back_link: '/teaching/introml'
back_text: 'Intro to ML'
---

For most non-trivial Bayesian models, the posterior \\(p(\mathbf{Z} \mid \mathbf{X})\\) has no closed form. Two attack vectors: **deterministic approximation** (this chapter — variational inference, EP) and **stochastic approximation** (Ch 11 — sampling). Variational methods replace the intractable posterior with a tractable surrogate and minimize the gap; sampling methods generate empirical samples and reason about them statistically. Each has its sweet spot.

## Prerequisites

- **EM** (Ch 9) — variational inference is EM with a constrained E-step.
- **Information theory** (Ch 1) — KL divergence, entropy.
- **Calculus of variations** — comfortable optimizing over functions, not just numbers.

---

## 10.1 The Variational Lower Bound

Recall (Ch 9): for any distribution \\(q(\mathbf{Z})\\),

$$
\ln p(\mathbf{X}) = \mathcal{L}(q) + \mathrm{KL}(q \\,\|\, p(\mathbf{Z} \mid \mathbf{X})),
\qquad
\mathcal{L}(q) = \int q(\mathbf{Z}) \ln \frac{p(\mathbf{X}, \mathbf{Z})}{q(\mathbf{Z})}\, \mathrm{d}\mathbf{Z}.
\tag{10.1}
$$

\\(\mathcal{L}\\) is the **evidence lower bound (ELBO)**. Minimizing KL is equivalent to maximizing \\(\mathcal{L}\\) — but minimizing KL of an *intractable* posterior would still leave us stuck. The variational approach **restricts \\(q\\)** to a tractable family and maximizes \\(\mathcal{L}\\) within that family. The result is a tractable approximation to the true posterior; the tightness of the gap is exactly the residual KL.

---

## 10.2 Mean-Field Variational Inference

The classical restriction: \\(q\\) factorizes over disjoint groups of latent variables,

$$
q(\mathbf{Z}) = \prod_{i=1}^{M} q_i(\mathbf{Z}_i).
\tag{10.2}
$$

This is **mean field**. Maximizing \\(\mathcal{L}\\) under this constraint, holding \\(q\_{j \neq i}\\) fixed, gives the optimal \\(q\_i\\):

$$
\ln q_i^{\star}(\mathbf{Z}_i) = \mathbb{E}_{j \neq i}\bigl[\ln p(\mathbf{X}, \mathbf{Z})\bigr] + \mathrm{const}.
\tag{10.3}
$$

i.e., take the log-joint, average it over all *other* factors, and exponentiate. Iterate over \\(i\\) until convergence — same alternating-optimization structure as EM, just with each E-step approximating the true conditional posterior by its expectation under the rest of \\(q\\).

### Example — Variational Gaussian Mixture

Take the Bayesian GMM from Ch 9 (Dirichlet prior on \\(\pi\\), Gaussian-Wishart on \\((\boldsymbol{\mu}\_k, \boldsymbol{\Lambda}\_k)\\)). Mean-field with the factorization \\(q(\mathbf{Z})\, q(\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Lambda})\\) gives closed-form updates for both factors. The result automatically prunes unused components (their posterior \\(q(\pi\_k)\\) collapses around 0), so you can set \\(K\\) generously and let the model decide. Hugely useful in practice.

### Why KL(q||p) and not KL(p||q)?

Mean-field minimizes \\(\mathrm{KL}(q \\| p)\\), which has a **mode-seeking** character: \\(q\\) tends to cover the *largest* mode of \\(p\\) and ignore others. The reverse, \\(\mathrm{KL}(p \\| q)\\), is **mode-covering** — \\(q\\) spreads to cover all modes, smearing density across them. Mode-seeking is computationally convenient (gradients of \\(\mathrm{KL}(q\\|p)\\) involve only expectations under \\(q\\)) and usually gives reasonable point estimates, but it under-estimates posterior variance. **Expectation propagation** below uses \\(\mathrm{KL}(p\\|q)\\) and trades convenience for fidelity.

<figure class="figure">
  <img src="{{ '/assets/img/prml/Figure10.3.png' | relative_url }}" alt="Mean-field approximation" loading="lazy">
  <figcaption><strong>Fig 10.3.</strong> Mean-field approximation (red) to a 2-D Gaussian (green) with strong correlation. The factorized \(q\) recovers the modes correctly but underestimates the variance along the correlated direction.</figcaption>
</figure>

---

## 10.3 Variational Logistic Regression

Logistic regression's posterior over \\(\mathbf{w}\\) is non-Gaussian (no conjugate). In Ch 4 we used the Laplace approximation; the variational alternative bounds the logistic sigmoid below by a Gaussian-shaped lower bound (the **Jaakkola–Jordan bound**):

$$
\sigma(z) \geq \sigma(\xi)\, \exp\\!\left( \tfrac{z - \xi}{2} - \lambda(\xi)(z^{2} - \xi^{2}) \right),
$$

with \\(\lambda(\xi) = \tfrac{1}{2\xi}(\sigma(\xi) - \tfrac{1}{2})\\) and a free variational parameter \\(\xi\\). Plugging in turns the integral over \\(\mathbf{w}\\) into a Gaussian integral with closed form. Adjust \\(\xi\\) per data point to tighten the bound; alternate with parameter updates. Yields a Gaussian \\(q(\mathbf{w})\\) and a tighter approximation than Laplace.

---

## 10.4 Expectation Propagation

A different decomposition. Suppose the posterior factorizes as a product over data points:

$$
p(\boldsymbol{\theta}) \propto p_0(\boldsymbol{\theta}) \prod_{n=1}^{N} f_n(\boldsymbol{\theta}).
$$

EP approximates each factor \\(f\_n\\) by a member of an exponential family \\(\widetilde{f}\_n\\), giving a tractable global approximation \\(\prod \widetilde{f}\_n\\). Updates are local: pick a factor, remove its current approximation, replace with the projection of \\(p\_0 \widetilde{f}\_{\setminus n} f\_n\\) onto the chosen family by minimizing \\(\mathrm{KL}(p \\| \widetilde{p})\\) (the *reverse* KL from VI).

Why bother? EP minimizes mode-covering KL, so it captures the posterior's mass more faithfully — particularly important for predictive uncertainty. Trade-offs: no convergence guarantees in general; updates can oscillate. Used heavily in Gaussian-process classification.

---

## 10.5 Variational vs Sampling

| | Variational | Sampling (Ch 11) |
|---|---|---|
| Output | Parametric \\(q^{\star}\\) | Empirical samples \\(\lbrace \mathbf{z}^{(t)}\rbrace\\) |
| Bias | Approximation gap (positive) | Asymptotically zero |
| Variance | Deterministic | Monte Carlo noise |
| Speed | Fast — gradient-based | Slow — many samples to converge |
| Scalability to large data | Mini-batch friendly (SVI, BBVI) | Per-step cost grows with data |
| Practical win | Posterior approximations for huge models | Gold standard when feasible |

**Mod**ern rule of thumb: variational for scale, sampling for accuracy. For deep generative models (VAEs, normalizing flows) variational is essentially the only option.

---

## 10.6 Modern Variational Inference

PRML's book-era VI was per-model derivations. Today:

- **Black-box VI (BBVI).** Define an arbitrary parameterized \\(q\_{\boldsymbol{\phi}}\\); estimate gradients of \\(\mathcal{L}\\) by Monte Carlo sampling from \\(q\\). Works for any model with a tractable joint.
- **Reparameterization trick.** When \\(q\_{\boldsymbol{\phi}}\\) is reparameterizable (e.g., Gaussian), \\(\nabla\_{\boldsymbol{\phi}} \mathbb{E}\_{q}[f(\mathbf{Z})]\\) becomes a low-variance gradient. Foundation of the variational autoencoder.
- **Stochastic VI.** For huge datasets, replace the full sum in \\(\mathcal{L}\\) with mini-batch estimates; converges much faster than full-batch updates.
- **Normalizing flows.** Build expressive \\(q\\) by composing invertible transformations of a base distribution. Sidesteps the mean-field underestimation problem.

---

## Takeaways

- **ELBO** \\(\mathcal{L}(q)\\) is the central object: \\(\ln p(\mathbf{X}) = \mathcal{L} + \mathrm{KL}\\); maximizing \\(\mathcal{L}\\) over a tractable family minimizes the gap.
- **Mean-field VI** factorizes \\(q\\) over latent groups — closed-form coordinate updates, mode-seeking, underestimates variance.
- **EP** approximates factor by factor with mode-covering KL — better posterior mass, no convergence guarantees.
- **Modern BBVI** + reparameterization makes variational inference essentially black-box; the engine behind VAEs and large-scale Bayesian neural networks.

Forward to <a href="{{ '/assets/courses/introml/11_sampling_methods/' | relative_url }}">Ch 11 — Sampling Methods</a>.

---

<small><em>Figures from Bishop, <em>Pattern Recognition and Machine Learning</em> (Springer, 2006), © Springer / C. M. Bishop. Reproduced for educational use.</em></small>
