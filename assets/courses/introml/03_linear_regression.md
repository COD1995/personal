---
layout: page
title: "Ch 3 — Linear Models for Regression"
description: Basis functions, the bias–variance decomposition, Bayesian linear regression, the evidence approximation.
math: true
slides: true
back_link: '/teaching/introml'
back_text: 'Intro to ML'
---

Chapter 1 introduced polynomial regression as a special case of fitting a function that's *linear in its parameters*. This chapter generalizes that idea: any function of the form

$$
y(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{M-1} w_j \phi_j(\mathbf{x}) = \mathbf{w}^{\top} \boldsymbol{\phi}(\mathbf{x})
$$

— where the \\(\phi\_j\\) are fixed nonlinear **basis functions** of the input — is a *linear model* and gets the same closed-form least-squares treatment we used before. The flexibility comes from choosing rich basis functions (polynomial, Gaussian, sigmoidal, splines, …); the tractability comes from the model's linearity in \\(\mathbf{w}\\).

## Prerequisites

- **Linear algebra:** matrix transpose/inverse, the pseudo-inverse, eigendecomposition.
- **Calculus:** matrix-valued gradients (\\(\nabla\_{\mathbf{w}} \\|\mathbf{y} - \mathbf{X}\mathbf{w}\\|^2\\) and the like). A short refresher: for any vector \\(\mathbf{v}\\) and matrix \\(\mathbf{A}\\), \\(\nabla\_{\mathbf{w}} (\mathbf{v}^{\top}\mathbf{w}) = \mathbf{v}\\) and \\(\nabla\_{\mathbf{w}} (\mathbf{w}^{\top}\mathbf{A}\mathbf{w}) = (\mathbf{A} + \mathbf{A}^{\top})\mathbf{w}\\).
- **Probability:** Gaussians (Ch 2), Bayes' rule (Ch 1).

---

## 3.1 Linear Basis Function Models

A few common basis-function choices:

- **Polynomial**: \\(\phi\_j(x) = x^j\\). Global support — every basis function is non-zero everywhere, so changing the data far from a region still shifts the fit there.
- **Gaussian RBF**: \\(\phi\_j(\mathbf{x}) = \exp\\!\left(-\\|\mathbf{x} - \boldsymbol{\mu}\_j\\|^{2} / (2 s^{2})\right)\\). Local support, smooth, easy to interpret.
- **Sigmoidal**: \\(\phi\_j(\mathbf{x}) = \sigma((\mathbf{x} - \mu\_j)/s)\\) with \\(\sigma(a) = 1/(1 + e^{-a})\\). Makes a "soft step" — useful for piecewise responses.

The key ingredient is the **design matrix** \\(\boldsymbol{\Phi}\\) whose entries are \\(\Phi\_{nj} = \phi\_j(\mathbf{x}\_n)\\). Once you've built it, every formula below is a one-liner in linear algebra.

### Maximum likelihood and least squares

Assume targets are noisy versions of the model output: \\(t = y(\mathbf{x}, \mathbf{w}) + \epsilon\\) with \\(\epsilon \sim \mathcal{N}(0, \beta^{-1})\\). The log-likelihood for an i.i.d. dataset becomes

$$
\ln p(\mathbf{t} \mid \mathbf{w}, \beta)
  = \frac{N}{2} \ln \beta - \frac{N}{2} \ln (2\pi) - \beta E_D(\mathbf{w}),
\qquad
E_D(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^{N} \lbrace t_n - \mathbf{w}^{\top}\boldsymbol{\phi}(\mathbf{x}_n)\rbrace^{2}.
$$

Maximizing in \\(\mathbf{w}\\) ↔ minimizing \\(E\_D\\). Setting \\(\nabla\_{\mathbf{w}} E\_D = 0\\) gives the **normal equations**

$$
\mathbf{w}_{\mathrm{ML}} = (\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^{\top}\mathbf{t}.
\tag{3.1}
$$

The matrix \\((\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^{\top}\\) is the **Moore–Penrose pseudo-inverse** of \\(\boldsymbol{\Phi}\\); when \\(\boldsymbol{\Phi}\\) is square and invertible, it reduces to \\(\boldsymbol{\Phi}^{-1}\\).

### Regularized least squares

Adding a quadratic penalty \\(\frac{\lambda}{2}\mathbf{w}^{\top}\mathbf{w}\\) gives

$$
\mathbf{w}_{\mathrm{ridge}} = (\lambda \mathbf{I} + \boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^{\top}\mathbf{t}.
\tag{3.2}
$$

The \\(\lambda \mathbf{I}\\) term also fixes numerical conditioning when \\(\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}\\) is nearly singular. More general penalties \\(\sum\_j |w\_j|^q\\) lead to **lasso** (\\(q=1\\), produces sparse \\(\mathbf{w}\\)) and others; only the quadratic case has a closed form.

---

## 3.2 The Bias–Variance Decomposition

How do you choose model complexity? Before defending a particular regularizer or model order, look at how prediction error decomposes.

For squared loss, the expected error of an estimator \\(y(\mathbf{x}; \mathcal{D})\\) trained on a dataset \\(\mathcal{D}\\), averaged over datasets, splits into

$$
\mathbb{E}_{\mathcal{D}}\bigl[(y(\mathbf{x};\mathcal{D}) - h(\mathbf{x}))^{2}\bigr]
  = \underbrace{(\mathbb{E}_{\mathcal{D}}[y] - h)^{2}}_{\text{bias}^{2}}
   + \underbrace{\mathbb{E}_{\mathcal{D}}[(y - \mathbb{E}_{\mathcal{D}}[y])^{2}]}_{\text{variance}}.
\tag{3.3}
$$

with \\(h(\mathbf{x}) = \mathbb{E}[t \mid \mathbf{x}]\\) the optimal regression function.

- **Bias** measures how far the *average* model is from the truth. Rigid models have high bias.
- **Variance** measures how much the model wiggles between datasets. Flexible models have high variance.

The two trade off: increasing model complexity *typically* lowers bias and raises variance. The total expected error is bias² + variance + irreducible noise. The minimum lies in the dip — neither too rigid nor too flexible.

<figure class="figure">
  <img src="{{ '/assets/img/prml/Figure3.5.png' | relative_url }}" alt="Bias-variance tradeoff" loading="lazy">
  <figcaption><strong>Fig 3.5.</strong> Bias² (blue), variance (red), and bias² + variance (purple) as a function of the regularization strength \(\ln \lambda\). The crossing point identifies the optimal trade-off.</figcaption>
</figure>

The decomposition is conceptually beautiful but practically limited — averaging over datasets is what you'd ideally know, but in practice you're stuck with the one dataset you have. Bayesian methods (next section) sidestep the issue by integrating over \\(\mathbf{w}\\) directly.

---

## 3.3 Bayesian Linear Regression

Replace the point estimate \\(\mathbf{w}\_{\mathrm{ML}}\\) with a full posterior \\(p(\mathbf{w} \mid \mathcal{D})\\). Putting a Gaussian prior \\(p(\mathbf{w}) = \mathcal{N}(\mathbf{w} \mid \mathbf{m}\_0, \mathbf{S}\_0)\\) and combining with the Gaussian likelihood gives a Gaussian posterior

$$
p(\mathbf{w} \mid \mathcal{D}) = \mathcal{N}(\mathbf{w} \mid \mathbf{m}_N, \mathbf{S}_N),
$$

with

$$
\mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + \beta \boldsymbol{\Phi}^{\top}\boldsymbol{\Phi},
\qquad
\mathbf{m}_N = \mathbf{S}_N (\mathbf{S}_0^{-1}\mathbf{m}_0 + \beta \boldsymbol{\Phi}^{\top}\mathbf{t}).
\tag{3.4}
$$

For a zero-mean isotropic prior \\(p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})\\), the posterior mean coincides with the **ridge** solution \\(\mathbf{w}\_{\mathrm{ridge}}\\) at \\(\lambda = \alpha/\beta\\) — same point estimate, plus an honest covariance.

### Predictive distribution

For a new input \\(\mathbf{x}\_\*\\), integrate over \\(\mathbf{w}\\):

$$
p(t_* \mid \mathbf{x}_*, \mathcal{D}) = \int p(t_* \mid \mathbf{x}_*, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{D})\, \mathrm{d}\mathbf{w}
  = \mathcal{N}(t_* \mid \mathbf{m}_N^{\top}\boldsymbol{\phi}(\mathbf{x}_*),\, \sigma_N^{2}(\mathbf{x}_*))
$$

with \\(\sigma\_N^{2}(\mathbf{x}\_\*) = 1/\beta + \boldsymbol{\phi}(\mathbf{x}\_\*)^{\top}\mathbf{S}\_N \boldsymbol{\phi}(\mathbf{x}\_\*)\\). The first term is irreducible noise; the second grows as \\(\mathbf{x}\_\*\\) moves away from training points — so the model *knows* when it's extrapolating.

<figure class="figure">
  <img src="{{ '/assets/img/prml/Figure3.8.png' | relative_url }}" alt="Bayesian predictive distribution" loading="lazy">
  <figcaption><strong>Fig 3.8.</strong> Predictive distribution for sinusoidal data with Gaussian basis functions. The shaded band is one std-dev under the posterior — narrow near training points, wide elsewhere.</figcaption>
</figure>

---

## 3.4 The Evidence Approximation

Instead of fixing hyperparameters \\((\alpha, \beta)\\) by hand, treat them as additional parameters and pick the values that maximize the **marginal likelihood** (a.k.a. **evidence**)

$$
p(\mathbf{t} \mid \alpha, \beta) = \int p(\mathbf{t} \mid \mathbf{w}, \beta)\, p(\mathbf{w} \mid \alpha)\, \mathrm{d}\mathbf{w}.
$$

For our Gaussian setup this integral is closed-form. Maximizing its log over \\(\alpha\\) and \\(\beta\\) (via fixed-point iterations) gives

$$
\alpha = \frac{\gamma}{\mathbf{m}_N^{\top}\mathbf{m}_N},
\qquad
\frac{1}{\beta} = \frac{1}{N - \gamma} \sum_{n} (t_n - \mathbf{m}_N^{\top}\boldsymbol{\phi}(\mathbf{x}_n))^{2},
$$

where \\(\gamma\\) is the **effective number of parameters** — the number of weights that have been "pulled" by the data away from the prior. The evidence framework automatically penalizes complexity: a model that uses too many degrees of freedom for too little data has lower evidence.

This is the workhorse Bayesian model-selection criterion for the rest of PRML.

---

## Takeaways

- **Linear-in-parameters** ≠ linear function. Polynomial, Gaussian-RBF, and sigmoidal bases all give closed-form fits.
- **Sum-of-squares** = MLE under Gaussian noise. **Ridge** = MAP under Gaussian prior. **Lasso** = MAP under Laplace prior (no closed form).
- **Bias–variance** is the conceptual lens; **Bayesian linear regression** is the operational tool that integrates over \\(\mathbf{w}\\).
- The **evidence framework** picks hyperparameters by maximizing marginal likelihood — Bayesian Occam's razor.

Forward to <a href="{{ '/assets/courses/introml/04_linear_classification/' | relative_url }}">Ch 4 — Linear Models for Classification</a>.

---

<small><em>Figures from Bishop, <em>Pattern Recognition and Machine Learning</em> (Springer, 2006), © Springer / C. M. Bishop. Reproduced for educational use.</em></small>
