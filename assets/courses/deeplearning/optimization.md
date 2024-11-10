---
layout: page
title: Optimization
description: 
related_publications: false
toc:
    sidebar: left
---
**Author**: Jue Guo

## Constraints

One of the nice properties of convex optimization is that it allows us to handle constraints efficiently. That is, it allows us to solve *constrained optimization* problems of the form:

$$
\begin{array}{l}
\underset{\mathbf{x}}{\operatorname{minimize}} f(\mathbf{x}) \\
\text { subject to } c_{i}(\mathbf{x}) \leq 0 \text { for all } i \in\{1, \ldots, n\}
\end{array}
$$

where $$f$$ is the objective and the functions $$c_{i}$$ are constraint functions.

**Question**: Imagine a unit ball $$c_{1}(\mathbf{x})=\|\mathbf{x}\|_{2}-1$$; Now we have a second constraint: $$c_{2}(\mathbf{x})=\mathbf{v}^{\top} \mathbf{x}+b$$. What does this visually represent?

### Lagrangian

A ball inside a box; the ball will roll to the place that is lowest, and the forces of gravity will be balanced out with the forces that the sides of the box can impose on the ball.

- The gradient of the objective function (gravity) will be offset by the gradient of the constraint function (the ball needs to remain inside the box by virtue of the walls "pushing back").

The above reasoning can be expressed via the following saddle point optimization problem:

$$
L\left(\mathbf{x}, \alpha_{1}, \ldots, \alpha_{n}\right) = f(\mathbf{x}) + \sum_{i=1}^{n} \alpha_{i} c_{i}(\mathbf{x}) \text{ where } \alpha_{i} \geq 0
$$

- The variables $$\alpha_{i}(i=1, \ldots, n)$$ are *Lagrange multipliers* that ensure that constraints are properly enforced.

**What are Lagrange Multipliers $$\alpha_{i}$$?**  
If the constraint $$c_{i}(x) \leq 0$$ is active (i.e., if $$c_{i}(x) = 0$$), the corresponding multiplier $$\alpha_{i}$$ can take any positive value. If the constraint $$c_{i}(x) < 0$$ is inactive (i.e., not binding or strictly less than zero), then the multiplier $$\alpha_{i}$$ is set to zero. *Question? Can you think of an example to demonstrate the activation of the constraint function?*

**Example**  
Suppose you have a function $$f(x) = x^{2}$$, and you want to minimize it, but with the constraint that $$x \geq 1$$ (meaning $$c(x) = 1 - x \leq 0$$).

- If you ignore the constraint, the minimum would be at $$x = 0$$, but this violates the constraint $$x \geq 1$$.
- The Lagrangian would add a term $$\alpha \cdot (1 - x)$$, where $$\alpha \geq 0$$.
- The solution to the Lagrangian will find $$x = 1$$ (the smallest value of $$x$$ that satisfies the constraint $$x \geq 1$$) and a corresponding value of $$\alpha$$ that ensures the constraint is properly enforced.

### Penalties

Rather than satisfying $$c_{i}(\mathrm{x}) \leq 0$$, we simply add $$\alpha_{i} c_{i}(\mathbf{x})$$ to the objective function $$f(x)$$. This ensures that the constraints will not be violated too badly. It is a common trick: we add $$\frac{\lambda}{2}\|\mathbf{w}\|^{2}$$ to the objective function to ensure that $$\mathbf{w}$$ does not grow too large. We can see this will ensure $$\|\mathbf{w}\|^{2} - r^{2} \leq 0$$ for some radius $$r$$.

In general, adding penalties is a good way of ensuring approximate constraint satisfaction. In practice, this turns out to be much more robust than exact satisfaction. Furthermore, for nonconvex problems, many of the properties that make the exact approach so appealing in the convex case (e.g., optimality) no longer hold.

## Gradient Descent

Consider some continuously differentiable real-valued function $$f: \mathbb{R} \rightarrow \mathbb{R}$$. Using a Taylor expansion, we obtain:

$$
f(x + \epsilon) = f(x) + \epsilon f^{\prime}(x) + O\left(\epsilon^{2}\right).
$$

It is not unreasonable to assume that for small $$\epsilon$$, moving in the direction of the negative gradient will decrease $$f$$. Under this assumption, we pick a fixed step size $$\eta > 0$$ and choose $$\epsilon = -\eta f^{\prime}(x)$$:

$$
f\left(x - \eta f^{\prime}(x)\right) = f(x) - \eta f^{\prime 2}(x) + O\left(\eta^{2} f^{\prime 2}(x)\right)
$$

**Observations**  
1. If the derivative $$f^{\prime}(x) \neq 0$$ does not vanish, we make progress since $$\eta f^{\prime 2}(x) > 0$$.  
2. A small enough $$\eta$$ will make the higher-order terms become irrelevant.

$$
f\left(x - \eta f^{\prime}(x)\right) \lesssim f(x).
$$

This means if we use $$x \leftarrow x - \eta f^{\prime}(x)$$ to iterate $$x$$, the value of the function $$f(x)$$ might decline.

### Multivariate Gradient Descent

Let $$\mathbf{x} = [x_{1}, x_{2}, \ldots, x_{d}]^{\top}$$; the objective function $$f: \mathbb{R}^{d} \rightarrow \mathbb{R}$$ maps vectors into scalars.

$$
\nabla f(\mathbf{x}) = \left[\frac{\partial f(\mathbf{x})}{\partial x_{1}}, \frac{\partial f(\mathbf{x})}{\partial x_{2}}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_{d}}\right]^{\top}
$$

Each partial derivative element $$\partial f(\mathbf{x}) / \partial x_{i}$$ in the gradient indicates the rate of change of $$f$$ at $$\mathbf{x}$$ with respect to the input $$x_i$$. Following the same idea from the one-dimensional case, we have:

$$
f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^{\top} \nabla f(\mathbf{x}) + O\left(\|\boldsymbol{\epsilon}\|^{2}\right)
$$

Choosing a suitable learning rate $$\eta > 0$$ yields the prototypical gradient descent algorithm:

$$
\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})
$$

### Adaptive Methods

Picking a suitable learning rate $$\eta$$ is tricky; *Too small, we learn too slow; Too large, the solution oscillates, and in the worst case, it diverges.* Can we determine $$\eta$$ automatically or eliminate the selection process? *Second-order methods* that consider not only the value and gradient of the objective function but also its *curvature* can help.

Please note that while these methods cannot be applied to deep learning directly due to the computational cost, they provide useful intuition into designing advanced optimization algorithms that mimic many of the desirable properties of these algorithms.

### Newton's Method

Reviewing the Taylor expansion of some function $$f: \mathbb{R}^{d} \rightarrow \mathbb{R}$$ shows that there is no need to stop after the first term.

$$
f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^{\top} \nabla f(\mathbf{x}) + \frac{1}{2} \epsilon^{\top} \nabla^{2} f(\mathbf{x}) \boldsymbol{\epsilon} + O\left(\|\boldsymbol{\epsilon}\|^{3}\right).
$$

To avoid cumbersome notation, we define $$\mathbf{H} \stackrel{\text { def }}{=} \nabla^{2} f(\mathbf{x})$$ to be the Hessian of $$f$$, which is a $$d \times d$$ matrix.

**Problems with this dimensionality?** For small $$d$$ and simple problems, $$\mathbf{H}$$ is easy to compute. For deep neural networks, on the other hand, $$\mathbf{H}$$ may be prohibitively large due to the cost of storing $$\mathcal{O}(d^2)$$ entries. Furthermore, it may be too expensive to compute via backpropagation.

Now we have:

$$
f(\mathbf{x} + \epsilon) = f(\mathbf{x}) + \epsilon^{T} \nabla f(\mathbf{x}) + \frac{1}{2} \epsilon^{T} \mathbf{H} \epsilon + O\left(\|\epsilon\|^{3}\right).
$$

**Objective: Find $$\epsilon$$**  
We want to find the value of $$\epsilon$$ that minimizes the function $$f(\mathbf{x} + \epsilon)$$. This is a standard second-order optimization technique where we use the gradient and the Hessian to adjust the value of $$\mathbf{x}$$ for a better approximation (closer to a minimum). Therefore:

$$
\frac{\partial}{\partial \epsilon}\left(f(\mathbf{x}) + \epsilon^{T} \nabla f(\mathbf{x}) + \frac{1}{2} \epsilon^{T} \mathbf{H} \epsilon\right).
$$

Taking the derivative and setting it to zero, we have:

$$
\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \text{ and hence } \boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).
$$

**Why are we going through all this trouble?**  
Let's say $$f(x) = \frac{1}{2} x^{2}$$, $$\nabla f(x) = \frac{d}{d x}\left(\frac{1}{2} x^{2}\right) = x$$, and $$\frac{d^{2}}{d x^{2}}\left(\frac{1}{2} x^{2}\right) = 1$$. Following Newton's method, the update step is $$\epsilon = -x$$.

### Convergence Analysis

Let's apply convergence analysis on Newton's method, specifically how quickly the error $$e^{(k)} = x^{(k)} - x^{*}$$ (the difference between the current iterate $$x^{(k)}$$ and the optimal solution $$x^{*}$$) decreases from one iteration to the next. Quick recap of the key concept:

- $$f\left(x^{*}\right)$$ is minimized when $$f^{\prime}\left(x^{*}\right) = 0$$.
- Newton's method updates the current point $$x^{(k)}$$ using the rule:

$$
x^{(k+1)} = x^{(k)} - \frac{f^{\prime}\left(x^{(k)}\right)}{f^{\prime \prime}\left(x^{(k)}\right)}
$$

The error at iteration $$k, e^{(k)} = x^{(k)} - x^{*}$$, is used in a **Taylor expansion** around the optimal point $$x^{*}$$ to analyze how the error behaves. We expand the derivative $$f^{\prime}\left(x^{*}\right) = 0$$ around $$x^{(k)}$$ as follows:

$$
0 = f^{\prime}\left(x^{*}\right) = f^{\prime}\left(x^{(k)}\right) - e^{(k)} f^{\prime \prime}\left(x^{(k)}\right) + \frac{1}{2}\left(e^{(k)}\right)^{2} f^{(3)}\left(\xi^{(k)}\right)
$$

This equation gives us an approximation of the derivative $$f^{\prime}\left(x^{*}\right)$$ using the second and third derivatives of $$f(x)$$ evaluated at points near $$x^{(k)}$$.

- $$e^{(k)} f^{\prime \prime}\left(x^{(k)}\right)$$: first-order correction using the second derivative.
- $$\frac{1}{2}\left(e^{(k)}\right)^{2} f^{(3)}\left(\xi^{(k)}\right)$$: second-order correction using the third derivative.

Here, $$\xi^{(k)}$$ is some point between $$x^{(k)}$$ and $$x^{*}$$ according to the Mean Value Theorem. To make the equation more tractable, we divide the Taylor expansion by $$f^{\prime \prime}\left(x^{(k)}\right)$$:

$$
e^{(k)} - \frac{f^{\prime}\left(x^{(k)}\right)}{f^{\prime \prime}\left(x^{(k)}\right)} = \frac{1}{2}\left(e^{(k)}\right)^{2} \frac{f^{(3)}\left(\xi^{(k)}\right)}{f^{\prime \prime}\left(x^{(k)}\right)}
$$

This equation tells us how far the next iterate $$x^{(k+1)}$$ is from the optimal solution $$x^{*}$$ after one Newton step. Now we plug in the update rule from Newton's method:

$$
x^{(k+1)} = x^{(k)} - \frac{f^{\prime}\left(x^{(k)}\right)}{f^{\prime \prime}\left(x^{(k)}\right)}
$$

The error $$e^{(k+1)}$$ at the next iteration is:

$$
e^{(k+1)} = x^{(k+1)} - x^{*} = x^{(k)} - \frac{f^{\prime}\left(x^{(k)}\right)}{f^{\prime \prime}\left(x^{(k)}\right)} - x^{*} = e^{(k)} - \frac{f^{\prime}\left(x^{(k)}\right)}{f^{\prime \prime}\left(x^{(k)}\right)}
$$

Recall the expanded form of $$f^{\prime}\left(x^{*}\right) = 0$$:

$$
f^{\prime}\left(x^{(k)}\right) = e^{(k)} f^{\prime \prime}\left(x^{(k)}\right) - \frac{1}{2}\left(e^{(k)}\right)^{2} f^{(3)}\left(\xi^{(k)}\right)
$$

Substitution:

$$
e^{(k+1)} = -\frac{e^{(k)} f^{\prime \prime}\left(x^{(k)}\right) - \frac{1}{2}\left(e^{(k)}\right)^{2} f^{(3)}\left(\xi^{(k)}\right)}{f^{\prime \prime}\left(x^{(k)}\right)}
$$

Simplification:

$$
e^{(k+1)} = -\left(e^{(k)} - \frac{1}{2}\left(e^{(k)}\right)^{2} \frac{f^{(3)}\left(\xi^{(k)}\right)}{f^{\prime \prime}\left(x^{(k)}\right)}\right)
$$

We can ignore $$e^{(k)}$$; why? **The error at iteration $$k+1$$ is mostly driven by the quadratic correction**:

1. At convergence, the linear term $$e^{(k)}$$ drives the approximation initially.
2. As the error gets small, Newton's method updates the solution in such a way that the remaining error becomes dominated by the quadratic term.

In other words, even though the quadratic term $$\left(e^{(k)}\right)^{2}$$ is smaller in magnitude, it determines the next iteration's error because the method's convergence accelerates rapidly near the optimum. Therefore:

$$
e^{(k+1)} = \frac{1}{2}\left(e^{(k)}\right)^{2} \frac{f^{(3)}\left(\xi^{(k)}\right)}{f^{\prime \prime}\left(x^{(k)}\right)}
$$

We want to analyze the magnitude regardless of the sign, therefore taking the absolute value of both sides, we get:

$$
\left|e^{(k+1)}\right| = \frac{1}{2}\left(e^{(k)}\right)^{2} \frac{\left|f^{(3)}\left(\xi^{(k)}\right)\right|}{f^{\prime \prime}\left(x^{(k)}\right)}
$$

With a bounded condition $$
\left|f^{\prime \prime \prime}\left(\xi^{(k)}\right)\right| /\left(2 f^{\prime \prime}\left(x^{(k)}\right)\right) \leq c
$$, we have:

$$
\left|e^{(k+1)}\right| \leq c\left(e^{(k)}\right)^{2}
$$

### Preconditioning

Preconditioning provides a cheaper alternative to using the full Hessian matrix. Instead of computing and storing the entire Hessian, preconditioning only uses the diagonal entries of the Hessian matrix. This dramatically simplifies the process while still providing useful information about the curvature of the function.

The update equation in this preconditioned version is:

$$
\mathbf{x} \leftarrow \mathbf{x} - \eta \operatorname{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x})
$$

where:

- $$\operatorname{diag}(\mathbf{H})^{-1}$$ is the inverse of this diagonal matrix.

**Real-World Analogy: Units Mismatch**  
Imagine you are optimizing a function where one variable is *height in millimeters* and another variable is *height in kilometers*.

If you try to use the same learning rate (step size) for both variables, it will create problems because the scales of the two variables are wildly different. This mismatch in units (millimeters vs. kilometers) will lead to slow convergence because one variable might need very small updates, while the other requires larger updates.

Preconditioning solves this problem by effectively allowing for different learning rates for each variable. It adjusts the step size based on the curvature (as estimated by the diagonal of the Hessian) so that each variable gets an update that is appropriate for its scale.

