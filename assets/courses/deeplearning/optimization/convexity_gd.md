---
layout: page
title: Convexity and Gradient Descent
description: 
related_publications: false
toc:
    sidebar: right
back_link: '/teaching/deeplearnig'
back_text: 'Deep Learning Course Page'
number_heading: false
enable_heading_styles: true
---

## Convexity
Convexity plays a vital role in the design of optimization algorithms. This is largely due to the fact that it is much easier to analyze and test algorithms in such a context.
- In other words, if the algorithm performs poorly even in the convex setting, typically we should not hope to see great results otherwise. Furthermore, even though the optimization problems in deep learning are generally nonconvex, they often exhibit some properties of convex ones near local minima.

### Definition

Before convex analysis, we need to define *convex sets* and *convex functions*. They lead to mathematical tools that are commonly applied to machine learning.

#### Convex Sets
Sets are the basis of convexity. Simply put, a set $$\mathcal{X}$$ in a vector space is convex if for any $$a, b \in \mathcal{X}$$ the line segment connecting $$a$$ and $$b$$ is also in $$\mathcal{X}$$. In mathematical terms this means that for all $$\lambda \in[0,1]$$ we have

$$
\lambda a+(1-\lambda) b \in \mathcal{X} \text { whenever } a, b \in \mathcal{X}
$$

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://d2l.ai/_images/pacman.svg" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
Fig. 1. The first set is nonconvex and the other two are convex.
</div>

Something useful: 
- Assume that $$\mathcal{X}$$ and $$\mathcal{Y}$$ are convex sets. Then $$\mathcal{X} \cap \mathcal{Y}$$ is also convex.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://d2l.ai/_images/convex-intersect.svg" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
Fig. 2. The intersection between two convex sets is convex.
</div>

- We can strengthen this result with little effort: given convex sets $$\mathcal{X}_{i}$$, their intersection $$\cap_{i} \mathcal{X}_{i}$$ is
convex. 
    - To see that the converse is not true, consider two disjoint sets $$\mathcal{X} \cap \mathcal{Y}=\emptyset$$. Now pick $$a \in \mathcal{X}$$ and $$b \in \mathcal{Y}$$. 
    - The line segment in [fig. 3](#fig3) connecting $$a$$ and $$b$$ needs to contain some part
    that is neither in $$\mathcal{X}$$ nor in $$\mathcal{Y}$$, since we assumed that $$\mathcal{X} \cap \mathcal{Y}=\emptyset$$. Hence the line segment is not in
    $$\mathcal{X} \cup \mathcal{Y}$$ either, thus proving that in general unions of convex sets need not be convex.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig3"></a>
        {% include figure.liquid loading="eager" path="https://d2l.ai/_images/nonconvex.svg" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
Fig. 3. The union of two convex sets need not be convex.
</div>
Typically the problems in deep learning are defined on convex sets. For instance, $$\mathbb{R}^{d}$$, the set of $$d$$ - dimensional vectors of real numbers, is a convex set (after all, the line between any two points in $$\mathbb{R}^{d}$$ remains in $$\mathbb{R}^{d}$$ ). In some cases we work with variables of bounded length, such as balls of radius $$r$$ as defined by $$\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \textrm{ and } \|\mathbf{x}\| \leq r\}$$.

### Convex Functions

Now that we have convex sets we can introduce *convex functions* $f$.
Given a convex set $$\mathcal{X}$$, a function $$f: \mathcal{X} \to \mathbb{R}$$ is *convex* if for all $$x, x' \in \mathcal{X}$$ and for all $$\lambda \in [0, 1]$$ we have

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

To illustrate this let's plot a few functions and check which ones satisfy the requirement.
Below we define a few functions, both convex and nonconvex.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://d2l.ai/_images/output_convexity_94e148_15_0.svg" class="img-fluid rounded" %}
    </div>
</div>

### Jensen's Inequality

Given a convex function $$f$$, one of the most useful mathematical tools is *Jensen's inequality*.
It amounts to a generalization of the definition of convexity:

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \textrm{ and }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$

where $$\alpha_i$$ are nonnegative real numbers such that $$\sum_i \alpha_i = 1$$ and $$X$$ is a random variable. In other words, the expectation of a convex function is no less than the convex function of an expectation, where the latter is usually a simpler expression. 
To prove the first inequality we repeatedly apply the definition of convexity to one term in the sum at a time.


One of the common applications of Jensen's inequality is to bound a more complicated expression by a simpler one.
 - For example, its application can be with regard to the log-likelihood of partially observed random variables. That is, we use

    $$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

    - since $$\int P(Y) P(X \mid Y) dY = P(X)$$. This can be used in variational methods. 
    - Here $$Y$$ is typically the unobserved random variable, $$P(Y)$$ is the best guess of how it might be distributed, 
    - and $$P(X)$$ is the distribution with $$Y$$ integrated out. For instance, in clustering $$Y$$ might be the cluster labels and $$P(X \mid Y)$$ is the generative model when applying cluster labels.

### Properties

Convex functions have many useful properties. We describe a few commonly-used ones below.

#### Local Minima Are Global Minima

This can be proved by [contradiction](https://en.wikipedia.org/wiki/Proof_by_contradiction): 

Consider a convex function $$f$$ defined on a convex set $$\mathcal{X}$$. 

  - Suppose that $$x^{*}\in \mathcal{X}$$ is a local minimum: there exists a small positive value $$p$$ so that for $$x \in \mathcal{X}$$ that satisfies 
  $$0<\left|x-x^{*}\right| \leq p$$ we have $$f\left(x^{*}\right)<f(x)$$

Now, let's make an assumption:

Assume that the local minimum $$x^{*}$$ is not the global minimum of $$f$$: there exists $$x^{\prime} \in \mathcal{X}$$ for which $$f\left(x^{\prime}\right)<f\left(x^{*}\right)$$. Remember the fact that we constrain a range for the condition of $$p$$, therefore we also need to make sure that the $$x^{\prime}$$ also exists within this range of $$p$$ to make the contradiction stands in the following proof.

Since $$\mathcal{X}$$ is convex, the line segment between $$x^{*}$$ and $$x^{\prime}$$ is entirely within $$\mathcal{X}$$. Any point $$x$$ on this line can be expressed as:

$$x=\lambda x^{*}+(1-\lambda) x^{\prime}, \quad \lambda \in[0,1]$$

The distance between $$x$$ and $$x^{*}$$ is:

$$
\left|x-x^{*}\right|=\left|\lambda x^{*}+(1-\lambda) x^{\prime}-x^{*}\right|=(1-\lambda)\left|x^{\prime}-x^{*}\right|
$$

To ensure $$\textcolor{red}{\left|x - x^{*}\right| \leq p}$$ and $$
\textcolor{red}{\left|x - x^{*}\right| > 0}
$$

$$
(1-\lambda)\left|x^{\prime}-x^{*}\right|=p
$$

Solving for $$\lambda$$, we get:

$$
1-\lambda=\frac{p}{\left|x^{\prime}-x^{*}\right|} \Longrightarrow \lambda=1-\frac{p}{\left|x^{\prime}-x^{*}\right|}
$$

However, according to the definition of convex functions, we have

$$
\begin{aligned} f\left(\lambda x^{*}+(1-\lambda) x^{\prime}\right) & \leq \lambda f\left(x^{*}\right)+(1-\lambda) f\left(x^{\prime}\right) \\ & <\lambda f\left(x^{*}\right)+(1-\lambda) f\left(x^{*}\right) \\ & =f\left(x^{*}\right)\end{aligned}
$$

which contradicts with our statement that $$x^{*}$$ is a local minimum. Therefore, there does not exist
$$x^{\prime} \in \mathcal{X}$$ for which $$f\left(x^{\prime}\right)<f\left(x^{*}\right)$$. The local minimum $$x^{*}$$ is also the global minimum.
#### Below Sets of Convex Functions Are Convex
We can conveniently define convex sets via *below sets* of convex functions. Concretely, given a convex function $$f$$ defined on a convex set $$\mathcal{X}$$, any below set

$$
\mathcal{S}_b \stackrel{\textrm{def}}{=} \{ x \in \mathcal{X} \mid f(x) \leq b \}
$$

is convex.

**Proof:**

Take any two points $$x, x' \in \mathcal{S}_b$$. By definition, this means that $$f(x) \leq b$$ and $$f(x') \leq b$$.

Consider any $$\lambda \in [0,1]$$ and define $$y = \lambda x + (1 - \lambda) x'$$. Since $$\mathcal{X}$$ is convex and $$x, x' \in \mathcal{X}$$, it follows that $$y \in \mathcal{X}$$.

Using the convexity of $$f$$, we have:

$$
f(y) = f(\lambda x + (1 - \lambda) x') \leq \lambda f(x) + (1 - \lambda) f(x') \leq \lambda b + (1 - \lambda) b = b.
$$

Therefore, $$f(y) \leq b$$, which means $$y \in \mathcal{S}_b$$.

Since any convex combination of points in $$\mathcal{S}_b$$ is also in $$\mathcal{S}_b$$, the set $$\mathcal{S}_b$$ is convex.

#### Convexity and Second Derivatives

Whenever the second derivative of a function $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ exists, it is easy to check whether $$f$$ is convex. All we need to do is check whether the Hessian of $$f$$ is positive semidefinite: $$\nabla^2 f \succeq 0$$, i.e., denoting the Hessian matrix $$\nabla^2 f$$ by $$\mathbf{H}$$, we have $$\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$$ for all $$\mathbf{x} \in \mathbb{R}^n$$.

For instance, the function $$f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$$ is convex since $$\nabla^2 f = \mathbf{I}$$, i.e., its Hessian is the identity matrix.

Formally, a twice-differentiable one-dimensional function $$f: \mathbb{R} \rightarrow \mathbb{R}$$ is convex if and only if its second derivative $$f''(x) \geq 0$$. For any twice-differentiable multidimensional function $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$, it is convex if and only if its Hessian $$\nabla^2 f \succeq 0$$.

**Proof:**

First, we prove the one-dimensional case.

Assume that $$f$$ is convex. Then, for any $$\epsilon > 0$$:

$$
\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left( \frac{x + \epsilon + x - \epsilon}{2} \right) = f(x).
$$

This inequality follows from the definition of convexity.

Since the second derivative is given by the limit over finite differences, it follows that:

$$
f''(x) = \lim_{\epsilon \to 0} \frac{f(x + \epsilon) + f(x - \epsilon) - 2 f(x)}{\epsilon^2} \geq 0.
$$

Thus, $$f''(x) \geq 0$$.

Conversely, suppose that $$f''(x) \geq 0$$. Then, $$f'$$ is a monotonically nondecreasing function.

Let $$a < x < b$$ be points in $$\mathbb{R}$$, where $$x = (1 - \lambda)a + \lambda b$$ and $$\lambda \in (0, 1)$$.

By the [mean value theorem](https://en.wikipedia.org/wiki/Mean_value_theorem), there exist $$\alpha \in [a, x]$$ and $$\beta \in [x, b]$$ such that:

$$
f'(\alpha) = \frac{f(x) - f(a)}{x - a}, \quad f'(\beta) = \frac{f(b) - f(x)}{b - x}.
$$

Since $$f'$$ is nondecreasing, $$f'(\beta) \geq f'(\alpha)$$.

Therefore:

$$
\frac{f(x) - f(a)}{x - a} \leq \frac{f(b) - f(x)}{b - x}.
$$

Cross-multiplying:

$$
(b - x)[f(x) - f(a)] \leq (x - a)[f(b) - f(x)].
$$

Rewriting (carefully expand it out on the previous step):

$$
\frac{x - a}{b - a}[f(b) - f(x)] + \frac{b - x}{b - a}[f(x) - f(a)] \geq 0.
$$

This implies(rewrite $$
\lambda=\frac{x-a}{b-a}
$$):

$$
(1 - \lambda) f(a) + \lambda f(b) \geq f(x).
$$

Thus, $$f$$ is convex.

Next, we introduce a lemma before proving the multidimensional case.

**Lemma:** A function $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ is convex if and only if for all $$\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$$:

$$
g(z) \stackrel{\textrm{def}}{=} f(z \mathbf{x} + (1 - z) \mathbf{y}), \quad z \in [0,1],
$$

is convex.

To show that convexity of $$f$$ implies that $$g$$ is convex, observe that for all $$a, b, \lambda \in [0, 1]$$:

$$
\begin{aligned}
g(\lambda a + (1 - \lambda) b) &= f\left( (\lambda a + (1 - \lambda) b) \mathbf{x} + \left(1 - (\lambda a + (1 - \lambda) b)\right) \mathbf{y} \right) \\
&= f\left( \lambda \left( a \mathbf{x} + (1 - a) \mathbf{y} \right) + (1 - \lambda) \left( b \mathbf{x} + (1 - b) \mathbf{y} \right) \right) \\
&\leq \lambda f(a \mathbf{x} + (1 - a) \mathbf{y}) + (1 - \lambda) f(b \mathbf{x} + (1 - b) \mathbf{y}) \\
&= \lambda g(a) + (1 - \lambda) g(b).
\end{aligned}
$$

Therefore, $$g$$ is convex.

Conversely, suppose $$g$$ is convex for all $$\mathbf{x}, \mathbf{y}$$. Then, for all $$\lambda \in [0, 1]$$:

$$
\begin{aligned}
f(\lambda \mathbf{x} + (1 - \lambda) \mathbf{y}) &= g(\lambda \cdot 1 + (1 - \lambda) \cdot 0) \\
&\leq \lambda g(1) + (1 - \lambda) g(0) \\
&= \lambda f(\mathbf{x}) + (1 - \lambda) f(\mathbf{y}).
\end{aligned}
$$

Thus, $$f$$ is convex.

Finally, using the lemma and the one-dimensional result, we can prove the multidimensional case.

A multidimensional function $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ is convex if and only if for all $$\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$$, the function $$g(z) = f(z \mathbf{x} + (1 - z) \mathbf{y})$$ is convex.

According to the one-dimensional case, this holds if and only if:

$$
g''(z) = (\mathbf{x} - \mathbf{y})^\top \nabla^2 f(z \mathbf{x} + (1 - z) \mathbf{y})(\mathbf{x} - \mathbf{y}) \geq 0,
$$

for all $$\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$$.

This condition is equivalent to $$\nabla^2 f \succeq 0$$, per the definition of positive semidefinite matrices.

### Constraints
One of the nice properties of convex optimization is that it allows us to handle constraints efficiently. That is, it allows us to solve *constrained optimization* problems of the form:

$$
\begin{array}{l}
\underset{\mathbf{x}}{\operatorname{minimize}} f(\mathbf{x}) \\
\text { subject to } c_{i}(\mathbf{x}) \leq 0 \text { for all } i \in\{1, \ldots, n\}
\end{array}
$$

where $$f$$ is the objective and the functions $$c_{i}$$ are constraint functions.

**Question**: Imagine a unit ball $$c_{1}(\mathbf{x})=\|\mathbf{x}\|_{2}-1$$; Now we have a second constraint: $$c_{2}(\mathbf{x})=\mathbf{v}^{\top} \mathbf{x}+b$$. What does this visually represent?

#### Lagrangian

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

#### Penalties

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