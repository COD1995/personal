---
layout: page
title: "Ch 8 — Graphical Models"
description: Bayesian networks, conditional independence, Markov random fields, message passing, junction trees.
math: true
slides: true
back_link: '/teaching/introml'
back_text: 'Intro to ML'
---

A probabilistic graphical model is a **diagram of a joint distribution**: nodes are random variables, edges encode statistical dependence, and the absence of edges encodes *independence*. Two reasons graphical models matter:

1. **Communication.** A picture exposes the structure of a model far more clearly than its factorized formula.
2. **Computation.** The graph structure dictates which variables can be marginalized cheaply, what message-passing schemes apply, and where conditional independencies live.

PRML Ch 8 distinguishes two flavors: **directed** graphs (Bayesian networks) and **undirected** graphs (Markov random fields).

## Prerequisites

- **Probability** (Chs 1–2): joint vs. conditional vs. marginal; chain rule.
- **Linear-algebra basics** for the Gaussian-MRF examples.
- **Optimization familiarity** for the inference cost discussion.

---

## 8.1 Bayesian Networks (Directed Graphs)

A **directed acyclic graph** (DAG) over variables \\(\lbrace x\_1, \ldots, x\_K\rbrace\\) represents the joint

$$
p(x_1, \ldots, x_K) = \prod_{k=1}^{K} p(x_k \mid \mathrm{pa}(x_k)),
\tag{8.1}
$$

where \\(\mathrm{pa}(x\_k)\\) are the parents of \\(x\_k\\) in the graph. Each variable is conditional only on its parents. The DAG is read off the factorization, and vice versa.

### Examples

- **Naive Bayes** — class \\(\mathcal{C}\\) at the root, features as children: \\(p(\mathcal{C}) \prod\_d p(x\_d \mid \mathcal{C})\\).
- **HMMs** — a chain \\(z\_1 \to z\_2 \to \cdots\\) of hidden states with observations \\(z\_t \to x\_t\\).
- **Polynomial regression as a graph** — \\(\mathbf{w}\\) at the root, training targets \\(t\_n\\) as children given \\(\mathbf{w}\\) and \\(\mathbf{x}\_n\\).

### Conditional independence — the d-separation criterion

Three patterns of three variables \\(a \to b \to c\\), \\(a \leftarrow b \to c\\), \\(a \to b \leftarrow c\\):

- **Tail-to-tail / head-to-tail** (chains and forks): \\(a \perp\\!\\!\!\perp c \mid b\\). Conditioning on \\(b\\) **breaks** the path.
- **Head-to-head** (collider): \\(a \perp\\!\\!\!\perp c\\) marginally; *not* conditional on \\(b\\). Conditioning on \\(b\\) (or any descendant) **opens** the path.

A path between \\(a\\) and \\(c\\) is **blocked** given \\(\mathbf{Z}\\) if it contains a non-collider in \\(\mathbf{Z}\\) or a collider not in \\(\mathbf{Z}\\) and with no descendant in \\(\mathbf{Z}\\). \\(a \perp\\!\\!\!\perp c \mid \mathbf{Z}\\) iff every path between \\(a\\) and \\(c\\) is blocked. This is **d-separation** — the topological criterion for reading conditional independence off a DAG.

### Plates

A grouping shorthand: a rectangle around a sub-graph means "repeat this for \\(n = 1, \ldots, N\\)." Common in writing down models that have one observation per data point — saves drawing \\(N\\) copies of the same node.

---

## 8.2 Markov Random Fields (Undirected Graphs)

For undirected graphs, the joint factorizes over **cliques** (fully connected subsets):

$$
p(\mathbf{x}) = \frac{1}{Z} \prod_{C} \psi_C(\mathbf{x}_C),
\qquad
Z = \sum_{\mathbf{x}} \prod_{C} \psi_C(\mathbf{x}_C).
\tag{8.2}
$$

Each \\(\psi\_C \geq 0\\) is a **potential function** (not a conditional probability — it can be unnormalized). \\(Z\\) is the **partition function**, generally intractable to compute.

### Conditional independence — graph separation

Much simpler than d-separation. \\(\mathbf{x}\_A \perp\\!\\!\!\perp \mathbf{x}\_B \mid \mathbf{x}\_C\\) iff every path from \\(A\\) to \\(B\\) passes through \\(C\\) — pure graph separation.

### Hammersley–Clifford

A positive distribution can be factorized as (8.2) iff it satisfies the conditional independence relations implied by the graph. The graph structure and the factorization are equivalent characterizations.

### Common MRF examples

- **Image models** — pixel labels as nodes, neighbor pixels connected to encourage locally-similar labels (image denoising, segmentation).
- **Boltzmann machines / Ising models** — binary variables on a grid with pairwise interactions.
- **Conditional random fields** — discriminative MRFs for sequence labeling (NER, chunk parsing).

---

## 8.3 Inference in Graphical Models

Three core tasks:

- **Marginalization.** Compute \\(p(x\_i)\\) by summing out the rest. \\(O(K^N)\\) brute force.
- **Conditioning.** Compute \\(p(x\_i \mid \mathbf{x}\_{\mathrm{evidence}})\\).
- **MAP.** Find the assignment that maximizes \\(p(\mathbf{x})\\) (or the posterior).

### Variable elimination

Choose an elimination order; for each variable, sum it out by gathering all factors involving it, multiplying, and summing — leaving a smaller factor. Cost is exponential in the **largest intermediate factor**, which depends critically on the elimination order. Finding the optimal order is NP-hard.

### Message passing on trees (sum–product algorithm)

When the graph is a **tree** (or factor graph that's a tree), exact inference is \\(O(K^2 N)\\) via two passes of message passing. Each node sends a message to its parent (upward pass) and receives one back (downward pass); marginals fall out at the end.

For general graphs, message passing on a **junction tree** (a tree of clique groupings) generalizes the algorithm. Cost depends on the largest clique — exponential in the *treewidth* of the graph.

### Loopy belief propagation

For graphs with cycles, run sum–product anyway — message passing iterates until (hopefully) convergent. No guarantees, but often surprisingly accurate, and the basis of many inference algorithms in computer vision.

### When all else fails

For graphs where exact inference is infeasible, fall back to:

- **Variational inference** (Ch 10) — fit a tractable approximation.
- **Sampling** (Ch 11) — Markov chain Monte Carlo.

---

## 8.4 Why Graphical Models Matter

Even when the graph isn't directly used for inference, it's a clean language:

- **Specifying complex models** — Latent Dirichlet Allocation, deep belief networks, neural turing machines all have natural graphical-model presentations.
- **Communicating modeling assumptions** — what's independent of what given what is the heart of any probabilistic model, and the graph makes it explicit.
- **Suggesting algorithms** — d-separation tells you what variables you can ignore; treewidth bounds tell you what's tractable; the absence of certain edges suggests where to apply variational approximations.

---

## Takeaways

- **DAGs** factorize the joint by conditional probabilities; **MRFs** factorize by clique potentials with a partition function.
- **d-separation** (DAGs) and **graph separation** (MRFs) read off conditional independence purely from graph structure.
- **Sum–product** on trees and junction trees gives exact inference; **loopy BP** extends approximately to general graphs.
- The graph language scales to arbitrarily complex models — most practical Bayesian models are graphical models in disguise.

Forward to <a href="{{ '/assets/courses/introml/09_mixture_models_em/' | relative_url }}">Ch 9 — Mixture Models &amp; EM</a>.

---

<small><em>Figures from Bishop, <em>Pattern Recognition and Machine Learning</em> (Springer, 2006), © Springer / C. M. Bishop. Reproduced for educational use.</em></small>
