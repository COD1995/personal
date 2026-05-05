---
layout: page
title: "Back-Propagation (Rumelhart, Hinton & Williams, 1986)"
description: "Reading note · Pattern Recognition · Neural Network Foundations"
back_link: '/teaching/pattern/'
back_text: "Back to Pattern Recognition"
---

## TL;DR

A neural network is just a stack of simple "units" that each take a weighted
sum of their inputs, squash it through a non-linear function, and pass the
result to the next layer. To **train** such a network we want to find the
weights that make its output match a target. **Back-propagation** is the
algorithm that says: *"Take the error you see at the output, and pass it
backwards through the network using the chain rule, so every weight learns
how much it personally contributed to the mistake."*

That's it. The rest of this note slowly builds up the math behind that one
sentence — assuming you are *not* fluent in calculus.

---

## Problem & Motivation

A single neuron (the *perceptron*, 1958) can only separate inputs with a
straight line. It cannot learn `XOR`:

| $$x_1$$ | $$x_2$$ | $$y$$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No single line cuts the four points correctly. To fix this we add an
intermediate ("hidden") layer of neurons — but now we face the
**credit-assignment problem**: when the network produces the wrong answer,
*which* hidden weight is at fault, and by how much?

Backprop answers that question. It is the gradient of the loss
with respect to every weight, computed efficiently by re-using intermediate
results.

---

## Math Prerequisites

If the next four pages feel like review, skip ahead. Otherwise, read slowly.

### 1. Derivatives — what does $$\dfrac{df}{dx}$$ mean?

If $$f(x)$$ is a function, then $$\dfrac{df}{dx}$$ at a point $$x$$ is the
**slope** of $$f$$ at that point. It tells you: *"if I nudge $$x$$ by a tiny
amount $$\Delta x$$, how much does $$f$$ change?"*

$$
f(x + \Delta x) \;\approx\; f(x) + \frac{df}{dx}\,\Delta x.
$$

Example. If $$f(x) = x^2$$, then $$\dfrac{df}{dx} = 2x$$. At $$x=3$$ the slope
is $$6$$, so nudging $$x$$ from $$3$$ to $$3.01$$ changes $$f$$ by roughly
$$6 \times 0.01 = 0.06$$.

### 2. Partial derivatives — when there are many inputs

Suppose $$E$$ depends on many weights $$w_1, w_2, \dots, w_n$$. The **partial
derivative** $$\dfrac{\partial E}{\partial w_i}$$ asks: *"if I nudge **only**
$$w_i$$ and hold everything else fixed, how much does $$E$$ change?"* The
notation $$\partial$$ (instead of $$d$$) is just a reminder that other
variables exist but we are pretending they're constants for this calculation.

Example. If $$E = w_1^2 + 3 w_1 w_2$$, then

$$
\frac{\partial E}{\partial w_1} = 2 w_1 + 3 w_2, \qquad
\frac{\partial E}{\partial w_2} = 3 w_1.
$$

### 3. The chain rule — composed functions

The chain rule says: *"if $$y$$ depends on $$u$$, and $$u$$ depends on $$x$$,
then a nudge in $$x$$ ripples through to $$y$$ via $$u$$."*

$$
\frac{dy}{dx} \;=\; \frac{dy}{du} \cdot \frac{du}{dx}.
$$

You **multiply** the slopes along the path from $$x$$ to $$y$$.

Concrete example. Let $$y = (3x+2)^2$$. Set $$u = 3x+2$$, so $$y = u^2$$.
Then $$\dfrac{dy}{du} = 2u$$ and $$\dfrac{du}{dx} = 3$$, giving

$$
\frac{dy}{dx} = 2u \cdot 3 = 6(3x+2).
$$

When the path branches (one $$x$$ feeds many $$u$$'s, all of which feed
$$y$$), you **add up** the contribution from each branch. This branching is
exactly what happens in a neural network — and exactly why backprop has the
form it does.

### 4. Gradient descent — using the slope to improve

To minimise $$E(w)$$ we repeatedly take small steps **opposite** to the slope:

$$
w \;\leftarrow\; w \;-\; \eta\, \frac{\partial E}{\partial w}.
$$

The number $$\eta$$ (eta, Greek "h") is the **learning rate** — how big a
step. If the slope is positive, $$E$$ is going *up* as $$w$$ increases, so we
move $$w$$ *down*. That's the minus sign.

### 5. The sigmoid activation and its derivative

The 1986 paper uses the **logistic sigmoid**:

$$
\sigma(z) \;=\; \frac{1}{1 + e^{-z}}.
$$

It squashes any real number into the range $$(0, 1)$$. Its derivative has a
beautiful property — it can be written entirely in terms of $$\sigma(z)$$
itself:

$$
\sigma'(z) \;=\; \sigma(z)\,\bigl(1 - \sigma(z)\bigr).
$$

**Why this is convenient.** During backprop we'll need $$\sigma'(z)$$. We
already have $$\sigma(z)$$ from the forward pass — no recomputation needed.

---

## Building Up: From One Neuron to a Network

We will build the algorithm in three stages, each adding one layer of
complication. By stage 3 you will have re-derived the algorithm yourself.

### Stage 1: A single neuron

The simplest possible network is one neuron with $$n$$ inputs:

$$
a \;=\; \sum_{i=1}^{n} w_i\, x_i, \qquad o \;=\; \sigma(a),
$$

where $$x_i$$ is the $$i$$-th input value and $$w_i$$ is its weight. We have
a target $$t$$ and use the **squared error**:

$$
E \;=\; \tfrac{1}{2}\,(t - o)^2.
$$

(The $$\tfrac{1}{2}$$ is just so the $$2$$ from differentiating cancels — it
doesn't change where the minimum is.)

**Goal:** find $$\dfrac{\partial E}{\partial w_i}$$, so we can update $$w_i$$.

The dependency chain is

$$
w_i \;\longrightarrow\; a \;\longrightarrow\; o \;\longrightarrow\; E.
$$

Apply the chain rule along this chain — multiplying the slopes:

$$
\frac{\partial E}{\partial w_i}
\;=\;
\underbrace{\frac{\partial E}{\partial o}}_{\text{step 1}}
\cdot
\underbrace{\frac{\partial o}{\partial a}}_{\text{step 2}}
\cdot
\underbrace{\frac{\partial a}{\partial w_i}}_{\text{step 3}}.
$$

Compute each piece:

- **Step 1.** $$E = \tfrac{1}{2}(t - o)^2$$, so $$\dfrac{\partial E}{\partial o} = -(t - o)$$.
- **Step 2.** $$o = \sigma(a)$$, so $$\dfrac{\partial o}{\partial a} = \sigma'(a) = o(1-o)$$.
- **Step 3.** $$a = \sum_j w_j x_j$$, so $$\dfrac{\partial a}{\partial w_i} = x_i$$.

Multiplying:

$$
\frac{\partial E}{\partial w_i}
\;=\;
-(t - o)\, \cdot \, o(1-o)\, \cdot \, x_i.
$$

This single formula already contains the whole idea of backprop. Read it
left-to-right:

1. *How wrong was the output?* &nbsp;&nbsp; $$-(t-o)$$
2. *How sensitive is the output to its raw input?* &nbsp;&nbsp; $$o(1-o)$$
3. *How much did this particular weight feed into that raw input?* &nbsp;&nbsp; $$x_i$$

Multiply them, and you have weight $$w_i$$'s share of the blame.

The update rule (gradient descent) is then

$$
w_i \;\leftarrow\; w_i \;-\; \eta\, \frac{\partial E}{\partial w_i}
\;=\; w_i \;+\; \eta\, (t-o)\, o(1-o)\, x_i.
$$

### Stage 1.5: A worked numerical example

Let's run the actual numbers through one full forward-and-backward pass.
To keep the arithmetic clean we use a **linear** neuron (no sigmoid) — the
chain-rule structure is identical, just with one fewer factor. This walkthrough
follows the [Backprop Explainer (Bertucci & Kahng, 2021)](https://xnought.github.io/backprop-explainer/),
which has a beautiful interactive version of exactly this example.

<figure class="bp-diagram" markdown="0">
<svg viewBox="0 0 720 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="bp-fb-title">
  <title id="bp-fb-title">Animated forward and backward pass through a single linear neuron</title>
  <defs>
    <marker id="bp-arr-fwd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#5f5f63"/>
    </marker>
    <marker id="bp-arr-bwd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#8b3a3a"/>
    </marker>
  </defs>

  <text x="360" y="22" text-anchor="middle" font-family="Inter, sans-serif" font-size="11" font-weight="600" letter-spacing="0.08em" fill="#5f5f63">FORWARD &nbsp;(values flow right)</text>

  <g font-family="Inter, sans-serif" font-size="14" fill="#1d1d1f">
    <!-- Input box (flashes when forward pulse leaves) -->
    <rect x="40" y="50" width="80" height="50" rx="4" fill="#fff" stroke="#5f5f63" stroke-width="1.2">
      <animate attributeName="stroke" values="#5f5f63;#5f5f63;#5f5f63;#5f5f63" dur="6s" repeatCount="indefinite"/>
    </rect>
    <text x="80" y="80" text-anchor="middle">x</text>

    <!-- Neuron box (flashes when forward pulse passes ~t=0.17, when backward pulse passes ~t=0.66 of 6s cycle) -->
    <rect x="220" y="50" width="180" height="50" rx="4" fill="#fff" stroke="#5f5f63" stroke-width="1.2">
      <animate attributeName="stroke"
               values="#5f5f63;#5f5f63;#56a8c7;#5f5f63;#5f5f63;#8b3a3a;#5f5f63"
               keyTimes="0;0.16;0.18;0.22;0.66;0.68;0.72"
               dur="6s" repeatCount="indefinite" fill="freeze"/>
      <animate attributeName="stroke-width"
               values="1.2;1.2;2.4;1.2;1.2;2.4;1.2"
               keyTimes="0;0.16;0.18;0.22;0.66;0.68;0.72"
               dur="6s" repeatCount="indefinite"/>
    </rect>
    <text x="310" y="80" text-anchor="middle">ŷ = w·x + b</text>

    <!-- Loss box (flashes when forward pulse arrives ~t=0.42) -->
    <rect x="500" y="50" width="180" height="50" rx="4" fill="#fff" stroke="#5f5f63" stroke-width="1.2">
      <animate attributeName="stroke"
               values="#5f5f63;#5f5f63;#56a8c7;#5f5f63"
               keyTimes="0;0.41;0.43;0.47"
               dur="6s" repeatCount="indefinite"/>
      <animate attributeName="stroke-width"
               values="1.2;1.2;2.4;1.2"
               keyTimes="0;0.41;0.43;0.47"
               dur="6s" repeatCount="indefinite"/>
    </rect>
    <text x="590" y="80" text-anchor="middle">E = (ŷ − y)²</text>
  </g>

  <!-- Forward arrows (top) -->
  <g stroke="#5f5f63" stroke-width="1.5" fill="none">
    <line x1="120" y1="75" x2="218" y2="75" marker-end="url(#bp-arr-fwd)"/>
    <line x1="400" y1="75" x2="498" y2="75" marker-end="url(#bp-arr-fwd)"/>
  </g>

  <!-- Forward pulse: travels 120 → 590 over first half of 6s cycle (active 0–3s, idle 3–6s) -->
  <circle cy="75" r="5" fill="#56a8c7" opacity="0">
    <animate attributeName="cx"
             values="120;590;590;120"
             keyTimes="0;0.45;0.50;1"
             dur="6s" repeatCount="indefinite"/>
    <animate attributeName="opacity"
             values="0;1;1;0;0;0"
             keyTimes="0;0.04;0.43;0.47;0.50;1"
             dur="6s" repeatCount="indefinite"/>
  </circle>
  <!-- Forward value label "x = 2.1" — first half of forward cycle -->
  <text font-family="JetBrains Mono, monospace" font-size="11" font-weight="600" fill="#56a8c7" text-anchor="middle" opacity="0">
    <animate attributeName="x" values="120;218" keyTimes="0;1" dur="2.4s" repeatCount="indefinite"/>
    <animate attributeName="y" values="65;65" keyTimes="0;1" dur="2.4s" repeatCount="indefinite"/>
    <animate attributeName="opacity"
             values="0;1;1;0;0"
             keyTimes="0;0.10;0.85;0.95;1"
             dur="2.4s" repeatCount="indefinite"/>
    x = 2.1
  </text>
  <!-- Forward value label "ŷ = 2.1" — second arrow segment -->
  <text font-family="JetBrains Mono, monospace" font-size="11" font-weight="600" fill="#56a8c7" text-anchor="middle" opacity="0">
    <animate attributeName="x" values="400;498" keyTimes="0;1" dur="2.4s" begin="1.2s" repeatCount="indefinite"/>
    <animate attributeName="y" values="65;65" keyTimes="0;1" dur="2.4s" begin="1.2s" repeatCount="indefinite"/>
    <animate attributeName="opacity"
             values="0;1;1;0;0"
             keyTimes="0;0.10;0.85;0.95;1"
             dur="2.4s" begin="1.2s" repeatCount="indefinite"/>
    ŷ = 2.1
  </text>

  <text x="360" y="135" text-anchor="middle" font-family="Inter, sans-serif" font-size="11" font-weight="600" letter-spacing="0.08em" fill="#8b3a3a">BACKWARD &nbsp;(gradients flow left, multiply along the path)</text>

  <!-- Backward arrows (bottom) -->
  <g stroke="#8b3a3a" stroke-width="1.5" fill="none" stroke-dasharray="5,4">
    <line x1="500" y1="160" x2="402" y2="160" marker-end="url(#bp-arr-bwd)"/>
    <line x1="220" y1="160" x2="122" y2="160" marker-end="url(#bp-arr-bwd)"/>
  </g>

  <g font-family="Source Serif 4, Georgia, serif" font-size="14" fill="#8b3a3a" text-anchor="middle">
    <text x="450" y="180">∂E/∂ŷ = 2(ŷ − y)</text>
    <text x="170" y="180">∂ŷ/∂w = x</text>
  </g>

  <!-- Backward pulse: travels 590 → 80 in second half of 6s cycle -->
  <circle cy="160" r="5" fill="#8b3a3a" opacity="0">
    <animate attributeName="cx"
             values="120;120;590;80;80"
             keyTimes="0;0.50;0.55;0.95;1"
             dur="6s" repeatCount="indefinite"/>
    <animate attributeName="opacity"
             values="0;0;1;1;0;0"
             keyTimes="0;0.50;0.54;0.93;0.95;1"
             dur="6s" repeatCount="indefinite"/>
  </circle>
  <!-- Backward value label "−3.8" — first backward segment -->
  <text font-family="JetBrains Mono, monospace" font-size="11" font-weight="600" fill="#8b3a3a" text-anchor="middle" opacity="0">
    <animate attributeName="x" values="498;402" keyTimes="0;1" dur="2.4s" begin="3s" repeatCount="indefinite"/>
    <animate attributeName="y" values="148;148" keyTimes="0;1" dur="2.4s" begin="3s" repeatCount="indefinite"/>
    <animate attributeName="opacity"
             values="0;1;1;0;0"
             keyTimes="0;0.10;0.85;0.95;1"
             dur="2.4s" begin="3s" repeatCount="indefinite"/>
    −3.8
  </text>
  <!-- Backward value label "−7.98" — second backward segment (after multiplication by x) -->
  <text font-family="JetBrains Mono, monospace" font-size="11" font-weight="600" fill="#8b3a3a" text-anchor="middle" opacity="0">
    <animate attributeName="x" values="218;122" keyTimes="0;1" dur="2.4s" begin="4.2s" repeatCount="indefinite"/>
    <animate attributeName="y" values="148;148" keyTimes="0;1" dur="2.4s" begin="4.2s" repeatCount="indefinite"/>
    <animate attributeName="opacity"
             values="0;1;1;0;0"
             keyTimes="0;0.10;0.85;0.95;1"
             dur="2.4s" begin="4.2s" repeatCount="indefinite"/>
    −7.98
  </text>

  <g font-family="Source Serif 4, Georgia, serif" font-size="13" fill="#8b3a3a" font-weight="600" text-anchor="middle">
    <text x="80" y="225">∂E/∂w = (∂E/∂ŷ) · (∂ŷ/∂w) = −7.98</text>
  </g>
</svg>
<figcaption>Forward (gray + blue pulse): the value <code>x = 2.1</code> flows into the neuron, becomes <code>ŷ = 2.1</code>, then enters the loss. Backward (red, dashed): the gradient <code>∂E/∂ŷ = −3.8</code> emerges from the loss, gets multiplied by <code>x</code> at the neuron, and arrives at the input as <code>∂E/∂w = −7.98</code> — the chain rule, animated.</figcaption>
</figure>

**Setup.**

- Network: $$\hat{y} = w x + b$$ &nbsp;(one linear neuron with weight and bias)
- One training example: $$x = 2.1$$, target $$y = 4$$
- Initial parameters: $$w = 1$$, $$b = 0$$
- Loss: squared error $$E = (\hat{y} - y)^2$$ &nbsp;(no $$\tfrac{1}{2}$$ this time, to match the explainer)
- Learning rate: $$\eta = 0.01$$

#### Step 1 — Forward pass

$$
\hat{y} = w\,x + b = (1)(2.1) + 0 = 2.1
$$

$$
E = (\hat{y} - y)^2 = (2.1 - 4)^2 = (-1.9)^2 = 3.61.
$$

So the prediction is $$2.1$$, the target is $$4$$, and the current loss is $$3.61$$. Clearly the prediction is too low.

#### Step 2 — Backward pass: compute the gradient

We want $$\dfrac{\partial E}{\partial w}$$ and $$\dfrac{\partial E}{\partial b}$$. The dependency chain is

$$
w,\, b \;\longrightarrow\; \hat{y} \;\longrightarrow\; E.
$$

**(2a) How does $$E$$ change with $$\hat{y}$$?**
For $$E = (\hat{y} - y)^2$$, the derivative (using the chain rule on the square) is

$$
\frac{\partial E}{\partial \hat{y}} = 2(\hat{y} - y) = 2(2.1 - 4) = 2(-1.9) = -3.8.
$$

The minus sign says: *if we increase $$\hat{y}$$, the loss goes down* — which makes sense, since we under-shot the target.

**(2b) How does $$\hat{y}$$ change with $$w$$ and $$b$$?**
For $$\hat{y} = wx + b$$,

$$
\frac{\partial \hat{y}}{\partial w} = x = 2.1, \qquad
\frac{\partial \hat{y}}{\partial b} = 1.
$$

**(2c) Chain them together.**

$$
\frac{\partial E}{\partial w}
\;=\; \frac{\partial E}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}
\;=\; (-3.8)(2.1) \;=\; -7.98.
$$

$$
\frac{\partial E}{\partial b}
\;=\; \frac{\partial E}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial b}
\;=\; (-3.8)(1) \;=\; -3.8.
$$

**Plain-English read of these two numbers:**
*"the loss is currently decreasing fastest if we make $$w$$ larger (because $$\partial E/\partial w$$ is negative — moving $$w$$ in the negative-of-negative direction lowers $$E$$). Same story for $$b$$."*

#### Step 3 — Gradient-descent update

Step in the direction that *lowers* the loss, scaled by the learning rate:

$$
w \;\leftarrow\; w - \eta\, \frac{\partial E}{\partial w}
\;=\; 1 - (0.01)(-7.98)
\;=\; 1 + 0.0798
\;=\; 1.0798.
$$

$$
b \;\leftarrow\; b - \eta\, \frac{\partial E}{\partial b}
\;=\; 0 - (0.01)(-3.8)
\;=\; 0 + 0.038
\;=\; 0.038.
$$

Both parameters increased — exactly what the gradient told us to do.

#### Step 4 — Verify the loss actually went down

Run a second forward pass with the new parameters:

$$
\hat{y}_{\text{new}} = (1.0798)(2.1) + 0.038 = 2.2676 + 0.038 = 2.3056.
$$

$$
E_{\text{new}} = (2.3056 - 4)^2 = (-1.6944)^2 \approx 2.87.
$$

**Loss dropped from $$3.61$$ to $$2.87$$ in one step — a decrease of about $$0.74$$.** Iterate this process thousands of times over many training examples, and the loss approaches zero.

#### Try it yourself: gradient descent on $$E(w)$$

The plot below shows the loss curve for our example
($$E(w) = (2.1\,w + 0 - 4)^2$$ with $$b$$ fixed at $$0$$). Drag the slider to
move $$w$$ and watch $$\hat y$$, $$E$$, and the gradient update. Click **Step**
to apply one gradient-descent update — exactly the math from Step 3 above.
Click **Auto-play** to watch the ball roll down the curve.

<div class="bp-demo" markdown="0">
<svg viewBox="0 0 480 270" id="bp-plot" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Loss curve E(w) with current point and tangent line">
  <g stroke="#e5e2dc" stroke-width="1" fill="none">
    <line x1="40" y1="225" x2="465" y2="225"/>
    <line x1="40" y1="20"  x2="40"  y2="225"/>
  </g>
  <g font-family="Inter, sans-serif" font-size="10" fill="#5f5f63">
    <text x="40"  y="240" text-anchor="middle">0</text>
    <text x="158" y="240" text-anchor="middle">1</text>
    <text x="276" y="240" text-anchor="middle">2</text>
    <text x="394" y="240" text-anchor="middle">3</text>
    <text x="252" y="258" text-anchor="middle" font-weight="600">w</text>
    <text x="34" y="228" text-anchor="end">0</text>
    <text x="34" y="24"  text-anchor="end">12</text>
    <text x="14" y="125" text-anchor="middle" font-weight="600" transform="rotate(-90 14 125)">E(w)</text>
  </g>
  <line id="bp-min"   stroke="#e5e2dc" stroke-width="1" stroke-dasharray="3,3"/>
  <text id="bp-min-label" font-family="Inter, sans-serif" font-size="10" fill="#5f5f63" text-anchor="middle">w* ≈ 1.905</text>
  <path     id="bp-curve"   stroke="#5f5f63" stroke-width="1.6" fill="none"/>
  <polyline id="bp-traj"    fill="none" stroke="#8b3a3a" stroke-width="1" stroke-opacity="0.45" stroke-dasharray="3,3"/>
  <g        id="bp-trail"></g>
  <line     id="bp-tangent" stroke="#8b3a3a" stroke-width="1.2" stroke-opacity="0.7"/>
  <circle   id="bp-ball"    r="6" fill="#8b3a3a" stroke="#fff" stroke-width="1.5"/>
</svg>

<div class="bp-readout">
  <div>w = <span id="bp-w">1.0000</span></div>
  <div>ŷ = w·x + b = <span id="bp-yhat">2.1000</span></div>
  <div>E = (ŷ − y)² = <span id="bp-loss">3.6100</span></div>
  <div>∂E/∂w = <span id="bp-grad">−7.9800</span></div>
  <div>step # <span id="bp-step-count">0</span></div>
</div>

<div class="bp-controls">
  <input type="range" id="bp-slider" min="0" max="3.5" step="0.001" value="1" aria-label="weight w">
  <button id="bp-btn-step"  type="button">Step</button>
  <button id="bp-btn-play"  type="button">Auto-play</button>
  <button id="bp-btn-reset" type="button">Reset</button>
</div>
</div>

<script>
(function () {
  var X = 2.1, Y = 4, B = 0, LR = 0.01;
  var W_MIN = 0, W_MAX = 3.5, E_MAX = 12;
  var L = 40, R = 465, T = 20, BTM = 225;
  var W_OPT = Y / X;

  var w = 1, steps = 0, traj = [[1, lossAt(1)]];

  function yhat(v)  { return v * X + B; }
  function lossAt(v){ return Math.pow(yhat(v) - Y, 2); }
  function gradAt(v){ return 2 * (yhat(v) - Y) * X; }
  function sx(v)    { return L + (v - W_MIN) / (W_MAX - W_MIN) * (R - L); }
  function sy(e)    { return T + (1 - Math.min(e, E_MAX) / E_MAX) * (BTM - T); }

  function byId(id){ return document.getElementById(id); }
  var els = {
    curve:   byId('bp-curve'),
    ball:    byId('bp-ball'),
    traj:    byId('bp-traj'),
    trail:   byId('bp-trail'),
    tangent: byId('bp-tangent'),
    min:     byId('bp-min'),
    minLbl:  byId('bp-min-label'),
    slider:  byId('bp-slider'),
    w:       byId('bp-w'),
    yhat:    byId('bp-yhat'),
    loss:    byId('bp-loss'),
    grad:    byId('bp-grad'),
    cnt:     byId('bp-step-count'),
    play:    byId('bp-btn-play')
  };

  // build curve
  var d = '';
  for (var i = 0; i <= 200; i++) {
    var wi = W_MIN + (W_MAX - W_MIN) * i / 200;
    d += (i ? ' L' : 'M') + sx(wi).toFixed(2) + ',' + sy(lossAt(wi)).toFixed(2);
  }
  els.curve.setAttribute('d', d);

  // optimum line
  els.min.setAttribute('x1', sx(W_OPT));
  els.min.setAttribute('x2', sx(W_OPT));
  els.min.setAttribute('y1', sy(0));
  els.min.setAttribute('y2', sy(E_MAX));
  els.minLbl.setAttribute('x', sx(W_OPT));
  els.minLbl.setAttribute('y', T - 6);

  function fmt(n) {
    var s = n.toFixed(4);
    return s.charAt(0) === '-' ? '−' + s.slice(1) : s;
  }

  function refresh() {
    els.w.textContent     = fmt(w);
    els.yhat.textContent  = fmt(yhat(w));
    els.loss.textContent  = fmt(lossAt(w));
    els.grad.textContent  = fmt(gradAt(w));
    els.cnt.textContent   = steps;

    els.ball.setAttribute('cx', sx(w));
    els.ball.setAttribute('cy', sy(lossAt(w)));

    var s = gradAt(w), dx = 0.45;
    var x1 = w - dx, x2 = w + dx;
    var y1 = lossAt(w) - s * dx, y2 = lossAt(w) + s * dx;
    els.tangent.setAttribute('x1', sx(x1));
    els.tangent.setAttribute('x2', sx(x2));
    els.tangent.setAttribute('y1', sy(Math.max(0, y1)));
    els.tangent.setAttribute('y2', sy(Math.max(0, y2)));

    els.traj.setAttribute('points',
      traj.map(function (p) { return sx(p[0]) + ',' + sy(p[1]); }).join(' '));

    var html = '';
    for (var k = 0; k < traj.length - 1; k++) {
      var op = 0.18 + 0.45 * (k / Math.max(1, traj.length - 1));
      html += '<circle cx="' + sx(traj[k][0]).toFixed(2) +
              '" cy="'      + sy(traj[k][1]).toFixed(2) +
              '" r="2.5" fill="#8b3a3a" fill-opacity="' + op.toFixed(2) + '"/>';
    }
    els.trail.innerHTML = html;
  }

  function doStep() {
    var g = gradAt(w);
    if (Math.abs(g) < 1e-4) return false;
    w = w - LR * g;
    if (w < W_MIN) w = W_MIN;
    if (w > W_MAX) w = W_MAX;
    steps++;
    traj.push([w, lossAt(w)]);
    if (traj.length > 80) traj.shift();
    els.slider.value = w;
    refresh();
    return true;
  }

  els.slider.addEventListener('input', function (e) {
    w = parseFloat(e.target.value);
    refresh();
  });
  byId('bp-btn-step').addEventListener('click', doStep);

  var timer = null;
  els.play.addEventListener('click', function () {
    if (timer) {
      clearInterval(timer);
      timer = null;
      els.play.textContent = 'Auto-play';
      els.play.classList.remove('active');
    } else {
      els.play.textContent = 'Pause';
      els.play.classList.add('active');
      timer = setInterval(function () {
        if (!doStep()) {
          clearInterval(timer);
          timer = null;
          els.play.textContent = 'Auto-play';
          els.play.classList.remove('active');
        }
      }, 220);
    }
  });
  byId('bp-btn-reset').addEventListener('click', function () {
    if (timer) {
      clearInterval(timer);
      timer = null;
      els.play.textContent = 'Auto-play';
      els.play.classList.remove('active');
    }
    w = 1; steps = 0; traj = [[1, lossAt(1)]];
    els.slider.value = 1;
    refresh();
  });

  refresh();
}());
</script>

> **Why we used a linear neuron here.** With sigmoid we'd multiply by an extra factor $$\sigma'(a) = o(1-o)$$ between Step 2a and 2b. The arithmetic gets uglier (try it: $$\sigma(2.1) \approx 0.891$$, $$\sigma'(2.1) \approx 0.097$$ → tiny gradients) but the *procedure* is identical. This is one reason modern networks prefer ReLU: its derivative is just $$1$$ for $$z > 0$$, so error signals don't shrink layer by layer.

### Stage 2: One hidden layer (the credit-assignment fix)

Now add a hidden layer. Inputs $$x_i$$ feed hidden units $$h_j$$, which feed
the output $$o$$. Let

- $$v_{ij}$$ = weight from input $$x_i$$ to hidden unit $$h_j$$,
- $$w_j$$ = weight from hidden unit $$h_j$$ to the single output.

Forward pass:

$$
a_j \;=\; \sum_i v_{ij}\, x_i,
\qquad
h_j \;=\; \sigma(a_j),
$$

$$
b \;=\; \sum_j w_j\, h_j,
\qquad
o \;=\; \sigma(b),
\qquad
E \;=\; \tfrac{1}{2}(t-o)^2.
$$

The output weights $$w_j$$ are easy — same as Stage 1 with $$h_j$$ playing
the role of input:

$$
\frac{\partial E}{\partial w_j}
\;=\;
-(t-o)\, o(1-o)\, h_j.
$$

The hard case is $$v_{ij}$$. Its dependency chain branches:

$$
v_{ij} \;\longrightarrow\; a_j \;\longrightarrow\; h_j \;\longrightarrow\; b \;\longrightarrow\; o \;\longrightarrow\; E.
$$

Apply the chain rule along the whole path — five steps multiplied:

$$
\frac{\partial E}{\partial v_{ij}}
\;=\;
\underbrace{\frac{\partial E}{\partial o}}_{=\,-(t-o)}
\cdot
\underbrace{\frac{\partial o}{\partial b}}_{=\,o(1-o)}
\cdot
\underbrace{\frac{\partial b}{\partial h_j}}_{=\,w_j}
\cdot
\underbrace{\frac{\partial h_j}{\partial a_j}}_{=\,h_j(1-h_j)}
\cdot
\underbrace{\frac{\partial a_j}{\partial v_{ij}}}_{=\,x_i}.
$$

Result:

$$
\frac{\partial E}{\partial v_{ij}}
\;=\;
-\,(t-o)\, o(1-o)\, w_j\, h_j(1-h_j)\, x_i.
$$

**Look at the structure.** The first three factors $$-(t-o)\, o(1-o)\, w_j$$
are exactly the error signal *as seen at hidden unit* $$j$$. They are the
error that arrived at $$o$$, multiplied by *how strongly $$h_j$$ was
connected to $$o$$*. The last two factors $$h_j(1-h_j)\, x_i$$ are the local
slope of $$h_j$$ and the input feeding $$v_{ij}$$.

This is the **back-propagation** of error: the output error gets
"transmitted" backward through $$w_j$$ to give an effective error at $$h_j$$,
and that effective error then determines $$v_{ij}$$'s gradient.

### Walk through it: a 1-hidden-layer network, one step at a time

Time to make Stage 2 concrete. Drag the slider below to advance through every step
of one forward + backward pass on the smallest non-trivial network: one input,
**two ReLU hidden units**, one output. Numbers chosen so every gradient is an
integer or a half — no calculator needed.

We use ReLU $$\text{ReLU}(z) = \max(0, z)$$ instead of sigmoid for clean
arithmetic. Its derivative is the simplest possible: $$\text{ReLU}'(z) = 1$$ if
$$z > 0$$, else $$0$$.

<div class="nn-walk" markdown="0">
<svg viewBox="0 0 640 340" xmlns="http://www.w3.org/2000/svg" id="nn-svg" role="img" aria-label="Step-by-step neural network forward and backward pass">
  <defs>
    <marker id="nn-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#5f5f63"/>
    </marker>
  </defs>
  <g id="nn-edges">
    <line id="e-x-h1" class="nn-edge" x1="80"  y1="170" x2="240" y2="100"/>
    <line id="e-x-h2" class="nn-edge" x1="80"  y1="170" x2="240" y2="240"/>
    <line id="e-h1-y" class="nn-edge" x1="240" y1="100" x2="460" y2="170"/>
    <line id="e-h2-y" class="nn-edge" x1="240" y1="240" x2="460" y2="170"/>
    <line id="e-y-e"  class="nn-edge" x1="460" y1="170" x2="560" y2="170"/>
  </g>
  <g font-family="JetBrains Mono, monospace" font-size="11" fill="#5f5f63">
    <text x="145" y="125" text-anchor="middle">v₁ = 1</text>
    <text x="145" y="225" text-anchor="middle">v₂ = −0.5</text>
    <text x="345" y="125" text-anchor="middle">w₁ = 0.5</text>
    <text x="345" y="225" text-anchor="middle">w₂ = 1</text>
  </g>
  <g id="nn-nodes">
    <circle id="n-x"  class="nn-node" cx="80"  cy="170" r="24"/>
    <circle id="n-h1" class="nn-node" cx="240" cy="100" r="24"/>
    <circle id="n-h2" class="nn-node" cx="240" cy="240" r="24"/>
    <circle id="n-y"  class="nn-node" cx="460" cy="170" r="24"/>
    <circle id="n-e"  class="nn-node" cx="560" cy="170" r="24"/>
  </g>
  <g font-family="Source Serif 4, Georgia, serif" font-size="15" fill="#1d1d1f">
    <text x="80"  y="175" text-anchor="middle">x</text>
    <text x="240" y="105" text-anchor="middle">h₁</text>
    <text x="240" y="245" text-anchor="middle">h₂</text>
    <text x="460" y="175" text-anchor="middle">ŷ</text>
    <text x="560" y="175" text-anchor="middle">E</text>
  </g>
  <g font-family="JetBrains Mono, monospace" font-size="11" font-weight="600" fill="#8b3a3a">
    <text id="val-x"  x="80"  y="215" text-anchor="middle"></text>
    <text id="val-h1" x="240" y="65"  text-anchor="middle"></text>
    <text id="val-h2" x="240" y="285" text-anchor="middle"></text>
    <text id="val-y"  x="460" y="215" text-anchor="middle"></text>
    <text id="val-e"  x="560" y="215" text-anchor="middle"></text>
  </g>
  <g id="nn-grad-labels" font-family="JetBrains Mono, monospace" font-size="10" font-weight="600" fill="#8b3a3a" opacity="0">
    <text id="g-w1" x="345" y="155" text-anchor="middle"></text>
    <text id="g-w2" x="345" y="195" text-anchor="middle"></text>
    <text id="g-v1" x="145" y="155" text-anchor="middle"></text>
    <text id="g-v2" x="145" y="195" text-anchor="middle"></text>
  </g>
  <g font-family="Inter, sans-serif" font-size="10" letter-spacing="0.06em" fill="#5f5f63">
    <text x="80"  y="320" text-anchor="middle">INPUT</text>
    <text x="240" y="320" text-anchor="middle">HIDDEN (ReLU)</text>
    <text x="460" y="320" text-anchor="middle">OUTPUT</text>
    <text x="560" y="320" text-anchor="middle">LOSS</text>
  </g>
</svg>

<div class="nn-step-info">
  <div class="nn-step-header">
    <span id="nn-step-num">Step 0 / 17</span>
    <span id="nn-step-phase">Initial state</span>
  </div>
  <div id="nn-step-title" class="nn-step-title">Drag the slider →</div>
  <div id="nn-step-formula" class="nn-step-formula"></div>
  <div id="nn-step-result" class="nn-step-result"></div>
</div>

<div class="nn-controls">
  <button id="nn-prev" type="button" aria-label="previous step">←</button>
  <input type="range" id="nn-slider" min="0" max="17" step="1" value="0" aria-label="step">
  <button id="nn-next" type="button" aria-label="next step">→</button>
</div>
</div>

<script>
(function () {
  var X = 2, Y = 1;
  var V1 = 1, C1 = 0, V2 = -0.5, C2 = 2;
  var W1 = 0.5, W2 = 1, B = 0.5;

  var z1 = V1 * X + C1;
  var h1 = Math.max(0, z1);
  var z2 = V2 * X + C2;
  var h2 = Math.max(0, z2);
  var yhat = W1 * h1 + W2 * h2 + B;
  var E = Math.pow(yhat - Y, 2);

  var dE_dy  = 2 * (yhat - Y);
  var dE_dw1 = dE_dy * h1;
  var dE_dw2 = dE_dy * h2;
  var dE_db  = dE_dy;
  var d1 = W1 * dE_dy * (z1 > 0 ? 1 : 0);
  var d2 = W2 * dE_dy * (z2 > 0 ? 1 : 0);
  var dE_dv1 = d1 * X;
  var dE_dc1 = d1;
  var dE_dv2 = d2 * X;
  var dE_dc2 = d2;

  function f(n) { return (n % 1 === 0) ? n.toString() : n.toFixed(2).replace(/\.?0+$/, ''); }
  function neg(s) { return s.charAt(0) === '-' ? '−' + s.slice(1) : s; }
  function fn(n) { return neg(f(n)); }

  var STEPS = [
    { phase: 'Setup', title: 'Network initialised — drag the slider →',
      formula: 'Inputs: x = 2, target y = 1.   Weights and biases shown on edges/nodes.',
      result: '',
      hl: { nodes: [], edges: [] }, vals: {} },
    // ===== FORWARD =====
    { phase: 'Forward · 1/7', title: 'Read the input',
      formula: 'x = ' + fn(X),
      result: 'x = ' + fn(X),
      hl: { nodes: ['n-x'], edges: [] },
      vals: { x: 'x = ' + fn(X) } },
    { phase: 'Forward · 2/7', title: 'Hidden pre-activation z₁',
      formula: 'z₁ = v₁·x + c₁ = (1)(2) + 0 = ' + fn(z1),
      result: 'z₁ = ' + fn(z1),
      hl: { nodes: ['n-x'], edges: ['e-x-h1'] },
      vals: { x: 'x = ' + fn(X) } },
    { phase: 'Forward · 3/7', title: 'Apply ReLU at h₁',
      formula: 'h₁ = ReLU(z₁) = max(0, ' + fn(z1) + ') = ' + fn(h1),
      result: 'h₁ = ' + fn(h1),
      hl: { nodes: ['n-h1'], edges: [] },
      vals: { x: 'x = ' + fn(X), h1: 'h₁ = ' + fn(h1) } },
    { phase: 'Forward · 4/7', title: 'Hidden pre-activation z₂',
      formula: 'z₂ = v₂·x + c₂ = (−0.5)(2) + 2 = ' + fn(z2),
      result: 'z₂ = ' + fn(z2),
      hl: { nodes: ['n-x'], edges: ['e-x-h2'] },
      vals: { x: 'x = ' + fn(X), h1: 'h₁ = ' + fn(h1) } },
    { phase: 'Forward · 5/7', title: 'Apply ReLU at h₂',
      formula: 'h₂ = ReLU(z₂) = max(0, ' + fn(z2) + ') = ' + fn(h2),
      result: 'h₂ = ' + fn(h2),
      hl: { nodes: ['n-h2'], edges: [] },
      vals: { x: 'x = ' + fn(X), h1: 'h₁ = ' + fn(h1), h2: 'h₂ = ' + fn(h2) } },
    { phase: 'Forward · 6/7', title: 'Combine into the output',
      formula: 'ŷ = w₁·h₁ + w₂·h₂ + b = (0.5)(2) + (1)(1) + 0.5 = ' + fn(yhat),
      result: 'ŷ = ' + fn(yhat),
      hl: { nodes: ['n-y'], edges: ['e-h1-y','e-h2-y'] },
      vals: { x: 'x = ' + fn(X), h1: 'h₁ = ' + fn(h1), h2: 'h₂ = ' + fn(h2), y: 'ŷ = ' + fn(yhat) } },
    { phase: 'Forward · 7/7', title: 'Compute the loss (target y = 1)',
      formula: 'E = (ŷ − y)² = (' + fn(yhat) + ' − 1)² = ' + fn(E),
      result: 'E = ' + fn(E),
      hl: { nodes: ['n-e'], edges: ['e-y-e'] },
      vals: { x: 'x = ' + fn(X), h1: 'h₁ = ' + fn(h1), h2: 'h₂ = ' + fn(h2), y: 'ŷ = ' + fn(yhat), E: 'E = ' + fn(E) } },
    // ===== BACKWARD =====
    { phase: 'Backward · 1/10', title: 'Output gradient',
      formula: '∂E/∂ŷ = 2(ŷ − y) = 2(2.5 − 1) = ' + fn(dE_dy),
      result: '∂E/∂ŷ = ' + fn(dE_dy),
      hl: { nodes: ['n-y','n-e'], edges: ['e-y-e'] },
      vals: { keepAll: true } },
    { phase: 'Backward · 2/10', title: 'Gradient w.r.t. w₁',
      formula: '∂E/∂w₁ = (∂E/∂ŷ)·h₁ = (3)(2) = ' + fn(dE_dw1),
      result: '∂E/∂w₁ = ' + fn(dE_dw1),
      hl: { nodes: ['n-h1','n-y'], edges: ['e-h1-y'] },
      grads: { w1: '∂E/∂w₁ = ' + fn(dE_dw1) },
      vals: { keepAll: true } },
    { phase: 'Backward · 3/10', title: 'Gradient w.r.t. w₂',
      formula: '∂E/∂w₂ = (∂E/∂ŷ)·h₂ = (3)(1) = ' + fn(dE_dw2),
      result: '∂E/∂w₂ = ' + fn(dE_dw2),
      hl: { nodes: ['n-h2','n-y'], edges: ['e-h2-y'] },
      grads: { w1: '∂E/∂w₁ = ' + fn(dE_dw1), w2: '∂E/∂w₂ = ' + fn(dE_dw2) },
      vals: { keepAll: true } },
    { phase: 'Backward · 4/10', title: 'Gradient w.r.t. output bias b',
      formula: '∂E/∂b = ∂E/∂ŷ · 1 = ' + fn(dE_db),
      result: '∂E/∂b = ' + fn(dE_db),
      hl: { nodes: ['n-y'], edges: [] },
      grads: { w1: '∂E/∂w₁ = ' + fn(dE_dw1), w2: '∂E/∂w₂ = ' + fn(dE_dw2) },
      vals: { keepAll: true } },
    { phase: 'Backward · 5/10', title: 'Hidden error δ₁ at h₁',
      formula: 'δ₁ = w₁·(∂E/∂ŷ)·ReLU′(z₁) = (0.5)(3)(1) = ' + fn(d1),
      result: 'δ₁ = ' + fn(d1),
      hl: { nodes: ['n-h1'], edges: ['e-h1-y'] },
      grads: { w1: '∂E/∂w₁ = ' + fn(dE_dw1), w2: '∂E/∂w₂ = ' + fn(dE_dw2) },
      vals: { keepAll: true } },
    { phase: 'Backward · 6/10', title: 'Hidden error δ₂ at h₂',
      formula: 'δ₂ = w₂·(∂E/∂ŷ)·ReLU′(z₂) = (1)(3)(1) = ' + fn(d2),
      result: 'δ₂ = ' + fn(d2),
      hl: { nodes: ['n-h2'], edges: ['e-h2-y'] },
      grads: { w1: '∂E/∂w₁ = ' + fn(dE_dw1), w2: '∂E/∂w₂ = ' + fn(dE_dw2) },
      vals: { keepAll: true } },
    { phase: 'Backward · 7/10', title: 'Gradient w.r.t. v₁',
      formula: '∂E/∂v₁ = δ₁·x = (1.5)(2) = ' + fn(dE_dv1),
      result: '∂E/∂v₁ = ' + fn(dE_dv1),
      hl: { nodes: ['n-x','n-h1'], edges: ['e-x-h1'] },
      grads: { w1: '∂E/∂w₁ = ' + fn(dE_dw1), w2: '∂E/∂w₂ = ' + fn(dE_dw2), v1: '∂E/∂v₁ = ' + fn(dE_dv1) },
      vals: { keepAll: true } },
    { phase: 'Backward · 8/10', title: 'Gradient w.r.t. c₁',
      formula: '∂E/∂c₁ = δ₁ · 1 = ' + fn(dE_dc1),
      result: '∂E/∂c₁ = ' + fn(dE_dc1),
      hl: { nodes: ['n-h1'], edges: [] },
      grads: { w1: '∂E/∂w₁ = ' + fn(dE_dw1), w2: '∂E/∂w₂ = ' + fn(dE_dw2), v1: '∂E/∂v₁ = ' + fn(dE_dv1) },
      vals: { keepAll: true } },
    { phase: 'Backward · 9/10', title: 'Gradient w.r.t. v₂',
      formula: '∂E/∂v₂ = δ₂·x = (3)(2) = ' + fn(dE_dv2),
      result: '∂E/∂v₂ = ' + fn(dE_dv2),
      hl: { nodes: ['n-x','n-h2'], edges: ['e-x-h2'] },
      grads: { w1: '∂E/∂w₁ = ' + fn(dE_dw1), w2: '∂E/∂w₂ = ' + fn(dE_dw2), v1: '∂E/∂v₁ = ' + fn(dE_dv1), v2: '∂E/∂v₂ = ' + fn(dE_dv2) },
      vals: { keepAll: true } },
    { phase: 'Backward · 10/10', title: 'Gradient w.r.t. c₂ — done!',
      formula: '∂E/∂c₂ = δ₂ · 1 = ' + fn(dE_dc2),
      result: 'All six parameter gradients computed. Ready to update.',
      hl: { nodes: ['n-h2'], edges: [] },
      grads: { w1: '∂E/∂w₁ = ' + fn(dE_dw1), w2: '∂E/∂w₂ = ' + fn(dE_dw2), v1: '∂E/∂v₁ = ' + fn(dE_dv1), v2: '∂E/∂v₂ = ' + fn(dE_dv2) },
      vals: { keepAll: true } }
  ];

  function gid(id){ return document.getElementById(id); }
  var slider = gid('nn-slider'), prev = gid('nn-prev'), next = gid('nn-next');
  slider.max = STEPS.length - 1;

  var FULL_VALS = {
    x:  'x = '  + fn(X),
    h1: 'h₁ = ' + fn(h1),
    h2: 'h₂ = ' + fn(h2),
    y:  'ŷ = '  + fn(yhat),
    E:  'E = '  + fn(E)
  };

  function apply(i) {
    var s = STEPS[i];
    gid('nn-step-num').textContent     = 'Step ' + i + ' / ' + (STEPS.length - 1);
    gid('nn-step-phase').textContent   = s.phase;
    gid('nn-step-title').textContent   = s.title;
    gid('nn-step-formula').textContent = s.formula || '';
    gid('nn-step-result').textContent  = s.result  || '';

    // reset highlights
    var nodes = document.querySelectorAll('.nn-node');
    for (var k = 0; k < nodes.length; k++) nodes[k].classList.remove('active');
    var edges = document.querySelectorAll('.nn-edge');
    for (var k = 0; k < edges.length; k++) edges[k].classList.remove('active');

    s.hl.nodes.forEach(function (id) { gid(id).classList.add('active'); });
    s.hl.edges.forEach(function (id) { gid(id).classList.add('active'); });

    var v = s.vals.keepAll ? FULL_VALS : s.vals;
    gid('val-x').textContent  = v.x  || '';
    gid('val-h1').textContent = v.h1 || '';
    gid('val-h2').textContent = v.h2 || '';
    gid('val-y').textContent  = v.y  || '';
    gid('val-e').textContent  = v.E  || '';

    var g = s.grads || {};
    gid('g-w1').textContent = g.w1 || '';
    gid('g-w2').textContent = g.w2 || '';
    gid('g-v1').textContent = g.v1 || '';
    gid('g-v2').textContent = g.v2 || '';
    gid('nn-grad-labels').setAttribute('opacity', Object.keys(g).length ? '1' : '0');
  }

  slider.addEventListener('input', function (e) { apply(parseInt(e.target.value, 10)); });
  prev.addEventListener('click', function () {
    var i = Math.max(0, parseInt(slider.value, 10) - 1);
    slider.value = i; apply(i);
  });
  next.addEventListener('click', function () {
    var i = Math.min(STEPS.length - 1, parseInt(slider.value, 10) + 1);
    slider.value = i; apply(i);
  });

  apply(0);
}());
</script>

### Stage 3: Many layers — the general recursion

To avoid the bookkeeping nightmare of writing out every chain, we define the
**error signal** at every unit:

$$
\delta_j^{(\ell)} \;\equiv\; \frac{\partial E}{\partial a_j^{(\ell)}}.
$$

Read this as: *"how much would the loss change if I nudged the raw input
$$a_j^{(\ell)}$$ of unit $$j$$ in layer $$\ell$$ by a tiny amount?"*

Once we have $$\delta_j^{(\ell)}$$ for every unit, the gradient with respect
to any weight is just one multiplication:

$$
\boxed{\;\;
\frac{\partial E}{\partial w_{ij}^{(\ell)}}
\;=\;
\delta_j^{(\ell)} \, o_i^{(\ell-1)}
\;\;}
$$

**Plain-English reading:** *"weight $$w_{ij}^{(\ell)}$$'s share of the blame
= (error signal at the unit it feeds) × (the value flowing in from the
previous layer)."*

The remaining task is to compute $$\delta$$ at every unit. We do it in two
steps.

**Output layer (start of the recursion).** The error signal is just the
direct loss derivative, and the chain rule gives

$$
\delta_j^{(L)} \;=\; \frac{\partial E}{\partial o_j^{(L)}} \cdot \sigma'\!\left(a_j^{(L)}\right)
\;=\; -\bigl(t_j - o_j^{(L)}\bigr)\, \sigma'\!\left(a_j^{(L)}\right).
$$

**Hidden layer (the recursion).** Hidden unit $$j$$ in layer $$\ell$$ feeds
*every* unit $$k$$ in layer $$\ell+1$$. The chain rule with branches says we
sum the contributions from each branch:

$$
\boxed{\;\;
\delta_j^{(\ell)}
\;=\;
\sigma'\!\left(a_j^{(\ell)}\right)\,
\sum_k w_{jk}^{(\ell+1)}\, \delta_k^{(\ell+1)}
\;\;}
$$

**Plain-English reading:** *"the error signal at hidden unit $$j$$ is the
weighted sum of the error signals from the next layer (gathered through the
weights $$w_{jk}^{(\ell+1)}$$ that connect $$j$$ to those next-layer units),
multiplied by the local slope $$\sigma'$$ of $$j$$ itself."*

Notice that this lets us compute $$\delta$$ layer by layer **from the output
back toward the input** — that is the "back-propagation" in the name.

### The full algorithm

Putting it all together, training proceeds by repeating:

1. **Forward pass.** Push $$x$$ through the network, recording every $$a^{(\ell)}$$ and $$o^{(\ell)}$$.
2. **Output error.** Compute $$\delta^{(L)}$$ from the targets.
3. **Backward pass.** For $$\ell = L-1, L-2, \dots, 1$$, apply the recursion to obtain $$\delta^{(\ell)}$$.
4. **Weight gradients.** For every weight, $$\dfrac{\partial E}{\partial w_{ij}^{(\ell)}} = \delta_j^{(\ell)}\, o_i^{(\ell-1)}$$.
5. **Update.** $$w_{ij}^{(\ell)} \leftarrow w_{ij}^{(\ell)} - \eta\, \delta_j^{(\ell)}\, o_i^{(\ell-1)}$$.
6. **Repeat** over training examples until convergence.

The paper additionally adds a **momentum** term, which carries a fraction of
the previous step into the next:

$$
\Delta w_{ij}^{(\ell)}(t)
\;=\;
-\eta\, \delta_j^{(\ell)}\, o_i^{(\ell-1)}
\;+\;
\alpha\, \Delta w_{ij}^{(\ell)}(t-1).
$$

Intuitively: if we keep stepping in roughly the same direction, momentum
accelerates us; if we keep oscillating back and forth, the previous step
cancels part of the new step and we stop wobbling.

---

## Key Results in the Paper

- **XOR.** A 2-2-1 network learns XOR — the smallest demonstration that hidden units can build a representation a perceptron cannot.
- **Symmetry detection.** A network with two hidden units learns to detect left/right mirror symmetry; the trained weights are interpretable feature detectors.
- **Family-tree relations.** Trained on relations among two artificial families, the network generalizes to held-out (person, relation, person) triples, and the hidden units develop *semantic* features (nationality, generation, branch) that were never given as supervision. This is the seed of the modern idea of "learned representations."

---

## Why It Matters / Limitations

- **Why it matters.** Backprop is the workhorse of every modern deep network. CNNs, RNNs, Transformers, diffusion models — all of them run the same chain-rule recursion at scale. Once you understand the math above, you understand the *core* of how every deep model on the planet is trained.
- **Limitations of the original.**
  - *Vanishing / exploding gradients.* When you multiply many sigmoids' derivatives ($$\sigma'(z) \le 1/4$$ everywhere), the error signal shrinks exponentially with depth. Tiny networks dodge this; deep networks needed ReLU + careful initialization + normalization.
  - *Local minima concerns.* The 1986 paper worried about getting stuck in bad local minima. Modern theory says local minima are rarely the obstacle in high dimensions; saddle points and plateaus matter more — and SGD's stochasticity helps escape both.
  - *Plain SGD on sigmoids.* Replaced in modern practice by ReLU activations, adaptive optimizers (Adam), batch / layer normalization, and skip connections. The *algorithm* (backprop) is unchanged; everything around it has been upgraded.

---

## Reading the Original

- [Original PDF (Nature, vol. 323, pp. 533–536, 1986)](https://gwern.net/doc/ai/nn/1986-rumelhart-2.pdf)

### Suggested Companion Reads

- [**Backprop Explainer**](https://xnought.github.io/backprop-explainer/) (Bertucci & Kahng, VISxAI 2021) — interactive in-browser visualization. Drag a slider to see how nudging a weight changes the loss; watch the gradient be computed and applied live. Highly recommended after this note.
- Werbos (1974) — the earliest formulation, in his PhD thesis.
- LeCun (1985) — independent rediscovery, formulated as a Lagrangian.
- Bishop, *Pattern Recognition and Machine Learning*, §5.3 — a clean modern matrix-notation treatment.
- 3Blue1Brown's "Neural Networks" YouTube series — the same derivation animated.
