---
layout: page
title: "Dropout (Srivastava et al., 2014)"
description: "Reading note · Pattern Recognition · Neural Network Foundations"
back_link: '/teaching/pattern/'
back_text: "Back to Pattern Recognition"
---

## TL;DR

A big neural network has so many parameters that it can memorize the training set even when there is no real signal. **Dropout** prevents that by, on every forward pass during training, *randomly turning off* a fraction of the hidden units — they output zero, and no gradient flows through them. So the network is forced to learn features that are useful **even when its neighbours go missing**, and at test time we recover the full network at almost no extra cost.

That single sentence is what every student should walk away with. The rest of this note unpacks the math and shows two small interactive demos so you can build intuition before reading the paper.

---

## Problem &amp; Motivation

A neural network with many parameters is a powerful function approximator. The flip side is that its capacity often exceeds what the training data can support. The result is **overfitting**: low training error, high test error.

Classical defences:

- **Early stopping** — stop training when validation error rises.
- **L\\(\_2\\) weight decay** — add \\(\frac{\lambda}{2}\\\|\mathbf{w}\\\|^2\\) to the loss.
- **Data augmentation** — generate more training points by transforming what you already have.
- **Model averaging** — train several models, average their predictions. Reduces variance, but training and storing \\(K\\) networks is \\(K\\) times the cost.

Srivastava and co-authors point out a deeper issue particular to deep nets: **co-adaptation**. Neurons learn to rely on the *exact* presence of certain other neurons. A unit might encode "feature \\(A\\) **and** feature \\(B\\) are both present" only because some downstream unit needs the conjunction. If \\(A\\) or \\(B\\) misfires, the whole circuit collapses. Co-adapted detectors are great on training data and brittle outside of it.

Dropout's premise is that **forcing every unit to function in the absence of any other unit** breaks this fragile co-adaptation. At the same time, training one network with dropout is approximately equivalent to averaging an exponentially large *ensemble* of "thinned" networks — without the expense of ever instantiating them.

---

## Setting &amp; Notation

A standard feed-forward layer computes

$$
\mathbf{z}^{(\ell+1)} = \mathbf{W}^{(\ell+1)}\, \mathbf{y}^{(\ell)} + \mathbf{b}^{(\ell+1)},
\qquad
\mathbf{y}^{(\ell+1)} = f(\mathbf{z}^{(\ell+1)}),
$$

with \\(\mathbf{y}^{(\ell)}\\) the activations of layer \\(\ell\\) and \\(f\\) a non-linearity (sigmoid, ReLU, ...).

Dropout introduces a binary **mask** \\(\mathbf{r}^{(\ell)} \in \\{0,1\\}^{n\_\ell}\\), one entry per unit, drawn independently each forward pass:

$$
r_j^{(\ell)} \sim \operatorname{Bernoulli}(p_\ell),
\qquad
\widetilde{\mathbf{y}}^{(\ell)} = \mathbf{r}^{(\ell)} \odot \mathbf{y}^{(\ell)}.
\tag{1}
$$

Here \\(\odot\\) is element-wise multiplication, and \\(p\_\ell\\) is the **keep probability** for layer \\(\ell\\) (often called the *retention rate*; the *drop rate* is \\(1 - p\_\ell\\)). The thinned activations \\(\widetilde{\mathbf{y}}^{(\ell)}\\) are what the next layer's weighted sum sees. Backpropagation runs exactly as before, except gradient updates only flow through units that survived the mask.

Convention: \\(p\_\ell = 0.5\\) is typical for hidden layers; \\(p\_\ell\\) closer to \\(1\\) (often \\(0.8\\)) is used for input layers because raw inputs are precious.

---

## Interactive Demo 1 — Sampling a Thinned Network

A small fully-connected network: 4 input units → 6 hidden units → 3 output units. Move the *drop rate* slider, then press **Sample mask** (or **Auto-play**) and watch the network re-thin itself. Each sample is a different sub-network that the optimizer has to make work.

<div class="bp-demo" markdown="0">
<svg viewBox="0 0 520 240" id="dropout-net" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Small neural network with units randomly dropped">
  <g class="nn-edges" stroke-linecap="round"></g>
  <g class="nn-nodes"></g>
  <g class="nn-labels" font-family="Inter, system-ui, sans-serif" font-size="11" fill="#5f5f63"></g>
</svg>

<div class="bp-readout" id="dropout-readout">
  <div>drop rate &nbsp;<span id="dropout-p">0.50</span></div>
  <div>hidden kept &nbsp;<span id="dropout-kept">— / 6</span></div>
  <div>samples &nbsp;<span id="dropout-count">0</span></div>
  <div>last seed &nbsp;<span id="dropout-seed">—</span></div>
</div>

<div class="bp-controls">
  <label style="font-family:var(--sans);font-size:0.86rem;color:var(--muted);min-width:78px;">drop rate</label>
  <input id="dropout-slider" type="range" min="0" max="0.9" step="0.05" value="0.5">
  <button id="dropout-sample" type="button">Sample mask</button>
  <button id="dropout-play"   type="button">Auto-play</button>
  <button id="dropout-reset"  type="button">Show full</button>
</div>
</div>

<script>
(function () {
  var svg = document.getElementById('dropout-net');
  if (!svg) return;
  var NS = 'http://www.w3.org/2000/svg';
  var edgesG = svg.querySelector('.nn-edges');
  var nodesG = svg.querySelector('.nn-nodes');
  var labelsG = svg.querySelector('.nn-labels');

  // Network shape
  var layers = [
    { n: 4, x: 70,  yTop: 30, yGap: 50, label: 'input'  },
    { n: 6, x: 250, yTop: 25, yGap: 36, label: 'hidden' },
    { n: 3, x: 430, yTop: 60, yGap: 60, label: 'output' },
  ];

  // Build positions
  var positions = layers.map(function (L) {
    return Array.from({ length: L.n }, function (_, i) {
      return { x: L.x, y: L.yTop + i * L.yGap };
    });
  });

  // Edges (between consecutive layers)
  var edges = [];
  for (var l = 0; l < layers.length - 1; l++) {
    var a = positions[l], b = positions[l + 1];
    for (var i = 0; i < a.length; i++) for (var j = 0; j < b.length; j++) {
      var line = document.createElementNS(NS, 'line');
      line.setAttribute('x1', a[i].x); line.setAttribute('y1', a[i].y);
      line.setAttribute('x2', b[j].x); line.setAttribute('y2', b[j].y);
      line.setAttribute('class', 'nn-edge');
      line.dataset.from = l + ':' + i;
      line.dataset.to   = (l + 1) + ':' + j;
      edgesG.appendChild(line);
      edges.push(line);
    }
  }

  // Nodes
  var nodes = [];
  for (var l = 0; l < layers.length; l++) {
    nodes[l] = [];
    for (var i = 0; i < layers[l].n; i++) {
      var c = document.createElementNS(NS, 'circle');
      c.setAttribute('cx', positions[l][i].x);
      c.setAttribute('cy', positions[l][i].y);
      c.setAttribute('r', 16);
      c.setAttribute('class', 'nn-node');
      nodesG.appendChild(c);
      nodes[l][i] = c;
    }
    var lbl = document.createElementNS(NS, 'text');
    lbl.setAttribute('x', layers[l].x);
    lbl.setAttribute('y', 225);
    lbl.setAttribute('text-anchor', 'middle');
    lbl.textContent = layers[l].label;
    labelsG.appendChild(lbl);
  }

  // Dropped (X mark) overlay group
  var droppedG = document.createElementNS(NS, 'g');
  droppedG.setAttribute('stroke', '#8b3a3a');
  droppedG.setAttribute('stroke-width', '2');
  droppedG.setAttribute('stroke-linecap', 'round');
  svg.appendChild(droppedG);

  // State
  var rate = 0.5;
  var sampleCount = 0;
  var seed = 0;
  var playTimer = null;

  // Tiny seedable RNG (Mulberry32)
  function mulberry32(s) {
    return function () {
      s = (s + 0x6D2B79F5) | 0;
      var t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function clearDrops() {
    while (droppedG.firstChild) droppedG.removeChild(droppedG.firstChild);
    nodes.forEach(function (layerNodes) {
      layerNodes.forEach(function (n) {
        n.style.opacity = '';
        n.classList.remove('dropped');
      });
    });
    edges.forEach(function (e) {
      e.style.opacity = '';
    });
  }

  function drawDropX(node) {
    var cx = +node.getAttribute('cx'), cy = +node.getAttribute('cy'), r = +node.getAttribute('r');
    var d = r * 0.65;
    [[cx - d, cy - d, cx + d, cy + d], [cx - d, cy + d, cx + d, cy - d]].forEach(function (pts) {
      var l = document.createElementNS(NS, 'line');
      l.setAttribute('x1', pts[0]); l.setAttribute('y1', pts[1]);
      l.setAttribute('x2', pts[2]); l.setAttribute('y2', pts[3]);
      droppedG.appendChild(l);
    });
  }

  function applyMask(rng) {
    clearDrops();
    // Drop hidden layer (index 1) units only, like in standard dropout examples.
    var dropped = [];
    for (var i = 0; i < nodes[1].length; i++) {
      if (rng() < rate) dropped.push(i);
    }
    // If everything was dropped, keep at least one (avoid degenerate display).
    if (dropped.length === nodes[1].length) dropped.pop();

    dropped.forEach(function (i) {
      var n = nodes[1][i];
      n.style.opacity = '0.18';
      n.classList.add('dropped');
      drawDropX(n);
    });
    var droppedSet = new Set(dropped.map(function (i) { return '1:' + i; }));
    edges.forEach(function (e) {
      if (droppedSet.has(e.dataset.from) || droppedSet.has(e.dataset.to)) {
        e.style.opacity = '0.12';
      }
    });

    sampleCount += 1;
    document.getElementById('dropout-kept').textContent = (nodes[1].length - dropped.length) + ' / ' + nodes[1].length;
    document.getElementById('dropout-count').textContent = sampleCount;
  }

  function sample() {
    seed = (Math.random() * 0xffffffff) >>> 0;
    document.getElementById('dropout-seed').textContent = seed.toString(16).padStart(8, '0');
    applyMask(mulberry32(seed));
  }

  function reset() {
    clearDrops();
    sampleCount = 0;
    document.getElementById('dropout-kept').textContent = nodes[1].length + ' / ' + nodes[1].length;
    document.getElementById('dropout-count').textContent = sampleCount;
    document.getElementById('dropout-seed').textContent = '—';
  }

  // Wire up
  var slider  = document.getElementById('dropout-slider');
  var pSpan   = document.getElementById('dropout-p');
  slider.addEventListener('input', function () {
    rate = parseFloat(slider.value);
    pSpan.textContent = rate.toFixed(2);
  });
  document.getElementById('dropout-sample').addEventListener('click', sample);
  document.getElementById('dropout-reset').addEventListener('click', reset);

  var playBtn = document.getElementById('dropout-play');
  playBtn.addEventListener('click', function () {
    if (playTimer) {
      clearInterval(playTimer);
      playTimer = null;
      playBtn.classList.remove('active');
      playBtn.textContent = 'Auto-play';
    } else {
      playTimer = setInterval(sample, 700);
      playBtn.classList.add('active');
      playBtn.textContent = 'Stop';
    }
  });

  reset();
})();
</script>

A few things to notice as you play:

- Each sample is *not* the same network. The optimizer is being asked to make the *expected* loss small over a distribution of subnetworks.
- Cranking the drop rate up to 0.9 makes the surviving network so tiny it can barely fit the data — too much regularization. Crank it down to 0.0 and you recover the original network with no regularization.
- The exponentially many subnetworks share weights — that's why the cost stays at a single network's worth of memory.

---

## What Dropout Computes — Math

### Forward pass with dropout

For one mask realization \\(\mathbf{r}\\), the layer-\\(\ell\\) computation becomes

$$
\widetilde{\mathbf{y}}^{(\ell)} = \mathbf{r}^{(\ell)} \odot \mathbf{y}^{(\ell)},
\qquad
\mathbf{z}^{(\ell+1)} = \mathbf{W}^{(\ell+1)}\, \widetilde{\mathbf{y}}^{(\ell)} + \mathbf{b}^{(\ell+1)}.
\tag{2}
$$

If you back-propagate through (2), the gradient with respect to \\(W\_{ij}^{(\ell+1)}\\) picks up a factor \\(r\_j^{(\ell)}\\). Concretely, units that were **dropped contribute zero gradient** for that step. They are momentarily invisible.

### Test time — the weight-scaling rule

At test time we want a single deterministic prediction. Two equivalent options:

1. **Monte Carlo estimate.** Average the network's output over many random masks: \\(\widehat{y} = \tfrac{1}{T}\sum\_{t=1}^{T} y(\mathbf{x};\, \mathbf{r}^{(t)})\\). Accurate as \\(T \to \infty\\) but expensive.
2. **Weight scaling.** Use the **full** network with no dropout, but multiply each weight by the keep probability \\(p\_\ell\\) of its source layer — equivalently, multiply each layer's activations by \\(p\_\ell\\) at test time.

Why are they equivalent (in expectation, for a *linear* layer)? Because

$$
\mathbb{E}_{\mathbf{r}}\\!\left[\mathbf{W}^{(\ell+1)}\, (\mathbf{r}^{(\ell)} \odot \mathbf{y}^{(\ell)})\right]
  = \mathbf{W}^{(\ell+1)}\, \mathbb{E}_{\mathbf{r}}[\mathbf{r}^{(\ell)}] \odot \mathbf{y}^{(\ell)}
  = \mathbf{W}^{(\ell+1)}\, p_\ell\, \mathbf{y}^{(\ell)}.
\tag{3}
$$

For non-linear activations, this is no longer exact — but Srivastava et al. show empirically that the weight-scaling rule is an excellent approximation across many architectures and datasets, with the bonus that test-time inference becomes a single forward pass through the original network.

In modern frameworks (PyTorch's `nn.Dropout`, TensorFlow's `keras.layers.Dropout`), the convention is to **scale during training** instead: divide kept activations by \\(p\_\ell\\) so the *expected* activation matches the no-dropout case. This is called *inverted dropout* and lets test-time code run with no special handling at all — `model.eval()` just disables the random masking.

---

## Interactive Demo 2 — Train Average Converges to Test

Equation (3) says: *if you average the training-time outputs over enough mask samples, you get the deterministic test-time output.* This demo shows that convergence happen in real time.

The setup is intentionally tiny — one weight, one input, one Bernoulli mask. With \\(w = 2\\), \\(x = 1\\), keep probability \\(p\\), the train-time output is

$$
y^{\text{train}} = r \cdot w \cdot x, \qquad r \sim \operatorname{Bernoulli}(p),
$$

so each sample is either **\\(0\\)** (the mask dropped this unit, prob \\(1-p\\)) or **\\(w \cdot x = 2\\)** (kept, prob \\(p\\)). The deterministic test-time output is \\(p \cdot w \cdot x = 2p\\) — the dashed grey line.

Click **Sample** to add one mask draw; click **Auto-play** to add ~10 per second. The blue dots are individual draws (clamped at \\(0\\) and \\(2\\)); the red curve is the running average so far. Watch it settle onto the dashed line.

<div class="bp-demo" markdown="0">
<svg viewBox="0 0 480 240" id="dropout-eq" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Running average of dropout samples converging to the test-time scaled value">
  <!-- Plot frame -->
  <rect x="50" y="20" width="410" height="180" fill="none" stroke="#e5e2dc" stroke-width="1"/>

  <!-- Y axis labels (output value, 0 to 2) -->
  <text x="44" y="200" text-anchor="end" font-family="Inter, system-ui, sans-serif" font-size="11" fill="#5f5f63">0</text>
  <text x="44" y="113" text-anchor="end" font-family="Inter, system-ui, sans-serif" font-size="11" fill="#5f5f63">1</text>
  <text x="44" y="26"  text-anchor="end" font-family="Inter, system-ui, sans-serif" font-size="11" fill="#5f5f63">2</text>
  <text x="22" y="115" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="11" fill="#5f5f63" transform="rotate(-90, 22, 115)">output</text>

  <!-- X axis -->
  <text x="50"  y="218" text-anchor="start" font-family="Inter, system-ui, sans-serif" font-size="11" fill="#5f5f63">1</text>
  <text x="460" y="218" text-anchor="end"   font-family="Inter, system-ui, sans-serif" font-size="11" fill="#5f5f63" id="eq-x-max">500</text>
  <text x="255" y="232" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="11" fill="#5f5f63">samples</text>

  <!-- Test-time scaled value (horizontal dashed line) -->
  <line id="eq-test-line" x1="50" y1="105" x2="460" y2="105" stroke="#5f5f63" stroke-width="1.4" stroke-dasharray="6 4"/>
  <text id="eq-test-label" x="464" y="108" text-anchor="start" font-family="Inter, system-ui, sans-serif" font-size="11" fill="#5f5f63">p·w·x</text>

  <!-- Sample dots and running-mean polyline get drawn into these groups -->
  <g id="eq-dots" fill="#3a6d8b" opacity="0.55"></g>
  <polyline id="eq-mean" fill="none" stroke="#8b3a3a" stroke-width="2"/>
</svg>

<div class="bp-readout">
  <div>drop rate &nbsp;<span id="eq-p">0.50</span></div>
  <div>samples T &nbsp;<span id="eq-T">0</span></div>
  <div>running mean &nbsp;<span id="eq-mean-val">—</span></div>
  <div>test target &nbsp;<span id="eq-test-val">1.000</span></div>
</div>

<div class="bp-controls">
  <label style="font-family:var(--sans);font-size:0.86rem;color:var(--muted);min-width:78px;">drop rate</label>
  <input id="eq-slider-p" type="range" min="0" max="0.9" step="0.05" value="0.5">
  <button id="eq-step"  type="button">Sample</button>
  <button id="eq-play"  type="button">Auto-play</button>
  <button id="eq-reset" type="button">Reset</button>
</div>
</div>

<script>
(function () {
  var sliderP = document.getElementById('eq-slider-p');
  if (!sliderP) return;
  var pSpan        = document.getElementById('eq-p');
  var tSpan        = document.getElementById('eq-T');
  var meanSpan     = document.getElementById('eq-mean-val');
  var testValSpan  = document.getElementById('eq-test-val');
  var testLine     = document.getElementById('eq-test-line');
  var testLabel    = document.getElementById('eq-test-label');
  var dotsG        = document.getElementById('eq-dots');
  var meanLine     = document.getElementById('eq-mean');
  var stepBtn      = document.getElementById('eq-step');
  var playBtn      = document.getElementById('eq-play');
  var resetBtn     = document.getElementById('eq-reset');
  var xMaxLabel    = document.getElementById('eq-x-max');

  // Plot bounds in SVG coords.
  var X0 = 50, X1 = 460, Y0 = 200, Y1 = 20;     // y inverted: Y0 = output 0, Y1 = output 2
  var T_MAX = 500;                              // x scale: 1 .. T_MAX

  var w = 2.0, x = 1.0, fullOut = w * x;        // bound output to [0, 2]

  function xPos(t) { return X0 + (t - 1) / (T_MAX - 1) * (X1 - X0); }
  function yPos(v) { return Y0 - (v / fullOut) * (Y0 - Y1); }

  var rate = parseFloat(sliderP.value);
  var samples = [];                             // raw outputs (each 0 or w*x)
  var sum = 0;
  var meanPath = '';
  var playTimer = null;

  function setTestLine() {
    var v = rate * w * x;
    var y = yPos(v);
    testLine.setAttribute('y1', y);
    testLine.setAttribute('y2', y);
    testLabel.setAttribute('y', y + 4);
    testValSpan.textContent = v.toFixed(3);
  }

  function reset() {
    samples = [];
    sum = 0;
    meanPath = '';
    while (dotsG.firstChild) dotsG.removeChild(dotsG.firstChild);
    meanLine.setAttribute('points', '');
    tSpan.textContent = '0';
    meanSpan.textContent = '—';
  }

  function addSample() {
    if (samples.length >= T_MAX) return;
    var keep = Math.random() < rate ? 1 : 0;
    var out  = keep * w * x;
    samples.push(out);
    sum += out;
    var t = samples.length;
    var mean = sum / t;

    // Dot for this draw. Slight horizontal jitter for visibility on stacks at 0 / 2.
    var cx = xPos(t);
    var cy = yPos(out);
    var dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('cx', cx);
    dot.setAttribute('cy', cy);
    dot.setAttribute('r', 2.2);
    dotsG.appendChild(dot);

    // Extend the running-mean polyline.
    meanPath += (meanPath ? ' ' : '') + cx.toFixed(1) + ',' + yPos(mean).toFixed(1);
    meanLine.setAttribute('points', meanPath);

    tSpan.textContent = t;
    meanSpan.textContent = mean.toFixed(3);
  }

  function setRate(v) {
    rate = v;
    pSpan.textContent = rate.toFixed(2);
    setTestLine();
  }

  sliderP.addEventListener('input', function () {
    setRate(parseFloat(sliderP.value));
    reset();   // changing rate restarts the experiment
  });
  stepBtn.addEventListener('click', addSample);
  resetBtn.addEventListener('click', reset);
  playBtn.addEventListener('click', function () {
    if (playTimer) {
      clearInterval(playTimer);
      playTimer = null;
      playBtn.classList.remove('active');
      playBtn.textContent = 'Auto-play';
    } else {
      playTimer = setInterval(function () {
        if (samples.length >= T_MAX) {
          clearInterval(playTimer);
          playTimer = null;
          playBtn.classList.remove('active');
          playBtn.textContent = 'Auto-play';
        } else {
          addSample();
        }
      }, 100);
      playBtn.classList.add('active');
      playBtn.textContent = 'Stop';
    }
  });

  xMaxLabel.textContent = T_MAX;
  setRate(rate);
  reset();
})();
</script>

What you should see while playing:

- Every blue dot lands at exactly **\\(0\\)** or **\\(2\\)**. Individual mask draws are extreme — there's no in-between.
- The red running-mean curve starts noisy (one or two samples gives huge swings) and tightens as more samples come in. Statistically, the std-dev of the running mean shrinks as \\(1/\sqrt{T}\\).
- That curve **converges to the dashed grey line**, which is the test-time scaled value \\(p \cdot w \cdot x\\). Click **Reset** and try a different drop rate — the line jumps to the new \\(p \cdot w \cdot x\\) and the convergence story repeats.

That's equation (3) in pictures: averaging over the random masks during training is, in expectation, the same as the deterministic weight-scaled forward pass at test time — which is why deployed networks pay no extra cost for using dropout during training.

---

## The Ensemble View

A network with \\(N\\) hidden units has \\(2^N\\) possible binary masks — i.e. \\(2^N\\) different "thinned" subnetworks. Training with dropout is equivalent to **simultaneously training all \\(2^N\\) of them with shared weights**, where each subnetwork is sampled with probability proportional to the product of its mask's Bernoulli probabilities.

At test time, exact ensembling would require summing the predictions of all \\(2^N\\) subnetworks — infeasible. The weight-scaling rule (Section "Test time" above) gives a closed-form approximation that is exact for linear models and remarkably close in practice for non-linear ones.

This ensemble interpretation is why dropout is often described as "model averaging without the cost". The catch is that the ensembles are highly *correlated* (they all share the same weight matrices), so the variance reduction is less than \\(K\\) independent models would give. But it is free relative to actually training and storing \\(K\\) networks.

---

## Hyperparameters &amp; Practical Choices

The paper's recommendations, which still hold up:

| Layer | Typical keep \\(p\\) | Notes |
|---|---|---|
| Input | \\(0.8\\) | Inputs are precious; drop too many and the network can't learn. |
| Hidden | \\(0.5\\) | The default. \\(0.5\\) is also the maximum-entropy choice — most regularization per parameter. |
| Convolutional feature maps | \\(0.7\\)–\\(0.9\\) | Conv layers already share parameters and are less over-parameterized; less aggressive dropout. |
| Batch-normalized layers | (often skip) | BN already provides a strong regularizing signal; combining the two can hurt. |

Other practical points:

- **Train longer.** Networks with dropout typically need 2–3× more epochs to converge, because the optimizer is solving a noisier problem.
- **Higher learning rates / momentum.** Effective in cancelling the optimization noise. Srivastava et al. recommend \\(10\times\\) or larger LRs than the no-dropout setup.
- **Max-norm weight constraint.** Constrain \\(\\\|\mathbf{w}\_j\\\|\_2 \leq c\\) per unit. Together with high LR + dropout, gives the configuration the paper found most effective.

---

## Empirical Results From the Paper

A summary of headline numbers (from the original 2014 paper, MNIST and CIFAR-10):

- **MNIST.** A standard fully-connected network drops from \\(\sim 1.6\%\\) test error (no dropout) to \\(\sim 1.05\%\\) with dropout. Convnets with dropout reach \\(\sim 0.79\%\\) — a significant relative improvement.
- **CIFAR-10.** Convnet test error drops from \\(\sim 14\%\\) to \\(\sim 12.6\%\\) with dropout, and to \\(\sim 11.9\%\\) with dropout + max-norm constraint.
- **Speech (TIMIT).** Phone error rates drop \\(\sim 5\%\\) relative.
- **Document classification (Reuters).** Similar magnitude improvements.

The paper's contribution is a *consistent* gain across very different domains with one tiny code change.

---

## Why It Works — The Co-Adaptation Argument

A unit's gradient depends on the other units in its layer being where the unit "expects" them. When 50 % of those neighbours might be missing, a feature that worked only because of a specific neighbour's contribution is no longer reliable. The optimizer is pushed toward features that are useful **on their own**.

Visualizing first-layer filters in a fully-connected network without dropout, the filters tend to look like noisy combinations — different units detecting overlapping features. With dropout, the filters become much more localized and interpretable: edges, blobs, corners. Each unit has been trained to do its own job.

Quantitative evidence in the paper: the *correlation* between activations of different units, measured on test data, drops sharply when dropout is used. Co-adaptation has been broken.

---

## Limitations &amp; When It Doesn't Help

Dropout is not a silver bullet:

- **Recurrent networks.** Naive dropout applied to recurrent connections destroys long-range dependencies. Specialized variants (variational dropout, zoneout, recurrent dropout with shared masks) work better. Modern transformers, in contrast, dropout is straightforward (applied within the FFN sub-layer of each block).
- **Over-regularization.** Combined with strong L\\(\_2\\), label smoothing, and modern data augmentation, dropout can over-regularize and *hurt* performance. The current best practice for image classification is *much less* dropout than the 2014 recommendations would suggest.
- **Batch normalization and layer normalization.** BN/LN provide a stochastic-noise regularization of their own (mini-batch statistics fluctuate). On well-tuned BN networks, additional dropout often makes no difference or hurts.
- **Very deep networks.** Replacing dropout with **stochastic depth** (skipping whole layers randomly) tends to do better. Same idea, larger granularity.

---

## Modern Context

Dropout's intellectual descendants are everywhere:

- **DropConnect** (Wan et al., 2013) — drop *weights* instead of units. Slightly more flexible; rarely beats vanilla dropout in practice.
- **Stochastic depth** (Huang et al., 2016) — skip whole residual blocks randomly during training. Standard in modern CNNs.
- **Data augmentation as dropout** — masking input pixels (Cutout) or feature-map regions (DropBlock) is dropout applied to the input or low layers.
- **Attention dropout** in transformers — drop entries of the attention probability matrix. Standard in BERT, GPT, ViT.
- **MC Dropout for uncertainty** (Gal &amp; Ghahramani, 2016) — keep dropout active at test time and compute predictive variance over Monte Carlo passes. A cheap probabilistic interpretation of an otherwise deterministic network.

---

## Reading the Paper

After this note, the paper itself ([PDF](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)) is much more approachable. The sections most worth reading carefully:

- §2: motivation through the lens of overfitting and model averaging.
- §3: the formal description of the dropout layer (matches our §2 above).
- §4: training tricks (max-norm, learning rates) — these matter more than the paper's prose suggests.
- §6: experimental results — useful as a benchmark of *what to expect* from dropout in different domains.
- §7.1: the visualizations of features learned with vs. without dropout — striking and the most convincing piece of evidence for the co-adaptation theory.

---

## Takeaways

- **Dropout is randomness as regularization.** Drop a fraction of units each forward pass; force the network to perform without any one of them.
- **Train ↔ test scaling.** Either inverted-dropout during training (modern default) or weight-scaling at test (original paper). Equivalent in expectation; inverted dropout is ergonomically better.
- **Ensemble interpretation.** \\(2^N\\) thinned subnetworks share weights and are trained simultaneously; the weight-scaling rule averages them in closed form.
- **Hyperparameter rule of thumb:** \\(p = 0.5\\) for hidden, \\(p = 0.8\\) for input, less aggressive on convolutional layers; train longer with a larger LR and a max-norm constraint on weights.
- **Modern context:** still useful, but on networks already heavily regularized by BatchNorm and modern augmentations, additional dropout is less impactful and sometimes harmful. Stochastic depth and attention dropout are its modern descendants.
