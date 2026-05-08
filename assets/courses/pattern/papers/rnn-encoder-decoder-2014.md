---
layout: page
title: "RNN Encoder–Decoder (Cho et al., 2014)"
description: "Reading note · Pattern Recognition · Neural Network Foundations"
back_link: '/teaching/pattern/'
back_text: "Back to Pattern Recognition"
---

## TL;DR

Two RNNs, glued together. The first ("encoder") reads an input sequence \\(\mathbf{x} = (x\_1, \ldots, x\_T)\\) and produces a single fixed-length **summary vector** \\(\mathbf{c}\\). The second ("decoder") reads \\(\mathbf{c}\\) and generates an output sequence \\(\mathbf{y} = (y\_1, \ldots, y\_{T'})\\) one token at a time, conditioning each token on \\(\mathbf{c}\\) and the tokens it has produced so far.

This 2014 paper introduces the encoder–decoder architecture in its modern form, names it that, applies it to statistical machine translation as a phrase-scoring component, and — almost as an afterthought — proposes the **GRU** (gated recurrent unit) as the recurrent cell. Both ideas became foundational: the encoder–decoder template is now the conceptual skeleton of every sequence-to-sequence model (including transformers), and the GRU remains a competitive RNN cell to this day.

---

## Problem &amp; Motivation

Pre-2014 statistical machine translation was dominated by **phrase-based SMT**: translate the source sentence by stitching together translations of overlapping phrases. The phrase-translation table assigns each source-phrase / target-phrase pair a probability, learned from massive parallel corpora.

The problem: the phrase table treats each phrase pair as an atomic unit. There's no way to say *"this rare source phrase should be translated by analogy to this similar phrase we've seen many times."* Generalization across paraphrases is brittle.

Cho et al. propose a neural model that scores phrase pairs — \\(p(\mathbf{y} \mid \mathbf{x})\\) for source phrase \\(\mathbf{x}\\) and target phrase \\(\mathbf{y}\\) — using a sequence-to-sequence neural network. The score is fed into the SMT system as an additional feature. Modest performance bumps, but the architecture introduced in passing has outlived the SMT context entirely.

---

## The Encoder

A standard RNN reads the source phrase token by token, updating a hidden state:

$$
\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}),
\qquad t = 1, \ldots, T,
$$

with \\(f\\) some recurrent cell (the paper uses GRU; we'll see it next). After the last token, the final hidden state \\(\mathbf{h}\_T\\) is taken as the **summary vector**

$$
\mathbf{c} = \mathbf{h}_T.
$$

\\(\mathbf{c}\\) is supposed to capture all of the information in the input sequence, distilled into a fixed-dimensional vector. It's a strong assumption — and the *fixed-length* part will turn out to be the architecture's main bottleneck — but the framing is clean.

---

## The Decoder

A second RNN generates the target sequence. At each time step \\(t\\), it computes a hidden state and emits a probability distribution over the target vocabulary, conditioned on (a) the summary, (b) the previously generated tokens, and (c) its own hidden state:

$$
\mathbf{h}'_t = f(\mathbf{h}'_{t-1}, y_{t-1}, \mathbf{c}),
$$

$$
p(y_t \mid y_{t-1}, \ldots, y_1, \mathbf{x}) = g(\mathbf{h}'_t, y_{t-1}, \mathbf{c}).
$$

The probability of the entire output sequence is

$$
p(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{T'} p(y_t \mid y_{t-1}, \ldots, y_1, \mathbf{c}).
$$

Training is straightforward: maximize the log-likelihood of correct target sequences, end-to-end via backpropagation through both RNNs.

The crucial design choice: \\(\mathbf{c}\\) is fed in at *every* decoder step, not just the first. So the decoder can "look back" at the source summary while generating each token. (Modern attention-based encoder–decoders generalize this: the decoder looks at *every* source position individually rather than a single summary.)

---

## The GRU

The second contribution. Standard RNNs (Elman networks) struggle with long-range dependencies because gradients vanish or explode through repeated multiplication. LSTMs (Hochreiter &amp; Schmidhuber, 1997) solved this with a complex cell containing input, forget, and output gates. The **Gated Recurrent Unit** is a simplified alternative.

A GRU has two gates: an **update gate** \\(z\_t\\) and a **reset gate** \\(r\_t\\), both in \\([0, 1]\\):

$$
\begin{aligned}
\mathbf{r}_t &= \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1}), \\
\mathbf{z}_t &= \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1}), \\
\tilde{\mathbf{h}}_t &= \tanh\\!\bigl(\mathbf{W} \mathbf{x}_t + \mathbf{U} (\mathbf{r}_t \odot \mathbf{h}_{t-1})\bigr), \\
\mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t.
\end{aligned}
$$

Read the equations:

- **Reset gate \\(\mathbf{r}\_t\\)** — element-wise multiplied with the previous hidden state when computing the candidate \\(\tilde{\mathbf{h}}\_t\\). When \\(\mathbf{r}\_t \approx 0\\), the candidate ignores the past entirely; when \\(\mathbf{r}\_t \approx 1\\), it uses everything.
- **Update gate \\(\mathbf{z}\_t\\)** — interpolates between the previous hidden state and the candidate. \\(\mathbf{z}\_t \approx 0\\) keeps the old state (long-term memory); \\(\mathbf{z}\_t \approx 1\\) jumps to the new candidate (full update).

The two-gate structure gives the GRU **selective memory**: it can keep information unchanged across many time steps when the update gate stays low, and overwrite the state quickly when the input is informative. That's the same long-range-dependency trick LSTMs do, with fewer parameters and (often) similar performance.

GRU vs LSTM in 2014–2016 was an active comparison. The community consensus that emerged: GRUs are slightly cheaper, LSTMs slightly more expressive, performance is roughly the same on most tasks. Modern practice picks whichever is the framework default.

---

## Why a Fixed-Length Summary is the Bottleneck

The encoder–decoder works, but compressing an arbitrary-length sequence into a fixed-dimensional \\(\mathbf{c}\\) is rough. The same paper notes that performance degrades on long sentences — the summary becomes lossy.

Bahdanau, Cho &amp; Bengio (2015) addressed this within months: instead of a single \\(\mathbf{c}\\), the decoder computes **attention weights** over all encoder hidden states \\(\mathbf{h}\_1, \ldots, \mathbf{h}\_T\\) and uses a different weighted summary at each decoding step. That's the seed of the attention mechanism.

Vaswani et al. (2017) eventually pushed this idea to its limit: drop the recurrence entirely, do **self-attention** at the encoder and decoder, and you get the **transformer**. Modern sequence models — BERT, GPT, T5, Whisper, CLIP — are all variations on the encoder–decoder template, just with attention instead of an RNN.

---

## Empirical Results From the Paper

Cho et al. plug the neural phrase-scorer into a Moses-based SMT system on English ↔ French (WMT '14). The numbers:

- **BLEU on test set:** the SMT-only baseline scores 30.64. Adding the neural phrase score brings it to 31.48 — a small but real improvement.
- **Qualitative analysis:** the neural model assigns higher probability to *literal* phrase translations and lower probability to canned/idiomatic phrases that happen to appear together in training data. The neural and statistical components are complementary.

The 2014 paper did not "win" machine translation — that came with Sutskever, Vinyals &amp; Le (also 2014, "Sequence to Sequence Learning with Neural Networks") and Bahdanau et al. (2015). But it introduced the architectural primitive everyone built on.

---

## What Made This Paper Important

- **Named the encoder–decoder template** as a reusable architecture for sequence-to-sequence problems.
- **Introduced the GRU**, which along with LSTM became the standard recurrent cell for sequence modeling for the next 5+ years.
- **Demonstrated end-to-end training of an MT-relevant neural model** — every modern NMT system descends from this thread of work.
- The paper's own SMT result was modest, but the architectural ideas have been ridiculously influential.

---

## Modern Context

- **Bahdanau attention** (2015): the natural extension — instead of one summary vector, the decoder attends over all encoder states.
- **Sutskever et al. seq2seq** (2014, parallel work): a pure NMT model with multi-layer LSTM encoder and decoder. Established that NMT can outperform SMT at scale.
- **Transformer** (Vaswani et al., 2017): drops recurrence; replaces every \\(f\\) with self-attention. Same encoder–decoder skeleton, totally different internal mechanics.
- **Decoder-only models** (GPT family): drop the encoder; the decoder attends to its own past tokens. The encoder–decoder split is no longer mandatory but the conceptual frame still applies.
- **Encoder-only models** (BERT, RoBERTa): drop the decoder; the encoder produces task-specific predictions directly.

The 2014 encoder–decoder is the architectural ancestor of nearly every sequence model today. The internal cells have changed; the high-level shape has not.

---

## Reading the Paper

The paper has two more-or-less independent contributions; treat them as separate:

- §2: encoder–decoder framework. Read for the architectural concept; equations are simple.
- §2.3: the GRU. Work through the equations on paper to internalize the gating mechanism.
- §3–§4: SMT integration and experiments. Less critical; skim unless you care about the SMT-era context.

Critical follow-ups:

- Sutskever, Vinyals &amp; Le, 2014 — pure neural seq2seq for MT.
- Bahdanau, Cho &amp; Bengio, 2015 — attention.
- Vaswani et al., 2017 — transformer.

---

## Takeaways

- **Encoder–decoder = two RNNs.** Encoder reads the source, produces a summary vector \\(\mathbf{c}\\); decoder generates the target conditioned on \\(\mathbf{c}\\) one token at a time.
- **GRU = simplified LSTM.** Two gates (reset, update) instead of three (input, forget, output). Comparable performance, fewer parameters.
- **The fixed-length summary is the obvious bottleneck.** Attention (2015) and transformers (2017) generalize the architecture to fix it.
- **Architectural skeleton with very long legs.** Modern sequence models — including transformer-based encoder–decoders like T5 — are direct descendants.
