---
layout: page
title: "Final Project — From a Transformer to a Reasoner"
description: "Capstone · CSE 455/555 Introduction to Pattern Recognition · Summer 2026"
back_link: '/teaching/pattern/'
back_text: "Back to Pattern Recognition"
---

## The project

There is **one** final project and everyone does it. No tracks, no menu of
easier options. You are graduate students building in the age of large language
models; the point of this capstone is to make you *build one yourself, end to
end, and then interrogate it like a researcher* — not to re-run a tutorial.

> **Build a small language model from scratch, post-train it into an
> instruction-follower, and then investigate one sharp question about reasoning
> and scale — with controlled experiments, baselines, and honest analysis.**

The project deliberately threads the entire back half of the syllabus: the
Transformer (Session 6), GPT-style language modeling (Session 8), scaling laws
(Session 9), RLHF / preference optimization (Session 10), and chain-of-thought
and emergence (Sessions 11–12). By the last week you will have re-derived, in
miniature, the pipeline behind every model you read about.

You may work **solo or in pairs**. A pair is expected to go proportionally
deeper — an extra model size in the scaling study, or a second post-training
method to compare.

---

## The three stages

You must complete all three. The *minimum* is the full pipeline working at small
scale; depth and scale beyond the minimum are where the top grades live.

### Stage 1 — Implement and pretrain

- **Implement the core yourself.** Write the decoder-only Transformer —
  multi-head self-attention, the block, the training loop — from scratch in
  PyTorch. You may *not* use `nn.Transformer`, a `Trainer` wrapper, or any
  high-level "fit" call for the core model and loop. (You may read
  [nanoGPT](https://github.com/karpathy/nanoGPT) for reference; you may not copy
  it wholesale.)
- **Pretrain a small model** on a public corpus — e.g.
  [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
  WikiText, or an OpenWebText subset. Target scale: roughly **10–125M
  parameters**, sized to your compute.
- **Scaling mini-study.** Train **at least three** model sizes *or* three token
  budgets under a fixed compute ceiling, and check whether your loss-vs-compute
  curve follows the power law from Session 9. Report whether your data was
  compute-optimal in the Chinchilla sense.

### Stage 2 — Post-train into an instruction-follower

- **Supervised fine-tune (SFT)** your pretrained model on a small instruction
  dataset (e.g. a subset of Alpaca / Dolly / OpenAssistant).
- **Add a preference-optimization step.** Apply **DPO** (recommended) or a full
  **reward-model + PPO** RLHF loop, and measure the change in
  instruction-following quality against the **SFT-only baseline**. For the
  preference step you may use a library such as
  [TRL](https://github.com/huggingface/trl) — the research is in the *experiment
  and analysis*, not in re-implementing PPO.
- Report *how* you measured "better" (win-rate via a judge model, a small
  human-rated set, or a held-out preference accuracy) and why that metric is
  trustworthy.

### Stage 3 — Probe reasoning and scale

- On a multi-step reasoning task — GSM8K-style arithmetic, or a synthetic
  multi-hop task you control — measure **chain-of-thought vs. direct answering**
  across your model sizes from Stage 1.
- Investigate one of these questions (or your own, equally sharp):
  - Does the CoT benefit **emerge** with scale, or is it present throughout?
  - Is the apparent emergence **a metric artifact** (smooth under one metric,
    discontinuous under another)?
  - Does post-training (Stage 2) change the reasoning behavior at all?

---

## What "rigorous" means here (non-negotiable)

This is the part that separates a graduate project from a weekend hack.

- **Baselines.** Every headline number is compared against at least one sensible
  baseline (prior method, ablated version, or a trivial control). A number with
  nothing to compare it to tells us nothing.
- **Ablations.** Change one thing at a time and show its effect — remove
  positional encodings, vary heads, turn off CoT, drop the DPO step.
- **Multiple seeds.** Any claim about a *difference* reports **mean ± standard
  deviation over ≥ 3 seeds**. One lucky run is not evidence.
- **Failure analysis.** Show *where* the model fails and offer a hypothesis for
  *why*, with qualitative examples.
- **Reproducible code.** A public repo with a README, pinned environment, fixed
  seeds, and a **one-command** way to regenerate your main result.
- **Compute realism.** Stay small — models ≤ a few hundred M parameters, public
  datasets, free/affordable compute (Colab, Kaggle, a single GPU). Designing an
  experiment that *fits* the budget is itself a graded skill; scope accordingly.

> **Negative and partial results are fully credited** when the investigation is
> sound. "DPO did *not* improve win-rate at my scale, and here is the controlled
> evidence" is a strong project. A pretty number with no baseline is a weak one.

**AI-tool policy.** You may use LLMs for coding and writing (disclose how in an
appendix). The Transformer and training loop must be *your* implementation, and
the research question, experimental design, analysis, and conclusions must be
your own work.

---

## Milestones and deadlines

| When | Deliverable | What is graded |
|---|---|---|
| **Wk 1 — Jun 26** | Team + topic registered (one paragraph) | Ungraded gate — must be on file to proceed |
| **Wk 2 — Jul 3** | **Proposal** (2 pp.): your reasoning question, the model sizes you'll train, datasets, compute plan, and a concrete success criterion | Feasibility & clarity; is it runnable in the budget? |
| **Wk 3 — Jul 10** | **Baseline checkpoint:** repo runs end-to-end, smallest model pretrains, one real loss number on the board | Evidence the pipeline works; de-risks the project early |
| **Wk 4 — Jul 17** | **Draft** (4–6 pp.) with preliminary results and at least one plot | Submitted for peer review |
| **Wk 5 — Jul 24** | **Peer reviews:** structured reviews of 2 classmates' drafts | Quality & usefulness of the reviews you write |
| **Wk 6 — Jul 29** | **Lightning talk** — 5 min + Q&A | Clarity of question, result, and honest limitations |
| **Wk 6 — Jul 31** | **Final report** (6–8 pp., paper format) **+ code repo** | The bulk of the grade — see rubric |

---

## Grading rubric (the 35%, broken out)

| Component | % | What earns full marks |
|---|---|---|
| Proposal | 4 | A sharp, falsifiable question with a realistic plan |
| Baseline checkpoint | 4 | Pipeline runs; smallest model pretrains on time |
| Peer reviews (given) | 4 | Specific, constructive feedback to 2 peers |
| Final report | 13 | Clear question, sound method, baseline + ablation, multi-seed results, honest analysis across all three stages |
| Code & reproducibility | 6 | One-command reproduction of the main result; clean README |
| Lightning talk | 4 | Communicates the finding and its limits in 5 minutes |
| **Total** | **35** | |

---

## The final report

Write it as a short research paper (NeurIPS/ICML format, 6–8 pages excluding
references and appendix):

1. **Abstract & question** — what you investigated and what you found, in five
   sentences.
2. **Method** — your architecture, training setup, datasets, and the exact
   experimental design (what's the independent variable, what's controlled).
3. **Results** — tables and plots with baselines, ablations, and error bars over
   seeds.
4. **Analysis** — where it works, where it fails, and your best explanation why.
5. **Limitations & what you'd do with more compute.**
6. **Appendix** — reproduction instructions and AI-tool disclosure.

---

## Starting points

These are launch pads, not crutches — the core implementation must be yours.

- **Model & training:** [nanoGPT](https://github.com/karpathy/nanoGPT) /
  [minGPT](https://github.com/karpathy/minGPT) for reference architecture.
- **Pretraining data:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
  WikiText-103, OpenWebText.
- **Instruction / preference data:** Alpaca, Dolly-15k, OpenAssistant.
- **Preference optimization:** [TRL](https://github.com/huggingface/trl) (DPO,
  PPO) — fine to use for Stage 2's optimizer.
- **Reasoning eval:** GSM8K, or a synthetic multi-hop arithmetic generator you
  write (which gives you perfect control over difficulty).
