# Computing Between Models with Residual Coupling

Pascal Ekin pfekin@gmail.com

> Residual Coupling (RC) is a third paradigm for multi-model computation that learns operators acting on the *differences* between latent states, enabling frozen models to produce structured corrective updates for one another without modifying their parameters.

---

## Overview

Current scaling paradigms for large language models rely on monolithic fine-tuning or competitive Mixture-of-Experts (MoE) routing. Both suffer from structural fragility: fine-tuning risks catastrophic forgetting, while MoE's winner-take-all routing leaves most specialised parameters idle on any given forward pass. RC couples a frozen generalist anchor and one or more frozen specialist modules through learned, directional, gated linear operators placed at intermediate transformer layers. New domains are incorporated as non-destructive plugins without modifying any existing component.

In the medical domain, the multi-bilateral topology reduces perplexity by **80.07%** relative to the frozen generalist baseline (~7× the gain achieved by MoE at 10.66%), and improves factual accuracy on TruthfulQA Health by up to **5.5 percentage points** over MoE.

---

## Experiments & Code

Three scripts reproduce the experiments reported in the paper. Each can be run on a standard Colab T4 instance (gradient accumulation over 8 steps keeps GPU memory within single-card constraints).

### Experiment 1 — Domain Generality · `lf_benchmark.py`

One generalist, one specialist, three topologies (unilateral, bilateral, MoE), across four domains. Demonstrates proportional depth alignment for heterogeneous model pairs. Results are embedded as comments at the end of the file.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/lf_benchmark.py)

---

### Experiment 2 — Multi-Specialist Topology Sweep · `lf_three.py`

One generalist, two specialist modules, four topologies, medical domain. Results embedded as comments at the end of the file.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/lf_three.py)

---

### Experiment 3 — Factual Accuracy · `lf_qa.py`

Same configuration as Experiment 2, extended with TruthfulQA Health evaluation under the MC1 protocol.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/lf_qa.py)

---

## Hyperparameters

| Parameter | `lf_benchmark.py` | `lf_three.py` | `lf_qa.py` |
|---|---|---|---|
| MAX\_STEPS | 2,000 | 2,000 | 2,000 |
| GRAD\_ACCUM | 8 | 8 | 8 |
| MAX\_SEQ\_LEN | 128 | 128 | 128 |
| TEST\_SAMPLES | 20 | 50 | 50 |
| Bridge LR | 1e-4 | 1e-4 | 1e-4 |
| Router / mixing LR | 5e-3 | 5e-3 | 5e-3 |
| Gate initialisation | −2.0 | −2.0 | −2.0 |
| Optimiser | AdamW | AdamW | AdamW |

## Implementation Notes

- **Vocabulary alignment** across heterogeneous model pairs is handled via `torch.clamp`, clamping token indices to the smaller vocabulary size before each model's embedding lookup.
- **Bridge projections** are implemented without bias terms (`bias=False`), consistent with the pure-projection geometric interpretation.
- All experiments use a causal language modelling objective.

