# Residual Coupling (RC)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Computing Between Models: Residual Coupling of Frozen Transformers

Paper: [link to paper]

---

## Overview

Residual Coupling (RC) is an architecture for coupling frozen transformer models through learned operators acting on differences between their latent states. Instead of selecting between models or merging their representations, RC enables models to produce corrective updates for one another during a shared forward pass.

Bidirectional coupling suppresses model-specific confabulation and reinforces shared factual signal. The effect emerges from the interaction between independently trained models rather than from additional supervision or fine-tuning.

All base models remain frozen. Only lightweight bridge projections are trained.

---

## Key Properties

* No modification of base model weights
* No catastrophic forgetting
* Parallel processing across all models
* Continuous latent communication instead of token-level routing
* Modular addition and removal of specialists

---

## Why this works

Private noise and shared signal:

Each model’s representation contains shared structure and model-specific noise. Noise is uncorrelated across independently trained models. During bidirectional coupling, bridge gates learn to amplify components that produce consistent cross-model updates and suppress components that do not. This acts as a sub-symbolic regulariser against hallucination.

Differences as computation:

The system does not operate on representations directly. It learns transformations between them. Bridge updates behave as directional corrections relative to the target model’s current state rather than as feature transfer.

---

## Architecture

Each model processes the same input sequence in parallel. At selected transformer layers, bridge projections map latent states from a source model into corrective updates applied to a target model’s residual stream.

Bridge update:

h_target ← h_target + σ(g) · W · h_source

The update is trained to act as a correction relative to the target model’s current state, not as a direct transfer of features.

<div align="center">
  <img src="architecture.png" alt="RC architecture." width="600"/>
  <p><em>Figure 1: RC architecture.</em></p>
</div>

Topologies:

* Unilateral: specialists inject into generalist only
* Star-bilateral: bidirectional between generalist and each specialist
* Multi-bilateral: bidirectional between all pairs
* MoE: routing baseline

<div align="center">
  <img src="topologies.png" alt="RC Topologies" width="600"/>
  <p><em>Figure 2: The four topologies side by side.</em></p>
</div>

---

## Experiments

All experiments use a causal language modelling objective with gradient accumulation over 8 steps.

Improvements over MoE are interpreted as evidence that collaborative coupling extracts useful signal while suppressing model-specific noise.

### Experiment 1: Domain Generality

| Domain     | Generalist          | Specialist                 | Frozen A PPL | Frozen B PPL | MoE PPL | Uni PPL | Bi PPL | Bi vs MoE |
| ---------- | ------------------- | -------------------------- | ------------ | ------------ | ------- | ------- | ------ | --------- |
| Medical    | GPT-2 Medium (345M) | DialoGPT-Medium            | 45.71        | 331.04       | 50.35   | 12.89   | 11.04  | +78.1%    |
| Scientific | GPT-2 Large (774M)  | gpt2-large-medical         | 35.82        | 34.32        | 31.94   | 21.68   | 21.57  | +32.5%    |
| Coding     | GPT-2 (124M)        | CodeGPT-small-py           | 18.54        | 5M           | 66.81   | 13.34   | 6.49   | +90.3%    |
| Legal      | GPT-2 (124M)        | open-australian-legal-gpt2 | 24.72        | 38.02        | 19.02   | 7.56    | 6.88   | +63.8%    |

Bilateral coupling outperforms both unilateral and MoE across all domains.

[`benchmark.py`](https://github.com/pfekin/residual-coupling/blob/main/benchmark.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/benchmark.py) 

---

### Experiment 2: Multi-Specialist Topology Sweep

| Topology          | PPL   | vs MoE | vs Frozen Generalist |
| ----------------- | ----- | ------ | -------------------- |
| Multi-Unilateral  | 12.90 | +74.7% | +77.4%               |
| Star-Bilateral    | 11.68 | +77.1% | +79.5%               |
| Multi-Bilateral   | 11.37 | +77.7% | +80.1%               |
| MoE               | 50.99 | —      | +10.7%               |
| Frozen Generalist | 57.08 | —      | —                    |

Multi-bilateral achieves the lowest perplexity.

[`three.py`](https://github.com/pfekin/residual-coupling/blob/main/three.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/three.py) 

---

### Experiment 3: Factual Accuracy (TruthfulQA Health)

| Topology         | PPL   | PPL vs MoE | TruthfulQA (%) | TQA vs MoE |
| ---------------- | ----- | ---------- | -------------- | ---------- |
| Multi-Unilateral | 12.90 | +74.7%     | 23.64          | +5.46      |
| Star-Bilateral   | 11.68 | +77.1%     | 21.82          | +3.64      |
| Multi-Bilateral  | 11.37 | +77.7%     | 23.64          | +5.46      |
| MoE              | 50.99 | —          | 18.18          | —          |

All RC topologies improve factual accuracy over MoE.

[`qa.py`](https://github.com/pfekin/residual-coupling/blob/main/three.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/qa.py) 

---

## Implementation

* Bridge layers selected proportionally to model depth
* No bias in bridge projections
* Vocabulary alignment via token clamping
* AdamW optimiser
* Learning rate: 1e-4 (bridges), 5e-3 (router/mixing)

---

## Parameter Overhead

Example configuration (3 models, d = 768, 5 bridge layers):

* Unilateral: ~4.7M
* Star-bilateral: ~9.4M
* Multi-bilateral: ~14.2M
* MoE router: ~2.3K
* Base model: ~124M

Bridge parameters remain small relative to frozen models.

---

## Usage

Basic workflow:

1. Load frozen generalist and specialist models
2. Insert bridge projections at selected layers
3. Train bridges on domain data
4. Run inference with all models in parallel

The base models remain unchanged throughout; all adaptation is located in the bridge parameters.

Adding a new domain:

* Train a new specialist model
* Train bridges between it and the existing system
* No retraining of existing models required

---

## Notes and Extensions

* Vocabulary alignment currently uses token clamping for heterogeneous model pairs. This can be extended to a learned soft alignment between embedding spaces.
* Multi-bilateral connectivity scales with the number of model pairs. In practice, the star-bilateral topology provides most of the observed gains with lower parameter cost and simpler scaling.
* Experiments are run on models up to 774M parameters. The mechanism is architecture-agnostic and can be applied to larger models.
* Cross-modal coupling is a direct extension of the same mechanism and remains to be evaluated.

---

## Citation

[link to paper]

## Contact

* **Author**: Pascal Ekin
* **Email**: [pfekin@gmail.com](mailto:pfekin@gmail.com)
* **Issues**: Use the GitHub issue tracker for bugs/requests
