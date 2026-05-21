# Experiments

Benchmark results, ablation study, reproduction instructions, and configuration details for
the residual coupling experiments reported in the
[paper](https://ssrn.com/abstract=6746521).

---

## 1. Two-model cross-domain benchmark

One GPT-2 generalist anchor paired with a domain specialist in each configuration. Bilateral
RC compared against the same two frozen models combined via MoE routing. All conditions
trained for 2,000 steps.

| Domain | Anchor | Specialist | Frozen A (PPL) | MoE (PPL) | Bilateral RC (PPL) | Reduction |
|--------|--------|------------|---------------:|----------:|-------------------:|----------:|
| Medical | GPT-2 Medium (345M) | DialoGPT-Medium (345M) | 50.05 | 64.66 | **12.01** | 76% |
| Legal | GPT-2 (124M) | open-australian-legal-gpt2 (124M) | 26.48 | 21.83 | **8.30** | 69% |
| Coding | GPT-2 (124M) | CodeGPT-small-py (124M) | 16.68 | 878.40 | **5.91** | 65% |
| Scientific | GPT-2 Large (774M) | gpt2-large-medical (774M) | 28.54 | 26.85 | **17.51** | 39% |

The coding domain is a structural stress test. CodeGPT-small-py uses a different tokenizer
from GPT-2, producing a frozen perplexity of approximately 7 million on general text. MoE
collapses to 878.40 because it operates on output logits after the specialist's output
projection has discarded latent structure. Bilateral RC reaches 5.91 by reading hidden states
before that collapse.

The scientific domain shows the smallest gain because the specialist's training distribution
overlaps substantially with the generalist's. The representational gap is smaller and the
bridge has less corrective work.

**Steered individual outputs.** In bilateral coupling both models improve individually, not
only through the fused output. In the medical domain the specialist's individual perplexity
drops from 317.89 to 22.59. In the legal domain it drops from 44.44 to 10.84. In the
scientific domain both models converge to near-identical individual perplexity (18.35 and
18.24) despite having been trained on different corpora.

The return bridge's effect is clearest in the coding domain: without it (unilateral mode),
the generalist's individual output degrades from its frozen baseline of 16.68 to 32.11, as
the training objective optimizes the fused output at the expense of the generalist's residual
stream. Bilateral coupling recovers the generalist's individual output to 11.29 while the
fused output reaches 5.91.

---

## 2. Three-model medical topology sweep

GPT-2 (124M) as the generalist anchor with two medical specialists: DialoGPT-small (117M)
and a fine-tuned medical QA model (124M). All topologies trained for 2,000 steps on medical
conversational data. TruthfulQA Health (MC1) evaluated on n=50 samples.

| Topology | Fused PPL | TruthfulQA Health | PPL reduction | TQA gain |
|----------|----------:|------------------:|--------------:|---------:|
| Frozen baseline | 57.08 | 16.36% | -- | -- |
| MoE | 56.80 | 20.00% | 0.5% | +3.6 pp |
| Multi-unilateral | 11.26 | 23.64% | 80.3% | +7.3 pp |
| Star-bilateral | 11.07 | 21.82% | 80.6% | +5.5 pp |
| Multi-bilateral | **11.02** | **25.45%** | 80.7% | +9.1 pp |
| Hybrid | 11.11 | 23.64% | 80.5% | +7.3 pp |

MoE reduces perplexity by 0.5% and improves factual accuracy by 3.6 percentage points. All
RC topologies reduce perplexity by approximately 80% and improve factual accuracy by 5 to 9
percentage points. The consistent factual accuracy improvement across all RC topologies
suggests that cross-model gating suppresses model-specific confabulation, not only noise in
the language modeling objective: each model's hallucinations are statistically uncorrelated
with the other's representations, so the gate scalars learn to amplify projections that
produce consistent cross-model updates and suppress those that do not.

---

## 3. Ablation: learned structure is required

Two ablation conditions in the three-model medical experiment test whether the gains require
learned projection structure or merely the presence of an additive bridge at a larger
parameter count. The first (`multi_bilateral_random`) freezes projection matrices at random
initialization and trains only gate values. The second (`multi_bilateral_no_gate`) removes
learned gates, setting all gate values to 1.0.

| Condition | Fused PPL | TruthfulQA Health |
|-----------|----------:|------------------:|
| Frozen baseline | 57.08 | 16.36% |
| MoE | 56.80 | 20.00% |
| Multi-bilateral (trained) | **11.02** | 25.45% |
| Multi-bilateral (no gate) | 16.42 | **30.91%** |
| Multi-bilateral (random projections) | 166.82 | 20.00% |

Random bridges make things significantly worse: 166.82 against a frozen baseline of 57.08.
Trained gate values alone cannot recover the loss. The gains therefore require learned
projection structure, not a parametric transformation of position.

The gate ablation is more nuanced. Removing learned gates worsens perplexity (16.42 against
11.02) while improving TruthfulQA accuracy (30.91% against 25.45%). In two-model experiments
ungated bilateral marginally outperforms gated in some domains, but the gate's stabilizing
effect becomes more important as the number of coupled models increases. The TruthfulQA
divergence between the two conditions is an open question: the no-gate condition may allow
more specialist signal to pass through, which helps factual accuracy while introducing enough
distributional noise to raise perplexity.

---

## Running the experiments

### Two-model cross-domain benchmark (`benchmark.py`)

Runs bilateral RC against MoE, unilateral, logit ensemble, no-gate, and random-projection
baselines across a single domain. Edit `DOMAIN` at the top of the file to switch between
`"medical"`, `"legal"`, `"coding"`, and `"scientific"`.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17EINsJ-s-3ZJb4TiGskqc7_7VI_wI7jc?usp=sharing)

```bash
pip install torch transformers datasets tqdm
python benchmark.py
```

### Three-model topology sweep with TruthfulQA (`three_qa.py`)

Runs all eight coupling conditions on three frozen medical models and evaluates each on
TruthfulQA Health (MC1).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KmglCM7f0m-qoiWryUlLMIfelowjhJgu?usp=sharing)

```bash
python three_qa.py
```

Results for each run are embedded as comments at the end of each script.

---

## Configuration

### Hyperparameters

| Parameter | `benchmark.py` | `three_qa.py` |
|-----------|---------------:|--------------:|
| MAX_STEPS | 2,000 | 2,000 |
| GRAD_ACCUM | 8 | 8 |
| MAX_SEQ_LEN | 128 | 128 |
| TEST_SAMPLES | 25 | 50 |
| Bridge LR | 1e-4 | 1e-4 |
| Router / mix LR | 5e-3 | 5e-3 |
| Gate initialization | -2.0 | -2.0 |
| Optimizer | AdamW | AdamW |

Peak VRAM usage is approximately 9 GB in both scripts with gradient accumulation over 8
steps. Bridge layers are distributed proportionally to model depth: every sixth layer for
36-layer models, every fourth for 24-layer, every third for 12-layer.

### Model catalog

**Generalists:** `gpt2` (124M), `gpt2-medium` (345M), `gpt2-large` (774M)

**Specialists:**

| Model | Parameters | Domain |
|-------|-----------|--------|
| `microsoft/DialoGPT-medium` | 345M | Conversational |
| `microsoft/DialoGPT-small` | 117M | Conversational |
| `nrslearning/finetuned-gpt2-medical-QA` | 124M | Medical QA |
| `isaacus/open-australian-legal-gpt2` | 124M | Legal |
| `microsoft/CodeGPT-small-py` | 124M | Python code |
| `Locutusque/gpt2-large-medical` | 774M | Medical literature |
