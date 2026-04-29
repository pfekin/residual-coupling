> ***"The pure present is an ungraspable advance of the past devouring the future.***
> ***In truth, all sensation is already memory."***
> 
> -- Henri Bergson, *Matter and Memory* (1896)<br/>
> Quoted by a character in Haruki Murakami’s *Kafka on the Shore* (2002), consistent with the verbatim recall of Murakami’s works documented by Liu et al. (2026).
---

<div align="center">

# Computing Between Models

**Residual Coupling of Frozen Transformers**

[![Paper PDF](https://img.shields.io/badge/paper-PDF-red?style=flat-square&logo=adobeacrobat)](https://colab.research.google.com/github/YOUR_USERNAME/differance-engine/blob/main/paper/differance_engine.pdf)

</div>

---

The standard approach to specialisation modifies models. Fine-tuning overwrites weights. Mixture-of-Experts routes tokens to a single expert and discards the rest. Agentic pipelines compress continuous internal geometry into discrete tokens between calls.

Residual Coupling (RC) implements a different approach. Frozen transformers function as non-linear memorizers: their weights define the boundaries of what the system can know. At each bridge layer, a small learned projection reads one model's hidden state and adds a corrective update directly into the other model's residual stream, the same stream that each transformer layer already writes into additively. The bridge does not replace or intercept the receiving model's computation, it perturbs the stream that computation reads from. The receiving model then processes that perturbed stream through its own frozen weights, unchanged. Bridge projections are linear operators that navigate the structured differences between those memorised spaces. Because they are linear, they physically cannot memorise in the way a non-linear layer can and they are constrained to learn continuous relational maps between what the frozen models have separately encoded.

 
Coupling two existing frozen models takes a few thousand training steps on a single GPU and no weights are modified.

## Results at a glance

### Two-model benchmark, four domains (`benchmark.py`)

| Domain | Model A | Model B | Frozen A | MoE | Bilateral | Gain vs. frozen |
|---|---|---|---:|---:|---:|---:|
| Medical | GPT-2 Medium | DialoGPT-Medium | 50.05 | 64.66 | **12.01** | +76% |
| Legal | GPT-2 | australian-legal-gpt2 | 26.48 | 21.83 | **8.30** | +69% |
| Coding | GPT-2 | CodeGPT-small-py | 16.68 | 878.40 | **5.91** | +65% |
| Scientific | GPT-2 Large | gpt2-large-medical | 28.54 | 26.85 | **17.51** | +39% |

> Perplexity (lower is better). Coding is the stress test: CodeGPT uses a different tokeniser, giving it a frozen perplexity of ~7 million on general text. Mixture of Experts (MoE) collapses to 878.40. Bilateral coupling reaches 5.91, below the frozen generalist's 16.68, by learning to extract latent signal without relying on Model B's token-level output.

### Three-model topology sweep, medical domain (`three_qa.py`)

| Topology | Fused PPL | TruthfulQA Health | vs. frozen baseline |
|---|---:|---:|---|
| Frozen baseline | 57.08 | 16.36% | -- |
| MoE | 56.80 | 20.00% | -0.5% PPL / +3.6 pp TQA |
| Multi-unilateral | 11.26 | 23.64% | -80.3% PPL / +7.3 pp TQA |
| Star-bilateral | 11.07 | 21.82% | -80.6% PPL / +5.5 pp TQA |
| **Multi-bilateral** | **11.02** | **25.45%** | **-80.7% PPL / +9.1 pp TQA** |
| Hybrid | 11.11 | 23.64% | -80.5% PPL / +7.3 pp TQA |

> TQA = TruthfulQA Health (MC1), n=50. Across all RC topologies, factual accuracy improves alongside perplexity, and all topologies outperform MoE on both metrics.

## Architecture

### Figure 1 -- Parallel frozen stacks connected by bridge projections

<div align="center">
  <img src="architecture.png" alt="Figure 1: RC architecture" width="600"/>
  <p><em>Figure 1: RC Architecture.</em></p>
</div>

The bridge at layer $\ell$ from model $B$ (specialist) to model $A$ (generalist) computes a correction $\delta \mathbf{h}_A^{(\ell)}$ and adds it to model $A$’s residual stream after layer $\ell$ and before layer $\ell+1$:

$$
\delta \mathbf{h}_A^{(\ell)} = \sigma(g^{(\ell, B \to A)}) W^{(\ell, B \to A)} \mathbf{h}_B^{(\ell)}
$$
$$
\mathbf{h}_A^{(\ell)} \leftarrow \mathbf{h}_A^{(\ell)} + \delta \mathbf{h}_A^{(\ell)}
$$

The frozen layer at $\ell$ then processes the perturbed $\mathbf{h}_A^{(\ell)}$ through its unchanged weights. The bridge does not enter the layer as it writes into the residual stream that the layer reads from, consistent with the standard transformer update pattern.

The scalar gate $g^{(\ell, B \to A)}$ is initialized to $-2$, so $\sigma(-2) \approx 0.12$ at step 0 and the additive correction is initially small, increasing only as supported by the training signal. The projection $W^{(\ell, B \to A)} \in \mathbb{R}^{d_A \times d_B}$ naturally handles dimensional mismatch between heterogeneous models. No attention heads are introduced and no sequence positions are added. The computational overhead per bridge layer is $O(d^2)$, rather than $O(L^2)$.




### Figure 2 -- The four coupling topologies

<div align="center">
  <img src="topologies.png" alt="Figure 2: The four topologies" width="600"/>
  <p><em>Figure 2: The four topologies.</em></p>
</div>

In the three-model medical experiment the PPL gap between multi-unilateral (11.26) and multi-bilateral (11.02) is small, though a single domain, dataset, and distribution is insufficient to generalise from. The gap is decisive in the two-model coding domain: unilateral coupling produces a fused PPL of 15.15 but degrades the generalist's individual output to 32.11, worse than its frozen baseline of 16.68. Bilateral coupling corrects this to 11.29 and reaches a fused PPL of 5.91. Without the return signal, the training objective optimises the fused output at the expense of the generalist's own residual stream.

## Get your hands dirty

### Run the two-model benchmark (four domains)

In Colab (no setup):


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17EINsJ-s-3ZJb4TiGskqc7_7VI_wI7jc?usp=sharing)

Set `DOMAIN` at the top of the notebook to `"medical"`, `"legal"`, `"coding"`, or `"scientific"`. Runs in ~25 minutes on a T4.

Locally:

```bash
pip install torch transformers datasets tqdm
python benchmark.py   # edit DOMAIN at the top of the file
```

Expected output (medical domain):
```
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 SYNERGY-X: STEERED MULTI-AGENT PERFORMANCE (Test Samples: 50)
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 MODE                      | GEN (A) PPL/TQA        | SPEC (B) PPL/TQA       | SPEC (C) PPL/TQA      
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 FROZEN BASELINES          | 57.08   / 16.36%       | 758.38  / 36.36%       | 9209.68 / 23.64%      
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 logit_ensemble            | 65.41   / 16.36%       | 752.60  / 36.36%       | 9209.68 / 23.64%      
 multi_unilateral          | 11.26   / 23.64%       | 752.60  / 36.36%       | 9209.68 / 23.64%      
 star_bilateral            | 11.07   / 21.82%       | 1241.14 / 38.18%       | 4881.03 / 23.64%      
 multi_bilateral           | 11.02   / 25.45%       | 1233.61 / 30.91%       | 7473.90 / 29.09%      
 hybrid_multi_bilateral    | 11.11   / 23.64%       | 50.02   / 32.73%       | 22.12   / 25.45%      
 multi_bilateral_no_gate   | 16.42   / 30.91%       | 10406328.12 / 25.45%   | 117917.34 / 32.73%    
 multi_bilateral_random    | 166.82  / 20.00%       | 805.35  / 36.36%       | 5856.65 / 30.91%      
 moe                       | 56.80   / 20.00%       | 216.08  / 32.73%       | 259.18  / 18.18%      
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
```

### Run the three-model topology sweep with TruthfulQA (`three_qa.py`)

In Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KmglCM7f0m-qoiWryUlLMIfelowjhJgu?usp=sharing)

Three frozen models (GPT-2, DialoGPT-small, finetuned-gpt2-medical-QA) across eight coupling topologies, with TruthfulQA Health evaluation for all outputs. Set `RUN_TRUTHFUL_QA = True` (default).

Locally:

```bash
python three_qa.py
```

## The core mechanism in ~20 lines

The bridge logic from `benchmark.py`. Everything else in the repository is data loading, model loading, and evaluation plumbing.

```python
class LatentBridge(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode

        # Specialist -> Generalist
        self.proj_b2a = nn.Linear(dim, dim, bias=False)
        self.gate_a   = nn.Parameter(torch.tensor([-2.0]))   # starts near-closed

        if "bilateral" in mode:
            # Generalist -> Specialist (the return signal)
            self.proj_a2b = nn.Linear(dim, dim, bias=False)
            self.gate_b   = nn.Parameter(torch.tensor([-2.0]))

    def forward(self, h_A, h_B):
        ga      = torch.sigmoid(self.gate_a) if "no_gate" not in self.mode else 1.0
        h_A_new = h_A + ga * self.proj_b2a(h_B)          # steer generalist

        if "bilateral" in self.mode:
            gb      = torch.sigmoid(self.gate_b) if "no_gate" not in self.mode else 1.0
            h_B_new = h_B + gb * self.proj_a2b(h_A)      # steer specialist
            return h_A_new, h_B_new

        return h_A_new, h_B                               # B unchanged (unilateral)
```

`RC` wraps two frozen models, inserts a `LatentBridge` at each designated layer, and produces three outputs: the fused logits and each model's individually steered logits. The frozen model parameters never receive gradients. Only the bridge projections and the final mixing scalar are trained.

## Ablation: why linearity is not a compromise

The `bilateral_random` condition is the most informative in the ablation. Projection matrices are fixed at random initialisation; only the gate scalars are trained. On the coding domain this produces a fused PPL of 499.93 against bilateral's 5.91. The gate alone, applied to a random matrix, recovers almost nothing. The bridge's gains require learned projection structure.

| Condition | Fused PPL |
|---|---:|
| Frozen baseline | 16.68 |
| Logit ensemble | 596.16 |
| MoE | 878.40 |
| Unilateral | 15.15 |
| **Bilateral** | **5.91** |
| Bilateral no-gate | 4.95 |
| Bilateral random | 499.93 |

In the legal and coding domains, `bilateral_no_gate` slightly outperforms gated bilateral (8.13 vs. 8.30; 4.95 vs. 5.91). The gate is a conservative default that prevents early-training instability at a small cost in cases where the corrective signal is unambiguous from step 0. In the three-model setting (`three_qa.py`) the pattern reverses: `multi_bilateral_no_gate` reaches 16.42 against `multi_bilateral`'s 11.02, because the more complex coupling surface needs the gate to regulate early contributions.

## Hyperparameters

| Parameter | `benchmark.py` | `three_qa.py` |
|---|---:|---:|
| MAX_STEPS | 2,000 | 2,000 |
| GRAD_ACCUM | 8 | 8 |
| MAX_SEQ_LEN | 128 | 128 |
| TEST_SAMPLES | 25 | 50 |
| Bridge LR | 1e-4 | 1e-4 |
| Router / mix LR | 5e-3 | 5e-3 |
| Gate init | -2.0 | -2.0 |
| Optimiser | AdamW | AdamW |
| Seed | 42 | 42 |

Bridge layers are selected proportionally to model depth: every 6th layer for 36-layer models, every 4th for 24-layer, every 3rd for 12-layer. This proportional depth alignment allows coupling between models with different layer counts without architectural modification of either.

## Models used

| Role | Model | Params | Domain |
|---|---|---:|---|
| Generalist A | `gpt2` | 124M | Medical / Legal / Coding |
| Generalist A | `gpt2-medium` | 345M | Medical (benchmark) |
| Generalist A | `gpt2-large` | 774M | Scientific |
| Specialist B | `microsoft/DialoGPT-medium` | 345M | Medical |
| Specialist B | `microsoft/DialoGPT-small` | 117M | Medical (three_qa) |
| Specialist B | `nrslearning/finetuned-gpt2-medical-QA` | 124M | Medical (three_qa) |
| Specialist B | `isaacus/open-australian-legal-gpt2` | 124M | Legal |
| Specialist B | `microsoft/CodeGPT-small-py` | 124M | Coding |
| Specialist B | `Locutusque/gpt2-large-medical` | 774M | Scientific |

All models are loaded via `AutoModelForCausalLM.from_pretrained`. Vocabulary alignment across heterogeneous pairs is handled by `torch.clamp` before each model's embedding lookup.

## Reading the paper

[![Paper PDF](https://img.shields.io/badge/paper-PDF-red?style=flat-square&logo=adobeacrobat)](https://colab.research.google.com/github/YOUR_USERNAME/differance-engine/blob/main/paper/differance_engine.pdf)

Section 2 situates RC relative to model stitching (Ainsworth et al. 2022; Stoica et al. 2023), representational convergence, and adapter-based methods. Section 3 describes the architecture: bridge layers read one model’s hidden state and inject a gated linear projection into the other model’s residual stream. Section 4 provides the theoretical account of, why frozen models can coordinate through linear operators (Platonic Representation Hypothesis, Huh et al. 2024), why linearity in the bridge is a design virtue rather than a constraint (the bridge cannot memorise; it can only map between memorised spaces), and why operational closure (Maturana and Varela 1980) makes catastrophic forgetting structurally impossible. Section 5 reports the four experiments. The conclusion draws the Mountcastle cortical column analogy: the column is the memorizer and the connectivity is the generalisation space.


## Acknowledgements

Models from the Hugging Face Hub. Datasets: `lavita/ChatDoctor-HealthCareMagic-100k`, `lex_glue/scotus`, `iamtarun/python_code_instructions_18k_alpaca`, `ccdv/pubmed-summarization`. TruthfulQA evaluation uses the Health category of the MC1 split (Lin et al. 2022).

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/)


</div>

