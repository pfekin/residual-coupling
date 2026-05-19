# Scaling LLMs Horizontally: Hidden-State Coupling Without Weight Modification

[![Paper PDF](https://img.shields.io/badge/paper-PDF-red?style=flat-square&logo=adobeacrobat)](https://ssrn.com/abstract=6746521)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/)

The standard approach to specialization modifies models. Fine-tuning overwrites weights. Mixture-of-Experts (MoE) routes tokens to a single expert and discards the rest. Agentic pipelines compress continuous internal geometry into discrete tokens between sequential calls.

**Residual Coupling (RC)** introduces a horizontal scaling axis for multi-model systems, moving beyond vertical scaling via larger monolithic models. This architecture establishes a two-step paradigm where frozen base models function as non-linear memorizers, while lightweight linear bridges handle cross-domain generalization and relational alignment. 

At each designated bridge layer, a small learned projection reads one model's hidden state and injects an additive correction directly into another model's residual stream. Constraining the bridges to purely linear maps restricts overfitting because they can only map existing geometric relationships between the frozen representation spaces. Furthermore, keeping the base weights completely frozen eliminates catastrophic forgetting entirely. The system maintains operational closure, transforming inputs through its existing structure rather than changing its fundamental weights to accommodate them.

Coupling existing frozen models takes a few thousand training steps on a single GPU without modifying a single base weight.

---

## Architecture & Topologies

### Parallel Frozen Stacks Connected by Bridge Projections

<div align="center">
  <img src="architecture.png" alt="Figure 1: RC architecture" width="600"/>
</div>

The bridge at layer $\ell$ from model $B$ (specialist) to model $A$ (generalist) computes an additive correction $\delta \mathbf{h}_A^{(\ell)}$ and injects it into model $A$’s residual stream before layer $\ell+1$ executes:

$$
\delta \mathbf{h}_A^{(\ell)} = \sigma(g^{(\ell, B \to A)}) W^{(\ell, B \to A)} \mathbf{h}_B^{(\ell)}
$$

The frozen layer at $\ell+1$ then processes the perturbed stream through its unchanged weights. The scalar gate $g^{(\ell, B \to A)}$ is initialized to $-2$ ($\sigma(-2) \approx 0.12$), ensuring the additive correction is initially conservative, scaling up only as supported by the training signal. 

The projection matrix $W^{(\ell, B \to A)} \in \mathbb{R}^{d_A \times d_B}$ natively handles dimensional mismatches between heterogeneous models. No attention heads are introduced, sequence positions are unaltered, and the computational overhead per bridge layer is bounded at $O(d^2)$ rather than $O(L^2)$.

### The Four Coupling Topologies

<div align="center">
  <img src="topologies.png" alt="Figure 2: The four topologies" width="600"/>
</div>

Residual Coupling can be deployed across multiple distinct configurations depending on system requirements:
* **Unilateral / Multi-Unilateral:** Directed information flow from specialist columns to a central generalist.
* **Bilateral / Multi-Bilateral:** Simultaneous cross-model bridges forming an inter-layer feedback loop that stabilizes both streams. In complex multi-model environments, bilateral loops ensure that optimization of the fused target objective does not degrade the internal representation spaces of the participating models.

---

## Core Framework Implementation

The following mechanism from `benchmark.py` defines the core bridge steering logic. 

```python
class LatentBridge(nn.Module):
    """A cross-model linear operator that enables inter-layer steering."""
    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode
        
        # Primary steering: Specialist (B) influences Generalist (A)
        self.proj_b2a = nn.Linear(dim, dim, bias=False)
        self.gate_a = nn.Parameter(torch.tensor([-2.0]))
        if "no_gate" in mode: self.gate_a.requires_grad = False
        
        # Secondary steering: Generalist (A) influences Specialist (B) (Bilateral Mode)
        if "bilateral" in mode:
            self.proj_a2b = nn.Linear(dim, dim, bias=False)
            self.gate_b = nn.Parameter(torch.tensor([-2.0]))
            if "no_gate" in mode: self.gate_b.requires_grad = False

    def forward(self, h_A, h_B):
        ga = torch.sigmoid(self.gate_a) if "no_gate" not in self.mode else 1.0
        h_A_new = h_A + ga * self.proj_b2a(h_B)
        
        if "bilateral" in self.mode:
            h_B_new = h_B + gb * self.proj_a2b(h_A)
            return h_A_new, h_B_new
            
        return h_A_new, h_B 
```

The framework's orchestration layer (`ResidualCoupler`) manages the simultaneous forward passes, automatically synchronizing hidden state progress across models with disparate layer counts by evaluating layers proportionally to their respective model depths. 

Heterogeneous vocabulary alignment is handled natively at the hidden-state layer. By clamping token indices before embedding lookups and padding logit bounds prior to mixing, RC allows models utilizing entirely different tokenizers to communicate natively without any shared token-level requirements.

---

## Empirical Benchmarks

### 1. Two-Model Cross-Domain Coupling (`benchmark.py`)

Evaluating bilateral RC against Mixture-of-Experts (MoE) routing across heterogeneous frozen model pairs yields the following perplexity bounds (lower is better):

| Domain | Model A | Model B | Frozen A | MoE | Bilateral | Gain vs. Frozen |
|---|---|---|---:|---:|---:|---:|
| Medical | GPT-2 Medium | DialoGPT-Medium | 50.05 | 64.66 | **12.01** | +76% |
| Legal | GPT-2 | australian-legal-gpt2 | 26.48 | 21.83 | **8.30** | +69% |
| Coding | GPT-2 | CodeGPT-small-py | 16.68 | 878.40 | **5.91** | +65% |
| Scientific | GPT-2 Large | gpt2-large-medical | 28.54 | 26.85 | **17.51** | +39% |

> *Note on Tokenizer Mismatch:* The Coding domain functions as a structural stress test. CodeGPT utilizes a completely different tokenizer, causing a frozen baseline perplexity of ~7M on general text. Traditional token-routing via MoE collapses to 878.40. Bilateral RC achieves a perplexity of 5.91 by extracting the latent underlying signal before the model's output projection collapses into discrete tokens.

### 2. Three-Model Medical Topology Sweep (`three_qa.py`)

Evaluated across the Health category of the TruthfulQA (MC1) split (n=50):

| Topology | Fused PPL | TruthfulQA Health | vs. Frozen Baseline |
|---|---:|---:|---|
| Frozen baseline | 57.08 | 16.36% | -- |
| MoE | 56.80 | 20.00% | -0.5% PPL / +3.6 pp TQA |
| Multi-unilateral | 11.26 | 23.64% | -80.3% PPL / +7.3 pp TQA |
| Star-bilateral | 11.07 | 21.82% | -80.6% PPL / +5.5 pp TQA |
| **Multi-bilateral** | **11.02** | **25.45%** | **-80.7% PPL / +9.1 pp TQA** |
| Hybrid | 11.11 | 23.64% | -80.5% PPL / +7.3 pp TQA |

### 3. Ablation: Structural Constraints vs. Random Matrices

| Condition | Fused PPL |
|---|---:|
| Frozen baseline | 16.68 |
| Logit ensemble | 596.16 |
| MoE | 878.40 |
| Unilateral | 15.15 |
| Bilateral | 5.91 |
| **Bilateral no-gate** | **4.95** |
| Bilateral random | 499.93 |

The `bilateral_random` condition confirms that the bridge's capacity depends entirely on learned projection matrices, as training the gate scalars alone over a random initialization recovers almost no performance. 

---

## Architectural Implications

Residual Coupling moves beyond a post-hoc engineering patch for fine-tuning or prompt-based coordination. It offers several structural advantages for future multi-model system designs:
* **Bypassing the Token Bottleneck**: Whether structured as sequential chains or iterative loops, agentic workflows using continuous hidden states eliminate the discretization loss and generation latency of text token handoffs. Instead of compressing rich geometric signals into discrete text, models interact directly within their representation spaces, preserving complete context and eliminating intermediate decoding overhead.
* **Mitigating Hallucinations:** Because the bridge projections are strictly linear, they can only map existing geometric structures between the representation spaces. As the bridges are optimized against ground-truth target data, they have no incentive to map ungrounded features such as individual models' hallucinations, effectively drowning them out as noise while reinforcing the factual signals.
* **A Path to Native Multi-Modal Integration:** By decoupling non-linear memorization from relational alignment, RC bridges provide a structural framework for scaling multi-model networks and offer a direct path toward native multi-modal integration without modifying base weights.

---

## Get Your Hands Dirty

### 1. Two-Model Cross-Domain Benchmark
Execute the primary benchmark across `"medical"`, `"legal"`, `"coding"`, or `"scientific"` domains:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17EINsJ-s-3ZJb4TiGskqc7_7VI_wI7jc?usp=sharing)

To run locally:
```bash
pip install torch transformers datasets tqdm
python benchmark.py   # Edit DOMAIN at the top of the file
```

### 2. Three-Model Topology Sweep with TruthfulQA
Evaluate eight distinct coupling topologies using three frozen models with active TruthfulQA evaluation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KmglCM7f0m-qoiWryUlLMIfelowjhJgu?usp=sharing)

To run locally:
```bash
python three_qa.py
```

---

## Configuration Details

### Hyperparameters
| Parameter | `benchmark.py` | `three_qa.py` |
|---|---:|---:|
| MAX_STEPS | 2,000 | 2,000 |
| GRAD_ACCUM | 8 | 8 |
| MAX_SEQ_LEN | 128 | 128 |
| TEST_SAMPLES | 25 | 50 |
| Bridge LR | 1e-4 | 1e-4 |
| Router / mix LR | 5e-3 | 5e-3 |
| Gate init | -2.0 | -2.0 |
| Optimizer | AdamW | AdamW |

*Note on Depth Mapping:* Bridge layers are mapped proportionally to model depth (e.g., every 6th layer for 36-layer structures, every 4th for 24-layer, and every 3rd for 12-layer columns). This proportional alignment enables direct coupling between models with vastly different layer counts without structural modification.

### Model Catalog
* **Generalists:** `gpt2` (124M), `gpt2-medium` (345M), `gpt2-large` (774M)
* **Specialists:** `microsoft/DialoGPT-medium` (345M), `microsoft/DialoGPT-small` (117M), `nrslearning/finetuned-gpt2-medical-QA` (124M), `isaacus/open-australian-legal-gpt2` (124M), `microsoft/CodeGPT-small-py` (124M), `Locutusque/gpt2-large-medical` (774M)

---

## Reference & Citation
For a comprehensive theoretical exploration of how frozen models coordinate through linear operators (drawing from the Platonic Representation Hypothesis) and why linearity prevents memorization. The text details how freezing base weights guarantees structural protection against catastrophic forgetting, while the decoupled training of models and bridges offers a novel implementation of Maturana and Varela’s operational closure. Read the full text: 

[![Paper PDF](https://img.shields.io/badge/paper-PDF-red?style=flat-square&logo=adobeacrobat)](https://ssrn.com/abstract=6746521)


If you use this framework or reference these theoretical findings in your research, please cite the work as follows:

**APA Style:**

Ekin, P. (2026). *Computing Between Models with Residual Coupling.* SSRN Electronic Journal. https://ssrn.com/abstract=6746521

**BibTeX:**
```bibtex
@article{residual_coupling_2026,
  title={Computing Between Models with Residual Coupling},
  author={Pascal Ekin},
  journal={SSRN Electronic Journal},
  year={2026},
  url={[https://ssrn.com/abstract=6746521](https://ssrn.com/abstract=6746521)}
}
```

## Acknowledgements
Models provided via Hugging Face Hub. Training and evaluation datasets utilized include `ChatDoctor-HealthCareMagic-100k`, `scotus`, `python_code_instructions_18k_alpaca`, and `pubmed-summarization`. TruthfulQA evaluation relies on the Health category of the MC1 split.
