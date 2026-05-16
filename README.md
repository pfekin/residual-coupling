# Scaling LLMs Horizontally: Hidden-State Coupling Without Weight Modification

[![Paper PDF](https://img.shields.io/badge/paper-PDF-red?style=flat-square&logo=adobeacrobat)](https://ssrn.com/abstract=6746521)

The standard approach to specialization modifies models. Fine-tuning overwrites weights. Mixture-of-Experts routes tokens to a single expert and discards the rest. Agentic pipelines compress continuous internal geometry into discrete tokens between calls.

Residual Coupling (RC) introduces a horizontal scaling axis for multi-model systems. This architecture establishes a two-step paradigm where base models function as non-linear memorizers, while lightweight linear bridges handle cross-domain generalization and relational alignment.At each bridge layer, a small learned projection reads one model's hidden state and adds a corrective update directly into the other model's residual stream, the same stream that each transformer layer already writes into additively. The bridge does not replace or intercept the receiving model's computation, it perturbs the stream that computation reads from. The receiving model then processes that perturbed stream through its own frozen weights, unchanged. 

Bridge projections are linear operators that navigate the structured differences between those memorized spaces. Because they are linear, their capacity to represent arbitrary mappings is sharply constrained compared to a non-linear layer, so what they learn is limited to the linear structure already present in the relationship between the two frozen models’ representation spaces. As the bridges are optimized against ground-truth target data, they have no incentive to map ungrounded features such as individual models' hallucinations.
 
 
Coupling two existing frozen models takes a few thousand training steps on a single GPU and no weights are modified.

## Results at a glance

### Two-model benchmark, four domains (`benchmark.py`)

| Domain | Model A | Model B | Frozen A | MoE | Bilateral | Gain vs. frozen |
|---|---|---|---:|---:|---:|---:|
| Medical | GPT-2 Medium | DialoGPT-Medium | 50.05 | 64.66 | **12.01** | +76% |
| Legal | GPT-2 | australian-legal-gpt2 | 26.48 | 21.83 | **8.30** | +69% |
| Coding | GPT-2 | CodeGPT-small-py | 16.68 | 878.40 | **5.91** | +65% |
| Scientific | GPT-2 Large | gpt2-large-medical | 28.54 | 26.85 | **17.51** | +39% |

> Perplexity (lower is better). Coding is the stress test: CodeGPT uses a different tokenizer, giving it a frozen perplexity of ~7 million on general text. Mixture of Experts (MoE) collapses to 878.40. Bilateral coupling reaches 5.91, below the frozen generalist's 16.68, by learning to extract latent signal without relying on Model B's token-level output.

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


The frozen layer at $\ell$ then processes the perturbed $\mathbf{h}_A^{(\ell)}$ through its unchanged weights. The bridge writes into the residual stream that the layer reads from, consistent with the standard transformer update pattern.

The scalar gate $g^{(\ell, B \to A)}$ is initialized to $-2$, so $\sigma(-2) \approx 0.12$ at step 0 and the additive correction is initially small, increasing only as supported by the training signal. The projection $W^{(\ell, B \to A)} \in \mathbb{R}^{d_A \times d_B}$ naturally handles dimensional mismatch between heterogeneous models. No attention heads are introduced and no sequence positions are added. The computational overhead per bridge layer is $O(d^2)$, rather than $O(L^2)$.




### Figure 2 -- The four coupling topologies

<div align="center">
  <img src="topologies.png" alt="Figure 2: The four topologies" width="600"/>
  <p><em>Figure 2: The four topologies.</em></p>
</div>

In the three-model medical experiment the PPL gap between multi-unilateral (11.26) and multi-bilateral (11.02) is small, though a single domain, dataset, and distribution is insufficient to generalize from. The gap is decisive in the two-model coding domain: unilateral coupling produces a fused PPL of 15.15 but degrades the generalist's individual output to 32.11, worse than its frozen baseline of 16.68. Bilateral coupling corrects this to 11.29 and reaches a fused PPL of 5.91. Without the return signal, the training objective optimises the fused output at the expense of the generalist's own residual stream.

## Get your hands dirty

### Run the two-model benchmark (four domains)

In Colab (no setup):


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17EINsJ-s-3ZJb4TiGskqc7_7VI_wI7jc?usp=sharing)

Set `DOMAIN` at the top of the notebook to `"medical"`, `"legal"`, `"coding"`, or `"scientific"`. Runs in ~25 minutes per domain on a T4.

Locally:

```bash
pip install torch transformers datasets tqdm
python benchmark.py   # edit DOMAIN at the top of the file
```

### Run the three-model topology sweep with TruthfulQA (`three_qa.py`)

In Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KmglCM7f0m-qoiWryUlLMIfelowjhJgu?usp=sharing)

Three frozen models (GPT-2, DialoGPT-small, finetuned-gpt2-medical-QA) across eight coupling topologies, with TruthfulQA Health evaluation for all outputs. Set `RUN_TRUTHFUL_QA = True` (default).

Locally:

```bash
python three_qa.py
```

## The Core Mechanism: Decoupling Memory from Alignment

The system splits the computational workload into a two-step process. Heavy factual acquisition is contained within the frozen, non-linear base networks. Cross-model coordination is offloaded to lightweight linear bridges trained on target datasets. 

The core bridge logic from `benchmark.py` demonstrates this architecture:

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
        # Steer Model A using B's features
        ga = torch.sigmoid(self.gate_a) if "no_gate" not in self.mode else 1.0
        h_A_new = h_A + ga * self.proj_b2a(h_B)
        
        if "bilateral" in self.mode:
            # Steer Model B using A's features
            gb = torch.sigmoid(self.gate_b) if "no_gate" not in self.mode else 1.0
            h_B_new = h_B + gb * self.proj_a2b(h_A)
            return h_A_new, h_B_new
            
        return h_A_new, h_B # Unilateral: B remains un-steered
```

`ResidualCoupler` wraps two frozen models, inserts a `LatentBridge` at each designated layer, and produces three outputs: the fused logits and each model's individually steered logits. One non-obvious detail is vocabulary alignment: when the two models use different tokenizers, `torch.clamp` maps out-of-range token indices to the highest valid index before each model's embedding lookup, keeping both forward passes valid without any shared vocabulary requirement. Logits are then padded to the larger vocabulary size before mixing.

```python
class ResidualCoupler(nn.Module):
    """Wraps two frozen models and manages inter-layer communication via bridges."""
    def __init__(self, model_A, model_B, mode):
        super().__init__()
        self.A, self.B, self.mode = model_A, model_B, mode
        self.v_A, self.v_B = model_A.config.vocab_size, model_B.config.vocab_size
        
        # Detect actual layer counts from model configs
        self.L_A = model_A.config.n_layer
        self.L_B = model_B.config.n_layer
        
        # One LatentBridge per designated layer, keyed by layer index.
        # Bridge layers are selected proportionally to model depth:
        # every 3rd layer for 12-layer models, every 4th for 24-layer,
        # every 6th for 36-layer.
        self.bridges = nn.ModuleDict({str(l): LatentBridge(C["dim"], mode) for l in BRIDGE_LAYERS})
        
        # Learned mixing scalar for the fused output.
        # sigmoid(0.0) = 0.5 at initialization: both models contribute equally
        # before any training begins.
        self.mix = nn.Parameter(torch.tensor([0.0]))

    def forward(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        
        # Clamp token indices to each model's vocabulary size before embedding.
        # This handles heterogeneous tokenizers: indices that exceed a model's
        # vocabulary are mapped to the highest valid index rather than raising
        # an error, with no tokenizer alignment or shared vocabulary required.
        h_A = self.A.transformer.wte(ids.clamp(0, self.v_A-1)) + self.A.transformer.wpe(pos)
        h_B = self.B.transformer.wte(ids.clamp(0, self.v_B-1)) + self.B.transformer.wpe(pos)

        curr_B = 0
        # Iterate through the Generalist's depth
        for i in range(self.L_A):
            # Model A always executes one layer per loop step
            h_A = self.A.transformer.h[i](h_A)[0]
            
            # Model B executes layers proportionally (waiting or running extra blocks)
            # to stay in sync with Model A's fractional progress.
            target_B = int((i + 1) * self.L_B / self.L_A)
            while curr_B < target_B:
                h_B = self.B.transformer.h[curr_B](h_B)[0]
                curr_B += 1
                
            # Bridges are applied relative to Model A's layer index
            if str(i) in self.bridges:
                h_A, h_B = self.bridges[str(i)](h_A, h_B)
	
	# Project final hidden states to logits using each model's own LM head.
        l_A, l_B = self.A.lm_head(self.A.transformer.ln_f(h_A)), self.B.lm_head(self.B.transformer.ln_f(h_B))
        
        # Pad the smaller logit tensor to the larger vocabulary size.
        # Extra positions receive a large negative value so they do not
        # contribute to the probability distribution after softmax.
        max_v = max(self.v_A, self.v_B)
        def pad(l, v): return torch.cat([l, torch.full((*l.shape[:-1], max_v-v), -1e4, device=DEVICE)], dim=-1) if l.size(-1)<max_v else l
        l_A, l_B = pad(l_A, self.v_A), pad(l_B, self.v_B)
        
        # Fused output: weighted combination of both models' logit distributions,
        # with the mixing weight learned alongside the bridge projections.
        m = torch.sigmoid(self.mix)
        return (m * l_A) + ((1-m) * l_B), l_A, l_B
```

## Ablation: why linearity is not a compromise

The `bilateral_random` condition is the most informative in the ablation. Projection matrices are fixed at random initialization and only the gate scalars are trained. On the coding domain this produces a fused PPL of 499.93 against bilateral's 5.91. The gate alone, applied to a random matrix, recovers almost nothing. The bridge's gains require learned projection structure.

| Condition | Fused PPL |
|---|---:|
| Frozen baseline | 16.68 |
| Logit ensemble | 596.16 |
| MoE | 878.40 |
| Unilateral | 15.15 |
| Bilateral | 5.91 |
| **Bilateral no-gate** | **4.95** |
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
| Optimizer | AdamW | AdamW |
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
[![Paper PDF](https://img.shields.io/badge/paper-PDF-red?style=flat-square&logo=adobeacrobat)](https://ssrn.com/abstract=6746521)

Section 2 situates RC relative to model stitching (Ainsworth et al. 2022; Stoica et al. 2023), representational convergence, and adapter-based methods. Section 3 describes the architecture: bridge layers read one model’s hidden state and inject a gated linear projection into the other model’s residual stream. Section 4 provides the theoretical account of why frozen models can coordinate through linear operators (Platonic Representation Hypothesis, Huh et al. 2024), why linearity in the bridge is a design virtue rather than a constraint (the bridge cannot memorize; it can only map between memorized spaces), and why operational closure (Maturana and Varela 1980) makes catastrophic forgetting structurally impossible. Section 5 reports the four experiments. The conclusion draws the Mountcastle cortical column analogy: in RC the column is a memorizer and the connectivity is the generalization space. Crucially, the cycle does not stop at one iteration: new specialists can be added to an existing bridged ensemble at any time, each trained on the manifold established by all preceding frozen columns, with the prior state left intact and recoverable. Maturana and Varela’s (1980) concept of operational closure formalizes why this is structurally guaranteed: a frozen model transforms its input according to its own invariant internal organization rather than being modified by it.

## Architectural Implications

Residual Coupling moves beyond a post-hoc engineering patch for fine-tuning or prompt-based coordination. It offers several structural advantages for future system design:

* **Eliminating Multi-Turn Latency:** Passing continuous hidden states across parallel columns collapses an iterative, multi-turn text prompting loop into a single parallel forward pass. In some specific scenarios, this significantly boosts the speed and capacity of agentic workflows.
* **The Cortical Column Analogy:** This architecture maps directly to Mountcastle's cortical column framework. In the case of RC, localized columns function as invariant domain memorizers, while the cross-model bridge connectivity defines the generalization space.
* **A Path to Multi-Modal Integration:** By decoupling non-linear memorization from relational alignment, RC bridges provide a framework for scaling multi-model systems and offer a path toward native multi-modal integration without modifying base weights.

## Acknowledgements

Models from the Hugging Face Hub. Datasets: `lavita/ChatDoctor-HealthCareMagic-100k`, `lex_glue/scotus`, `iamtarun/python_code_instructions_18k_alpaca`, `ccdv/pubmed-summarization`. TruthfulQA evaluation uses the Health category of the MC1 split (Lin et al. 2022).


---




[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/)



