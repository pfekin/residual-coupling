# Computing Between Models: Residual Coupling of Frozen Transformers

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/)

**Paper:** [link to paper]

---

Transformer language models memorise. They distil statistical patterns from training text at sufficient scale to appear to generalise, but that generalisation is an effect of scale rather than a distinct architectural mechanism. Fine-tuning does not teach a model to understand a new domain. Recent work demonstrates that it reactivates memorised content already present from pretraining [Liu et al., 2026]. The dominant paradigms for specialisation share this premise: capability lives inside individual models, and the question is how to configure or select among them.

Residual Coupling (RC) proposes a different architecture. Models are frozen. Rather than treating any individual model as the locus of capability, RC trains small linear operators on the differences between frozen models' latent representations. The operators are what learns. The models are the substrate.

The frozen models carry absolute knowledge: fixed, parametric, the accumulated trace of their training. The bridge operators carry relational knowledge: continuous, interpolatable, defined by the structured difference between what two closed systems know. The two modes are complementary. The system's capability is a product of both.

Because each model is frozen, it does not adapt to incoming bridge signals. It processes them through its own unchanged weights, maintaining its identity while transforming external input according to its own internal organisation. This is what prevents catastrophic forgetting: not regularisation, but the fact that the condition causing forgetting is never met. Maturana and Varela called this operational closure.

The architecture takes its name from this: the Différance Engine. The bridges are difference operators. What the system can do is a property of both the components and the relations between them.

Two frozen off-the-shelf models can be coupled in a few thousand gradient steps on a single GPU. No base model is modified. No training data is duplicated. The computational cost of specialisation is reduced to the cost of training the bridges alone.

---

## Results

### Experiment 1: Domain generality

One generalist, one specialist, four domains. Bilateral coupling outperforms MoE across all domains.

| Domain | Frozen A | Frozen B | MoE | Unilateral | Bilateral |
|---|---|---|---|---|---|
| Medical | 45.71 | 331.04 | 50.35 | 12.89 | **11.04** (+78.1% vs MoE) |
| Scientific | 35.82 | 34.32 | 31.94 | 21.68 | **21.57** (+32.5%) |
| Coding | 18.54 | ~5.9M | 66.81 | 13.34 | **6.49** (+90.3%) |
| Legal | 24.72 | 38.02 | 19.02 | 7.56 | **6.88** (+63.8%) |

The coding result is notable. The specialist's frozen perplexity on the evaluation text is approximately 5.9 million, placing it entirely out of distribution. MoE achieves 66.81, worse than the frozen generalist alone. Bilateral coupling achieves 6.49. The bridge gates suppress the specialist's failures and extract only the components of its representation that produce consistent corrective updates in the generalist.

[`benchmark.py`](benchmark.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VgymuxR8cDHJ-rIGpU3EnSc-f_fd4dxg?usp=sharing)

### Experiment 2: Multi-specialist topology sweep

Three models, medical domain. With three frozen models, multi-bilateral couples all pairs and produces the lowest perplexity. Star-bilateral couples each specialist to the generalist but not to each other and produces perplexity within 0.31 of multi-bilateral at roughly two thirds of the bridge parameter count.

| Topology | PPL | vs MoE | vs Frozen generalist |
|---|---|---|---|
| Multi-Unilateral | 12.90 | +74.7% | +77.4% |
| Star-Bilateral | 11.68 | +77.1% | +79.5% |
| **Multi-Bilateral** | **11.37** | **+77.7%** | **+80.1%** |
| MoE | 50.99 | baseline | +10.7% |
| Frozen generalist | 57.08 | reference | reference |

[`three.py`](three.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-1YW5g4HsQxCo725dZc0k8OPXj0KK-iE?usp=sharing)

### Experiment 3: Factual accuracy (TruthfulQA Health, MC1)

Perplexity measures fit to the training distribution. Experiment 3 asks whether the noise-suppression mechanism produces measurable improvements in factual accuracy on verifiable questions. All RC topologies improve over MoE. Star-bilateral provides most of the factual accuracy gain at lower parameter cost.

| Topology | PPL vs MoE | TruthfulQA (%) | vs MoE |
|---|---|---|---|
| Multi-Unilateral | +74.7% | 23.64 | +5.46 pp |
| Star-Bilateral | +77.1% | 21.82 | +3.64 pp |
| **Multi-Bilateral** | **+77.7%** | **23.64** | **+5.46 pp** |
| MoE | baseline | 18.18 | reference |

[`qa.py`](qa.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Zt8TwtmLYhGd59mxpOaLjtk_zPtPlBLQ?usp=sharing)

---

## Architecture

<div align="center">
  <img src="architecture.png" alt="Figure 1: RC architecture" width="600"/>
  <p><em>Figure 1: RC architecture.</em></p>
</div>

Bridge update at layer $\ell$ from model $j$ to model $i$:

$$h_i \leftarrow h_i + \sigma(g_{ij}) \cdot W_{j \to i} \cdot h_j$$

$\sigma$ is the sigmoid function. The scalar gate $g_{ij}$ is initialised at $-2.0$, giving $\sigma(-2.0) \approx 0.12$, so bridge contributions begin near zero and grow only as the training objective provides gradient signal. Bridge projections omit bias terms, keeping each bridge as a pure linear map between latent manifolds. The effect on perplexity is negligible.

The bridge gate learns an operation analogous to common-mode rejection in a differential amplifier: components present in both models pass through attenuated, while components absent from the partner model are amplified if they produce consistent corrective updates. In the bilateral case, each model's updated state feeds back into the bridge running in the reverse direction, closing a correction loop across the full depth of the network.

Models of different sizes and depths are coupled through non-square projections and proportional depth alignment: a bridge at generalist layer $\ell$ reads from specialist layer $\lfloor \ell \times L_B / L_A \rceil$.

**Topologies:** unilateral (specialists inject into generalist only), star-bilateral (bidirectional between generalist and each specialist; specialists do not exchange directly), multi-bilateral (bidirectional between all pairs), MoE (routing baseline).

<div align="center">
  <img src="topologies.png" alt="Figure 2: The four topologies" width="600"/>
  <p><em>Figure 2: The four topologies.</em></p>
</div>

**Parameter overhead** for three models, d = 768, five bridge layers: unilateral ~4.7M, star-bilateral ~9.4M, multi-bilateral ~14.2M, against 124M per frozen base model and ~2.3K for the MoE router.

---

## Usage

```bash
pip install torch transformers datasets
```

The script below is the two-model bilateral case in full. Set `DOMAIN = "legal"` to switch domains. Results reproduce those in the paper.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
import random
import numpy as np
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
DOMAIN         = "coding"    # "coding", "legal", "scientific", "medical"
MODE           = "bilateral"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42
MAX_STEPS      = 2000
GRAD_ACCUM     = 8
MAX_SEQ_LEN    = 128
TEST_SAMPLES   = 20

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

CONFIGS = {
    "medical": {
        "A": "gpt2-medium", "B": "microsoft/DialoGPT-medium",
        "dataset": "lavita/ChatDoctor-HealthCareMagic-100k",
        "dim": 1024, "layers": 24,
        "map": lambda x: f"Patient: {x.get('instruction', '')[:200]} Doctor: {x.get('output', '')[:200]}"
    },
    "scientific": {
        "A": "gpt2-large", "B": "Locutusque/gpt2-large-medical",
        "dataset": "ccdv/pubmed-summarization", "subset": "document",
        "dim": 1280, "layers": 36, "map": lambda x: x["article"][:600]
    },
    "legal": {
        "A": "gpt2", "B": "isaacus/open-australian-legal-gpt2",
        "dataset": "lex_glue", "subset": "scotus",
        "dim": 768, "layers": 12, "map": lambda x: x["text"][:600]
    },
    "coding": {
        "A": "gpt2", "B": "microsoft/CodeGPT-small-py",
        "dataset": "iamtarun/python_code_instructions_18k_alpaca",
        "dim": 768, "layers": 12,
        "map": lambda x: f"Instruction: {x['instruction']}\nCode: {x['output'][:400]}"
    }
}

C = CONFIGS[DOMAIN]
if C["layers"] == 36:   BRIDGE_LAYERS = [6, 12, 18, 24, 30]
elif C["layers"] == 24: BRIDGE_LAYERS = [4, 8, 12, 16, 20]
else:                   BRIDGE_LAYERS = [3, 6, 9]

# =============================================================================
# ARCHITECTURE
# =============================================================================

class LatentBridge(nn.Module):
    def __init__(self, dim, mode="bilateral"):
        super().__init__()
        self.mode = mode
        self.proj_a2b = nn.Linear(dim, dim, bias=False)
        self.gate_b   = nn.Parameter(torch.tensor(-2.0))
        if mode == "bilateral":
            self.proj_b2a = nn.Linear(dim, dim, bias=False)
            self.gate_a   = nn.Parameter(torch.tensor(-2.0))

class LatentMoE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.router = nn.Linear(dim, 2)
    def forward(self, h_A, h_B):
        w = torch.softmax(self.router(h_A + h_B), dim=-1)
        fused = w[:, :, 0:1] * h_A + w[:, :, 1:2] * h_B
        return fused, fused

class DifferanceEngine(nn.Module):
    def __init__(self, model_A, model_B, mode="bilateral"):
        super().__init__()
        self.mode    = mode
        self.model_A = model_A
        self.model_B = model_B
        self.v_A     = model_A.config.vocab_size
        self.v_B     = model_B.config.vocab_size
        set_seed(SEED)
        if mode == "moe":
            self.bridges = nn.ModuleDict({str(l): LatentMoE(C["dim"]) for l in BRIDGE_LAYERS})
        else:
            self.bridges = nn.ModuleDict({str(l): LatentBridge(C["dim"], mode) for l in BRIDGE_LAYERS})
        self.final_mix = nn.Parameter(torch.tensor(0.0))

    def forward(self, ids):
        ids_A = torch.clamp(ids, 0, self.v_A - 1)
        ids_B = torch.clamp(ids, 0, self.v_B - 1)
        pos   = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h_A   = (self.model_A.transformer.wte(ids_A) + self.model_A.transformer.wpe(pos)).clone()
        h_B   = (self.model_B.transformer.wte(ids_B) + self.model_B.transformer.wpe(pos)).clone()

        for i in range(C["layers"]):
            h_A_next = self.model_A.transformer.h[i](h_A)[0]
            h_B_next = self.model_B.transformer.h[i](h_B)[0]
            if str(i) in self.bridges:
                br = self.bridges[str(i)]
                if self.mode == "moe":
                    h_A_next, h_B_next = br(h_A_next, h_B_next)
                else:
                    delta_B  = br.proj_a2b(h_A_next) * torch.sigmoid(br.gate_b)
                    h_B_next = h_B_next + delta_B
                    if self.mode == "bilateral":
                        delta_A  = br.proj_b2a(h_B_next) * torch.sigmoid(br.gate_a)
                        h_A_next = h_A_next + delta_A
            h_A, h_B = h_A_next, h_B_next

        l_A = self.model_A.lm_head(self.model_A.transformer.ln_f(h_A))
        l_B = self.model_B.lm_head(self.model_B.transformer.ln_f(h_B))
        max_v = max(self.v_A, self.v_B)
        if l_A.size(-1) < max_v:
            l_A = torch.cat([l_A, torch.full((*l_A.shape[:-1], max_v - self.v_A),
                             -1e4, device=ids.device, dtype=l_A.dtype)], dim=-1)
        if l_B.size(-1) < max_v:
            l_B = torch.cat([l_B, torch.full((*l_B.shape[:-1], max_v - self.v_B),
                             -1e4, device=ids.device, dtype=l_B.dtype)], dim=-1)
        mix = torch.sigmoid(self.final_mix)
        return mix * l_A + (1 - mix) * l_B

# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def run_experiment():
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(C["A"])
    tokenizer.pad_token = tokenizer.eos_token

    model_A = AutoModelForCausalLM.from_pretrained(C["A"]).to(DEVICE)
    model_B = AutoModelForCausalLM.from_pretrained(C["B"]).to(DEVICE)
    for p in list(model_A.parameters()) + list(model_B.parameters()):
        p.requires_grad = False

    subset = C.get("subset")
    ds = load_dataset(C["dataset"], subset, split="train", streaming=True,
                      trust_remote_code=True) if subset \
         else load_dataset(C["dataset"], split="train", streaming=True,
                           trust_remote_code=True)
    ds = ds.shuffle(seed=SEED, buffer_size=1000)
    it = iter(ds)

    test_texts = [C["map"](next(it)) for _ in range(TEST_SAMPLES)]
    test_ids   = tokenizer(test_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(DEVICE)

    engine    = DifferanceEngine(model_A, model_B, mode=MODE).to(DEVICE).to(model_A.dtype)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in engine.named_parameters()
                    if p.requires_grad and "mix" not in n], "lr": 1e-4},
        {"params": [engine.final_mix],                      "lr": 5e-3},
    ])

    engine.train()
    for i in tqdm(range(MAX_STEPS)):
        try:    ex = next(it)
        except StopIteration: break
        ids = tokenizer(C["map"](ex), return_tensors="pt", truncation=True,
                        max_length=MAX_SEQ_LEN).input_ids.to(DEVICE)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(
            engine(ids)[:, :-1].reshape(-1, max(engine.v_A, engine.v_B)),
            ids[:, 1:].reshape(-1)
        )
        (loss / GRAD_ACCUM).backward()
        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    engine.eval()
    with torch.no_grad():
        mask = test_ids[:, 1:] != tokenizer.pad_token_id

        def get_ppl(logits, ids):
            ls = logits[:, :-1].contiguous().float()
            ts = ids[:, 1:].contiguous()
            ts = torch.clamp(ts, 0, ls.size(-1) - 1)
            loss = F.cross_entropy(ls.view(-1, ls.size(-1)), ts.view(-1), reduction="none")
            avg  = loss.view(ts.size(0), -1)[mask].mean().item()
            return math.exp(avg) if avg < 20 else 1e12

        ppl_A   = get_ppl(model_A(test_ids.clamp(0, engine.v_A - 1)).logits, test_ids)
        ppl_B   = get_ppl(model_B(test_ids.clamp(0, engine.v_B - 1)).logits, test_ids)
        ppl_syn = get_ppl(engine(test_ids), test_ids)
        gain    = (min(ppl_A, ppl_B) - ppl_syn) / min(ppl_A, ppl_B) * 100

        print(f"\nFrozen A:  {ppl_A:.2f}")
        print(f"Frozen B:  {ppl_B:.2f}")
        print(f"Bilateral: {ppl_syn:.2f}  ({gain:+.2f}% vs best frozen)")
        # coding: Frozen A 18.54  Frozen B ~5.9M  Bilateral  6.49  (+65.00%)
        # legal:  Frozen A 24.72  Frozen B 38.02  Bilateral  6.88  (+72.15%)

if __name__ == "__main__":
    run_experiment()
```

Modules follow a last-in-first-out protocol for clean removal: a specialist can be detached without retraining provided no further bridges have been trained on top of it. Columns could also be trained sequentially from scratch and frozen in turn, each one added to an existing coupled system, accumulating domain knowledge incrementally without disturbing what has already been learned.

---

## Notes

Vocabulary alignment uses token clamping for heterogeneous model pairs. A learned soft mapping between embedding matrices is the natural extension.

Multi-bilateral scales as O(n²) in bridge parameters. Star-bilateral is sufficient for most deployments and provides comparable factual accuracy gains.

A proprietary specialist can run on a local or edge device with no exposure of its weights, communicating with a remotely hosted generalist anchor through bridge tensors and latent activations only. The bridge parameter count for a single specialist at d = 768 over five layers is approximately 4.7M, well within the memory envelope of current edge hardware.

All experiments cover 124M to 774M parameter models. The mechanism is not architecture-specific. Behaviour at larger scales has not yet been examined.

Cross-modal coupling is a direct extension of the same mechanism and remains to be evaluated. Joint Embedding Predictive Architectures [LeCun, 2022] share the same commitment to latent-space computation as the medium of inter-component coordination; RC applies an analogous principle to corrective coupling between independently trained models rather than predictive learning between views.

---

## Citation

```bibtex
@article{ekin2026rc,
  title={Computing Between Models: Residual Coupling of Frozen Transformers},
  author={Ekin, Pascal},
  year={2026}
}
```

Pascal Ekin — [pfekin@gmail.com](mailto:pfekin@gmail.com)
