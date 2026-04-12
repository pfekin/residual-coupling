# Computing Between Models: Residual Coupling of Frozen Transformers

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/)

**Paper:** [link to paper]

---

Residual Coupling trains small linear operators on the *differences* between frozen models' latent states. All models process the same input in parallel. At selected transformer layers, each model receives a learned corrective update derived from the others' hidden states. No base model is ever modified.

The standard alternative is to route: pick one expert and discard the rest. RC does not route. It asks what the difference between models contains and learns to make that difference useful.

Because each model is frozen, it does not adapt to incoming bridge signals. It processes them through its own unchanged weights. The signal that propagates forward is not the bridge update but the bridge update as interpreted by a system that has remained itself. This is what prevents catastrophic forgetting: not regularisation, but the fact that the condition causing forgetting is never met. Maturana and Varela called this operational closure. The bridge parameters are the coupling site between closed systems, not a modification of either.

Independently trained models share factual structure but have uncorrelated confabulation. During bidirectional coupling, shared signal is present in both representations and reinforces. Model-specific noise is absent from the partner model and does not. The TruthfulQA results below show this is not only a perplexity effect: all RC topologies improve factual accuracy over MoE on medical questions.

The architecture takes its name from this: the Différance Engine. The bridges are difference operators. The system's capability resides in the gap between its components rather than in any component alone.

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

[`benchmark.py`](benchmark.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/benchmark.py)

### Experiment 2: Multi-specialist topology sweep

Three models, medical domain. Multi-bilateral achieves the lowest perplexity, with star-bilateral close behind at lower parameter cost.

| Topology | PPL | vs MoE | vs Frozen generalist |
|---|---|---|---|
| Multi-Unilateral | 12.90 | +74.7% | +77.4% |
| Star-Bilateral | 11.68 | +77.1% | +79.5% |
| **Multi-Bilateral** | **11.37** | **+77.7%** | **+80.1%** |
| MoE | 50.99 | baseline | +10.7% |
| Frozen generalist | 57.08 | reference | reference |

Star-Bilateral and Multi-bilateral achieves roughly seven times the perplexity gain of MoE. In this example the 0.31 PPL advantage over star-bilateral comes at roughly 50% more bridge parameters.

[`three.py`](three.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/three.py)

### Experiment 3: Factual accuracy (TruthfulQA Health, MC1)

| Topology | PPL vs MoE | TruthfulQA (%) | vs MoE |
|---|---|---|---|
| Multi-Unilateral | +74.7% | 23.64 | +5.46 pp |
| Star-Bilateral | +77.1% | 21.82 | +3.64 pp |
| **Multi-Bilateral** | **+77.7%** | **23.64** | **+5.46 pp** |
| MoE | baseline | 18.18 | reference |

All RC topologies improve factual accuracy over MoE. Star-bilateral provides most of the factual accuracy gain at lower parameter cost and is the practical choice where perplexity is not the primary objective.

[`qa.py`](qa.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/qa.py)

---

## Architecture

Bridge update at layer $ℓ$ from model $j$ to model $i$:

$h_i ← h_i + σ(g_{ij}) · W_{j→i} · h_j$

$σ$ is the sigmoid function. The scalar gate $g_{ij}$ is initialised at −2.0, giving $σ(−2.0) ≈ 0.12$, so bridge contributions begin near zero and grow only as the training objective provides gradient signal. Bridge projections omit bias terms, keeping each bridge as a pure linear map between latent manifolds. The effect on perplexity is negligible.

Models of different sizes and depths are coupled through non-square projections and proportional depth alignment: a bridge at generalist layer ℓ reads from specialist layer $floor(ℓ × L_B / L_A)$.

**Topologies:** unilateral (specialists inject into generalist only), star-bilateral (bidirectional between generalist and each specialist and specialists do not exchange directly), multi-bilateral (bidirectional between all pairs), MoE (routing baseline).

**Parameter overhead** for three models, d = 768, five bridge layers: unilateral ~4.7M, star-bilateral ~9.4M, multi-bilateral ~14.2M, against 124M per frozen base model and ~2.3K for the MoE router.

---

## Usage

```bash
pip install torch transformers datasets
```

The script below is the two-model bilateral case in full. Set `DOMAIN = "legal"` to switch domains. Results are reproduced from `benchmark.py`.

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
# STANDALONE CONFIGURATION
# =============================================================================
DOMAIN         = "legal"     # Options: "scientific", "legal", "coding", "medical"
MODE           = "bilateral"  # Options: "unilateral", "bilateral", "moe"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42

# Replicating high-gain hyperparameters from benchmark.py
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
        "dim": 1280, "layers": 36, "map": lambda x: x['article'][:600]
    },
    "legal": {
        "A": "gpt2", "B": "isaacus/open-australian-legal-gpt2",
        "dataset": "lex_glue", "subset": "scotus",
        "dim": 768, "layers": 12, "map": lambda x: x['text'][:600]
    },
    "coding": {
        "A": "gpt2", "B": "microsoft/CodeGPT-small-py",
        "dataset": "iamtarun/python_code_instructions_18k_alpaca",
        "dim": 768, "layers": 12, "map": lambda x: f"Instruction: {x['instruction']}\nCode: {x['output'][:400]}"
    }
}

C = CONFIGS[DOMAIN]
# Dynamic bridge selection mapped to model depth
if C["layers"] == 36: BRIDGE_LAYERS = [6, 12, 18, 24, 30]
elif C["layers"] == 24: BRIDGE_LAYERS = [4, 8, 12, 16, 20]
else: BRIDGE_LAYERS = [3, 6, 9]

# =============================================================================
# ARCHITECTURE MODULES (Replicating benchmark.py logic)
# =============================================================================

class LatentBridge(nn.Module):
    def __init__(self, dim, mode="bilateral"):
        super().__init__()
        self.mode = mode
        # Initializing gates at -2.0 for stability as seen in source
        self.proj_a2b, self.gate_b = nn.Linear(dim, dim), nn.Parameter(torch.tensor([-2.0]))
        if mode == "bilateral":
            self.proj_b2a, self.gate_a = nn.Linear(dim, dim), nn.Parameter(torch.tensor([-2.0]))

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
        self.mode, self.model_A, self.model_B = mode, model_A, model_B
        self.v_A, self.v_B = self.model_A.config.vocab_size, self.model_B.config.vocab_size
        set_seed(SEED)
        if mode == "moe":
            self.bridges = nn.ModuleDict({str(l): LatentMoE(C["dim"]) for l in BRIDGE_LAYERS})
        else:
            self.bridges = nn.ModuleDict({str(l): LatentBridge(C["dim"], mode) for l in BRIDGE_LAYERS})
        self.final_mix = nn.Parameter(torch.tensor([0.0]))

    def forward(self, ids):
        # Safety: Per-model clamping prevents CUDA Assert errors during vocab mismatch
        ids_A, ids_B = torch.clamp(ids, 0, self.v_A-1), torch.clamp(ids, 0, self.v_B-1)
        pos = torch.arange(0, ids.size(1), device=ids.device).unsqueeze(0)
        h_A = (self.model_A.transformer.wte(ids_A) + self.model_A.transformer.wpe(pos)).clone()
        h_B = (self.model_B.transformer.wte(ids_B) + self.model_B.transformer.wpe(pos)).clone()

        for i in range(C["layers"]):
            h_A_next, h_B_next = self.model_A.transformer.h[i](h_A)[0], self.model_B.transformer.h[i](h_B)[0]
            if str(i) in self.bridges:
                br = self.bridges[str(i)]
                if self.mode == "moe": h_A_next, h_B_next = br(h_A_next, h_B_next)
                else:
                    # Bounded updates using sigmoid-gated projections
                    delta_B = br.proj_a2b(h_A_next) * torch.sigmoid(br.gate_b)
                    h_B_fused = h_B_next + delta_B
                    if self.mode == "bilateral":
                        delta_A = br.proj_b2a(h_B_next) * torch.sigmoid(br.gate_a)
                        h_A_next = h_A_next + delta_A
                    h_B_next = h_B_fused
            h_A, h_B = h_A_next, h_B_next

        l_A, l_B = self.model_A.lm_head(self.model_A.transformer.ln_f(h_A)), self.model_B.lm_head(self.model_B.transformer.ln_f(h_B))
        
        # Logit Alignment: Neutral padding for missing vocabulary indices
        max_v = max(self.v_A, self.v_B)
        if l_A.size(-1) < max_v: 
            l_A = torch.cat([l_A, torch.full((*l_A.shape[:-1], max_v - self.v_A), -1e4, device=DEVICE, dtype=l_A.dtype)], dim=-1)
        if l_B.size(-1) < max_v: 
            l_B = torch.cat([l_B, torch.full((*l_B.shape[:-1], max_v - self.v_B), -1e4, device=DEVICE, dtype=l_B.dtype)], dim=-1)
        
        mix = torch.sigmoid(self.final_mix)
        return (mix * l_A) + ((1-mix) * l_B)

# =============================================================================
# DATA & REPRODUCTION SUITE
# =============================================================================

def run_experiment():
    set_seed(SEED)
    print(f"Initializing {DOMAIN.upper()} Benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(C["A"])
    tokenizer.pad_token = tokenizer.eos_token

    model_A = AutoModelForCausalLM.from_pretrained(C["A"]).to(DEVICE)
    model_B = AutoModelForCausalLM.from_pretrained(C["B"]).to(DEVICE)
    for p in model_A.parameters(): p.requires_grad = False
    for p in model_B.parameters(): p.requires_grad = False

    # Dynamic dataset loading matching benchmark.py
    subset = C.get("subset")
    ds = load_dataset(C["dataset"], subset, split="train", streaming=True, trust_remote_code=True) if subset \
         else load_dataset(C["dataset"], split="train", streaming=True, trust_remote_code=True)

    ds = ds.shuffle(seed=SEED, buffer_size=1000)
    it = iter(ds)
    test_texts = [C["map"](next(it)) for _ in range(TEST_SAMPLES)]
    test_ids = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(DEVICE)

    # Bridge selection and engine setup
    engine = DifferanceEngine(model_A, model_B, mode=MODE).to(DEVICE).to(model_A.dtype)
    
    # Differential learning rates for routers and mix parameters
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in engine.named_parameters() if p.requires_grad and "router" not in n and "mix" not in n], "lr": 1e-4},
        {"params": [p for n, p in engine.named_parameters() if p.requires_grad and ("router" in n or "mix" in n)], "lr": 5e-3}
    ])

    print(f"Training (Dtype: {model_A.dtype} | Mode: {MODE.upper()} | Steps: {MAX_STEPS})")
    engine.train()
    pbar = tqdm(range(MAX_STEPS))
    for i in pbar:
        try: ex = next(it)
        except StopIteration: break
        ids = tokenizer(C["map"](ex), return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(DEVICE)
        if ids.size(1) < 2: continue
        
        logits = engine(ids)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
        (loss / GRAD_ACCUM).backward()
        
        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0) # Added clipping for production stability
            optimizer.step(); optimizer.zero_grad()
            pbar.set_description(f"Loss: {loss.item():.4f}")

    engine.eval()
    with torch.no_grad():
        mask = test_ids[:, 1:] != tokenizer.pad_token_id
        
        # Reproducing Benchmark Perplexity Passes
        syn_logits = engine(test_ids)
        ids_A, ids_B = torch.clamp(test_ids, 0, engine.v_A - 1), torch.clamp(test_ids, 0, engine.v_B - 1)
        pure_l_A = engine.model_A(ids_A).logits
        pure_l_B = engine.model_B(ids_B).logits

        def get_ppl(l, t, m):
            l_s, t_s = l[:, :-1, :].contiguous(), t[:, 1:].contiguous()
            if l_s.size(-1) < t_s.max().item() + 1: t_s = torch.clamp(t_s, 0, l_s.size(-1) - 1)
            loss = F.cross_entropy(l_s.view(-1, l_s.size(-1)).float(), t_s.view(-1), reduction='none')
            avg_loss = loss.view(t_s.size(0), -1)[m].mean().item()
            return math.exp(avg_loss) if avg_loss < 20 else 1e12

        ppl_a, ppl_b, ppl_syn = get_ppl(pure_l_A, test_ids, mask), get_ppl(pure_l_B, test_ids, mask), get_ppl(syn_logits, test_ids, mask)
        gain_pct = ((min(ppl_a, ppl_b) - ppl_syn) / min(ppl_a, ppl_b)) * 100

        print("\n" + "═"*55)
        print(f" FINAL REPRODUCTION: {DOMAIN.upper()} | {MODE.upper()}")
        print("─"*55)
        print(f" Frozen Generalist (A) PPL:  {ppl_a:.2f}")
        print(f" Frozen Specialist (B) PPL:  {ppl_b:.2f}")
        print(f" SYNERGY Result PPL:        {ppl_syn:.2f}")
        print("─"*55)
        print(f" SYNERGY GAIN:              {gain_pct:+.2f}%")
        print(f" Logit Mixture (A Weight):   {torch.sigmoid(engine.final_mix).item():.2%}")
        print("═"*55)

if __name__ == "__main__":
    run_experiment()
```

Modules follow a last-in-first-out protocol for clean removal: a specialist can be detached without retraining provided no further bridges have been trained on top of it.

---

## Notes

Vocabulary alignment uses token clamping for heterogeneous model pairs. A learned soft mapping between embedding matrices is the natural extension.

Multi-bilateral scales as O(n²) in bridge parameters. Star-bilateral is sufficient for most deployments and provides comparable factual accuracy gains.

All experiments cover 124M to 774M parameter models. The mechanism is not architecture-specific. Behaviour at larger scales has not yet been examined.

Cross-modal coupling is a direct extension of the same mechanism and remains to be evaluated.

---

## Citation

```bibtex
@article{ekin2025rc,
  title={Computing Between Models: Residual Coupling of Frozen Transformers},
  author={Ekin, Pascal},
  year={2026}
}
```

Pascal Ekin — [pfekin@gmail.com](mailto:pfekin@gmail.com)
