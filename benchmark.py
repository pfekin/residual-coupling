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
# SETTINGS & CONFIG
# =============================================================================
DOMAIN         = "medical"     # Options: "scientific", "legal", "coding", "medical"
MODE           = "moe"   # Options: "unilateral", "bilateral", "moe"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42

# HYPERPARAMETERS
MAX_STEPS      = 500
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
        "A": "gpt2-medium",
        "B": "microsoft/DialoGPT-medium",
        "dataset": "lavita/ChatDoctor-HealthCareMagic-100k",
        "subset": None,
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
        "dataset": "iamtarun/python_code_instructions_18k_alpaca", "subset": None,
        "dim": 768, "layers": 12, "map": lambda x: f"Instruction: {x['instruction']}\nCode: {x['output'][:400]}"
    }
}

C = CONFIGS[DOMAIN]
# Dynamic bridge selection based on model depth
if C["layers"] == 36: BRIDGE_LAYERS = [6, 12, 18, 24, 30]
elif C["layers"] == 24: BRIDGE_LAYERS = [4, 8, 12, 16, 20]
else: BRIDGE_LAYERS = [3, 6, 9]

# =============================================================================
# ARCHITECTURE MODULES
# =============================================================================

class LatentBridge(nn.Module):
    def __init__(self, dim, mode="bilateral"):
        super().__init__()
        self.mode = mode
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
                    delta_B = br.proj_a2b(h_A_next) * torch.sigmoid(br.gate_b)
                    h_B_fused = h_B_next + delta_B
                    if self.mode == "bilateral":
                        delta_A = br.proj_b2a(h_B_next) * torch.sigmoid(br.gate_a)
                        h_A_next = h_A_next + delta_A
                    h_B_next = h_B_fused
            h_A, h_B = h_A_next, h_B_next

        l_A, l_B = self.model_A.lm_head(self.model_A.transformer.ln_f(h_A)), self.model_B.lm_head(self.model_B.transformer.ln_f(h_B))
        max_v = max(self.v_A, self.v_B)
        if l_A.size(-1) < max_v: l_A = torch.cat([l_A, torch.full((*l_A.shape[:-1], max_v - self.v_A), -1e4, device=DEVICE, dtype=l_A.dtype)], dim=-1)
        if l_B.size(-1) < max_v: l_B = torch.cat([l_B, torch.full((*l_B.shape[:-1], max_v - self.v_B), -1e4, device=DEVICE, dtype=l_B.dtype)], dim=-1)
        mix = torch.sigmoid(self.final_mix)
        return (mix * l_A) + ((1-mix) * l_B), l_A, l_B

# =============================================================================
# DATA & EXECUTION
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

    # Dataset selection
    if C["subset"]: ds = load_dataset(C["dataset"], C["subset"], split="train", streaming=True, trust_remote_code=True)
    else: ds = load_dataset(C["dataset"], split="train", streaming=True, trust_remote_code=True)

    ds = ds.shuffle(seed=SEED, buffer_size=1000)
    it = iter(ds)
    test_texts = [C["map"](next(it)) for _ in range(TEST_SAMPLES)]
    test_ids = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(DEVICE)

    # Engine & Dtype Sync
    engine = DifferanceEngine(model_A, model_B, mode=MODE).to(DEVICE).to(model_A.dtype)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in engine.named_parameters() if p.requires_grad and "router" not in n and "mix" not in n], "lr": 1e-4},
        {"params": [p for n, p in engine.named_parameters() if p.requires_grad and ("router" in n or "mix" in n)], "lr": 5e-3}
    ])

    print(f"Training Started (Dtype: {model_A.dtype} | Mode: {MODE.upper()})")
    engine.train()
    pbar = tqdm(range(MAX_STEPS))
    for i in pbar:
        try: ex = next(it)
        except StopIteration: break
        ids = tokenizer(C["map"](ex), return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(DEVICE)
        if ids.size(1) < 2: continue
        logits, _, _ = engine(ids)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
        (loss / GRAD_ACCUM).backward()
        if (i + 1) % GRAD_ACCUM == 0:
            optimizer.step(); optimizer.zero_grad()
            pbar.set_description(f"Loss: {loss.item():.4f}")

    # --- EVALUATION ---
    engine.eval()
    with torch.no_grad():
        mask = test_ids[:, 1:] != tokenizer.pad_token_id

        # Benchmarking passes
        syn_logits, _, _ = engine(test_ids)
        ids_A, ids_B = torch.clamp(test_ids, 0, engine.v_A - 1), torch.clamp(test_ids, 0, engine.v_B - 1)
        pure_l_A = engine.model_A(ids_A).logits
        pure_l_B = engine.model_B(ids_B).logits

        def get_ppl(l, t, m):
            l_s, t_s = l[:, :-1, :].contiguous(), t[:, 1:].contiguous()
            if l_s.size(-1) < t_s.max().item() + 1: t_s = torch.clamp(t_s, 0, l_s.size(-1) - 1)
            loss = F.cross_entropy(l_s.view(-1, l_s.size(-1)).float(), t_s.view(-1), reduction='none')
            return math.exp(loss.view(t_s.size(0), -1)[m].mean().item())

        ppl_a, ppl_b, ppl_syn = get_ppl(pure_l_A, test_ids, mask), get_ppl(pure_l_B, test_ids, mask), get_ppl(syn_logits, test_ids, mask)

        # Calculate Synergy Gain % relative to the best baseline
        best_baseline = min(ppl_a, ppl_b)
        gain_pct = ((best_baseline - ppl_syn) / best_baseline) * 100

        print("\n" + "═"*55)
        print(f" FINAL BENCHMARK: {DOMAIN.upper()} | {MODE.upper()}")
        print("─"*55)
        print(f" Frozen Generalist (A) PPL:  {ppl_a:.2f}")
        print(f" Frozen Specialist (B) PPL:  {ppl_b:.2f}")
        print(f" Differance Result PPL:       {ppl_syn:.2f}")
        print("─"*55)
        print(f" DIFFERANCE GAIN:               {gain_pct:+.2f}%")
        print(f" Logit Mixture (A Weight):   {torch.sigmoid(engine.final_mix).item():.2%}")
        print("═"*55)

if __name__ == "__main__":
    run_experiment()

"""
MAX_STEPS      = 2000
═══════════════════════════════════════════════════════
 FINAL BENCHMARK: MEDICAL | MOE
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:  47.51
 Frozen Specialist (B) PPL:  331.04
 DIFFERANCE Result PPL:       50.35
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:            -6.00%
 Logit Mixture (A Weight):   74.47%
═══════════════════════════════════════════════════════
 FINAL BENCHMARK: MEDICAL | UNILATERAL
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:  47.51
 Frozen Specialist (B) PPL:  331.04
 DIFFERANCE Result PPL:       12.89
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:           +72.87%
 Logit Mixture (A Weight):   55.58%
═══════════════════════════════════════════════════════
 FINAL BENCHMARK: MEDICAL | BILATERAL
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:   47.51
 Frozen Specialist (B) PPL:  331.04
 DIFFERANCE Result PPL:       11.04
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:           +76.76%
 Logit Mixture (A Weight):   64.43%
═══════════════════════════════════════════════════════




═══════════════════════════════════════════════════════
 FINAL BENCHMARK: SCIENTIFIC | MOE
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:   35.82
 Frozen Specialist (B) PPL:   34.32
 DIFFERANCE Result PPL:       31.94
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:            +6.92%
 Logit Mixture (A Weight):   55.81%
═══════════════════════════════════════════════════════
 FINAL BENCHMARK: SCIENTIFIC | UNILATERAL
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:   35.82
 Frozen Specialist (B) PPL:   34.32
 DIFFERANCE Result PPL:       21.68
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:            +36.82%
 Logit Mixture (A Weight):    34.31%
═══════════════════════════════════════════════════════
 FINAL BENCHMARK: SCIENTIFIC | BILATERAL
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:  35.82
 Frozen Specialist (B) PPL:  34.32
 DIFFERANCE Result PPL:      21.57
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:               +37.14%
 Logit Mixture (A Weight):   49.88%
═══════════════════════════════════════════════════════


═══════════════════════════════════════════════════════
 FINAL BENCHMARK: CODING | MOE
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:  18.54
 Frozen Specialist (B) PPL:  5954296.87
 DIFFERANCE Result PPL:      66.81
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:          -260.39%
 Logit Mixture (A Weight):   71.91%
═══════════════════════════════════════════════════════
 FINAL BENCHMARK: CODING | UNILATERAL
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:  18.54
 Frozen Specialist (B) PPL:  5954296.87
 DIFFERANCE Result PPL:       13.34
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:            +28.05%
 Logit Mixture (A Weight):    65.88%
═══════════════════════════════════════════════════════
 FINAL BENCHMARK: CODING | BILATERAL
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:  18.54
 Frozen Specialist (B) PPL:  5954296.87
 DIFFERANCE Result PPL:       6.49
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:           +65.00%
 Logit Mixture (A Weight):   62.54%
═══════════════════════════════════════════════════════


═══════════════════════════════════════════════════════
 FINAL BENCHMARK: LEGAL | MOE
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:  24.72
 Frozen Specialist (B) PPL:  38.02
 DIFFERANCE Result PPL:       19.02
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:            +23.05%
 Logit Mixture (A Weight):   72.51%
═══════════════════════════════════════════════════════
 FINAL BENCHMARK: LEGAL | UNILATERAL
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:  24.72
 Frozen Specialist (B) PPL:  38.02
 DIFFERANCE Result PPL:       7.56
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:           +69.40%
 Logit Mixture (A Weight):   35.34%
═══════════════════════════════════════════════════════
 FINAL BENCHMARK: LEGAL | BILATERAL
───────────────────────────────────────────────────────
 Frozen Generalist (A) PPL:  24.72
 Frozen Specialist (B) PPL:  38.02
 DIFFERANCE Result PPL:       6.88
───────────────────────────────────────────────────────
 DIFFERANCE GAIN:           +72.15%
 Logit Mixture (A Weight):   54.57%
═══════════════════════════════════════════════════════

"""

