import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
import random
import numpy as np
from tqdm import tqdm
import gc

# =============================================================================
# SETTINGS & CONFIG
# =============================================================================
DOMAIN         = "medical_multi" 
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42

# Increase samples for statistical significance
MAX_STEPS      = 2000           
GRAD_ACCUM     = 8           
MAX_SEQ_LEN    = 128          
TEST_SAMPLES   = 50  # Larger test set for stable PPL

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

CONFIGS = {
    "medical_multi": {
        "A": "gpt2",
        "B_list": [
            "microsoft/DialoGPT-small",
            "nrslearning/finetuned-gpt2-medical-QA"
        ],
        "dataset": "lavita/ChatDoctor-HealthCareMagic-100k",
        "dim": 768, "layers": 12,
        "map": lambda x: f"Patient: {x.get('instruction', '')[:200]} Doctor: {x.get('output', '')[:200]}"
    }
}

C = CONFIGS[DOMAIN]
BRIDGE_LAYERS = [2, 4, 6, 8, 10]

# =============================================================================
# ARCHITECTURE (Multi-Bridge)
# =============================================================================

class MultiLatentBridge(nn.Module):
    def __init__(self, dim, num_models, mode):
        super().__init__()
        self.projections = nn.ModuleDict()
        for i in range(num_models):
            for j in range(num_models):
                if i == j: continue
                if mode == "multi_unilateral" and i != 0: continue
                if mode == "star_bilateral" and (i != 0 and j != 0): continue
                self.projections[f"{j}_to_{i}"] = nn.Linear(dim, dim)
        self.gates = nn.Parameter(torch.full((num_models, num_models), -2.0))

    def forward(self, h_list):
        new_h = [h.clone() for h in h_list]
        for i in range(len(h_list)):
            delta = 0
            for j in range(len(h_list)):
                key = f"{j}_to_{i}"
                if key in self.projections:
                    delta += self.projections[key](h_list[j]) * torch.sigmoid(self.gates[i, j])
            new_h[i] = new_h[i] + delta
        return new_h

class MultiLatentMoE(nn.Module):
    def __init__(self, dim, num_models):
        super().__init__()
        self.router = nn.Linear(dim, num_models)
    def forward(self, h_list):
        h_avg = torch.stack(h_list, dim=0).mean(dim=0)
        w = torch.softmax(self.router(h_avg), dim=-1)
        fused = sum(h * w[:, :, i:i+1] for i, h in enumerate(h_list))
        return [fused for _ in h_list]

class DifferanceEngine(nn.Module):
    def __init__(self, model_A, specialist_list, mode):
        super().__init__()
        self.models = nn.ModuleList([model_A] + specialist_list)
        self.vocabs = [m.config.vocab_size for m in self.models]
        self.bridges = nn.ModuleDict({
            str(l): MultiLatentMoE(C["dim"], len(self.models)) if mode == "moe" 
            else MultiLatentBridge(C["dim"], len(self.models), mode)
            for l in BRIDGE_LAYERS
        })
        self.final_mix = nn.Parameter(torch.zeros(len(self.models)))

    def forward(self, ids):
        h_list = []
        pos = torch.arange(0, ids.size(1), device=ids.device).unsqueeze(0)
        for i, m in enumerate(self.models):
            h_list.append(m.transformer.wte(torch.clamp(ids, 0, self.vocabs[i]-1)) + m.transformer.wpe(pos))
        
        for l in range(C["layers"]):
            for i in range(len(self.models)):
                h_list[i] = self.models[i].transformer.h[l](h_list[i])[0]
            if str(l) in self.bridges: h_list = self.bridges[str(l)](h_list)
        
        max_v, logits_list = max(self.vocabs), []
        for i, m in enumerate(self.models):
            l = m.lm_head(m.transformer.ln_f(h_list[i]))
            if l.size(-1) < max_v:
                l = torch.cat([l, torch.full((*l.shape[:-1], max_v - self.vocabs[i]), -1e4, device=DEVICE, dtype=l.dtype)], dim=-1)
            logits_list.append(l)
        
        w = torch.softmax(self.final_mix, dim=0)
        return sum(l * w[i] for i, l in enumerate(logits_list)), logits_list

# =============================================================================
# SWEEP ENGINE
# =============================================================================

def run_comprehensive_sweep():
    set_seed(SEED)
    print("--- PREPARING UNIFIED BENCHMARK ---")
    tokenizer = AutoTokenizer.from_pretrained(C["A"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Load Models Once
    model_A = AutoModelForCausalLM.from_pretrained(C["A"]).to(DEVICE)
    specialists = [AutoModelForCausalLM.from_pretrained(b).to(DEVICE) for b in C["B_list"]]
    for m in [model_A] + specialists:
        for p in m.parameters(): p.requires_grad = False

    # 2. Extract Fixed Test Set
    ds_stream = load_dataset(C["dataset"], split="train", streaming=True, trust_remote_code=True)
    ds_iter = iter(ds_stream.shuffle(seed=SEED, buffer_size=1000))
    test_texts = [C["map"](next(ds_iter)) for _ in range(TEST_SAMPLES)]
    test_ids = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(DEVICE)
    mask = test_ids[:, 1:] != tokenizer.pad_token_id

    def get_ppl(logits, ids):
        l_s, t_s = logits[:, :-1, :].contiguous().float(), ids[:, 1:].contiguous()
        if l_s.size(-1) < t_s.max().item() + 1: t_s = torch.clamp(t_s, 0, l_s.size(-1) - 1)
        loss = F.cross_entropy(l_s.view(-1, l_s.size(-1)), t_s.view(-1), reduction='none')
        return math.exp(loss.view(t_s.size(0), -1)[mask].mean().item())

    # Calculate Frozen Baselines once for the whole sweep
    with torch.no_grad():
        base_ppl_a = get_ppl(model_A(torch.clamp(test_ids, 0, model_A.config.vocab_size-1)).logits, test_ids)
        base_ppls_b = [get_ppl(m(torch.clamp(test_ids, 0, m.config.vocab_size-1)).logits, test_ids) for m in specialists]
        best_baseline = min([base_ppl_a] + base_ppls_b)

    results = {}
    modes = ["multi_unilateral", "star_bilateral", "multi_bilateral", "moe"]

    for mode in modes:
        print(f"\nTraining Mode: {mode.upper()}...")
        set_seed(SEED) # Reset seed for weight init consistency
        net = DifferanceEngine(model_A, specialists, mode).to(DEVICE).to(model_A.dtype)
        
        opt = torch.optim.AdamW([
            {"params": [p for n, p in net.named_parameters() if p.requires_grad and not ("router" in n or "mix" in n)], "lr": 1e-4},
            {"params": [p for n, p in net.named_parameters() if p.requires_grad and ("router" in n or "mix" in n)], "lr": 5e-3}
        ])

        # Standard Training
        net.train()
        # Explicitly use 'step' and ensure range is cast correctly
        for step in tqdm(range(int(MAX_STEPS))):
            try: 
                ex = next(ds_iter)
                ids = tokenizer(C["map"](ex), return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(DEVICE)
            except StopIteration: 
                break
                
            if ids.size(1) < 2: continue
            
            logits, _ = net(ids)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            (loss / GRAD_ACCUM).backward()
            
            # Check step + 1
            if (step + 1) % GRAD_ACCUM == 0:
                opt.step()
                opt.zero_grad()

        # Final Eval
        net.eval()
        with torch.no_grad():
            final_logits, _ = net(test_ids)
            syn_ppl = get_ppl(final_logits, test_ids)
            results[mode] = syn_ppl
        
        del net; torch.cuda.empty_cache(); gc.collect()

    # --- FINAL COMPARISON TABLE ---
    print("\n" + "═"*65)
    print(f" UNIFIED ARCHITECTURE COMPARISON (Test Samples: {TEST_SAMPLES})")
    print("─"*65)
    print(f" Frozen Generalist (A) Baseline: {base_ppl_a:.2f}")
    for i, p in enumerate(base_ppls_b):
        print(f" Frozen Specialist (B{i+1}) Baseline: {p:.2f}")
    print("─"*65)
    print(f" {'MODE':<20} | {'DIFFERANCE PPL':<15} | {'GAIN %':<10}")
    print("─"*65)
    for mode, ppl in results.items():
        gain = ((best_baseline - ppl) / best_baseline) * 100
        print(f" {mode:<20} | {ppl:<15.2f} | {gain:>+7.2f}%")
    print("═"*65)

if __name__ == "__main__":
    run_comprehensive_sweep()






"""
═════════════════════════════════════════════════════════════════
 UNIFIED ARCHITECTURE COMPARISON (Test Samples: 50)
─────────────────────────────────────────────────────────────────
 Frozen Generalist (A) Baseline: 57.08
 Frozen Specialist (B1) Baseline: 758.38
 Frozen Specialist (B2) Baseline: 9209.68
─────────────────────────────────────────────────────────────────
 MODE                 | DIFFERANCE PPL     | GAIN %    
─────────────────────────────────────────────────────────────────
 multi_unilateral     | 12.90           |  +77.40%
 star_bilateral       | 11.68           |  +79.53%
 multi_bilateral      | 11.37           |  +80.07%
 moe                  | 50.99           |  +10.66%
═════════════════════════════════════════════════════════════════
"""
