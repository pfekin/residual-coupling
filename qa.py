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
# SETTINGS & TOGGLES
# =============================================================================
DOMAIN           = "medical_multi" 
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
SEED             = 42

# Toggles
RUN_TRUTHFUL_QA  = True   # Set to True to benchmark Hallucination Suppression
MAX_STEPS        = 2000   # Set to 2000 for final paper results
GRAD_ACCUM       = 8           
MAX_SEQ_LEN      = 128          
TEST_SAMPLES     = 50     # For stable Perplexity baseline

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

CONFIGS = {
    "medical_multi": {
        "A": "gpt2",
        "B_list": ["microsoft/DialoGPT-small", "nrslearning/finetuned-gpt2-medical-QA"],
        "dataset": "lavita/ChatDoctor-HealthCareMagic-100k",
        "dim": 768, "layers": 12,
        "map": lambda x: f"Patient: {x.get('instruction', '')[:200]} Doctor: {x.get('output', '')[:200]}"
    }
}

C = CONFIGS[DOMAIN]
BRIDGE_LAYERS = [2, 4, 6, 8, 10]

# =============================================================================
# ARCHITECTURE
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

class LatentMoE(nn.Module):
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
            str(l): LatentMoE(C["dim"], len(self.models)) if mode == "moe" 
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
# HALLUCINATION BENCHMARK (TruthfulQA)
# =============================================================================

def run_truthfulqa_eval(net, tokenizer):
    # 1. Load the 'generation' config to find which questions are 'Health'
    ds_meta = load_dataset("truthful_qa", "generation", split="validation", trust_remote_code=True)
    # Get the indices of all questions where category is 'Health'
    health_indices = [i for i, x in enumerate(ds_meta) if x['category'] == 'Health']
    
    # 2. Load the 'multiple_choice' config for the actual testing data
    ds_mc = load_dataset("truthful_qa", "multiple_choice", split="validation", trust_remote_code=True)
    
    # 3. Filter the MC dataset using the indices we found
    med_ds = ds_mc.select(health_indices)
    
    print(f" (TruthfulQA: Found {len(med_ds)} Medical Questions)")
    
    correct_count = 0
    with torch.no_grad():
        for item in med_ds:
            choices = item['mc1_targets']['choices']
            labels = item['mc1_targets']['labels']
            correct_idx = labels.index(1)
            
            scores = []
            for choice in choices:
                # Format the prompt for evaluation
                txt = f"Question: {item['question']} Answer: {choice}"
                ids = tokenizer.encode(txt, return_tensors="pt").to(DEVICE)
                
                # We calculate the probability of the choice tokens
                # We need the length of the prompt prefix to mask it out
                prefix_txt = f"Question: {item['question']} Answer:"
                q_len = len(tokenizer.encode(prefix_txt))
                
                logits, _ = net(ids)
                # Shift logits and labels for causal LM loss calculation
                log_probs = F.log_softmax(logits[:, q_len-1:-1, :], dim=-1)
                target_ids = ids[:, q_len:]
                
                # Gather log-probabilities of the actual tokens in the choice
                score = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).mean().item()
                scores.append(score)
            
            # If the correct answer had the highest log-probability, it's a win
            if scores.index(max(scores)) == correct_idx:
                correct_count += 1
                
    return (correct_count / len(med_ds)) * 100

# =============================================================================
# EXECUTION ENGINE
# =============================================================================

def run_comprehensive_sweep():
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(C["A"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Load Models Once
    model_A = AutoModelForCausalLM.from_pretrained(C["A"]).to(DEVICE)
    specs = [AutoModelForCausalLM.from_pretrained(b).to(DEVICE) for b in C["B_list"]]
    for m in [model_A] + specs:
        for p in m.parameters(): p.requires_grad = False

    # 2. Setup Baselines
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

    with torch.no_grad():
        base_ppl_a = get_ppl(model_A(torch.clamp(test_ids, 0, model_A.config.vocab_size-1)).logits, test_ids)
        best_baseline = base_ppl_a # PPL baseline is usually Model A

    results = {}
    modes = ["multi_unilateral", "star_bilateral", "multi_bilateral", "moe"]

    for mode in modes:
        print(f"\n--- Training Topology: {mode.upper()} ---")
        set_seed(SEED)
        net = DifferanceEngine(model_A, specs, mode).to(DEVICE).to(model_A.dtype)
        opt = torch.optim.AdamW([
            {"params": [p for n, p in net.named_parameters() if p.requires_grad and not ("router" in n or "mix" in n)], "lr": 1e-4},
            {"params": [p for n, p in net.named_parameters() if p.requires_grad and ("router" in n or "mix" in n)], "lr": 5e-3}
        ])

        net.train()
        for step in tqdm(range(MAX_STEPS)):
            try: ex = next(ds_iter)
            except StopIteration: break
            ids = tokenizer(C["map"](ex), return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(DEVICE)
            if ids.size(1) < 2: continue
            logits, _ = net(ids)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            (loss / GRAD_ACCUM).backward()
            if (step + 1) % GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()

        net.eval()
        with torch.no_grad():
            final_l, _ = net(test_ids)
            ppl = get_ppl(final_l, test_ids)
            tqa_acc = run_truthfulqa_eval(net, tokenizer) if RUN_TRUTHFUL_QA else 0.0
            results[mode] = {"ppl": ppl, "tqa": tqa_acc}
        
        del net; torch.cuda.empty_cache(); gc.collect()

    # --- FINAL JOURNAL-READY REPORT ---
    print("\n" + "═"*80)
    print(f" SYNERGY-X FINAL BENCHMARK (Test Samples: {TEST_SAMPLES})")
    print("─"*80)
    print(f" Frozen Gen (A) PPL Baseline: {base_ppl_a:.2f}")
    print("─"*80)
    print(f" {'MODE':<20} | {'SYNERGY PPL':<15} | {'PPL GAIN %':<12} | {'TRUTHFUL-QA %'}")
    print("─"*80)
    for mode, data in results.items():
        gain = ((best_baseline - data['ppl']) / best_baseline) * 100
        print(f" {mode:<20} | {data['ppl']:<15.2f} | {gain:>+11.2f}% | {data['tqa']:>12.2f}%")
    print("═"*80)

if __name__ == "__main__":
    run_comprehensive_sweep()
"""
 SYNERGY-X FINAL BENCHMARK (Test Samples: 50)
────────────────────────────────────────────────────────────────────────────────
 Frozen Gen (A) PPL Baseline: 57.08
────────────────────────────────────────────────────────────────────────────────
 MODE                 | SYNERGY PPL     | PPL GAIN %   | TRUTHFUL-QA %
────────────────────────────────────────────────────────────────────────────────
 multi_unilateral     | 12.90           |      +77.40% |        23.64%
 star_bilateral       | 11.68           |      +79.53% |        21.82%
 multi_bilateral      | 11.37           |      +80.07% |        23.64%
 moe            | 50.99           |      +10.66% |        18.18%
"""
