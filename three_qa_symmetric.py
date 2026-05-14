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
# SETTINGS & DATA CONFIGURATION
# =============================================================================
DOMAIN           = "medical_multi" 
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
SEED             = 42

# Execution Toggles
RUN_TRUTHFUL_QA  = True   
MAX_STEPS        = 2000   
GRAD_ACCUM       = 8           
MAX_SEQ_LEN      = 128          
TEST_SAMPLES     = 50     

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Setup for the multi-agent ensemble (GPT-2 + Medical Specialists)
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
BRIDGE_LAYERS = [2, 4, 6, 8, 10] # Layers where cross-model communication occurs

# =============================================================================
# ARCHITECTURE: MULTI-AGENT RESIDUAL COUPLING
# =============================================================================

class MultiLatentBridge(nn.Module):
    """
    Implements various communication topologies between N frozen models.
    Supports Unilateral, Star-Bilateral, and Full-Bilateral (Multi) connections.
    """
    def __init__(self, dim, num_models, mode):
        super().__init__()
        self.mode = mode
        self.projections = nn.ModuleDict()
        
        # Initialize projections based on the selected topology
        for i in range(num_models):
            for j in range(num_models):
                if i == j: continue
                # multi_unilateral: Only specialists help the generalist
                if mode == "multi_unilateral" and i != 0: continue
                # star_bilateral: Generalist acts as a hub for all specialists
                if mode == "star_bilateral" and (i != 0 and j != 0): continue
                
                # Linear cross-model projection
                self.projections[f"{j}_to_{i}"] =  nn.Linear(dim, dim, bias=False) 
        
        # Ablation Support: Random projections are frozen during training
        if "random" in mode:
            for p in self.projections.parameters():
                p.requires_grad = False
                
        # Learnable gating for each communication link
        self.gates = nn.Parameter(torch.full((num_models, num_models), -2.0))

    def forward(self, h_list):
        new_h = [h.clone() for h in h_list]
        for i in range(len(h_list)):
            delta = 0
            for j in range(len(h_list)):
                key = f"{j}_to_{i}"
                if key in self.projections:
                    gate = 1.0 if "no_gate" in self.mode else torch.sigmoid(self.gates[i, j])
                    delta += self.projections[key](h_list[j]) * gate
            new_h[i] = new_h[i] + delta 
        return new_h

class LatentMoE(nn.Module):
    """A latent-space Mixture-of-Experts baseline that routes to a single representation."""
    def __init__(self, dim, num_models):
        super().__init__()
        self.router = nn.Linear(dim, num_models)
    def forward(self, h_list):
        h_avg = torch.stack(h_list, dim=0).mean(dim=0)
        w = torch.softmax(self.router(h_avg), dim=-1)
        fused = sum(h * w[:, :, i:i+1] for i, h in enumerate(h_list))
        return [fused for _ in h_list]

class ResidualCoupler(nn.Module):
    """The main engine wrapping multiple frozen models with learnable bridges."""
    def __init__(self, model_A, specialist_list, mode):
        super().__init__()
        self.mode = mode
        self.models = nn.ModuleList([model_A] + specialist_list)
        self.vocabs = [m.config.vocab_size for m in self.models]
        self.bridges = nn.ModuleDict({
            str(l): LatentMoE(C["dim"], len(self.models)) if "moe" in mode 
            else MultiLatentBridge(C["dim"], len(self.models), mode)
            for l in BRIDGE_LAYERS
        })
        self.final_mix = nn.Parameter(torch.zeros(len(self.models)))

    def forward(self, ids):
        h_list = []
        pos = torch.arange(0, ids.size(1), device=ids.device).unsqueeze(0)
        
        # Initial Embedding
        for i, m in enumerate(self.models):
            h_list.append(m.transformer.wte(torch.clamp(ids, 0, self.vocabs[i]-1)) + m.transformer.wpe(pos))
            
        # Parallel Block Execution with Inter-Layer Bridges
        for l in range(C["layers"]):
            for i in range(len(self.models)):
                h_list[i] = self.models[i].transformer.h[l](h_list[i])[0]
                
            if str(l) in self.bridges and "logit_ensemble" not in self.mode: 
                h_list = self.bridges[str(l)](h_list)
                
        # Logit Generation & Alignment
        max_v, logits_list = max(self.vocabs), []
        for i, m in enumerate(self.models):
            l = m.lm_head(m.transformer.ln_f(h_list[i]))
            if l.size(-1) < max_v:
                l = torch.cat([l, torch.full((*l.shape[:-1], max_v - self.vocabs[i]), -1e4, device=DEVICE, dtype=l.dtype)], dim=-1)
            logits_list.append(l)
            
        # Final Output Routing
        if any(k in self.mode for k in ["ensemble", "hybrid", "moe"]):
            w = torch.softmax(self.final_mix, dim=0)
            final_out = sum(l * w[i] for i, l in enumerate(logits_list))
        else:
            final_out = logits_list[0] # Generalist output steered by specialists
            
        return final_out, logits_list

# =============================================================================
# EVALUATION: TRUTHFULQA (HEALTH)
# =============================================================================

def run_truthfulqa_eval(net, tokenizer, target_idx=0):
    """Evaluates multiple-choice accuracy on Health-related questions in TruthfulQA."""
    ds_meta = load_dataset("truthful_qa", "generation", split="validation", trust_remote_code=True)
    health_indices = [i for i, x in enumerate(ds_meta) if x['category'] == 'Health']
    ds_mc = load_dataset("truthful_qa", "multiple_choice", split="validation", trust_remote_code=True)
    med_ds = ds_mc.select(health_indices)
    
    correct_count = 0
    net.eval() 
    with torch.no_grad():
        for item in med_ds:
            choices = item['mc1_targets']['choices']
            labels = item['mc1_targets']['labels']
            correct_idx = labels.index(1)
            
            scores = []
            for choice in choices:
                txt = f"Question: {item['question']} Answer: {choice}"
                ids = tokenizer.encode(txt, return_tensors="pt").to(DEVICE)
                prefix_txt = f"Question: {item['question']} Answer:"
                q_len = len(tokenizer.encode(prefix_txt))
                
                outputs = net(ids)
                
                # Check if it's our ResidualCoupler (tuple) or a standard HF model (ModelOutput)
                if isinstance(outputs, tuple) and not hasattr(outputs, "logits"):
                    logits = outputs[1][target_idx] 
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                # Mean log-probability of the target answer tokens
                log_probs = F.log_softmax(logits[:, q_len-1:-1, :].float(), dim=-1)
                target_ids = ids[:, q_len:]
                
                if target_ids.max() >= log_probs.size(-1):
                    target_ids = torch.clamp(target_ids, 0, log_probs.size(-1) - 1)
                
                score = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).mean().item()
                scores.append(score)
            
            if scores and scores.index(max(scores)) == correct_idx:
                correct_count += 1
                
    return (correct_count / len(med_ds)) * 100

# =============================================================================
# TRAINING & BENCHMARK SWEEP
# =============================================================================

def run_comprehensive_sweep():
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(C["A"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load frozen base models
    print("Loading Base Models...")
    model_A = AutoModelForCausalLM.from_pretrained(C["A"]).to(DEVICE)
    specs = [AutoModelForCausalLM.from_pretrained(b).to(DEVICE) for b in C["B_list"]]
    for m in [model_A] + specs:
        for p in m.parameters(): p.requires_grad = False

    # Prep testing data
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

    # Baseline calculations
    print("Calculating Baselines...")
    with torch.no_grad():
        base_ppl_a = get_ppl(model_A(torch.clamp(test_ids, 0, model_A.config.vocab_size-1)).logits, test_ids)
        base_tqa_a = run_truthfulqa_eval(model_A, tokenizer) if RUN_TRUTHFUL_QA else 0.0
        spec_baselines = [(i+1, get_ppl(spec(torch.clamp(test_ids, 0, spec.config.vocab_size-1)).logits, test_ids), run_truthfulqa_eval(spec, tokenizer)) for i, spec in enumerate(specs)]

    results = {}
    modes = ["logit_ensemble", "multi_unilateral", "star_bilateral", "multi_bilateral", "hybrid_multi_bilateral", "multi_bilateral_no_gate", "multi_bilateral_random", "moe"]

    for mode in modes:
        print(f"\n--- Training Topology: {mode.upper()} ---")
        set_seed(SEED)
        ds_iter = iter(ds_stream.shuffle(seed=SEED, buffer_size=1000))
        net = ResidualCoupler(model_A, specs, mode).to(DEVICE).to(model_A.dtype)
        
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

        # Evaluation
        net.eval()
        with torch.no_grad():
            final_l, logits_list = net(test_ids)
            results[mode] = {"ppl": get_ppl(final_l, test_ids), "tqa": run_truthfulqa_eval(net, tokenizer, target_idx=0)}
            if len(logits_list) > 1:
                results[mode]["ppl_B"] = get_ppl(logits_list[1], test_ids)
                results[mode]["tqa_B"] = run_truthfulqa_eval(net, tokenizer, target_idx=1)
            if len(logits_list) > 2:
                results[mode]["ppl_C"] = get_ppl(logits_list[2], test_ids)
                results[mode]["tqa_C"] = run_truthfulqa_eval(net, tokenizer, target_idx=2)
        
        del net; torch.cuda.empty_cache(); gc.collect()

    # Reporting results
    print("\n" + "═"*115)
    print(f" STEERED MULTI-AGENT PERFORMANCE (Test Samples: {TEST_SAMPLES})")
    print("─"*115)
    print(f" {'MODE':<25} | {'GEN (A) PPL/TQA':<22} | {'SPEC (B) PPL/TQA':<22} | {'SPEC (C) PPL/TQA':<22}")
    print("─"*115)
    print(f" {'FROZEN BASELINES':<25} | {f'{base_ppl_a:<7.2f} / {base_tqa_a:>5.2f}%':<22} | {f'{spec_baselines[0][1]:<7.2f} / {spec_baselines[0][2]:>5.2f}%':<22} | {f'{spec_baselines[1][1]:<7.2f} / {spec_baselines[1][2]:>5.2f}%':<22}")
    print("─"*115)
    for mode, data in results.items():
        str_A = f"{data['ppl']:<7.2f} / {data['tqa']:>5.2f}%"
        str_B = f"{data.get('ppl_B', 0):<7.2f} / {data.get('tqa_B', 0):>5.2f}%" if 'ppl_B' in data else "      - / -      "
        str_C = f"{data.get('ppl_C', 0):<7.2f} / {data.get('tqa_C', 0):>5.2f}%" if 'ppl_C' in data else "      - / -      "
        print(f" {mode:<25} | {str_A:<22} | {str_B:<22} | {str_C:<22}")
    print("═"*115)

if __name__ == "__main__":
    run_comprehensive_sweep()
