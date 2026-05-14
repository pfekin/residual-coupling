import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math, random, numpy as np, gc
from tqdm import tqdm

# =============================================================================
# SETTINGS & DOMAIN CONFIGURATION
# =============================================================================
DOMAIN         = "legal"  # Options: "medical", "legal", "coding", "scientific"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42

MAX_STEPS      = 2000      
GRAD_ACCUM     = 8         
MAX_SEQ_LEN    = 128
TEST_SAMPLES   = 25
STREAMING      = False     

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Mapping domains to specific Generalist (A) and Specialist (B) models
CONFIGS = {
    "medical": {"A": "gpt2-medium", "B": "microsoft/DialoGPT-medium", "dim": 1024, "layers": 24, "dataset": "lavita/ChatDoctor-HealthCareMagic-100k", "map": lambda x: f"Patient: {x.get('instruction','')[:200]} Doctor: {x.get('output','')[:200]}"},
    "legal": {"A": "gpt2", "B": "isaacus/open-australian-legal-gpt2", "dim": 768, "layers": 12, "dataset": "lex_glue", "subset": "scotus", "map": lambda x: x['text'][:600]},
    "coding": {"A": "gpt2", "B": "microsoft/CodeGPT-small-py", "dim": 768, "layers": 12, "dataset": "iamtarun/python_code_instructions_18k_alpaca", "map": lambda x: f"Instruction: {x['instruction']}\nCode: {x['output'][:400]}"},
    "scientific": {"A": "gpt2-large", "B": "Locutusque/gpt2-large-medical", "dim": 1280, "layers": 36, "dataset": "ccdv/pubmed-summarization", "subset": "document", "map": lambda x: x['article'][:600]}
}
C = CONFIGS[DOMAIN]
# Dynamic selection of bridge insertion layers based on model depth
BRIDGE_LAYERS = [6, 12, 18, 24, 30] if C["layers"] == 36 else [4, 8, 12, 16, 20] if C["layers"] == 24 else [3, 6, 9]

# =============================================================================
# ARCHITECTURE: LATENT BRIDGES
# =============================================================================

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

        # Ablation for random/frozen projection weights
        if "random" in mode:
            for p in self.proj_b2a.parameters(): p.requires_grad = False
            if "bilateral" in mode:
                for p in self.proj_a2b.parameters(): p.requires_grad = False

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

class LatentMoE(nn.Module):
    """Router-based Mixture of Experts for latent representations."""
    def __init__(self, dim):
        super().__init__()
        self.router = nn.Linear(dim, 2)
    def forward(self, h_A, h_B):
        w = torch.softmax(self.router(h_A + h_B), dim=-1)
        fused = w[:,:,0:1] * h_A + w[:,:,1:2] * h_B
        return fused, fused

class ResidualCoupler(nn.Module):
    """Wraps two frozen models and manages inter-layer communication via bridges."""
    def __init__(self, model_A, model_B, mode):
        super().__init__()
        self.A, self.B, self.mode = model_A, model_B, mode
        self.v_A, self.v_B = model_A.config.vocab_size, model_B.config.vocab_size
        
        if mode == "logit_ensemble":
            self.bridges = nn.ModuleDict() 
        elif mode == "moe":
            self.bridges = nn.ModuleDict({str(l): LatentMoE(C["dim"]) for l in BRIDGE_LAYERS})
        else:
            self.bridges = nn.ModuleDict({str(l): LatentBridge(C["dim"], mode) for l in BRIDGE_LAYERS})
            
        # Learnable mixing coefficient for final output logits
        self.mix = nn.Parameter(torch.tensor([0.0]))

    def forward(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h_A = self.A.transformer.wte(ids.clamp(0, self.v_A-1)) + self.A.transformer.wpe(pos)
        h_B = self.B.transformer.wte(ids.clamp(0, self.v_B-1)) + self.B.transformer.wpe(pos)

        # Iterate through transformer blocks
        for i in range(C["layers"]):
            h_A, h_B = self.A.transformer.h[i](h_A)[0], self.B.transformer.h[i](h_B)[0]
            if str(i) in self.bridges:
                h_A, h_B = self.bridges[str(i)](h_A, h_B) # Apply bridge coupling

        l_A, l_B = self.A.lm_head(self.A.transformer.ln_f(h_A)), self.B.lm_head(self.B.transformer.ln_f(h_B))
        
        # Vocabulary alignment via padding
        max_v = max(self.v_A, self.v_B)
        def pad(l, v): return torch.cat([l, torch.full((*l.shape[:-1], max_v-v), -1e4, device=DEVICE)], dim=-1) if l.size(-1)<max_v else l
        l_A, l_B = pad(l_A, self.v_A), pad(l_B, self.v_B)
        
        m = torch.sigmoid(self.mix)
        return (m * l_A) + ((1-m) * l_B), l_A, l_B

# =============================================================================
# RUNTIME & BENCHMARKING
# =============================================================================

def run():
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(C["A"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and freeze models
    model_A = AutoModelForCausalLM.from_pretrained(C["A"]).to(DEVICE)
    model_B = AutoModelForCausalLM.from_pretrained(C["B"]).to(DEVICE)
    for p in list(model_A.parameters()) + list(model_B.parameters()): p.requires_grad = False

    subset = C.get("subset")
    ds = load_dataset(C["dataset"], subset, split="train", streaming=STREAMING, trust_remote_code=True)
    it = iter(ds.shuffle(seed=SEED))
    
    # Static test set for perplexity comparison
    test_texts = [C["map"](next(it)) for _ in range(TEST_SAMPLES)]
    test_ids = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(DEVICE)
    mask = test_ids[:, 1:] != tokenizer.pad_token_id

    def get_ppl(l):
        loss = F.cross_entropy(l[:, :-1, :].reshape(-1, l.size(-1)), test_ids[:, 1:].reshape(-1), reduction='none')
        return math.exp(loss.reshape(test_ids.size(0), -1)[mask].mean().item())

    # Pre-training baselines
    print("Calculating Frozen Baselines...")
    with torch.no_grad():
        v_A, v_B = model_A.config.vocab_size, model_B.config.vocab_size
        max_v = max(v_A, v_B)
        def pad_logits(l, v): return torch.cat([l, torch.full((*l.shape[:-1], max_v-v), -1e4, device=DEVICE)], dim=-1) if l.size(-1)<max_v else l
        ppl_A_base = get_ppl(pad_logits(model_A(test_ids.clamp(0, v_A-1)).logits, v_A))
        ppl_B_base = get_ppl(pad_logits(model_B(test_ids.clamp(0, v_B-1)).logits, v_B))
    
    results = {}
    modes = ["logit_ensemble", "moe", "unilateral", "bilateral", "bilateral_no_gate", "bilateral_random"]

    # Sweep through topologies
    for mode in modes:
        print(f"\nTraining Mode: {mode.upper()}")
        it = iter(ds.shuffle(seed=SEED))
        engine = ResidualCoupler(model_A, model_B, mode).to(DEVICE).to(model_A.dtype)
        opt = torch.optim.AdamW(engine.parameters(), lr=1e-4)
        engine.train()
        
        pbar = tqdm(total=MAX_STEPS // GRAD_ACCUM)
        for i in range(MAX_STEPS):
            try: batch = next(it)
            except StopIteration: break
            ids = tokenizer(C["map"](batch), return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).input_ids.to(DEVICE)
            if ids.size(1) < 2: continue
            out, _, _ = engine(ids)
            loss = F.cross_entropy(out[:, :-1, :].reshape(-1, out.size(-1)), ids[:, 1:].reshape(-1))
            (loss / GRAD_ACCUM).backward()
            if (i+1) % GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad(); pbar.update(1)
        pbar.close()

        # Evaluation
        engine.eval()
        with torch.no_grad():
            fused, steered_A, steered_B = engine(test_ids)
            results[mode] = {"fused": get_ppl(fused), "steered_A": get_ppl(steered_A), "steered_B": get_ppl(steered_B)}
        del engine; gc.collect(); torch.cuda.empty_cache()

    # Final Summary Table
    print("\n" + "═"*90)
    print(f" FINAL BENCHMARK, STEERED AGENTS: {DOMAIN.upper()}")
    print("─"*90)
    print(f" {'MODE':<22} | {'FUSED PPL':<12} | {'STEERED A':<12} | {'STEERED B':<12} | {'DOMAIN GAIN'}")
    print("─"*90)
    print(f" {'Baseline (Frozen)':<22} | {'-':<12} | {ppl_A_base:<12.2f} | {ppl_B_base:<12.2f} |")
    print("─"*90)
    for m, data in results.items():
        gain = f"({((ppl_A_base - data['fused'])/ppl_A_base)*100:+.2f}%)"
        print(f" {m:<22} | {data['fused']:<12.2f} | {data['steered_A']:<12.2f} | {data['steered_B']:<12.2f} | {gain}")
    print("═"*90)

if __name__ == "__main__": run()
