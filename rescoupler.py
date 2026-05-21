import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import math
import time

# =============================================================================
# 1. CORE COUPLING COMPONENTS
# =============================================================================

class MultiLatentBridge(nn.Module):
    """
    Direct token-free alignment network layer for multi-model cross-talk.

    Establishes trainable projection matrices and gating parameters to route hidden 
    states among multiple distinct models in a shared latent space without relying 
    on token-level generation.

    Attributes:
        mode (str): Architecture variant ("multi_bilateral", "multi_unilateral", 
            "star_bilateral", "multi_bilateral_no_gate", "multi_bilateral_random").
        num_models (int): Total number of models connected in the workspace (anchor + specialists).
        projections (nn.ModuleDict): Collection of Linear projection weights mapping
            hidden states from model J to model I.
        gates (nn.Parameter): Learnable routing matrices representing connection strengths.
    """
    def __init__(self, dim, num_models, mode="multi_bilateral", trainable=True):
        super().__init__()
        self.mode = mode
        self.num_models = num_models
        self.projections = nn.ModuleDict()
        
        for i in range(num_models):
            for j in range(num_models):
                if i == j: continue
                if mode == "multi_unilateral" and i != 0: continue  
                if mode == "star_bilateral" and (i != 0 and j != 0): continue  
                
                proj = nn.Linear(dim, dim, bias=False)
                if "random" in mode:
                    proj.weight.requires_grad_(False)
                else:
                    proj.weight.requires_grad_(trainable)
                self.projections[f"{j}_to_{i}"] = proj
        
        self.gates = nn.Parameter(torch.full((num_models, num_models), -2.0))
        if "no_gate" in mode:
            self.gates.requires_grad_(False)
        else:
            self.gates.requires_grad_(trainable)

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
    """
    Mixture of Experts (MoE) fusion layer for hidden state spaces.

    Computes an dynamic average hidden state across all aligned model checkpoints,
    projects router scores across models via a linear gating layer, and returns a 
    fused softmax-weighted representation back into each model stream.

    Attributes:
        router (nn.Linear): Linear mapping layer that produces model-wise selection probabilities
            from the shared latent hidden vectors.
    """
    def __init__(self, dim, num_models):
        super().__init__()
        self.router = nn.Linear(dim, num_models)

    def forward(self, h_list):
        h_avg = torch.stack(h_list, dim=0).mean(dim=0)
        w = torch.softmax(self.router(h_avg), dim=-1)
        fused = sum(h * w[:, :, i:i+1] for i, h in enumerate(h_list))
        return [fused for _ in h_list]


# =============================================================================
# 2. RESIDUAL COUPLER ENGINE
# =============================================================================

class ResidualCoupler(nn.Module):
    """
    Sequential execution engine and orchestrator for parallel model alignment.

    Iterates synchronistically layer-by-layer through a frozen generalist (anchor) 
    model and one or more specialized models. Intercepts hidden states at specified 
    bridge layer milestones, computes cross-network structural alignment using learnable 
    latent projections, and blends final outputs into an aligned representation matrix.

    Supported Topologies (`mode`):
        - "multi_bilateral": Full cross-talk where every model communicates bi-directionally 
          with all other models via learnable gated linear bridges.
        - "multi_unilateral": Uni-directional communication where only the anchor model 
          streams representations down to the specialists.
        - "star_bilateral": Centralized hub routing where specialists can only talk to 
          the anchor, and the anchor talks back to all specialists.
        - "multi_bilateral_no_gate": Linear cross-talk similar to multi_bilateral, but with 
          the connection sigmoid gates frozen at a static weight coefficient of 1.0.
        - "multi_bilateral_random": Bi-directional projection bridges with entirely frozen, 
          randomly initialized projection weights.
        - "moe": Token-free Mixture of Experts routing where hidden states are dynamically 
          fused across models via a softmax-weighted linear router network at each bridge milestone.
        - "logit_ensemble": Bypasses all intermediate hidden-state bridges during the forward 
          pass and applies a weighted softmax mixture across the models' final output heads.

    Args:
        anchor_model (nn.Module): The core baseline/generalist language model backbone.
        specialist_models (list): A sequence of specialist models (or Hugging Face repository 
            strings) to couple with the anchor.
        mode (str, optional): Routing and graph configuration topology. Defaults to "multi_bilateral".
        bridge_layers (list, optional): Explicit layer indices where latent bridges should be applied. 
            If None, distributes bridges evenly across the layout depth.
        num_bridge_layers (int, optional): Number of bridge layers to build if bridge_layers is unspecified. 
            Defaults to 5.
        freeze_transformers (bool, optional): If True, locks all base transformer weights, rendering 
            only the intermediate routing layers trainable. Defaults to True.
        trainable_bridges (bool, optional): If True, enables parameter updates for routing nodes. 
            Defaults to True.
        device (str, optional): Target execution hardware environment ("cuda" or "cpu"). 
            Defaults to "cuda" if available.

    Returns:
        tuple: A clean tracking structure containing:
            - final_out (torch.Tensor): Softmax-mixed ensemble output logits.
            - logits_list (list): Raw separate logits derived individually from each model in the graph.
    """
    def __init__(
        self, 
        anchor_model, 
        specialist_models: list, 
        mode="multi_bilateral",
        bridge_layers=None, 
        num_bridge_layers=5,
        freeze_transformers=True,
        trainable_bridges=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.mode = mode
        self.device = device

        self.anchor = anchor_model
        if freeze_transformers:
            for p in self.anchor.parameters(): p.requires_grad = False

        self.specialists = nn.ModuleList()
        self.depths = [self._get_model_depth(self.anchor)]
        self.vocabs = [self.anchor.config.vocab_size]

        for spec in specialist_models:
            if isinstance(spec, str):
                m = AutoModelForCausalLM.from_pretrained(spec).to(device)
            else:
                m = spec
            if freeze_transformers:
                for p in m.parameters(): p.requires_grad = False
            self.specialists.append(m)
            self.depths.append(self._get_model_depth(m))
            self.vocabs.append(m.config.vocab_size)

        anchor_layers = self.depths[0]
        if bridge_layers is not None:
            self.bridge_layers = [int(l) for l in bridge_layers]
        else:
            step = max(1, anchor_layers // (num_bridge_layers + 1))
            self.bridge_layers = [step * i for i in range(1, num_bridge_layers + 1)]

        self.hidden_dim = getattr(self.anchor.config, "n_embd", getattr(self.anchor.config, "hidden_size", None))
        
        num_total_models = 1 + len(self.specialists)
        self.bridges = nn.ModuleDict({
            str(l): LatentMoE(self.hidden_dim, num_total_models) if "moe" in mode
            else MultiLatentBridge(self.hidden_dim, num_total_models, mode, trainable=trainable_bridges)
            for l in self.bridge_layers
        })
        
        model_dtype = next(self.anchor.parameters()).dtype
        self.bridges.to(device=device, dtype=model_dtype) 
        
        self.final_mix = nn.Parameter(torch.zeros(num_total_models, device=device, dtype=model_dtype))
        if not trainable_bridges or "random" in mode:
            self.final_mix.requires_grad_(False)

    def _get_model_depth(self, model):
        return getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", None))

    def _get_local_layers_and_embeds(self, model):
        if hasattr(model, "transformer"):
            return model.transformer.h, model.transformer.wte, model.transformer.wpe, model.transformer.ln_f
        elif hasattr(model, "model"):
            return model.model.layers, model.model.embed_tokens, None, model.model.norm
        raise AttributeError("Unsupported transformer layout format.")

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        h_list = []
        
        layers_A, embed_A, pos_A, ln_A = self._get_local_layers_and_embeds(self.anchor)
        clamped_ids_A = torch.clamp(input_ids, 0, self.vocabs[0] - 1)
        h_A = embed_A(clamped_ids_A)
        if pos_A is not None:
            pos = torch.arange(0, input_ids.size(1), device=self.device).unsqueeze(0)
            h_A = h_A + pos_A(pos)
        h_list.append(h_A)

        for i, spec in enumerate(self.specialists, start=1):
            clamped_ids = torch.clamp(input_ids, 0, self.vocabs[i] - 1)
            _, embed_S, pos_S, _ = self._get_local_layers_and_embeds(spec)
            h_spec = embed_S(clamped_ids)
            if pos_S is not None:
                pos = torch.arange(0, input_ids.size(1), device=self.device).unsqueeze(0)
                h_spec = h_spec + pos_S(pos)
            h_list.append(h_spec)

        curr_indices = [0] * (1 + len(self.specialists))
        L_A = self.depths[0]

        for l in range(L_A):
            outputs = layers_A[l](h_list[0])
            h_list[0] = outputs[0] if isinstance(outputs, tuple) else outputs
            curr_indices[0] += 1

            for i, spec in enumerate(self.specialists, start=1):
                target_i = int((l + 1) * self.depths[i] / L_A)
                start_idx = curr_indices[i]
                
                if start_idx < target_i:
                    layers_S, _, _, _ = self._get_local_layers_and_embeds(spec)
                    spec_dtype = next(spec.parameters()).dtype
                    h_list[i] = h_list[i].to(dtype=spec_dtype)
                    
                    for idx in range(start_idx, target_i):
                        outputs_S = layers_S[idx](h_list[i])
                        h_list[i] = outputs_S[0] if isinstance(outputs_S, tuple) else outputs_S
                    curr_indices[i] = target_i

            if str(l) in self.bridges and "logit_ensemble" not in self.mode:
                bridge_dtype = next(self.bridges[str(l)].parameters()).dtype
                h_list = [h.to(dtype=bridge_dtype) for h in h_list]
                h_list = self.bridges[str(l)](h_list)

        max_v = max(self.vocabs)
        logits_list = []

        l_out_A = self.anchor.lm_head(ln_A(h_list[0]))
        if l_out_A.size(-1) < max_v:
            l_out_A = torch.cat([l_out_A, torch.full((*l_out_A.shape[:-1], max_v - self.vocabs[0]), -1e4, device=self.device, dtype=l_out_A.dtype)], dim=-1)
        logits_list.append(l_out_A)

        for i, spec in enumerate(self.specialists, start=1):
            _, _, _, ln_S = self._get_local_layers_and_embeds(spec)
            l_out = spec.lm_head(ln_S(h_list[i]))
            
            if l_out.size(-1) < max_v:
                l_out = torch.cat([l_out, torch.full((*l_out.shape[:-1], max_v - self.vocabs[i]), -1e4, device=self.device, dtype=l_out.dtype)], dim=-1)
            logits_list.append(l_out)

        if any(k in self.mode for k in ["ensemble", "hybrid", "moe"]):
            w = torch.softmax(self.final_mix, dim=0)
            final_out = sum(l * w[idx] for idx, l in enumerate(logits_list))
        else:
            final_out = logits_list[0] 

        return final_out, logits_list


# =============================================================================
# 3. SILENT TRAINING ENGINE
# =============================================================================

class SteeredTrainer:
    """
    Lightweight, pickling-safe training engine for steering decoupled model layers.

    Performs custom downstream step optimization over the token-free parameter paths 
    of a ResidualCoupler model graph. Operates silently without emitting screen-corrupting 
    step logs to preserve terminal clarity during high-throughput benchmarking runs.

    Args:
        model (nn.Module): The integrated target `ResidualCoupler` object graph.
        optimizer (torch.optim.Optimizer): Standard PyTorch optimizer initialized over active paths.
        train_stream (generator): A stream generator yielding tensor input ID tokens.
        eval_fn (callable, optional): Evaluation hook execution closure invoked at specific step intervals. 
            Defaults to None.
        eval_steps (int, optional): Frequency of training step iterations before triggering evaluation hooks. 
            Defaults to 250.
        gradient_accumulation_steps (int, optional): Virtual batch optimization scaling divider. 
            Defaults to 8.
            
    Example:
        >>> trainer = SteeredTrainer(model, optimizer, train_stream(), eval_fn=my_hook, eval_steps=100)
        >>> trainer.train(max_steps=1000)
    """
    def __init__(self, model, optimizer, train_stream, eval_fn=None, eval_steps=250, gradient_accumulation_steps=8):
        self.model = model
        self.optimizer = optimizer
        self.train_stream = train_stream
        self.eval_fn = eval_fn
        self.eval_steps = eval_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def train(self, max_steps):
        self.model.train()
        self.optimizer.zero_grad()
        
        for step in range(1, max_steps + 1):
            try:
                ids = next(self.train_stream)
            except StopIteration:
                break
                
            if ids.size(1) < 2: 
                continue
                
            logits, _ = self.model(ids)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            
            scaled_loss = loss / self.gradient_accumulation_steps
            scaled_loss.backward()
            
            if step % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            # Run evaluations gracefully on schedule without step logs corruption
            if self.eval_fn and step % self.eval_steps == 0:
                self.model.eval()
                with torch.no_grad():
                    self.eval_fn(self.model)
                self.model.train()
