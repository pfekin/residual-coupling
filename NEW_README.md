# Residual coupling

[![Paper](https://img.shields.io/badge/paper-PDF-red?style=flat-square&logo=adobeacrobat)](https://ssrn.com/abstract=6746521)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/)

`rescoupler` connects frozen language models through small learned bridge projections. At
each designated layer, a bridge reads one model's hidden state and injects an additive
correction into another model's residual stream. No base weights are modified at any point.

Standard approaches to model specialization all modify or discard: fine-tuning overwrites
weights and risks catastrophic forgetting, MoE routing commits each token to one expert and
discards the rest, agentic pipelines compress continuous hidden geometry into discrete tokens
at every handoff. Residual Coupling trains only the map between what frozen models have
separately memorized.

---

## How it works

<div align="center">
  <img src="images/architecture.png" alt="RC architecture: two frozen model stacks connected by bridge projections" width="600"/>
</div>

The bridge at layer $\ell$ from specialist $S$ to generalist $G$ computes:

$$\delta \mathbf{h}_G^{(\ell)} = \sigma(g^{(\ell,\, S \to G)})\, W^{(\ell,\, S \to G)}\, \mathbf{h}_S^{(\ell)}$$

and adds it to the generalist's residual stream before layer $\ell + 1$ executes. The gate
$g$ initializes at $-2$ ($\sigma(-2) \approx 0.12$), so bridge contributions begin
small and scale up only as the training signal supports them. In bilateral mode a return
bridge runs simultaneously from $G$ to $S$, stabilizing both residual streams rather than
optimizing the fused output at the expense of either model's individual representations.

Projection matrices $W \in \mathbb{R}^{d_A \times d_B}$ handle dimensional mismatch between
heterogeneous models natively. Layer-count mismatch is resolved by proportional depth
alignment: the bridge at anchor layer $\ell$ reads from specialist layer
$\lfloor \ell \cdot L_S / L_A \rfloor$. Vocabulary mismatches across different tokenizers
are handled by clamping token indices before embedding lookups and padding logit bounds
before output mixing.

Because the bridge projections are linear, they can only navigate geometric relationships
that already exist between the frozen models' representation spaces. This is tractable
because independently trained transformers converge toward structurally compatible internal
geometries: the relative positions of concepts are approximately preserved across models
trained on different data. The linearity also means the bridge has no mechanism to propagate
model-specific confabulation. During training, the gate scalars learn to amplify projections
that produce consistent updates across both models and suppress projections that appear on
only one side.

### Topologies

<div align="center">
  <img src="images/topologies.png" alt="The four coupling topologies" width="600"/>
</div>

| Mode | Description |
|------|-------------|
| `multi_bilateral` | All model pairs exchange bidirectional bridge updates |
| `star_bilateral` | Generalist and each specialist exchange bidirectionally; specialists do not bridge each other |
| `multi_unilateral` | Specialists inject into the generalist only; no return flow |
| `moe` | Latent-space MoE baseline: soft-routes hidden states via a learned router at each bridge layer |

## Architectural implications

Because bridge operations happen on hidden states rather than output tokens, RC preserves
geometric structure that token-level handoffs discard. Standard agentic pipelines compress
each model's continuous intermediate representations into a discrete token sequence at every
handoff. RC replaces sequential multi-pass text exchange with a single parallel forward pass,
and the coding experiment makes the practical difference concrete: any method that operates
on the out-of-distribution specialist's output logits fails entirely, while RC reaches a
fused perplexity of 5.91 by reading hidden states before the output projection collapses
them.

Bridge linearity constrains what the projections can learn. A linear map can only navigate
geometric relationships that already exist between the frozen models' representation spaces,
so the bridge has no mechanism to amplify model-specific confabulation. During training, gate
scalars learn to reinforce projections that produce consistent cross-model updates while
suppressing those that appear on only one side, where each model's private errors live.

By keeping memorization inside frozen base models and relational alignment inside bridges,
the architecture separates the two functions structurally rather than as a training
objective. The same pattern extends naturally to multimodal settings: a language model and a
vision encoder, both frozen, could be coupled through bridge projections on their residual
streams without architectural modification of either.

---

## Installation

```bash
pip install torch transformers datasets tqdm
```

Clone the repository and import directly:

```bash
git clone https://github.com/pfekin/residual-coupling.git
cd residual-coupling
```

```python
from rescoupler import ResidualCoupler, SteeredTrainer
```

---

## Quickstart

The example below couples a GPT-2 generalist with a DialoGPT specialist on medical
conversational data and runs a 2,000-step bridge training session.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rescoupler import ResidualCoupler, SteeredTrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and base models
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

generalist = AutoModelForCausalLM.from_pretrained("gpt2")
specialist = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Initialize the coupler (base weights are frozen by default)
model = ResidualCoupler(
    anchor_model=generalist,
    specialist_models=[specialist],
    mode="multi_bilateral",
    device=DEVICE
).to(DEVICE)

# Only bridge parameters are trainable
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)

# Connect a streaming dataset
raw_dataset = load_dataset(
    "lavita/ChatDoctor-HealthCareMagic-100k", split="train", streaming=True
)

def train_stream():
    for ex in raw_dataset:
        text = (
            f"Patient: {ex.get('instruction', '')[:100]} "
            f"Doctor: {ex.get('output', '')[:100]}"
        )
        yield tokenizer(
            text, return_tensors="pt", max_length=128, truncation=True
        ).input_ids.to(DEVICE)

def quick_eval(model):
    test_batch = next(train_stream())
    with torch.no_grad():
        final_logits, _ = model(test_batch)
        loss = F.cross_entropy(
            final_logits[:, :-1, :].reshape(-1, final_logits.size(-1)),
            test_batch[:, 1:].reshape(-1)
        )
    print(f"Sample perplexity: {torch.exp(loss).item():.2f}")

trainer = SteeredTrainer(
    model=model,
    optimizer=optimizer,
    train_stream=train_stream(),
    eval_fn=quick_eval,
    eval_steps=500,
    gradient_accumulation_steps=4
)

trainer.train(max_steps=2000)
```

`ResidualCoupler` accepts either loaded `nn.Module` objects or Hugging Face model ID strings
in `specialist_models`. It resolves layer counts, hidden dimensions, and vocabulary sizes
automatically. Supported architectures include GPT-2 family (`model.transformer.h`) and
LLaMA / Mistral family (`model.model.layers`).

### `ResidualCoupler` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `anchor_model` | required | Generalist model (`nn.Module`) |
| `specialist_models` | required | List of specialists (`nn.Module` or HF model ID strings) |
| `mode` | `"multi_bilateral"` | Coupling topology (see table above; also `"multi_bilateral_no_gate"`, `"multi_bilateral_random"`, `"logit_ensemble"`) |
| `bridge_layers` | `None` | Explicit anchor layer indices for bridges; if `None`, distributed evenly |
| `num_bridge_layers` | `5` | Number of bridge layers when `bridge_layers` is unspecified |
| `freeze_transformers` | `True` | Freeze all base model weights |
| `trainable_bridges` | `True` | Allow bridge parameter updates |
| `device` | auto | `"cuda"` or `"cpu"` |

---

## Benchmark results and experiment scripts

Full results across four domains and three-model topology sweeps, ablation study, Colab
notebooks, and configuration details: [EXPERIMENTS.md](EXPERIMENTS.md)

---

## Citation

The paper covers the theoretical account of why frozen models can coordinate through linear
operators, grounded in the Platonic Representation Hypothesis and Maturana and Varela's
operational closure; why the linearity constraint prevents the bridge from memorizing rather
than generalizing; and why freezing base weights makes catastrophic forgetting structurally
impossible rather than a property to be regularized away.

[![Paper](https://img.shields.io/badge/paper-PDF-red?style=flat-square&logo=adobeacrobat)](https://ssrn.com/abstract=6746521)
Ekin, P. (2026). *Computing Between Models with Residual Coupling.* SSRN Electronic Journal.
https://ssrn.com/abstract=6746521

**BibTeX:**
```bibtex
@article{residual_coupling_2026,
  title   = {Computing Between Models with Residual Coupling},
  author  = {Pascal Ekin},
  journal = {SSRN Electronic Journal},
  year    = {2026},
  url     = {https://ssrn.com/abstract=6746521}
}
```

## Acknowledgements

Models via Hugging Face Hub. Datasets: `ChatDoctor-HealthCareMagic-100k`, `scotus`,
`python_code_instructions_18k_alpaca`, `pubmed-summarization`. TruthfulQA evaluation uses
the Health category of the MC1 split.
