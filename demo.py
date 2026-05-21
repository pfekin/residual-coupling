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
