import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rescoupler import ResidualCoupler, SteeredTrainer

# 1. Load Tokenizer & Base Models (Using small, open ungated models for a seamless demo)
print("Loading pretrained tokenizer and models...")
model_id_A = "gpt2"
model_id_B = "microsoft/DialoGPT-small"

tokenizer = AutoTokenizer.from_pretrained(model_id_A)
tokenizer.pad_token = tokenizer.eos_token

generalist = AutoModelForCausalLM.from_pretrained(model_id_A)
specialist = AutoModelForCausalLM.from_pretrained(model_id_B)

# 2. Initialize the Orchestrator Graph
print("Initializing ResidualCoupler graph...")
model = ResidualCoupler(generalist, [specialist], mode="multi_bilateral", device=DEVICE).to(DEVICE)

# 3. Filter for ONLY the trainable bridge parameters
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# 4. Connect a raw streaming data engine (Using a small, fast-loading subset)
print("Connecting demo streaming dataset...")
raw_dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train", streaming=True)

def train_stream():
    for ex in raw_dataset:
        text = f"Patient: {ex.get('instruction', '')[:100]} Doctor: {ex.get('output', '')[:100]}" 
        yield tokenizer(text, return_tensors="pt", max_length=128, truncation=True).input_ids.to(model.device)

def quick_eval(model):
    print("\n[DEMO EVALUATION]")
    # Grab a small sample token batch to measure perplexity
    test_batch = next(train_stream())
    with torch.no_grad():
        final_logits, _ = model(test_batch)
        loss = F.cross_entropy(final_logits[:, :-1, :].reshape(-1, final_logits.size(-1)), test_batch[:, 1:].reshape(-1))
        ppl = torch.exp(loss).item()
    print(f"Mode: MULTI_BILATERAL | Sample Sequence Perplexity: {ppl:.2f}\n")

print("Starting demo run...")
trainer = SteeredTrainer(
    model=model,
    optimizer=optimizer,
    train_stream=train_stream(),
    eval_fn=quick_eval,       
    eval_steps=500,           
    gradient_accumulation_steps=4
)

trainer.train(max_steps=2000)
