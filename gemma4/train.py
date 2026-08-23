"""Train the tiny Gemma-style model on Turkish names.

Run:  python train.py
"""

import os

import torch

from config import ModelConfig
from model import TinyGemma
from tokenizer import CharTokenizer

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "temiz_isimler.txt")
BATCH_SIZE = 64
BLOCK_SIZE = 16
STEPS = 3000
LEARNING_RATE = 3e-3
EVAL_EVERY = 200
SEED = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Tokenizer + data
# ---------------------------------------------------------------------------
tokenizer = CharTokenizer.from_file(DATA_FILE)
vocab_size = tokenizer.vocab_size

text = open(DATA_FILE, encoding="utf-8").read()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)


def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
cfg = ModelConfig(vocab_size=vocab_size)
model = TinyGemma(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"device={device}  vocab_size={vocab_size}  parameters={n_params:,}")

# Show what each layer actually is: sliding vs full attention, whether it reuses
# an earlier layer's keys/values, and whether it got the double-wide MLP.
print("layer schedule:")
for i, layer_type in enumerate(cfg.layer_types):
    tags = []
    if cfg.is_kv_shared(i):
        tags.append("kv-shared")
    if cfg.stores_shared_kv(i):
        tags.append("stores-kv")
    if cfg.uses_double_wide_mlp(i):
        tags.append("2x-mlp")
    print(f"  layer {i}: {layer_type:18s} head_dim={cfg.head_dim_for(i):3d}"
          f"  {' '.join(tags)}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def sample_names(n: int = 10, max_new_tokens: int = 20):
    model.eval()
    start = torch.full((n, 1), tokenizer.newline_id, dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=max_new_tokens, temperature=1.0,
                         top_k=None, eos_id=tokenizer.eos_id)
    model.train()
    return [tokenizer.decode(row[1:]).split("\n")[0] for row in out.tolist()]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for step in range(1, STEPS + 1):
    x, y = get_batch()
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % EVAL_EVERY == 0 or step == 1:
        print(f"step {step:5d}  loss {loss.item():.4f}")

print("\nbaseline loss (uniform guessing): %.4f" % torch.log(torch.tensor(float(vocab_size))))
print("\nsample names:")
for name in sample_names(10):
    print("  ", name)

torch.save({"model": model.state_dict(), "chars": tokenizer.chars, "cfg": cfg},
           "tiny_gemma.pt")
print("\nsaved checkpoint to tiny_gemma.pt")
