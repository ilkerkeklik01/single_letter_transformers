"""Train the tiny Qwen3.8 model on Turkish names.

Run:  python train.py

Same linear, torch-only loop as the other folders - only the model underneath
changed. It prints the layer schedule first, because which layers are linear
attention and which are full attention is the single most important thing about
this architecture.
"""

import os

import torch

from config import TextConfig
from model import Qwen38Text
from tokenizer import CharTokenizer

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "temiz_isimler.txt")
BATCH_SIZE = 64
BLOCK_SIZE = 16
# Fewer steps than the other folders on purpose: the Gated DeltaNet layers are
# by far the most expensive thing in this repo. Written readably, the delta rule
# is a sequential loop, and backward through it costs ~60x a full-attention
# layer at this scale. The real model uses fused kernels; clarity is the trade
# this folder makes.
STEPS = 1000
LEARNING_RATE = 3e-3
EVAL_EVERY = 100
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
# chunk_size is a pure speed/parallelism knob - the chunked delta rule is an
# algebraic rearrangement of the recurrence, so the answer does not depend on it
# (verify_against_hf.py check [1] proves this). The config default is 64 to match
# the reference exactly; here we use the training block size, which avoids
# padding 16 tokens out to 64 and shortens the chunk-inversion loop 4x.
cfg = TextConfig(vocab_size=vocab_size, chunk_size=BLOCK_SIZE)
model = Qwen38Text(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
mtp_params = sum(p.numel() for p in model.mtp.parameters()) if cfg.mtp_num_hidden_layers else 0
print(f"device={device}  vocab_size={vocab_size}  parameters={n_params:,}"
      f"  (of which {mtp_params:,} are the MTP head)")
print("layer schedule:")
for i, layer_type in enumerate(cfg.layer_types):
    kind = "Gated DeltaNet" if layer_type == "linear_attention" else "full attention"
    print(f"  layer {i}: {kind}")

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
           "tiny_qwen38.pt")
print("\nsaved checkpoint to tiny_qwen38.pt")
