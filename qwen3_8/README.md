# qwen3_8 — Qwen3.8, complete, at any size

**Qwen3.8 is not a new architecture.** Its `config.json` says
`model_type: "qwen3_5"` and `architectures: ["Qwen3_5ForConditionalGeneration"]` —
it is the **Qwen3.5 architecture with new weights**. So this folder is
[`../qwen3_5/`](../qwen3_5/) done exactly instead of simplified, and, like
[`../muse_glimmer/`](../muse_glimmer/) and [`../gemma4/`](../gemma4/), it is
**size-blind**: `VLConfig()` gives you a toy, `VLConfig.qwen3_8_27b()` and
`VLConfig.qwen3_8_2_4t_a95b()` give you the released models, key-for-key and
shape-for-shape compatible with the real checkpoints.

**Read in this order:** `config.py` → `rms_norm.py` → `rotary.py` → `mlp.py`
→ **`gated_deltanet.py`** → `attention.py` → `moe.py` → `block.py` → `model.py`,
then `mtp.py`, `vision.py`, `vl_model.py`.

Run: `python3 train.py` then `python3 generate.py 20`, and `python3 demo.py` to
see the delta rule, the vision tower and the MTP head with real shapes.

## The two releases

| | `qwen3_8_27b` | `qwen3_8_2_4t_a95b` |
| --- | --- | --- |
| layers | 64 | 92 |
| hidden | 5120 | 8192 |
| feed-forward | dense, 17408 | **512 experts, top-10**, + shared expert |
| attention heads | 24 (4 kv) | 64 (4 kv) |
| linear value heads | 48 | 128 |
| vision tower | yes | **none** |
| MTP head | yes | yes |
| checkpoint tensors | 1199 | 1609 |
| parameters | 27,781,427,952 | **2,446,182,725,504** |

Both use the same 3:1 layer schedule, the same 256-wide heads, and the same
25% partial rotary.

## What makes it Qwen3.8

**Gated DeltaNet — 3 of every 4 layers.** Instead of keeping every past token,
each layer keeps **one fixed-size memory matrix** and updates it per token with
the gated delta rule:

```
S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1} k_t) k_t^T
```

`alpha` decays old memory, `beta` controls how hard to write, and the
`(v - prediction)` delta means only what the memory got *wrong* gets stored —
which is what keeps a fixed-size memory from saturating. Reading is `q^T S`.
Cost is linear in sequence length and nothing grows with context.

`gated_deltanet.py` gives you the rule **twice**: as the literal token-by-token
loop, and as the chunked form that actually trains. The chunked version is an
algebraic rearrangement, not an approximation — the `attn` inversion loop builds
the triangular operator that undoes the within-chunk sequential dependency. The
verification proves they agree to float32 noise at any chunk size, so chunk size
is purely a parallelism knob.

**The output gate hides inside `q_proj`.** Full-attention layers build `q_proj`
at *twice* the width the queries need (`num_heads * head_dim * 2`); the first
half is the query, the second becomes `sigmoid(gate)` and multiplies the
attention output. There is no separate gate tensor to spot in the checkpoint —
this is what the config's `attn_output_gate = True` means, and it is why
`q_proj` is `(12288, 5120)` on a model with 24 heads of 256.

**Partial rotary, dealt round-robin across three axes.** Only 25% of each head
rotates; the rest passes through untouched. And positions are 3-D (time, height,
width) with the frequencies **interleaved** `THWTHW…` rather than blocked
`TTT…HHH…WWW`, so every axis gets the full frequency spectrum instead of time
hogging the high frequencies. For pure text all three axes carry the same index
and this collapses to ordinary RoPE — but the plumbing has to be 3-D anyway.

**A shared expert alongside the routed ones.** In the sparse release every token
passes through a shared expert *in addition to* its top-10 routed experts, and
the shared path is itself gated by `sigmoid(shared_expert_gate(x))`. Routed
experts can only specialize freely if something else covers what is common to
all tokens; that is the shared expert's job. Hence "A95B": 2.4T parameters
total, ~95B active per token.

**Multi-Token Prediction.** Both checkpoints carry an `mtp.*` block that predicts
the token *two* ahead, by fusing the main model's hidden state at position `i`
with the embedding of the token at `i+1`. It is extra training signal, and at
inference it is the draft model for self-speculative decoding.

## Verified against the reference

Because Qwen3.8 declares `model_type: "qwen3_5"`, `transformers` 5.15 is a valid
reference for everything **except the MTP head**. `python3 verify_against_hf.py`
checks six things; every logit comparison came back **exactly 0.0, bit-exact**:

1. **The two delta-rule forms agree** — chunked vs the token-by-token loop, at
   chunk sizes 8, 16 and 64, to ~1e-7.
2. **Dense text parity** vs `Qwen3_5ForCausalLM` — Gated DeltaNet, gated
   attention, partial RoPE, interleaved MRoPE, the 3:1 schedule.
3. **Sparse text parity** vs `Qwen3_5MoeForCausalLM` — routed experts plus the
   gated shared expert, against the reference's own PyTorch expert loop.
4. **Vision parity** vs `Qwen3_5VisionModel` — two differently sized images packed
   into one sequence.
5. **File roundtrip** — `save_pretrained` output loads back through
   `hf_weights.load_hf_checkpoint`.
6. **Shape check against both released checkpoints** — meta-device build vs the
   real safetensors headers (no weights downloaded), in both directions:
   **all 1199 tensors** for the 27B and **all 1609** for the 2.4T.

### The one thing that is not bit-exact verified

`transformers` **throws the MTP head away**: both `Qwen3_5PreTrainedModel` and its
MoE sibling declare `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`, and no
released model in the library implements the module. So load Qwen3.8 with the
library and you silently get a model with no MTP head at all.

This folder builds it, which means check 6 covers those 15 (27B) and 19 (2.4T)
tensors **by name and shape only**. The names and shapes do match the real
checkpoint exactly. What cannot be checked without a reference is the **order of
the concatenation** into `fc` — embedding-first (the default here) versus
hidden-first. Both fit `fc`'s `(hidden, 2*hidden)` weight and only one is right;
`MTPHead(concat_embedding_first=...)` flips it. That caveat is stated in
`mtp.py` and printed by the verification script rather than buried here.

## Loading the real weights

```python
from config import VLConfig
from vl_model import Qwen38
from hf_weights import load_hf_checkpoint

model = Qwen38(VLConfig.qwen3_8_27b())
load_hf_checkpoint(model, "Qwen/Qwen3.8-27B")
```

Text-only is a first-class path — `Qwen38Text(TextConfig.qwen3_8_27b())` loads
from the same checkpoint and never downloads the vision shards. A checkpoint
saved by `transformers` has no MTP weights; the loader accepts that, leaves the
head at its initialization, and says so instead of failing.

## A note on speed

This is the slowest folder in the repo, and honestly so. Written readably, the
delta rule is a sequential loop, and backward through it costs roughly **60x a
full-attention layer** at this scale — so `train.py` runs 1000 steps rather than
3000. The real model uses fused FLA kernels; legibility is the trade made here.
`chunk_size` is exposed if you want to experiment, and cannot change the answer.

## Scope

The image processor and chat template are tokenizer-side concerns, not
architecture — `demo.py` shows the input format the tower expects, but for real
images use `AutoProcessor`. Like every folder here there is no KV cache, so
generation recomputes the prefix each step; that also means the linear layers
re-run their recurrence from scratch rather than carrying `S` forward, which is
the one place a real deployment would look very different.
