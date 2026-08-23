# muse_glimmer — Muse Glimmer 30B, complete, at any size

**Muse Glimmer 30B** (the 28B dense text **+ vision** model, added to `transformers`
in 5.15.0), rebuilt from scratch in the style of the other folders — with one extra
promise the others don't make: **the model code is size-blind.** `VLConfig()` gives
you a 64K-parameter toy; `VLConfig.muse_glimmer_30b()` gives you the released model,
key-for-key and shape-for-shape compatible with the real checkpoint — **all 1436
tensors**, vision tower included. Nothing in the modules changes, only the numbers.

**Read the text stack first:** `config.py` → `rms_norm.py` → `rotary.py` → `mlp.py`
→ `attention.py` → `block.py` → `model.py`.
**Then the vision side:** `vision_windows.py` → `vision_embed.py` →
`vision_attention.py` → `vision_block.py` → `vision_model.py` → `vl_model.py`.

Run: `python3 train.py` then `python3 generate.py 20` (text, the Turkish-names task),
and `python3 vision_demo.py` to watch an image become tokens.

## The text stack — where all the novelty is

- **Normed embeddings**: a weight-less RMSNorm right after the embedding lookup —
  every token enters the stack at unit RMS
- **Gated attention**: merged head outputs are multiplied by `sigmoid(gate_proj(x))`
  before the output projection
- **QK-Norm with a scale factor**: weight-less (one module serves q and k), and q is
  multiplied by a constant **3.87** on top of the usual `1/sqrt(head_dim)`
- **3-sliding + 1-full layer schedule**, counted backward from the last layer;
  sliding layers get **RoPE**, full layers get **NoPE** (no positions at all)
- **Sandwich norms** like Gemma, but *centered* (`x_hat * (1 + w)`, zero-init) and
  with a tighter epsilon (1e-8) on the two post-norms
- **Decoupled head width**: `num_heads * head_dim` ≠ `hidden_size` (4096 vs 6656)
- **Untied embeddings** and **softcapped logits**:
  `20 * tanh(logits * output_multiplier / 20)`, `output_multiplier = 1/sqrt(hidden/256)`

## The vision tower — a plain ViT, packed and windowed

The contrast is the lesson: almost none of the text stack's exotic machinery
reappears here. No GQA, no QK-norm, no gating, no causality — biases on every
projection, plain LayerNorm, plain GELU MLP. What *is* distinctive:

- **Packed sequences, not batches.** Every patch of every image lives in one long
  sequence; attention is scoped by boundary lists, not masks. Full layers attend
  within a **frame**, window layers within a **32×32-patch tile** — and the tokens
  are physically reordered so each tile is contiguous, then unsorted afterward
  (`vision_windows.py`).
- **Its own layer schedule, running the opposite way**: full attention every 4th
  layer counted *forward*, plus the last layer.
- **Linear patchifier** (no Conv2d) over pre-flattened patches, plus a learned
  32×32 position table **bilinearly interpolated** to whatever grid the image
  actually has (`vision_embed.py`).
- **2D RoPE**: frequencies over half the head dim, interleaved `[col, row, col, row]`.
- **Pixel shuffle** at the end: each 2×2 block of patches folds into the channel
  dimension, so `1536 → 6144` and a 32×32-patch image becomes 16×16 LLM tokens.

Then the bridge: adapter (two bias-free linears, GELU after **each**) → projection
into text hidden size → `perception_emb_norm`, a weight-less RMSNorm that is the
exact mirror of `embed_norm` — **both modalities enter the decoder at unit RMS.**
Features are scattered into the sequence wherever the placeholder token id appears.

## Same code, two configs

| setting | `VLConfig()` (tiny) | `VLConfig.muse_glimmer_30b()` |
| --- | --- | --- |
| vocab_size | 30 | 202,048 |
| text hidden / layers | 32 / 4 (`s s s f`) | 6,656 / 52 (`s s s f` ×13) |
| text heads / kv / head_dim | 4 / 2 / 8 | 32 / 2 / 128 |
| intermediate_size | 64 | 19,968 |
| sliding_window | 8 | 2,048 |
| rope_theta (sliding layers) | 10,000 | 500,000 |
| vision hidden / layers / heads | 32 / 2 / 2 | 1,536 / 50 / 16 |
| patch_size / temporal / pos grid | 2 / 1 / 4×4 | 14 / 2 / 32×32 |
| merged vision token width | 128 | 6,144 |
| **parameters** | **64,352** | **29,776,626,688** |
| ↳ breakdown | 43,424 text + 18,112 vision + 2,816 bridge | 27.85B text + 1.85B vision + 69M bridge |

`layer_types`, `layer_rope_theta`, `output_multiplier`, `out_hidden_size` and
`patch_dim` are all **derived** in `__post_init__` by the same rules the reference
configs use, so they stay correct at any depth or width you pick. Both presets were
checked field-for-field against the released `config.json`, including all 52 text and
50 vision layer-type entries.

## Verified against the reference

`python3 verify_against_hf.py` (needs `transformers >= 5.15.0`) proves the claim
four ways — every logit comparison below came back **exactly 0.0, bit-exact**:

1. **Text parity** — tiny HF `MuseGlimmerForConditionalGeneration` vs our
   `MuseGlimmer`, same weights, at sequence lengths below and above the sliding window.
2. **Vision parity** — vs our complete `MuseGlimmerVL`, fed **two differently-sized
   images** in one sequence (so windowing hits ragged edge tiles) plus a **2-frame
   video** checked at feature level (per-frame segments, temporal pixel shuffle).
3. **File roundtrip** — `save_pretrained` output loads straight into a fresh
   `MuseGlimmerVL` via `hf_weights.load_hf_checkpoint`; still bit-exact.
4. **30B shape check** — the 30B config built on the meta device vs the real
   `unsloth/Muse-Glimmer-30B` safetensors headers (no weights downloaded):
   **all 1436 checkpoint tensors match name-for-name and shape-for-shape, in both
   directions** — nothing of ours missing, nothing of theirs unclaimed.

## Loading the real 30B

The checkpoint is `lm_head.weight` plus everything under `model.`; our submodule
names match after prefix surgery, which is all `hf_weights.py` does. With ~60 GB of RAM:

```python
from config import VLConfig
from vl_model import MuseGlimmerVL
from hf_weights import load_hf_checkpoint

model = MuseGlimmerVL(VLConfig.muse_glimmer_30b())
load_hf_checkpoint(model, "unsloth/Muse-Glimmer-30B")   # bf16 repo, ~56 GB download
```

Text-only is still a first-class path — `MuseGlimmer(ModelConfig.muse_glimmer_30b())`
loads from the same checkpoint and the loader skips the vision tensors.

Notes on scope: the image **processor** (resize, normalize, patchify) and the ATEM
chat template are tokenizer-side concerns, not architecture — `vision_demo.py`
includes the patchify step so you can see the input format, but for real images use
`AutoProcessor`. Like the other folders there is no KV cache: generation recomputes
the prefix each step (the vision tower still runs only once), which is the right
trade at this scale.
