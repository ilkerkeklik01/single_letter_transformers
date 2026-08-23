# gemma4 — Gemma 4, complete, at any size

**Gemma 4** (added to `transformers` in 5.15.0) rebuilt from scratch — text stack,
vision tower, and audio Conformer — and, like [`muse_glimmer/`](../muse_glimmer/),
**size-blind**: `VLConfig()` gives you a toy, `VLConfig.gemma4_e2b()` and
`VLConfig.gemma4_26b_a4b()` give you the released models, key-for-key and
shape-for-shape compatible with the real checkpoints. The model code never
changes; only the numbers do.

**Read the text stack first:** `config.py` → `rms_norm.py` → `rotary.py` →
`mlp.py` → `moe.py` → `attention.py` → `block.py` → `model.py`.
**Then vision:** `clipped_linear.py` → `vision_embed.py` → `vision_encoder.py` →
`vision_model.py`. **Then audio:** `audio_conv.py` → `audio_attention.py` →
`audio_model.py`. **Then** `vl_model.py` ties all three together.

Run: `python3 train.py` then `python3 generate.py 20` (text, the Turkish-names
task), and `python3 media_demo.py` to watch an image and an audio clip become tokens.

## One architecture, two models that share almost nothing

This is the folder's real lesson. Google shipped two Gemma 4 models whose feature
sets barely overlap — the same code with different switches flipped:

| | `gemma4_e2b` | `gemma4_26b_a4b` |
| --- | --- | --- |
| text layers | 35, 4 sliding : 1 full | 30, 5 sliding : 1 full |
| per-layer embeddings | **yes**, width 256 | no (width 0) |
| double-wide MLP | **yes** | no |
| KV-shared layers | **20 of 35** | none |
| MoE experts | no | **128, top-8** |
| k and v share weights | no | **yes** (full layers) |
| vision tower | 768 wide, 16 layers, **clipped linears** | 1152 wide, 27 layers, **standardized output** |
| audio tower | 12-layer Conformer | **none at all** |
| total parameters | 5,104,298,467 | 25,805,936,206 |

## What makes the text stack Gemma 4

- **Sandwich norms** — RMSNorm before *and* after each sub-layer. But note:
  **the Gemma `(1 + weight)` quirk is gone.** Gemma 1–3 stored a zero-init weight
  and applied `x_hat * (1 + w)`; Gemma 4 stores a ones-init weight and applies
  `x_hat * w`, like everyone else.
- **No attention scaling.** `scale = 1.0` — logits are the raw q·k. That works only
  because q and k are RMS-normalized per head first, so the `q_norm`/`k_norm`
  weights *are* the temperature. It became a parameter instead of a constant.
- **Q, K and V are all normalized** — but `v_norm` is weight-less.
- **Proportional RoPE** on full-attention layers, and it is not implemented by
  slicing: the frequency table is built at full width and the trailing 75% of the
  angle pairs are set to **exactly zero**. Zero frequency means `cos=1, sin=0`, so
  those features pass through unrotated. Partial rotary as one uniform multiply.
- **Full layers are a different shape** — 512-wide heads vs 256, sometimes fewer
  KV heads. Two attention geometries in one stack.
- **KV sharing** — the last N layers build *no* k/v projections or norms at all,
  reusing the keys/values of the last non-shared layer of their own type. Those
  same layers get a **double-wide MLP**: Gemma spends the saved parameters back on
  a fatter feed-forward.
- **MoE that adds rather than replaces.** When enabled, the dense MLP still runs on
  every token; the sparse experts run *alongside* it and the two are summed. The
  router reads the raw layer input, not the normalized copy the dense MLP consumed.
- **Per-Layer Embeddings (PLE)** — each layer gets its own vector per token, built
  from two averaged halves: a packed second embedding table for token identity, and
  a projection of the embeddings for context. It arrives as a **gate**: project down
  to PLE width, GELU, *multiply* by the PLE vector, project back up. On e2b that
  table is 262144×8960 — bigger than the main embedding, and the reason "E2B" counts
  *effective* parameters.

## The two encoders

**Vision** is a bidirectional ViT that shares the text stack's sandwich norms and
q/k/v-norm attention (`scale = 1.0` again), minus causality and GQA. What is its
own: positions come from **two learned 1-D tables** (one for x, one for y) that are
summed rather than one 2-D table; each head is split in half and rotated by x and y
separately; and the encoder output is **average-pooled over 3×3 patch blocks**, so
2520 patches become the 280 soft tokens the text side expects.

**Audio** is a **Conformer**, not a transformer: each layer is
`FFN → attention → lightweight conv → FFN`, with half-step "macaron" residuals on
the two feed-forwards. Attention is **chunked and local** — the sequence is cut into
blocks of 12 and each block sees itself plus 12 frames of history and none of the
future — with **Transformer-XL relative positions** (`matrix_ac + matrix_bd` and the
`_rel_shift` trick), tanh-softcapped logits, and a learned per-dimension query scale.
The convolution is depthwise and causal, so the tower can stream.

Both end at a `MultimodalEmbedder`: a weight-less RMSNorm and one bias-free Linear
into text hidden size. That is the entire bridge.

## Verified against the reference

`python3 verify_against_hf.py` (needs `transformers >= 5.15.0`) checks five things.
**Every logit comparison below came back exactly 0.0, bit-exact:**

1. **Dense text parity** — the e2b recipe (PLE + KV sharing + double-wide MLP), at
   sequence lengths below and above the sliding window.
2. **MoE text parity** — the 26B-A4B recipe (sparse experts + k=v, no PLE), against
   the reference's own PyTorch expert loop. The script then re-runs against the fused
   `grouped_mm` and `batched_mm` kernels, which land ~5e-7 and ~1e-6 away: those
   kernels reorder the expert accumulation and disagree with the eager path *and with
   each other*. It is a kernel difference, not an architecture one, and the script
   prints it rather than hiding it.
3. **Multimodal parity** — the complete model, an image and an audio clip scattered
   between text tokens. Run twice: the e2b wiring, and the 26B wiring (MoE text,
   standardizing vision tower, no audio tower).
4. **File roundtrip** — `save_pretrained` output loads back through
   `hf_weights.load_hf_checkpoint`, text and multimodal alike.
5. **Shape check against both released checkpoints** — built on the meta device and
   compared to the real safetensors headers (no weights downloaded), in both
   directions. For e2b that is **1951 tensors matched plus the 60 deliberately
   ignored** — the whole 2011-tensor checkpoint accounted for; for 26B-A4B, **1013**.

## Loading the real weights

```python
from config import VLConfig
from vl_model import Gemma4
from hf_weights import load_hf_checkpoint

model = Gemma4(VLConfig.gemma4_e2b())
load_hf_checkpoint(model, "google/gemma-4-e2b-it")
```

Text-only is a first-class path too — `TinyGemma(ModelConfig.gemma4_e2b())` loads
from the same checkpoint, and the loader skips the vision and audio shards entirely
rather than downloading them.

Two wrinkles it handles for you. `lm_head.weight` is **tied** to
`embed_tokens.weight`, so the checkpoints never store it. And KV-shared layers *do*
carry `k_proj`/`v_proj`/`k_norm` in the checkpoint even though the architecture
ignores them — the reference lists them in `_keys_to_ignore_on_load_unexpected`, and
`unexpected_kv_keys()` reports exactly which 60 tensors those are, so the
verification can prove they are skipped on purpose rather than lost by accident.

## Scope

The image processor, feature extractor and chat template are tokenizer-side
concerns, not architecture — `media_demo.py` shows the input formats the towers
expect, but for real images and audio use `AutoProcessor`. Like every folder here
there is no KV cache: generation recomputes the prefix each step (the encoders still
run only once).
