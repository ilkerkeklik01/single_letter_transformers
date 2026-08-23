"""Show the three things that make Qwen3.8 what it is, with real shapes.

Run:  python demo.py

Nothing here is trained. It demonstrates:

  1. the two forms of the gated delta rule agreeing, so you can see for yourself
     that the chunked version is a rearrangement and not an approximation;
  2. an image becoming tokens through the vision tower;
  3. the MTP head predicting the token two positions ahead.
"""

import torch

from config import TextConfig, VLConfig
from model import Qwen38Text
from vl_model import Qwen38
from gated_deltanet import chunked_delta_rule, recurrent_delta_rule


def main():
    torch.manual_seed(0)

    # --- 1. the delta rule, two ways -------------------------------------
    B, T, H, K, V = 1, 40, 2, 8, 8
    q, k, v = torch.randn(B, T, H, K), torch.randn(B, T, H, K), torch.randn(B, T, H, V)
    g, beta = -torch.rand(B, T, H) * 0.5, torch.rand(B, T, H)
    loop = recurrent_delta_rule(q, k, v, g, beta)

    print("gated delta rule")
    print(f"  token-by-token recurrence     {tuple(loop.shape)}")
    for chunk in (8, 16, 64):
        out = chunked_delta_rule(q, k, v, g, beta, chunk_size=chunk)
        print(f"  chunked, chunk_size={chunk:<3d}       max diff vs the loop = "
              f"{(out - loop).abs().max().item():.2e}")
    print("  (the chunk size is a parallelism knob, not an approximation)")

    # --- 2. text: which layer is which ------------------------------------
    cfg = VLConfig()
    model = Qwen38(cfg).eval()
    print("\nlayer schedule")
    for i, layer_type in enumerate(cfg.text.layer_types):
        kind = "Gated DeltaNet (linear)" if layer_type == "linear_attention" else "full attention"
        print(f"  layer {i}: {kind}")
    print(f"  rotary: only {cfg.text.rotary_dim} of {cfg.text.head_dim} channels per head rotate"
          f"  (partial_rotary_factor={cfg.text.partial_rotary_factor})")

    # --- 3. vision --------------------------------------------------------
    v_cfg = cfg.vision
    grid = torch.tensor([[1, 4, 6]])          # one image, 4x6 patches
    n_patches = int(grid.prod())
    pixel_values = torch.randn(n_patches, v_cfg.patch_dim)
    with torch.no_grad():
        features = model.get_image_features(pixel_values, grid)

    merged = v_cfg.spatial_merge_size
    print("\nvision tower")
    print(f"  patches in            {tuple(pixel_values.shape)}"
          f"   ({n_patches} patches of {v_cfg.patch_dim} values each)")
    print(f"  tokens out            {tuple(features.shape)}"
          f"   ({merged}x{merged} patches merge into one)")

    ids = torch.tensor([[3] + [cfg.image_token_id] * features.shape[0] + [7]])
    with torch.no_grad():
        embeds, position_ids = model.embed(ids, pixel_values, grid)
        logits, _ = model(ids, pixel_values=pixel_values, grid_thw=grid)
    print(f"  token sequence        {tuple(ids.shape)}   logits {tuple(logits.shape)}")
    print("  3-D position ids (time / height / width), per token:")
    for axis, name in enumerate(("time  ", "height", "width ")):
        print(f"    {name}  {position_ids[axis, 0].tolist()}")

    # --- 4. multi-token prediction ---------------------------------------
    text = Qwen38Text(TextConfig()).eval()
    ids = torch.randint(0, 30, (1, 8))
    with torch.no_grad():
        next_logits, _ = text(ids)
        next_next_logits = text.predict_next_next(ids)
    print("\nmulti-token prediction")
    print(f"  main stack  -> logits for token i+1   {tuple(next_logits.shape)}")
    print(f"  MTP head    -> logits for token i+2   {tuple(next_next_logits.shape)}")
    print("  (the MTP head is what transformers drops when it loads Qwen3.8)")


if __name__ == "__main__":
    main()
