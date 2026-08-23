"""Walk one image through the complete model, printing every shape.

Run:  python vision_demo.py

Nothing here is trained - the point is to see the pipeline's shapes, which is
where a vision-language model actually becomes understandable: an image is
just a batch of tokens that arrive by a different road, and the merge step is
what makes their number manageable.

`patchify` below mirrors what `MuseGlimmerImageProcessor` does after resizing
and normalizing: cut the image into patch_size x patch_size tiles and flatten
each tile to (temporal, channel, row, col). A still image is repeated across
the temporal axis, which is why a 3-channel still fills a
`patch_temporal * 3 * patch * patch` vector.
"""

import torch

from config import VLConfig
from vl_model import MuseGlimmerVL


def patchify(image: torch.Tensor, patch_size: int, patch_temporal: int):
    """[3, H, W] -> ([grid_h * grid_w, patch_dim], grid_h, grid_w)."""
    channels, height, width = image.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = image.view(channels, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4)              # [gh, gw, C, p, p]
    patches = patches.unsqueeze(2).expand(-1, -1, patch_temporal, -1, -1, -1)
    return patches.reshape(grid_h * grid_w,
                           patch_temporal * channels * patch_size**2), grid_h, grid_w


def main():
    torch.manual_seed(0)
    cfg = VLConfig()
    model = MuseGlimmerVL(cfg).eval()
    v = cfg.vision

    # A 16x16 pixel "image" -> 8x8 patches at patch_size=2.
    image = torch.randn(3, 16, 16)
    pixel_values, grid_h, grid_w = patchify(image, v.patch_size, v.patch_temporal)
    grid_thw = torch.tensor([[1, grid_h, grid_w]])

    print("vision tower")
    print(f"  image                     {tuple(image.shape)}")
    print(f"  patchified                {tuple(pixel_values.shape)}"
          f"   ({grid_h}x{grid_w} patches of {v.patch_size}x{v.patch_size},"
          f" each flattened to {v.patch_dim})")
    print(f"  layer schedule            {v.layer_types}")
    print(f"  window tiles              {v.pos_emb_side}x{v.pos_emb_side} patches")

    with torch.no_grad():
        merged = model.vision_tower(pixel_values, grid_thw)
        features = model.get_image_features(pixel_values, grid_thw)
    print(f"  after tower + shuffle     {tuple(merged.shape)}"
          f"   ({v.merge_size}x{v.merge_size} patches merged into one token)")
    print(f"  after adapter+projection  {tuple(features.shape)}"
          f"   (now in text hidden size {cfg.text.hidden_size})")
    print(f"  feature RMS               {features.pow(2).mean().sqrt():.4f}"
          f"   (perception_emb_norm puts it at 1, like text tokens)")

    # Weave the image into a text sequence: one placeholder per merged token.
    n_tokens = features.shape[0]
    ids = torch.tensor([[3] + [cfg.image_token_id] * n_tokens + [7, 9]])
    with torch.no_grad():
        logits, _ = model(ids, pixel_values=pixel_values, image_grid_thw=grid_thw)

    print("\nfull model")
    print(f"  token sequence            {tuple(ids.shape)}"
          f"   (1 text + {n_tokens} image placeholders + 2 text)")
    print(f"  logits                    {tuple(logits.shape)}")
    print(f"  logit range               [{logits.min():.2f}, {logits.max():.2f}]"
          f"   (softcap keeps it inside +/-{cfg.text.final_logit_softcapping:.0f})")

    text_params = sum(p.numel() for p in model.language_model.parameters())
    vision_params = sum(p.numel() for p in model.vision_tower.parameters())
    bridge_params = (sum(p.numel() for p in model.vision_adapter.parameters())
                     + sum(p.numel() for p in model.vision_projection.parameters()))
    print(f"\nparameters: {text_params:,} text + {vision_params:,} vision"
          f" + {bridge_params:,} bridge"
          f" = {text_params + vision_params + bridge_params:,}")


if __name__ == "__main__":
    main()
