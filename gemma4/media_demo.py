"""Walk an image and an audio clip through the complete model, printing shapes.

Run:  python media_demo.py

Nothing here is trained - the point is to watch two very different inputs get
reduced to the same thing: a short run of vectors in the language model's
hidden size, indistinguishable from text embeddings by the time the decoder
sees them.

Both paths compress hard, and that is the whole trick of a multimodal model:
36 patches become 4 tokens (3x3 average pooling), and 40 frames of spectrogram
become 10 tokens (two stride-2 convolutions). Without that, media would swamp
the context window - the released e2b tower turns 2520 patches into the 280
soft tokens the text side expects.
"""

import torch

from config import VLConfig
from vl_model import Gemma4


def main():
    torch.manual_seed(0)
    cfg = VLConfig()
    model = Gemma4(cfg).eval()
    v, a = cfg.vision, cfg.audio

    # --- vision: a 6x6 grid of patches ---
    side = 6
    n_patches = side * side
    pixel_values = torch.rand(1, n_patches, v.patch_dim)
    xs, ys = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    position_ids = torch.stack([xs.flatten(), ys.flatten()], dim=-1)[None]

    with torch.no_grad():
        patches = model.vision_tower(pixel_values, position_ids)
        image_features = model.get_image_features(pixel_values, position_ids)

    print("vision tower")
    print(f"  image                 {side * v.patch_size}x{side * v.patch_size} pixels"
          f" -> {tuple(pixel_values.shape)}")
    print(f"  after encoder+pool    {tuple(patches.shape)}"
          f"   ({v.pooling_kernel_size}x{v.pooling_kernel_size} patches -> 1 soft token)")
    print(f"  after embed_vision    {tuple(image_features.shape)}"
          f"   (now in text hidden size {cfg.text.hidden_size})")

    # --- audio: 40 frames of "spectrogram" ---
    frames = 40
    input_features = torch.randn(1, frames, a.mel_bins)
    features_mask = torch.ones(1, frames, dtype=torch.bool)

    with torch.no_grad():
        subsampled, _ = model.audio_tower.subsample_conv_projection(input_features, features_mask)
        audio_features = model.get_audio_features(input_features, features_mask)

    print("\naudio tower")
    print(f"  spectrogram           {tuple(input_features.shape)}"
          f"   ({frames} frames x {a.mel_bins} mel bins)")
    print(f"  after subsampling     {tuple(subsampled.shape)}"
          f"   (two stride-2 convs quarter the time axis)")
    print(f"  after embed_audio     {tuple(audio_features.shape)}"
          f"   (also text hidden size {cfg.text.hidden_size})")
    print(f"  attention span        chunk {a.attention_chunk_size},"
          f" {a.attention_context_left - 1} frames of history,"
          f" {a.attention_context_right} of future")

    # --- both, woven into one token sequence ---
    n_img, n_aud = image_features.shape[0], audio_features.shape[0]
    ids = torch.tensor([[3] + [cfg.image_token_id] * n_img
                        + [5] + [cfg.audio_token_id] * n_aud + [7]])
    with torch.no_grad():
        logits, _ = model(ids, pixel_values=pixel_values, pixel_position_ids=position_ids,
                          input_features=input_features, input_features_mask=features_mask)

    print("\nfull model")
    print(f"  token sequence        {tuple(ids.shape)}"
          f"   (1 text + {n_img} image + 1 text + {n_aud} audio + 1 text)")
    print(f"  logits                {tuple(logits.shape)}")
    print(f"  logit range           [{logits.min():.2f}, {logits.max():.2f}]"
          f"   (softcap keeps it inside +/-{cfg.text.final_logit_softcapping:.0f})")

    counts = {"text": model.language_model, "vision": model.vision_tower,
              "audio": model.audio_tower}
    parts = {k: sum(p.numel() for p in m.parameters()) for k, m in counts.items()}
    bridge = sum(p.numel() for m in (model.embed_vision, model.embed_audio)
                 for p in m.parameters())
    print("\nparameters: " + " + ".join(f"{n:,} {k}" for k, n in parts.items())
          + f" + {bridge:,} bridge = {sum(parts.values()) + bridge:,}")


if __name__ == "__main__":
    main()
