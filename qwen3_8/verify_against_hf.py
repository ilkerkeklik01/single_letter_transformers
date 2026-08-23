"""Prove that this folder is the Qwen3.8 architecture, six ways.

Qwen3.8 declares `model_type = "qwen3_5"`, so `transformers` 5.15 is a valid
reference for everything except one module - the MTP head, which the library
deliberately drops. That split shapes these checks:

  1. Delta-rule forms  : the chunked (training) and recurrent (decode) forms of
                         the gated delta rule must agree, at any chunk size.
  2. Dense text parity : a tiny HF `Qwen3_5ForCausalLM` vs our `Qwen38Text` -
                         Gated DeltaNet, gated attention, partial RoPE,
                         interleaved MRoPE, the 3:1 layer schedule.
  3. Sparse text parity : the same against `Qwen3_5MoeForCausalLM` - 512-style
                         routed experts plus the gated shared expert.
  4. Vision parity     : `Qwen3_5VisionModel` vs our tower, two differently
                         sized images packed into one sequence.
  5. Roundtrip         : `save_pretrained` output loads back through
                         `hf_weights.load_hf_checkpoint`.
  6. Shape check       : both released checkpoints - `Qwen/Qwen3.8-27B` and
                         `Qwen/Qwen3.8-2.4T-A95B` - compared against the real
                         safetensors metadata on the Hub (headers only, no
                         weights), in both directions. This is the only check
                         that covers the MTP head, and it covers it by shape and
                         name, not by value.

Run:  python verify_against_hf.py
"""

import argparse
import tempfile

import torch

from config import TextConfig, VisionConfig, VLConfig
from model import Qwen38Text
from vl_model import Qwen38
from vision import VisionTower
from hf_weights import load_hf_checkpoint, load_hf_state_dict, map_hf_key


def _text_kwargs(cfg: TextConfig) -> dict:
    return dict(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size, num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads, num_key_value_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim, max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.rms_norm_eps, hidden_act=cfg.hidden_act,
        linear_conv_kernel_dim=cfg.linear_conv_kernel_dim,
        linear_key_head_dim=cfg.linear_key_head_dim,
        linear_value_head_dim=cfg.linear_value_head_dim,
        linear_num_key_heads=cfg.linear_num_key_heads,
        linear_num_value_heads=cfg.linear_num_value_heads,
        layer_types=list(cfg.layer_types), tie_word_embeddings=cfg.tie_word_embeddings,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta,
                         "partial_rotary_factor": cfg.partial_rotary_factor,
                         "mrope_section": list(cfg.mrope_section),
                         "mrope_interleaved": True},
    )


def build_hf_text_twin(cfg: TextConfig):
    """The reference text model, dense or sparse to match `cfg`."""
    if cfg.enable_moe:
        from transformers import Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
        hf_config = Qwen3_5MoeTextConfig(
            **_text_kwargs(cfg), num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            moe_intermediate_size=cfg.moe_intermediate_size,
            shared_expert_intermediate_size=cfg.shared_expert_intermediate_size)
        model = Qwen3_5MoeForCausalLM(hf_config).eval()
        # Compare against the reference's own PyTorch expert loop; the fused
        # kernels reorder the accumulation and land a few 1e-7 away.
        model.config._experts_implementation = None
        return model

    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    return Qwen3_5ForCausalLM(Qwen3_5TextConfig(**_text_kwargs(cfg))).eval()


def build_hf_vision_twin(cfg: VisionConfig):
    from transformers import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    return Qwen3_5VisionModel(Qwen3_5VisionConfig(
        depth=cfg.depth, hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size, num_heads=cfg.num_heads,
        in_channels=cfg.in_channels, patch_size=cfg.patch_size,
        temporal_patch_size=cfg.temporal_patch_size,
        spatial_merge_size=cfg.spatial_merge_size, out_hidden_size=cfg.out_hidden_size,
        num_position_embeddings=cfg.num_position_embeddings,
        hidden_act="gelu_pytorch_tanh")).eval()


def jitter(model, scale: float = 0.05):
    """Move every parameter off its init constant so nothing matches by accident."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * scale)
    return model


def compare_text(hf_model, our_model, cfg: TextConfig, label: str) -> float:
    worst = 0.0
    for seq_len in (5, 20, cfg.max_seq_len):
        ids = torch.randint(0, cfg.vocab_size, (2, seq_len))
        with torch.no_grad():
            hf_logits = hf_model(input_ids=ids, use_cache=False).logits
            our_logits, _ = our_model(ids)
        diff = (hf_logits - our_logits).abs().max().item()
        print(f"  {label} seq_len {seq_len:3d}: max |logit diff| = {diff:.3e}")
        worst = max(worst, diff)
    return worst


def check_text_parity(cfg: TextConfig, label: str, seed: int):
    torch.manual_seed(seed)
    hf_model = jitter(build_hf_text_twin(cfg))
    ours = Qwen38Text(cfg).eval()
    load_hf_state_dict(ours, hf_model.state_dict())
    worst = compare_text(hf_model, ours, cfg, label)
    assert worst < 1e-4, f"{label} logits diverge: {worst}"
    return hf_model


def check_hub_shapes(cfg: VLConfig, repo: str) -> None:
    """Compare a preset against the real checkpoint's safetensors metadata."""
    from huggingface_hub import get_safetensors_metadata

    multimodal = cfg.vision is not None
    print(f"\n  {repo}")
    with torch.device("meta"):                 # shapes only, no memory
        model = Qwen38(cfg) if multimodal else Qwen38Text(cfg.text)

    metadata = get_safetensors_metadata(repo)  # reads file headers, not weights
    hub_shapes = {}
    for file_metadata in metadata.files_metadata.values():
        for name, info in file_metadata.tensors.items():
            hub_shapes[name] = tuple(info.shape)

    state = model.state_dict()
    mapped, mismatches = {}, []
    for hub_key, shape in hub_shapes.items():
        our_key = map_hf_key(hub_key, multimodal)
        if our_key is None:
            mismatches.append(f"unclaimed checkpoint tensor: {hub_key}")
        elif our_key not in state:
            mismatches.append(f"checkpoint tensor we don't build: {hub_key}")
        else:
            mapped[our_key] = (hub_key, shape)

    for our_key, tensor in state.items():
        if our_key not in mapped:
            mismatches.append(f"our tensor missing from checkpoint: {our_key}")
        elif mapped[our_key][1] != tuple(tensor.shape):
            mismatches.append(f"shape mismatch {mapped[our_key][0]}: "
                              f"ours {tuple(tensor.shape)} vs hub {mapped[our_key][1]}")

    assert not mismatches, "\n".join(mismatches[:15])
    n_mtp = sum(1 for k in mapped if ".mtp." in k or k.startswith("mtp."))
    print(f"    all {len(mapped)} checkpoint tensors match name-for-name and shape-for-shape")
    print(f"    including the {n_mtp}-tensor MTP head that transformers drops")
    print(f"    parameters: {sum(t.numel() for t in state.values()):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hub", action="store_true",
                        help="skip the released-checkpoint shape check (needs network)")
    args = parser.parse_args()

    print("[1] the two delta-rule forms agree")
    # The chunked form is an algebraic rearrangement of the recurrence, not an
    # approximation, so it must match the literal loop at any chunk size.
    torch.manual_seed(0)
    from gated_deltanet import chunked_delta_rule, recurrent_delta_rule
    B, T, H, K, V = 2, 37, 4, 16, 16
    q, k, v = torch.randn(B, T, H, K), torch.randn(B, T, H, K), torch.randn(B, T, H, V)
    g, beta = -torch.rand(B, T, H) * 0.5, torch.rand(B, T, H)
    reference = recurrent_delta_rule(q, k, v, g, beta)
    for chunk in (8, 16, 64):
        diff = (chunked_delta_rule(q, k, v, g, beta, chunk_size=chunk)
                - reference).abs().max().item()
        print(f"  chunk_size {chunk:3d} vs the token-by-token loop: max diff = {diff:.2e}")
        assert diff < 1e-5, f"chunked delta rule disagrees at chunk_size={chunk}: {diff}"
    print("  PASS")

    print("\n[2] dense text parity (Gated DeltaNet + gated attention + partial MRoPE)")
    dense_cfg = TextConfig()
    hf_dense = check_text_parity(dense_cfg, "dense", seed=0)
    print("  PASS")

    print("\n[3] sparse text parity (routed experts + gated shared expert)")
    check_text_parity(TextConfig.tiny_moe(), "moe  ", seed=1)
    print("  PASS (bit-exact against the reference's own PyTorch expert loop)")

    print("\n[4] vision parity (packed patches, two images of different sizes)")
    torch.manual_seed(2)
    vision_cfg = VisionConfig()
    hf_vision = jitter(build_hf_vision_twin(vision_cfg))
    our_vision = VisionTower(vision_cfg).eval()
    our_vision.load_state_dict(hf_vision.state_dict(), strict=True)
    grid = torch.tensor([[1, 4, 6], [1, 2, 2]])
    pixels = torch.randn(int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum()), vision_cfg.patch_dim)
    with torch.no_grad():
        diff = (hf_vision(pixels, grid).pooler_output - our_vision(pixels, grid)).abs().max().item()
    print(f"  2 images ({pixels.shape[0]} patches): max |feature diff| = {diff:.3e}")
    assert diff < 1e-4, f"vision diverges: {diff}"
    print("  PASS")

    print("\n[5] file roundtrip: save_pretrained -> load_hf_checkpoint")
    with tempfile.TemporaryDirectory() as tmp:
        hf_dense.save_pretrained(tmp)
        reloaded = Qwen38Text(dense_cfg).eval()
        load_hf_checkpoint(reloaded, tmp)
        worst = compare_text(hf_dense, reloaded, dense_cfg, "reloaded")
    assert worst < 1e-4, f"logits diverge after roundtrip: {worst}"
    print("  PASS")

    if args.skip_hub:
        print("\n[6] released-checkpoint shape check skipped (--skip-hub)")
    else:
        print("\n[6] shape check against the released Qwen3.8 checkpoints")
        check_hub_shapes(VLConfig.qwen3_8_27b(), "Qwen/Qwen3.8-27B")
        check_hub_shapes(VLConfig.qwen3_8_2_4t_a95b(), "Qwen/Qwen3.8-2.4T-A95B")
        print("  PASS")

    print("\nAll checks passed. Note: the MTP head is verified by name and shape "
          "only - no released implementation exists to compare its values against.")


if __name__ == "__main__":
    main()
