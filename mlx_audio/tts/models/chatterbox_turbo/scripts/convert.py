#!/usr/bin/env python3
"""
Convert Chatterbox Turbo/Nano weights from PyTorch to MLX format.

Mirrors ``chatterbox/scripts/convert.py`` for the GPT-2 T3 variants. The two
differ only in backbone size and source repo, so one script handles both:

    # Nano, 8-bit T3 backbone
    python -m mlx_audio.tts.models.chatterbox_turbo.scripts.convert \\
        --variant nano --quantize --q-bits 8

    # Nano, fp16
    python -m mlx_audio.tts.models.chatterbox_turbo.scripts.convert --variant nano

Quantization is selective: only ``t3.tfmr.h.*`` Linear layers are quantized,
matching the published ``mlx-community/Chatterbox-Turbo-TTS-8bit`` layout. The
embeddings and heads stay in fp16, as they are sensitive to quantization.

Only T3 is converted. Nano and Turbo ship byte-identical ``ve.safetensors`` and
``s3gen_meanflow.safetensors``, so ``ve.*`` and ``s3gen.*`` are copied from the
published MLX Turbo checkpoint instead, which already carries them converted.
Converting them here would silently produce wrong weights: VoiceEncoder has no
``sanitize`` to split PyTorch's fused LSTM gates into ``Wx``/``Wh``/``bias``,
and ``S3Gen.sanitize`` is a shape heuristic that leaves the ``flow.``-prefixed
estimator blocks unmapped.

S3Tokenizer is not included; it loads at runtime from
``mlx-community/S3TokenizerV2``.
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.tts.models.chatterbox_turbo import T3, T3Config
from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import (
    VARIANT_REPO_IDS,
    _conds_from_pt,
    _t3_config_for,
)

# Source of the already-converted ve.* and s3gen.* tensors. Both published
# builds carry them byte-identically in fp16, so one source serves both.
GRAFT_REPO_ID = "mlx-community/Chatterbox-Turbo-TTS-fp16"
GRAFT_PREFIXES = ("ve.", "s3gen.")

# Tokenizer and vocabulary files copied verbatim from the source checkpoint.
# AutoTokenizer reads these directly; no tokenizer.json needs to be built.
TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def load_pytorch_safetensors(path: Path) -> dict:
    """Load a PyTorch safetensors file as numpy arrays."""
    from safetensors.numpy import load_file

    return load_file(str(path))


def numpy_to_mlx(weights: dict) -> dict:
    """Convert numpy arrays to MLX arrays."""
    return {k: mx.array(v) for k, v in weights.items()}


def save_mlx_quantized(weights: dict, path: Path):
    """Save MLX weights, preserving the packed uint32 form of quantized tensors."""
    mx.save_safetensors(str(path), weights, metadata={"format": "mlx"})
    print(f"Saved: {path} ({len(weights)} tensors)")


def download_weights(repo_id: str, cache_dir: Path = None) -> Path:
    """Download a source checkpoint from the Hub."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt"],
        )
    )


def load_graft_weights(cache_dir: Path = None) -> dict:
    """Load the already-converted ve.* and s3gen.* tensors from published Turbo."""
    graft_dir = download_weights(GRAFT_REPO_ID, cache_dir)
    weights = mx.load(str(graft_dir / "model.safetensors"))
    return {k: v for k, v in weights.items() if k.startswith(GRAFT_PREFIXES)}


def quantize_t3_backbone(model, bits: int = 8, group_size: int = 64) -> int:
    """Quantize only the GPT-2 transformer blocks (``tfmr.h.*``).

    The AR converter's equivalent matches Llama's ``tfmr.model.layers``; the
    GPT-2 backbone nests its blocks under ``tfmr.h`` instead.
    """
    quantized_count = [0]

    def should_quantize(path, module):
        if isinstance(module, nn.Linear) and "tfmr.h." in path:
            quantized_count[0] += 1
            return True
        return False

    nn.quantize(model, bits=bits, group_size=group_size, class_predicate=should_quantize)
    return quantized_count[0]


# GPT-2 stores these as Conv1D, whose weight is [in, out]; nn.Linear needs
# [out, in]. Matched by name, not shape: mlp.c_proj is [inner, n_embd], which
# already looks like [out, in], and attn.c_proj is square.
CONV1D_WEIGHTS = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def transpose_conv1d_weights(weights: dict) -> dict:
    """Transpose the GPT-2 Conv1D weights in a T3 state dict to nn.Linear layout."""
    return {
        k: (v.T if k.startswith("tfmr.h.") and k.endswith(CONV1D_WEIGHTS) else v)
        for k, v in weights.items()
    }


def _load_component(
    model, weights_path: Path, label: str, transpose: bool = False, strict: bool = False
) -> dict:
    """Load one component's weights, sanitizing them if the class supports it."""
    weights = numpy_to_mlx(load_pytorch_safetensors(weights_path))
    if transpose:
        weights = transpose_conv1d_weights(weights)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    # T3 loads strict: load_weights does not skip a shape-mismatched key, it
    # assigns the array anyway, so a bad orientation survives conversion and only
    # fails later inside quantized_matmul. VoiceEncoder and S3Gen cannot be
    # strict -- their PyTorch state dicts are keyed differently from the MLX
    # modules (fused LSTM gates, flow.-prefixed estimator blocks).
    model.load_weights(list(weights.items()), strict=strict)
    mx.eval(model.parameters())
    print(f"  {label}: {len(weights)} weights from {weights_path.name}")
    return weights


def convert(
    variant: str = "nano",
    output_dir: Path = None,
    cache_dir: Path = None,
    quantize: bool = False,
    bits: int = 8,
    group_size: int = 64,
) -> Path:
    """Convert a Turbo-family checkpoint to a single prefixed model.safetensors."""
    hp: T3Config = _t3_config_for(variant)
    repo_id = VARIANT_REPO_IDS[variant]

    if output_dir is None:
        suffix = f"{bits}bit" if quantize else "fp16"
        output_dir = Path(f"./Chatterbox-{variant.capitalize()}-TTS-{suffix}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {variant} ({repo_id})")
    ckpt_dir = download_weights(repo_id, cache_dir)

    t3_files = sorted(ckpt_dir.glob("t3*.safetensors"))
    if not t3_files:
        raise FileNotFoundError(f"No T3 safetensors in {ckpt_dir}")

    print("\nLoading components...")
    t3 = T3(hp)
    _load_component(t3, t3_files[0], "T3", transpose=True, strict=True)

    graft = load_graft_weights(cache_dir)
    print(f"  ve + s3gen: {len(graft)} weights from {GRAFT_REPO_ID}")

    if quantize:
        print(f"\nApplying {bits}-bit quantization to the T3 backbone...")
        n = quantize_t3_backbone(t3, bits=bits, group_size=group_size)
        mx.eval(t3.parameters())
        print(f"  Quantized {n} Linear layers")

    # Re-flatten from the live model so quantized tensors keep their packed
    # uint32 form and every key matches what load_weights expects.
    new_weights = {f"t3.{k}": v for k, v in tree_flatten(t3.parameters())}

    # Everything not packed into uint32 by quantization is stored as fp16, so a
    # quantized checkpoint keeps its unquantized tensors (the embeddings and
    # heads) at the same width as the fp16 build.
    new_weights = {
        k: (v.astype(mx.float16) if v.dtype in (mx.float32, mx.bfloat16) else v)
        for k, v in new_weights.items()
    }
    new_weights.update(graft)

    size_gb = sum(v.nbytes for v in new_weights.values()) / 1e9
    print(f"\nSaving {len(new_weights)} tensors ({size_gb:.3f} GB)...")
    save_mlx_quantized(new_weights, output_dir / "model.safetensors")

    # conds.pt -> conds.safetensors, so the checkpoint carries its built-in voice
    # in the format load_model prefers.
    conds = _conds_from_pt(ckpt_dir / "conds.pt")
    conds_flat = {
        "t3.speaker_emb": conds.t3.speaker_emb,
        "t3.cond_prompt_speech_tokens": conds.t3.cond_prompt_speech_tokens,
    }
    for k, v in conds.gen.items():
        if isinstance(v, mx.array):
            conds_flat[f"gen.{k}"] = v
    mx.save_safetensors(str(output_dir / "conds.safetensors"), conds_flat)
    print(f"Saved: conds.safetensors ({len(conds_flat)} tensors)")

    for name in TOKENIZER_FILES:
        src = ckpt_dir / name
        if src.exists():
            shutil.copy(src, output_dir / name)

    config = {
        "model_type": "chatterbox_turbo",
        "variant": variant,
        "version": "1.0",
        "dtype": "float16",
    }
    if quantize:
        config["quantization"] = {
            "bits": bits,
            "group_size": group_size,
            "quantized_components": ["t3.tfmr.h"],
        }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConversion complete: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / (1024 * 1024):.1f} MB")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert Chatterbox Turbo/Nano weights to MLX format"
    )
    parser.add_argument(
        "--variant", choices=sorted(VARIANT_REPO_IDS), default="nano",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--quantize", "-q", action="store_true")
    parser.add_argument("--q-bits", type=int, default=8, choices=[2, 3, 4, 8])
    parser.add_argument("--q-group-size", type=int, default=64)
    args = parser.parse_args()

    convert(
        variant=args.variant,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        quantize=args.quantize,
        bits=args.q_bits,
        group_size=args.q_group_size,
    )


if __name__ == "__main__":
    main()
