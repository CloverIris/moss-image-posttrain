"""Trainable FP32 LoRA factors; frozen MOSS weights retain their native layout.

This is the ordinary additive LoRA used by KohakuMossTrainer/LyCORIS LoCon:
delta W = (alpha / rank) * up @ down. Down is Kaiming-uniform, up starts
at zero. Factors and optimizer masters are FP32; the residual matmuls use
FP32 and are rounded to the base activation dtype at the addition.

Export uses Kohaku's accepted diffusion_model.<module>.lora_{down,up}.weight
keys. Import also accepts its lora_unet_ / lycoris_ names and normalizes
per-module alpha into up. DoRA/LoHa/LoKr/fused-qkv files are rejected.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ATTENTION_TARGETS = ("proj.q", "proj.k", "proj.v", "proj.o")
ALL_TARGETS = ATTENTION_TARGETS + ("proj.ffn_up_gate", "proj.ffn_down")
DOWN = ".lora_down.weight"
UP = ".lora_up.weight"


def target_modules(base: Mapping[str, Any], targets: str | Sequence[str] = "all") -> list[str]:
    if isinstance(targets, str):
        if targets == "all":
            targets = ALL_TARGETS
        elif targets == "attention":
            targets = ATTENTION_TARGETS
        else:
            targets = tuple(x.strip() for x in targets.split(",") if x.strip())
    selected = []
    for key, value in base.items():
        if key.startswith("layers.") and key.endswith(".weight") and len(value.shape) == 2:
            module = key[:-7]
            if any(module == item or module.endswith("." + item) for item in targets):
                selected.append(module)
    if not selected:
        raise ValueError(f"No DiT linear modules match LoRA targets: {targets}")
    return sorted(selected)


def init_lora(base: Mapping[str, Any], rank: int = 8, *, seed: int = 0,
              targets: str | Sequence[str] = "all", device=None) -> dict[str, jax.Array]:
    if type(rank) is not int or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer.")
    # NumPy initialization is deterministic independent of GPU count/backend.
    rng = np.random.default_rng(seed)
    result = {}
    for prefix in target_modules(base, targets):
        out_dim, in_dim = base[prefix + ".weight"].shape
        bound = in_dim ** -0.5  # torch.nn.init.kaiming_uniform_(a=sqrt(5))
        down = rng.uniform(-bound, bound, (rank, in_dim)).astype(np.float32)
        up = np.zeros((out_dim, rank), dtype=np.float32)
        result[prefix + DOWN] = jax.device_put(down, device)
        result[prefix + UP] = jax.device_put(up, device)
    return result


def linear(base, adapters, prefix, x, scale: float):
    from vendor.moss_jax import dit
    if prefix + ".mx_scale" in base:
        from .mxfp8 import linear as mx_linear
        result = mx_linear(base, prefix, x)
    else:
        result = dit.linear(base, prefix, x)
    down = adapters.get(prefix + DOWN)
    if down is None:
        return result
    up = adapters[prefix + UP]
    residual = jnp.matmul(jnp.matmul(x.astype(jnp.float32), down.T, precision=jax.lax.Precision.HIGHEST),
                          up.T, precision=jax.lax.Precision.HIGHEST)
    return dit._cast(result.astype(jnp.float32) + jnp.float32(scale) * residual, result.dtype)


def parameter_count(adapters) -> int:
    return sum(int(np.prod(value.shape)) for value in adapters.values())


def export_safetensors(path, adapters, *, alpha: float, metadata: Mapping[str, Any] | None = None,
                       style: str = "diffusion_model") -> None:
    from safetensors.numpy import save_file
    if style not in ("diffusion_model", "lora_unet", "native"):
        raise ValueError("style must be diffusion_model, lora_unet, or native.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    result = {}
    modules = sorted(key[:-len(DOWN)] for key in adapters if key.endswith(DOWN))
    if not modules or len(adapters) != 2 * len(modules):
        raise ValueError("Expected only paired LoRA down/up factors.")
    for module in modules:
        stem = (f"diffusion_model.{module}" if style == "diffusion_model" else
                f"lora_unet_{module.replace('.', '_')}" if style == "lora_unet" else module)
        down = np.asarray(jax.device_get(adapters[module + DOWN]), dtype=np.float32)
        up = np.asarray(jax.device_get(adapters[module + UP]), dtype=np.float32)
        if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
            raise ValueError(f"Invalid LoRA factor shapes for {module}.")
        if not np.isfinite(down).all() or not np.isfinite(up).all():
            raise ValueError(f"Nonfinite LoRA factors for {module}.")
        result[stem + DOWN] = np.ascontiguousarray(down)
        result[stem + UP] = np.ascontiguousarray(up)
        result[stem + ".alpha"] = np.asarray(alpha, dtype=np.float32)
    meta = {"format": "moss-art-jax-lora-v1", "scale_rule": "alpha/rank"}
    meta.update({str(k): (v if isinstance(v, str) else json.dumps(v)) for k, v in (metadata or {}).items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(result, str(temporary), metadata=meta)
    temporary.replace(path)


def import_safetensors(path, template, *, alpha: float, device=None) -> dict[str, jax.Array]:
    """Load compatible LoRA only, rejecting missing/extra tensors and wrong shapes.

    template is model.init_lora(...); rank/targets come from this template.
    File alpha values are preserved by rescaling each loaded up factor.
    """
    from safetensors.numpy import load_file
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    raw = load_file(str(path))
    modules = sorted(key[:-len(DOWN)] for key in template if key.endswith(DOWN))
    aliases = {}
    for module in modules:
        for alias in (module, "diffusion_model." + module, "transformer." + module,
                      "lora_unet_" + module.replace(".", "_"),
                      "lycoris_" + module.replace(".", "_")):
            aliases[alias] = module
    grouped = {}
    for key, value in raw.items():
        suffix = next((s for s in (DOWN, UP, ".alpha") if key.endswith(s)), None)
        if suffix is None or key[:-len(suffix)] not in aliases:
            raise ValueError(f"Unsupported or untargeted adapter tensor: {key}")
        module = aliases[key[:-len(suffix)]]
        if suffix in grouped.setdefault(module, {}):
            raise ValueError(f"Duplicate adapter factor for {module}: {suffix}")
        grouped[module][suffix] = value
    if set(grouped) != set(modules):
        raise ValueError("Adapter modules do not exactly match model LoRA targets.")
    result = {}
    for module, parts in grouped.items():
        if DOWN not in parts or UP not in parts:
            raise ValueError(f"Missing LoRA down/up for {module}")
        rank = template[module + DOWN].shape[0]
        for suffix in (DOWN, UP):
            if parts[suffix].shape != template[module + suffix].shape:
                raise ValueError(f"Shape mismatch for {module + suffix}: {parts[suffix].shape}")
        if not np.isfinite(parts[DOWN]).all() or not np.isfinite(parts[UP]).all():
            raise ValueError(f"Nonfinite LoRA factors for {module}")
        alpha_array = np.asarray(parts.get(".alpha", rank))
        if alpha_array.size != 1:
            raise ValueError(f"alpha must be scalar for {module}")
        saved_alpha = float(alpha_array.reshape(-1)[0])
        if not np.isfinite(saved_alpha) or saved_alpha <= 0:
            raise ValueError(f"Invalid alpha for {module}")
        result[module + DOWN] = jax.device_put(parts[DOWN].astype(np.float32), device)
        result[module + UP] = jax.device_put(parts[UP].astype(np.float32) * (saved_alpha / alpha), device)
    return result
