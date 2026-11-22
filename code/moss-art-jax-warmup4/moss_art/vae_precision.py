"""Experimental VAE operand precision without changing FP32 sensitive math.

Only selected convolution and attention projection operands are rounded to
BF16/FP16. Selected convolutions return low-precision output, immediately cast
back to FP32 before bias; linear projections request FP32 results. Residuals,
normalization, SiLU, BN unpacking, attention scores/softmax/value reduction, and
blending stay FP32. cuDNN controls internal convolution accumulation.
The default XLA FP32 path delegates directly to the read-only native decoder.
The optional bounded cuDNN backend changes convolution algorithms only.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from vendor.moss_jax import vae

Array = jax.Array
Params = Mapping[str, Any]
STAGES = ("input", "mid", "up0", "up1", "up2", "up3", "output")
PRECISIONS = ("fp32", "bfloat16", "float16")


def _options(precision, low_precision_stages):
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if isinstance(low_precision_stages, (str, bytes)):
        raise ValueError("low_precision_stages must be a collection of stage names, not a string")
    try:
        stages = frozenset(low_precision_stages)
    except TypeError as exc:
        raise ValueError("low_precision_stages must be a collection of stage names") from exc
    unknown = stages.difference(STAGES)
    if unknown:
        raise ValueError(f"Unknown VAE stage(s): {sorted(unknown, key=str)}; expected {STAGES}")
    dtype = {"fp32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}[precision]
    return dtype, stages


def _conv_options(backend, workspace_mib, precision):
    if backend not in ("xla", "cudnn_bounded", "tiled_gemm"):
        raise ValueError("VAE convolution backend must be xla, cudnn_bounded or tiled_gemm")
    if type(workspace_mib) is not int or not 0 <= workspace_mib <= 512:
        raise ValueError("VAE convolution workspace must be an integer from 0 to 512 MiB")
    if backend != "xla" and precision == "bfloat16":
        raise ValueError("Optimized convolutions currently support FP32 or float16 VAE operands")


def _conv2d(params: Params, x: Array, prefix: str, operand_dtype,
            conv_backend="xla", conv_workspace_mib=256) -> Array:
    if conv_backend == "xla" and operand_dtype == jnp.float32:
        return vae._conv2d(params, x, prefix)
    weight = vae._param(params, prefix + ".weight")
    if conv_backend in ("cudnn_bounded", "tiled_gemm"):
        if conv_backend == "cudnn_bounded":
            from .cudnn_conv import conv2d_nchw
            x = conv2d_nchw(x.astype(operand_dtype), weight.astype(operand_dtype),
                           workspace_mib=conv_workspace_mib)
        else:
            from .tiled_conv import conv2d_nchw
            x = conv2d_nchw(x.astype(operand_dtype), weight.astype(operand_dtype))
        x = x.astype(jnp.float32)
        if prefix + ".bias" in params:
            x = x + vae._param(params, prefix + ".bias")[None, :, None, None]
        return x
    if conv_backend != "xla":
        raise ValueError(f"Unknown VAE convolution backend: {conv_backend!r}")
    pad_h, pad_w = weight.shape[-2] // 2, weight.shape[-1] // 2
    x = lax.conv_general_dilated(
        x.astype(operand_dtype), weight.astype(operand_dtype),
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        precision=lax.Precision.HIGHEST,
        # FP32 preferred output makes XLA promote the entire cuDNN convolution
        # to FP32 on this runtime. Retain low-precision conv input/output, then
        # restore FP32 for bias and every subsequent sensitive operation.
        preferred_element_type=operand_dtype,
    ).astype(jnp.float32)
    if prefix + ".bias" in params:
        x = x + vae._param(params, prefix + ".bias")[None, :, None, None]
    return x


def _linear(params: Params, x: Array, prefix: str, operand_dtype) -> Array:
    if operand_dtype == jnp.float32:
        return vae._linear(params, x, prefix)
    x = jnp.matmul(
        x.astype(operand_dtype), vae._param(params, prefix + ".weight").T.astype(operand_dtype),
        precision=lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )
    if prefix + ".bias" in params:
        x = x + vae._param(params, prefix + ".bias")
    return x


def _resnet(params: Params, x: Array, prefix: str, groups: int, operand_dtype,
            conv_backend="xla", conv_workspace_mib=256) -> Array:
    def conv(value, name):
        return _conv2d(params, value, name, operand_dtype,
                       conv_backend, conv_workspace_mib)
    residual = x
    x = conv(vae._silu(vae._group_norm(params, x, prefix + ".norm1", groups)), prefix + ".conv1")
    x = conv(vae._silu(vae._group_norm(params, x, prefix + ".norm2", groups)), prefix + ".conv2")
    if prefix + ".conv_shortcut.weight" in params:
        residual = conv(residual, prefix + ".conv_shortcut")
    return residual + x


def _attention(params: Params, x: Array, prefix: str, groups: int,
               query_chunk_size: int, operand_dtype) -> Array:
    """Native full attention; only Q/K/V/output projection operands may change."""
    residual = x
    n, c, h, w = x.shape
    x = vae._group_norm(params, x, prefix + ".group_norm", groups)
    x = x.reshape(n, c, h * w).transpose(0, 2, 1)
    query = _linear(params, x, prefix + ".to_q", operand_dtype)
    key = _linear(params, x, prefix + ".to_k", operand_dtype)
    value = _linear(params, x, prefix + ".to_v", operand_dtype)
    tokens, head_dim = query.shape[1:]
    chunk_size = min(int(query_chunk_size), tokens)
    if chunk_size <= 0:
        raise ValueError("attention_chunk_size must be positive")
    pad = (-tokens) % chunk_size
    padded = jnp.pad(query, ((0, 0), (0, pad), (0, 0)))
    chunks = padded.reshape(n, -1, chunk_size, head_dim).transpose(1, 0, 2, 3)
    key_t = key.transpose(0, 2, 1)
    scale = jnp.asarray(head_dim ** -0.5, dtype=jnp.float32)

    def attend(q: Array) -> Array:
        scores = jnp.matmul(
            q, key_t, precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * scale
        probs = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(
            probs, value, precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

    attended = lax.map(attend, chunks)
    x = attended.transpose(1, 0, 2, 3).reshape(n, -1, head_dim)[:, :tokens]
    x = _linear(params, x, prefix + ".to_out.0", operand_dtype)
    return x.transpose(0, 2, 1).reshape(n, c, h, w) + residual


def _decode_direct(params: Params, z: Array, config: Any, attention_chunk_size: int,
                   low_dtype, stages: frozenset, conv_backend="xla",
                   conv_workspace_mib=256) -> Array:
    dtype = lambda stage: low_dtype if stage in stages else jnp.float32
    def conv(value, name, operand_dtype):
        return _conv2d(params, value, name, operand_dtype, conv_backend, conv_workspace_mib)
    def resnet(value, name, groups, operand_dtype):
        return _resnet(params, value, name, groups, operand_dtype,
                       conv_backend, conv_workspace_mib)
    groups = int(vae._get(config, "norm_num_groups"))
    if vae._get(config, "act_fn") not in ("silu", "swish"):
        raise ValueError("Only the SiLU FLUX.2 decoder is supported")
    if bool(vae._get(config, "use_post_quant_conv")):
        z = conv(z, "post_quant_conv", dtype("input"))
    x = conv(z, "decoder.conv_in", dtype("input"))
    x = resnet(x, "decoder.mid_block.resnets.0", groups, dtype("mid"))
    if bool(vae._get(config, "mid_block_add_attention")):
        x = _attention(params, x, "decoder.mid_block.attentions.0", groups,
                       attention_chunk_size, dtype("mid"))
    x = resnet(x, "decoder.mid_block.resnets.1", groups, dtype("mid"))
    up_blocks = tuple(vae._get(config, "up_block_types"))
    for block_index, block_type in enumerate(up_blocks):
        if block_type != "UpDecoderBlock2D":
            raise ValueError(f"Unsupported decoder up block: {block_type}")
        prefix = f"decoder.up_blocks.{block_index}"
        stage_dtype = dtype(f"up{block_index}")
        for resnet_index in range(int(vae._get(config, "layers_per_block")) + 1):
            x = resnet(x, f"{prefix}.resnets.{resnet_index}", groups, stage_dtype)
        if block_index < len(up_blocks) - 1:
            x = jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)
            x = conv(x, prefix + ".upsamplers.0.conv", stage_dtype)
    x = vae._silu(vae._group_norm(params, x, "decoder.conv_norm_out", groups))
    return conv(x, "decoder.conv_out", dtype("output"))


def decode_vae(
    params: Params, z: Any, config: Any = None, *,
    precision: str = "fp32", low_precision_stages=(),
    conv_backend: str = "xla", conv_workspace_mib: int = 256,
    tiling: bool = False, slicing: bool = False,
    attention_chunk_size: int = 256, tile_overlap_factor: float = 0.25,
) -> Array:
    """Decode unpacked latents with native slicing/tiling and FP32 blending."""
    low_dtype, stages = _options(precision, low_precision_stages)
    _conv_options(conv_backend, conv_workspace_mib, precision)
    if conv_backend == "xla" and (precision == "fp32" or not stages):
        return vae.decode_vae(
            params, z, config, tiling=tiling, slicing=slicing,
            attention_chunk_size=attention_chunk_size, tile_overlap_factor=tile_overlap_factor,
        )
    z = jnp.asarray(z, dtype=jnp.float32)
    if z.ndim != 4:
        raise ValueError(f"Expected unpacked NCHW latents, got shape {z.shape}")
    if slicing and z.shape[0] > 1:
        return jnp.concatenate([
            decode_vae(
                params, z[i:i + 1], config, precision=precision, low_precision_stages=stages,
                conv_backend=conv_backend, conv_workspace_mib=conv_workspace_mib,
                tiling=tiling, attention_chunk_size=attention_chunk_size,
                tile_overlap_factor=tile_overlap_factor,
            )
            for i in range(z.shape[0])
        ], axis=0)
    sample_size = vae._get(config, "sample_size")
    sample_size = int(sample_size[0] if isinstance(sample_size, (tuple, list)) else sample_size)
    scale_factor = 2 ** (len(vae._get(config, "block_out_channels")) - 1)
    tile_latent_size = sample_size // scale_factor
    if not tiling or max(z.shape[-2:]) <= tile_latent_size:
        return _decode_direct(params, z, config, attention_chunk_size, low_dtype, stages,
                              conv_backend, conv_workspace_mib)
    overlap_size = int(tile_latent_size * (1 - tile_overlap_factor))
    blend_extent = int(sample_size * tile_overlap_factor)
    row_limit = sample_size - blend_extent
    if overlap_size <= 0:
        raise ValueError("tile_overlap_factor must leave a positive tile stride")
    rows = [
        [
            _decode_direct(
                params, z[:, :, i:i + tile_latent_size, j:j + tile_latent_size],
                config, attention_chunk_size, low_dtype, stages,
                conv_backend, conv_workspace_mib,
            )
            for j in range(0, z.shape[3], overlap_size)
        ]
        for i in range(0, z.shape[2], overlap_size)
    ]
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i:
                tile = vae._blend_v(rows[i - 1][j], tile, blend_extent)
            if j:
                tile = vae._blend_h(row[j - 1], tile, blend_extent)
            row[j] = tile
            result_row.append(tile[:, :, :row_limit, :row_limit])
        result_rows.append(jnp.concatenate(result_row, axis=3))
    return jnp.concatenate(result_rows, axis=2)


def decode_latents(
    params: Params, latents: Any, config: Any = None, *,
    precision: str = "fp32", low_precision_stages=(),
    conv_backend: str = "xla", conv_workspace_mib: int = 256,
    tiling: bool = False, slicing: bool = False,
    attention_chunk_size: int = 256, tile_overlap_factor: float = 0.25,
    clamp: bool = False,
) -> Array:
    """Decode packed normalized MOSS latents to FP32 NCHW RGB.

    Precision, stages and convolution options must be static under jit.
    Empty stage selections retain FP32 operands. The default XLA backend
    preserves the exact native FP32 decoder; bounded cuDNN selects other plans.
    Clamp and tile thresholds have the same semantics as the native decoder.
    """
    _, stages = _options(precision, low_precision_stages)
    _conv_options(conv_backend, conv_workspace_mib, precision)
    if conv_backend == "xla" and (precision == "fp32" or not stages):
        return vae.decode_latents(
            params, latents, config, tiling=tiling, slicing=slicing,
            attention_chunk_size=attention_chunk_size,
            tile_overlap_factor=tile_overlap_factor, clamp=clamp,
        )
    z = vae.unpack_latents(params, latents, config)
    result = decode_vae(
        params, z, config, precision=precision, low_precision_stages=stages,
        conv_backend=conv_backend, conv_workspace_mib=conv_workspace_mib,
        tiling=tiling, slicing=slicing, attention_chunk_size=attention_chunk_size,
        tile_overlap_factor=tile_overlap_factor,
    )
    return jnp.clip(result, -1, 1) if clamp else result
