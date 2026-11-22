"""Diagnostic VAE with FP16 decoder weights and persistent FP16 activations.

BN denormalization/unpacking stays FP32 outside the decoder. Decoder convolution,
linear, bias, residual, SiLU and attention input/output tensors use the selected
dtype throughout. GroupNorm statistics and affine computation use FP32 (stored FP16
scale/bias are promoted for the affine), then cast back; QK and PV use selected-dtype operands with FP32 accumulation; softmax is
stable FP32 and probabilities cast back before PV. Native convolution accumulation
follows XLA/backend defaults and any caller-supplied environment; this module and
its benchmark do not force a global precision environment policy.
For images above the tiling threshold, blending uses FP32 opmath with a return
to model dtype after each vertical/horizontal blend. At 1024 no tiling occurs.
The returned raw RGB is cast FP32 only after decoding for storage/comparison.
This experiment is separate from production stage-selective mixed precision.
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
_get = vae._get
def _blend_v(above, tile, extent):
    # Preserve the original FP32 blend opmath and per-direction FP16 rounding,
    # without relying on unsafe FP32-to-FP16 scatter conversion.
    return vae._blend_v(above.astype(jnp.float32), tile.astype(jnp.float32), extent).astype(tile.dtype)


def _blend_h(left, tile, extent):
    return vae._blend_h(left.astype(jnp.float32), tile.astype(jnp.float32), extent).astype(tile.dtype)


class Decoder:
    def __init__(self, precision="float16"):
        if precision not in ("fp32", "float16"):
            raise ValueError("precision must be fp32 or float16")
        self.dtype = jnp.float16 if precision == "float16" else jnp.float32

    def _param(self, params, name):
        return jnp.asarray(params[name], self.dtype)

    def _group_norm(self, params, x, prefix, groups, eps=1e-6):
        # FP32 statistics avoid FP16 reduction overflow; return to FP16 before
        # the next activation or convolution, as in ordinary mixed-opmath GN.
        return vae._group_norm(params, x.astype(jnp.float32), prefix, groups, eps).astype(self.dtype)

    def _silu(self, x):
        return x * jax.nn.sigmoid(x)

    def _conv2d(self, params: Params, x: Array, prefix: str) -> Array:
        weight = self._param(params, prefix + ".weight")
        pad_h, pad_w = weight.shape[-2] // 2, weight.shape[-1] // 2
        x = lax.conv_general_dilated(
            x,
            weight,
            window_strides=(1, 1),
            padding=((pad_h, pad_h), (pad_w, pad_w)),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=self.dtype,
        )
        if prefix + ".bias" in params:
            x = x + self._param(params, prefix + ".bias")[None, :, None, None]
        return x

    def _linear(self, params: Params, x: Array, prefix: str) -> Array:
        x = jnp.matmul(
            x, self._param(params, prefix + ".weight").T,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=self.dtype,
        )
        if prefix + ".bias" in params:
            x = x + self._param(params, prefix + ".bias")
        return x

    def _resnet(self, params: Params, x: Array, prefix: str, groups: int) -> Array:
        residual = x
        x = self._conv2d(params, self._silu(self._group_norm(params, x, prefix + ".norm1", groups)), prefix + ".conv1")
        x = self._conv2d(params, self._silu(self._group_norm(params, x, prefix + ".norm2", groups)), prefix + ".conv2")
        if prefix + ".conv_shortcut.weight" in params:
            residual = self._conv2d(params, residual, prefix + ".conv_shortcut")
        return residual + x

    def _attention(self,
        params: Params, x: Array, prefix: str, groups: int, query_chunk_size: int,
    ) -> Array:
        """The decoder's single-head spatial self attention, in bounded memory.

        Chunking only the query axis leaves each softmax over the complete key
        sequence, so this is ordinary full attention, not windowed attention.
        """
        residual = x
        n, c, h, w = x.shape
        x = self._group_norm(params, x, prefix + ".group_norm", groups)
        x = x.reshape(n, c, h * w).transpose(0, 2, 1)
        query = self._linear(params, x, prefix + ".to_q")
        key = self._linear(params, x, prefix + ".to_k")
        value = self._linear(params, x, prefix + ".to_v")
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
            probs = jax.nn.softmax(scores, axis=-1).astype(self.dtype)
            return jnp.matmul(
                probs, value, precision=lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            ).astype(self.dtype)

        # lax.map lowers to a loop rather than materializing all score matrices.
        attended = lax.map(attend, chunks)
        x = attended.transpose(1, 0, 2, 3).reshape(n, -1, head_dim)[:, :tokens]
        x = self._linear(params, x, prefix + ".to_out.0")
        return x.transpose(0, 2, 1).reshape(n, c, h, w) + residual

    def _decode_direct(self,
        params: Params, z: Array, config: Any, attention_chunk_size: int,
    ) -> Array:
        groups = int(_get(config, "norm_num_groups"))
        if _get(config, "act_fn") not in ("silu", "swish"):
            raise ValueError("Only the SiLU FLUX.2 decoder is supported")
        if bool(_get(config, "use_post_quant_conv")):
            z = self._conv2d(params, z, "post_quant_conv")
        x = self._conv2d(params, z, "decoder.conv_in")
        x = self._resnet(params, x, "decoder.mid_block.resnets.0", groups)
        if bool(_get(config, "mid_block_add_attention")):
            x = self._attention(params, x, "decoder.mid_block.attentions.0", groups, attention_chunk_size)
        x = self._resnet(params, x, "decoder.mid_block.resnets.1", groups)
        up_blocks = tuple(_get(config, "up_block_types"))
        for block_index, block_type in enumerate(up_blocks):
            if block_type != "UpDecoderBlock2D":
                raise ValueError(f"Unsupported decoder up block: {block_type}")
            prefix = f"decoder.up_blocks.{block_index}"
            for resnet_index in range(int(_get(config, "layers_per_block")) + 1):
                x = self._resnet(params, x, f"{prefix}.resnets.{resnet_index}", groups)
            if block_index < len(up_blocks) - 1:
                x = jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)
                x = self._conv2d(params, x, prefix + ".upsamplers.0.conv")
        x = self._silu(self._group_norm(params, x, "decoder.conv_norm_out", groups))
        return self._conv2d(params, x, "decoder.conv_out")

    def decode_vae(self,
        params: Params,
        z: Any,
        config: Any = None,
        *,
        tiling: bool = False,
        slicing: bool = False,
        attention_chunk_size: int = 256,
        tile_overlap_factor: float = 0.25,
    ) -> Array:
        """Decode already-unpacked, denormalized VAE latents (like vae.decode).

        ``tiling=True`` reproduces diffusers' overlapping-tile decoding, including
        its strict size threshold and sequential vertical-then-horizontal blend.
        A 1024x1024 output uses the direct path with the model's default config.
        """
        z = jnp.asarray(z, dtype=self.dtype)
        if z.ndim != 4:
            raise ValueError(f"Expected unpacked NCHW latents, got shape {z.shape}")
        if slicing and z.shape[0] > 1:
            return jnp.concatenate([
                self.decode_vae(params, z[i:i + 1], config, tiling=tiling,
                           attention_chunk_size=attention_chunk_size,
                           tile_overlap_factor=tile_overlap_factor)
                for i in range(z.shape[0])
            ], axis=0)
        sample_size = _get(config, "sample_size")
        sample_size = int(sample_size[0] if isinstance(sample_size, (tuple, list)) else sample_size)
        scale_factor = 2 ** (len(_get(config, "block_out_channels")) - 1)
        tile_latent_size = sample_size // scale_factor
        if not tiling or max(z.shape[-2:]) <= tile_latent_size:
            return self._decode_direct(params, z, config, attention_chunk_size)
        overlap_size = int(tile_latent_size * (1 - tile_overlap_factor))
        blend_extent = int(sample_size * tile_overlap_factor)
        row_limit = sample_size - blend_extent
        if overlap_size <= 0:
            raise ValueError("tile_overlap_factor must leave a positive tile stride")
        rows = [
            [
                self._decode_direct(params, z[:, :, i:i + tile_latent_size, j:j + tile_latent_size],
                               config, attention_chunk_size)
                for j in range(0, z.shape[3], overlap_size)
            ]
            for i in range(0, z.shape[2], overlap_size)
        ]
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i:
                    tile = _blend_v(rows[i - 1][j], tile, blend_extent)
                if j:
                    tile = _blend_h(row[j - 1], tile, blend_extent)
                # Torch mutates the original tiles while blending; preserve those
                # updated values for subsequent tiles above/left of their neighbor.
                tile = tile.astype(self.dtype)
                row[j] = tile
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(jnp.concatenate(result_row, axis=3))
        return jnp.concatenate(result_rows, axis=2)

def decode_latents(
    params: Params, latents: Any, config: Any = None, *,
    precision="float16", tiling=False, slicing=False,
    attention_chunk_size=256, tile_overlap_factor=.25, clamp=False,
) -> Array:
    """Decode packed latents; all decoder-stage activations use selected dtype."""
    decoder = Decoder(precision)
    unpacked = vae.unpack_latents(params, latents, config)
    result = decoder.decode_vae(
        params, unpacked.astype(decoder.dtype), config, tiling=tiling, slicing=slicing,
        attention_chunk_size=attention_chunk_size, tile_overlap_factor=tile_overlap_factor,
    ).astype(jnp.float32)
    return jnp.clip(result, -1, 1) if clamp else result
