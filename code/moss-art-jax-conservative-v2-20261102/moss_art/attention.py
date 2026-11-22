"""MOSS attention with explicit native math or cuDNN fused attention.

Native preserves the released FP32 math and rematerializes query chunks during
reverse mode. cuDNN uses fused BF16/FP16 attention, with different accumulation
and rounding; it is numerically close, not bitwise identical to native math.
The vendor snapshot stays intact. A requested cuDNN backend never falls back.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from vendor.moss_jax import dit


def _native_attention(q: jax.Array, k: jax.Array, v: jax.Array,
                      rope: jax.Array, heads: int, sequence_lengths=None) -> jax.Array:
    """Native B,S,H,D full attention, rematerializing one 128-query chunk."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = dit.rotate(q.reshape(shape), rope).astype(v.dtype)
    k = dit.rotate(k.reshape(shape), rope).astype(v.dtype)
    v = v.reshape(shape)
    split_scale = jnp.float32(q.shape[-1] ** -0.25)
    key, value = k.astype(jnp.float32) * split_scale, v.astype(jnp.float32)

    def attend_queries(query, keys, values):
        scores = jnp.einsum("bthd,bshd->bhts",
                            query.astype(jnp.float32) * split_scale, keys,
                            precision=lax.Precision.HIGHEST)
        if sequence_lengths is not None:
            valid_keys = jnp.arange(keys.shape[1])[None, :] < sequence_lengths[:, None]
            scores = jnp.where(valid_keys[:, None, None, :], scores, -jnp.inf)
        probabilities = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("bhts,bshd->bthd", probabilities, values,
                          precision=lax.Precision.HIGHEST)

    chunk_size = 128
    length = q.shape[1]
    if length <= chunk_size:
        out = attend_queries(q, key, value)
    else:
        count = (length + chunk_size - 1) // chunk_size
        query = jnp.pad(q, ((0, 0), (0, count * chunk_size - length), (0, 0), (0, 0)))
        query = query.reshape(q.shape[0], count, chunk_size, heads, q.shape[-1]).transpose(1, 0, 2, 3, 4)
        # lax.map lowers to a scan. Its iteration boundary already prevents
        # cross-iteration CSE, so no extra checkpoint barriers are necessary.
        # Explicit key/value arguments retain shared inputs, never a stack of
        # every chunk's score/softmax arrays, for reverse-mode differentiation.
        recompute_chunk = jax.checkpoint(attend_queries, prevent_cse=False)
        out = lax.map(lambda block: recompute_chunk(block, key, value), query)
        out = out.transpose(1, 0, 2, 3, 4).reshape(q.shape[0], count * chunk_size, heads, q.shape[-1])[:, :length]
    if sequence_lengths is not None:
        out = jnp.where((jnp.arange(out.shape[1])[None, :] < sequence_lengths[:, None])[:, :, None, None], out, 0)
    return dit._cast(out, v.dtype).reshape(*q.shape[:2], -1)


def _cudnn_attention(q: jax.Array, k: jax.Array, v: jax.Array,
                     rope: jax.Array, heads: int, sequence_lengths=None) -> jax.Array:
    """Fused B,S,H,D attention; original RoPE positions and valid tokens only."""
    if v.dtype not in (jnp.bfloat16, jnp.float16) or q.dtype != v.dtype or k.dtype != v.dtype:
        raise ValueError("cuDNN attention requires Q/K/V with the same BF16 or FP16 dtype.")
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = dit.rotate(q.reshape(shape), rope).astype(v.dtype)
    k = dit.rotate(k.reshape(shape), rope).astype(v.dtype)
    v = v.reshape(shape)
    length = q.shape[1]
    padding = (-length) % 2
    lengths = {}
    if sequence_lengths is not None:
        lengths = dict(query_seq_lengths=sequence_lengths, key_value_seq_lengths=sequence_lengths)
    if padding:
        # JAX 0.10.2 cuDNN backward requires even Q/K sequence lengths.
        # Pad AFTER RoPE so image/text coordinates never change. Sequence
        # lengths exclude the dummy key from softmax and the dummy query from
        # gradients; unmasked zero padding would change the actual attention.
        pad_width = ((0, 0), (0, padding), (0, 0), (0, 0))
        q, k, v = (jnp.pad(x, pad_width) for x in (q, k, v))
        valid_lengths = (jnp.full((q.shape[0],), length, dtype=jnp.int32)
                         if sequence_lengths is None else sequence_lengths)
        lengths = dict(query_seq_lengths=valid_lengths, key_value_seq_lengths=valid_lengths)
    # cuDNN applies the full scale itself: do not reuse native split scaling.
    out = jax.nn.dot_product_attention(
        q, k, v, scale=float(shape[-1] ** -0.5), is_causal=False,
        implementation="cudnn", **lengths)
    out = out[:, :length]
    if sequence_lengths is not None:
        # cuDNN receives only prefix lengths, never a dense attention mask.
        # Explicitly clear invalid outputs before residual/FFN processing.
        valid = jnp.arange(length)[None, :] < sequence_lengths[:, None]
        out = jnp.where(valid[:, :, None, None], out, 0)
    return out.reshape(shape[0], length, heads * shape[-1])


def attention(q: jax.Array, k: jax.Array, v: jax.Array,
              rope: jax.Array, heads: int, backend: str = "native", sequence_lengths=None) -> jax.Array:
    """Select native reference math or explicitly requested cuDNN attention."""
    if sequence_lengths is not None:
        sequence_lengths = jnp.asarray(sequence_lengths, dtype=jnp.int32)
        if sequence_lengths.shape != (q.shape[0],):
            raise ValueError("sequence_lengths must have shape [batch]")
    if backend == "native":
        return _native_attention(q, k, v, rope, heads, sequence_lengths)
    if backend == "cudnn":
        return _cudnn_attention(q, k, v, rope, heads, sequence_lengths)
    raise ValueError(f"Unknown attention backend {backend!r}; expected 'native' or 'cudnn'.")
