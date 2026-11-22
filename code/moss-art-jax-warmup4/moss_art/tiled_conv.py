"""Experimental tiled same convolution for NCHW activations.

The patch matrix is bounded by tile_pixels * Kh * Kw * Cin elements instead of
N * H * W * Kh * Kw * Cin. Full inputs and output are still required. There are
no whole-image padded or transposed activation copies in the source algorithm.
Compiler temporary-memory analysis remains necessary for each compiled caller:
the logical patch bound alone is not a total-device-memory guarantee.
"""
from __future__ import annotations

import operator

import jax.numpy as jnp
from jax import lax


def conv2d_nchw(x, weight, *, tile_pixels: int = 4096):
    """NCHW x OIHW, unit-stride odd-kernel same convolution, without bias.

    Supports arbitrary positive batch/spatial/channel sizes and odd kernels
    including 1x1/3x3. Both arguments must share float32/float16/bfloat16 dtype.
    GEMM requests HIGHEST precision and FP32 accumulation/output, then casts the
    convolution result back to x.dtype. A caller adds bias after this boundary.

    tile_pixels is a static positive integer. The final tile may overlap the
    preceding tile; every overwritten position is recomputed identically. This
    avoids allocating a padded whole-image output for nondivisible image sizes.
    """
    x, weight = jnp.asarray(x), jnp.asarray(weight)
    if isinstance(tile_pixels, bool):
        raise ValueError("tile_pixels must be a positive integer")
    try:
        tile_pixels = operator.index(tile_pixels)
    except TypeError as error:
        raise ValueError("tile_pixels must be a static positive integer") from error
    if tile_pixels <= 0:
        raise ValueError("tile_pixels must be positive")
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("Expected x=NCHW and weight=OIHW")
    if any(size <= 0 for size in (*x.shape, *weight.shape)):
        raise ValueError("Convolution dimensions must be positive")
    batch_size, input_channels, height, width = x.shape
    output_channels, weight_channels, kh, kw = weight.shape
    if weight_channels != input_channels:
        raise ValueError("Input and weight channels differ")
    if kh % 2 != 1 or kw % 2 != 1:
        raise ValueError("Only odd kernels with symmetric same padding are supported")
    allowed = (jnp.dtype(jnp.float32), jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16))
    if x.dtype not in allowed or weight.dtype != x.dtype:
        raise ValueError("x and weight must share float32/float16/bfloat16 dtype")

    pixels = height * width
    tile = min(tile_pixels, pixels)
    tile_count = (pixels + tile - 1) // tile
    flattened = x.reshape(batch_size, input_channels, pixels)
    matrix = weight.transpose(2, 3, 1, 0).reshape(kh * kw * input_channels, output_channels)
    offsets = jnp.arange(tile, dtype=jnp.int32)
    channels = jnp.arange(input_channels, dtype=jnp.int32)[None, :]
    output = jnp.zeros((batch_size, output_channels, pixels), dtype=x.dtype)

    def body(index, result):
        batch = index // tile_count
        start = jnp.minimum((index % tile_count) * tile, pixels - tile)
        positions = start + offsets
        row, column = positions // width, positions % width

        def patch(dy, dx):
            rr, cc = row + dy - kh // 2, column + dx - kw // 2
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            spatial = jnp.clip(rr, 0, height - 1) * width + jnp.clip(cc, 0, width - 1)
            values = flattened[batch, channels, spatial[:, None]]
            return jnp.where(valid[:, None], values, jnp.zeros((), x.dtype))

        patches = jnp.concatenate([patch(dy, dx) for dy in range(kh) for dx in range(kw)], axis=1)
        values = jnp.matmul(
            patches, matrix, precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ).astype(x.dtype)
        return lax.dynamic_update_slice(result, values.T[None], (batch, 0, start))

    output = lax.fori_loop(0, batch_size * tile_count, body, output)
    return output.reshape(batch_size, output_channels, height, width)
