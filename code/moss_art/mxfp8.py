"""Frozen MXFP8 linears for Kohaku-style additive LoRA in native JAX.

E4M3 values and one power-of-two E8M0 scale per 32 contraction elements.
The format and optional separately quantized transpose follow the design in
KohakuMossTrainer quant/compute.py and kernels/fp8/formats.py. GPU execution
uses JAX's public scaled_matmul API, lowered to cuDNN blockScaledDot.
Only transformer block projections are converted. LoRA stays FP32.

The CPU implementation is a numerical reference for tests, never a GPU fallback.
BF16 dgrad dequantizes one frozen layer transiently. MXFP8 dgrad instead retains
an independently quantized W.T; both copies cost slightly more than BF16 W.
"""
from functools import partial
import jax
import jax.numpy as jnp

BLOCK = 32
SCALE_SUFFIX = '.mx_scale'
TRANSPOSE_SUFFIX = '.mx_transpose'
TRANSPOSE_SCALE_SUFFIX = '.mx_transpose_scale'


def quantization_contract():
    """Metadata only: scales are local deterministic functions of each input.

    No delayed amax, running statistics, calibration dataset or mutable state
    enters rollout. Frozen weight scales depend only on the frozen base;
    activation scales are recomputed on each actual invocation.
    """
    return dict(weight_format="e4m3fn", scale_format="e8m0fnu", block_size=BLOCK,
                scaling="ceil_power_of_two_amax_over_448", history=False,
                weight_scales="frozen_base_only", activation_scales="per_invocation",
                activation_dtype="bfloat16", accumulation_dtype="float32",
                scope="eligible_transformer_block_projections")


def eligible(key, value):
    return (key.startswith('layers.') and '.proj.' in key and key.endswith('.weight')
            and value.ndim == 2 and all(d % 128 == 0 for d in value.shape))


@jax.jit
def quantize(matrix):
    if matrix.ndim != 2 or matrix.shape[1] % BLOCK:
        raise ValueError('MXFP8 requires [rows, K] with K divisible by 32')
    blocks = matrix.astype(jnp.float32).reshape(matrix.shape[0], -1, BLOCK)
    amax = jnp.maximum(jnp.max(jnp.abs(blocks), axis=-1), jnp.float32(1e-30))
    # ceil(log2(amax / 448)), computed with FP32 exponent/mantissa bits to
    # keep the exact 448 * 2**e boundary independent of log2 rounding.
    bits = jax.lax.bitcast_convert_type(amax, jnp.uint32)
    exponent = ((bits >> 23) & 255).astype(jnp.int32) - 8
    exponent += ((bits & 0x7fffff) > 0x600000).astype(jnp.int32)
    scale_bits = jnp.clip(exponent, 0, 254).astype(jnp.uint8)
    scales = jax.lax.bitcast_convert_type(scale_bits, jnp.float8_e8m0fnu)
    values = jnp.clip(blocks / scales.astype(jnp.float32)[..., None], -448, 448)
    return values.astype(jnp.float8_e4m3fn).reshape(matrix.shape), scales


def dequantize(values, scales, dtype=jnp.bfloat16):
    blocks = values.astype(jnp.float32).reshape(values.shape[0], -1, BLOCK)
    return (blocks * scales.astype(jnp.float32)[..., None]).reshape(values.shape).astype(dtype)


def _matmul(x, weight, scales):
    shape = x.shape[:-1] + (weight.shape[0],)
    flat = x.reshape(-1, x.shape[-1])
    # M may include an arbitrary prompt capacity / final partial microbatch.
    rows = flat.shape[0]
    flat = jnp.pad(flat, ((0, -rows % 128), (0, 0)))
    values, x_scales = quantize(flat)
    if jax.default_backend() == 'cpu':
        # CPU-only numerical reference; no silent dequantization on CUDA.
        y = jnp.matmul(dequantize(values, x_scales, jnp.float32),
                       dequantize(weight, scales, jnp.float32).T,
                       precision=jax.lax.Precision.HIGHEST)
    else:
        y = jax.nn.scaled_matmul(values[None], weight[None], x_scales[None], scales[None],
                                preferred_element_type=jnp.float32)[0]
    return y[:rows].reshape(shape)


@jax.custom_vjp
def frozen_matmul(x, weight, scales, transposed=None, transposed_scales=None):
    return _matmul(x, weight, scales)


def _forward(x, weight, scales, transposed, transposed_scales):
    return _matmul(x, weight, scales), (weight, scales, transposed, transposed_scales,
                                      jnp.zeros((), x.dtype))


def _backward(saved, cotangent):
    weight, scales, transposed, transposed_scales, dtype_marker = saved
    if transposed is None:
        dense = dequantize(weight, scales, dtype_marker.dtype)
        dx = jnp.matmul(cotangent.astype(dtype_marker.dtype), dense,
                        precision=jax.lax.Precision.HIGHEST,
                        preferred_element_type=jnp.float32)
    else:
        dx = _matmul(cotangent.astype(dtype_marker.dtype), transposed, transposed_scales)
    return dx.astype(dtype_marker.dtype), None, None, None, None


frozen_matmul.defvjp(_forward, _backward)


def linear(base, prefix, x):
    from vendor.moss_jax.dit import _cast
    # DiT block activations remain BF16; only frozen linear operands are MXFP8.
    x = x.astype(jnp.bfloat16)
    y = frozen_matmul(x, base[prefix+'.weight'], base[prefix+SCALE_SUFFIX],
                     base.get(prefix+TRANSPOSE_SUFFIX), base.get(prefix+TRANSPOSE_SCALE_SUFFIX))
    if prefix+'.bias' in base:
        y = y + base[prefix+'.bias'].astype(jnp.float32)
    return _cast(y, jnp.bfloat16)