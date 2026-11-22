"""Numerical and shape tests for experimental tiled convolution."""
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from moss_art.tiled_conv import conv2d_nchw


def native(x, weight):
    kh, kw = weight.shape[2:]
    return lax.conv_general_dilated(
        x, weight, (1, 1), ((kh // 2, kh // 2), (kw // 2, kw // 2)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        precision=lax.Precision.HIGHEST, preferred_element_type=x.dtype,
    )


@pytest.mark.parametrize("shape,kernel,tile", [
    ((1, 3, 8, 8), (4, 3, 3, 3), 8),
    ((2, 3, 5, 7), (4, 3, 3, 3), 8),
    ((2, 3, 5, 7), (4, 3, 1, 1), 11),
    ((1, 2, 3, 5), (3, 2, 3, 5), 4),
    ((2, 2, 1, 1), (3, 2, 3, 3), 4096),
    ((1, 2, 3, 5), (3, 2, 3, 3), 1),
    ((1, 2, 3, 5), (3, 2, 1, 1), 4096),
])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
def test_no_bias_parity(shape, kernel, tile, dtype):
    rng = np.random.default_rng(91)
    x = jnp.asarray(rng.normal(size=shape) * .25, dtype)
    w = jnp.asarray(rng.normal(size=kernel) * .1, dtype)
    expected = np.asarray(jax.jit(native)(x, w), np.float32)
    actual = jax.jit(lambda a, b: conv2d_nchw(a, b, tile_pixels=tile))(x, w)
    assert actual.shape == expected.shape and actual.dtype == x.dtype
    tolerance = {jnp.float32: (2e-5, 2e-6), jnp.float16: (.003, .0005), jnp.bfloat16: (.02, .004)}[dtype]
    np.testing.assert_allclose(np.asarray(actual, np.float32), expected, rtol=tolerance[0], atol=tolerance[1])


def test_padding_and_last_overlap_are_correct():
    x = jnp.ones((2, 2, 5, 7), jnp.float32)
    w = jnp.ones((3, 2, 3, 3), jnp.float32)
    actual = np.asarray(jax.jit(lambda a, b: conv2d_nchw(a, b, tile_pixels=8))(x, w))
    expected = np.full((2, 3, 5, 7), 18, np.float32)
    expected[:, :, [0, -1], :] = 12
    expected[:, :, :, [0, -1]] = 12
    for row in (0, -1):
        for column in (0, -1):
            expected[:, :, row, column] = 8
    np.testing.assert_array_equal(actual, expected)


def test_jaxpr_has_no_convolution_and_uses_fp32_dot():
    x = jnp.ones((1, 3, 5, 7), jnp.float16)
    w = jnp.ones((4, 3, 3, 3), jnp.float16)
    graph = str(jax.make_jaxpr(lambda a, b: conv2d_nchw(a, b, tile_pixels=8))(x, w))
    assert "conv_general_dilated" not in graph
    assert "dot_general" in graph and "preferred_element_type=float32" in graph
    assert "HIGHEST" in graph


@pytest.mark.parametrize("tile", [0, -1, 1.5, True])
def test_reject_invalid_tile(tile):
    with pytest.raises(ValueError, match="tile_pixels"):
        conv2d_nchw(jnp.ones((1, 2, 3, 3)), jnp.ones((2, 2, 3, 3)), tile_pixels=tile)


def test_reject_invalid_inputs():
    x = jnp.ones((1, 2, 3, 3), jnp.float32)
    with pytest.raises(ValueError, match="channels"):
        conv2d_nchw(x, jnp.ones((2, 3, 3, 3), jnp.float32))
    with pytest.raises(ValueError, match="odd kernels"):
        conv2d_nchw(x, jnp.ones((2, 2, 2, 2), jnp.float32))
    with pytest.raises(ValueError, match="share"):
        conv2d_nchw(x, jnp.ones((2, 2, 3, 3), jnp.float16))
