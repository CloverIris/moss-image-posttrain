"""Regression checks for the isolated full-FP16 decoder experiment."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_vae_precision import fixture
from scripts.diagnostics.vae_full_fp16 import Decoder, decode_latents
from vendor.moss_jax import vae


@pytest.mark.parametrize("tiling", [False, True])
def test_parameterized_fp32_matches_native(fixture, tiling):
    params, latents, config = fixture
    kwargs = dict(tiling=tiling, slicing=True, attention_chunk_size=3)
    reference = jax.jit(lambda p, z: vae.decode_latents(p, z, config, **kwargs))(params, latents)
    actual = jax.jit(lambda p, z: decode_latents(p, z, config, precision="fp32", **kwargs))(params, latents)
    np.testing.assert_array_equal(actual, reference)


def half_params(params):
    return {key: value.astype(jnp.float16) if not key.startswith("bn.") else value
            for key, value in params.items()}


def test_half_model_has_half_cross_layer_activation_and_conv_results(fixture):
    params, latents, config = fixture
    params = half_params(params)
    decoder = Decoder("float16")
    z = vae.unpack_latents(params, latents[:1, :, :1, :1], config).astype(jnp.float16)
    direct = lambda p, x: decoder._decode_direct(p, x, config, 3)
    raw = jax.jit(direct)(params, z)
    assert raw.dtype == jnp.float16 and np.isfinite(raw).all()
    graph = jax.make_jaxpr(direct)(params, z).jaxpr
    convs = [eq for eq in graph.eqns if eq.primitive.name == "conv_general_dilated"]
    assert len(convs) == 19
    for eq in convs:
        assert eq.params["preferred_element_type"] == np.dtype("float16")
        assert eq.outvars[0].aval.dtype == jnp.float16
    output = decode_latents(params, latents[:1, :, :1, :1], config)
    assert output.dtype == jnp.float32 and np.isfinite(output).all()
    assert float(output.max()) > 1  # Preserve raw values for the pre-clipping finite check.


def test_group_norm_uses_float32_statistics_and_promoted_half_affine(fixture):
    params, _, _ = fixture
    params = half_params(params)
    # Squaring/reducing these values in FP16 would overflow.
    x = jnp.arange(24, dtype=jnp.float16).reshape(1, 4, 2, 3) * 100
    actual = Decoder("float16")._group_norm(params, x, "decoder.conv_norm_out", 2)
    expected = vae._group_norm(params, x.astype(jnp.float32), "decoder.conv_norm_out", 2).astype(jnp.float16)
    assert actual.dtype == jnp.float16 and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)


def test_half_tiling_keeps_explicit_cast_boundaries_without_unsafe_scatter(fixture):
    params, latents, config = fixture
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        actual = decode_latents(
            half_params(params), latents, config, precision="float16",
            tiling=True, slicing=True, attention_chunk_size=3,
        )
    assert actual.dtype == jnp.float32 and np.isfinite(actual).all()


def test_unknown_precision_rejected():
    with pytest.raises(ValueError, match="precision must be"):
        Decoder("auto")
