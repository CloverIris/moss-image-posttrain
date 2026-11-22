"""Reference parity for chunk recomputation, including all Q/K/V gradients."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.attention import attention
from moss_art.model import MossModel
from vendor.moss_jax import dit


def attention_case(length, dtype):
    rng = np.random.default_rng(121 + length)
    heads, head_dim = 2, 128
    shape = (1, length, heads * head_dim)
    q, k, v = [jnp.asarray(rng.normal(0, .3, shape), dtype) for _ in range(3)]
    rope = dit.rope_embeddings(1, length - 5, 5, head_dim, dtype=dtype)
    cotangent = jnp.asarray(rng.normal(0, .2, shape), dtype)
    return q, k, v, rope, cotangent, heads


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("length", [17, 128, 129, 257])
def test_forward_and_all_qkv_vjps_match_native(length, dtype):
    q, k, v, rope, cotangent, heads = attention_case(length, dtype)

    def forward_and_vjp(function):
        @jax.jit
        def compute(q, k, v, cotangent):
            out, pullback = jax.vjp(lambda q, k, v: function(q, k, v, rope, heads), q, k, v)
            return out, pullback(cotangent)
        return compute(q, k, v, cotangent)

    expected, expected_gradients = forward_and_vjp(dit.attention)
    actual, gradients = forward_and_vjp(attention)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    tolerance = dict(rtol=3e-5, atol=1e-6) if dtype == jnp.float32 else dict(rtol=2e-2, atol=2e-3)
    for label, gradient, reference in zip(("q", "k", "v"), gradients, expected_gradients):
        gradient, reference = np.asarray(gradient, np.float32), np.asarray(reference, np.float32)
        assert np.isfinite(gradient).all()
        assert np.count_nonzero(gradient)
        np.testing.assert_allclose(gradient, reference, err_msg=label, **tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
def test_chunked_model_zero_lora_matches_native(dtype):
    # 132 image tokens + 3 unpadded text tokens crosses a 128-query boundary.
    model = MossModel.tiny(seed=17, dtype=dtype)
    rng = np.random.default_rng(71)
    state = jnp.asarray(rng.normal(size=(1, model.latent_channels, 11, 12)), jnp.float32)
    context = jnp.asarray(rng.normal(size=(1, 3, model.text_dim)), jnp.float32)
    actual = model.predict(model.init_lora(), state, .4, context)
    expected = dit.compiled_transformer_forward(model.base, state, jnp.float32(400), context, config=model.config)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
