import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.model import MossModel
from vendor.moss_jax import dit


@pytest.fixture(scope="module")
def model():
    return MossModel.tiny(seed=7, rank=2)


def fixture(model):
    rng = np.random.default_rng(31)
    state = jnp.asarray(rng.normal(size=(1, model.latent_channels, 2, 2)), jnp.float32)
    context = jnp.asarray(rng.normal(size=(1, 3, model.text_dim)), jnp.float32)
    return state, context


def test_zero_lora_matches_native(model):
    state, context = fixture(model)
    adapters = model.init_lora(19)
    out = model.predict(adapters, state, 0.4, context)
    expected = dit.compiled_transformer_forward(model.base, state, jnp.float32(400), context, config=model.config)
    np.testing.assert_allclose(out, expected, rtol=2e-6, atol=2e-6)
    assert out.dtype == jnp.float32
    assert all(x.dtype == jnp.float32 for x in adapters.values())


def test_manual_vjp_matches_full_autodiff(model):
    state, context = fixture(model)
    adapters = model.init_lora(22)
    # Nonzero up ensures the down gradients are tested too.
    rng = np.random.default_rng(2)
    adapters = {k: v + jnp.asarray(rng.normal(0, 0.01, v.shape), jnp.float32) for k, v in adapters.items()}
    cotangent = jnp.asarray(rng.normal(size=state.shape), jnp.float32)
    prediction, pullback = model.predict_with_pullback(adapters, state, 0.71, context)
    manual = pullback(cotangent)
    ordinary = jax.grad(lambda a: jnp.sum(model.predict(a, state, 0.71, context) * cotangent))(adapters)
    assert set(manual) == set(adapters)
    for key in adapters:
        np.testing.assert_allclose(manual[key], ordinary[key], rtol=3e-5, atol=2e-6, err_msg=key)
        assert np.isfinite(manual[key]).all()
    # One descent step must lower a real velocity objective.
    target = jnp.zeros_like(prediction)
    value, grads = model.loss_and_grad(adapters, state, 0.71, context, lambda v: jnp.mean((v-target)**2))
    updated = jax.tree.map(lambda a, g: a - 0.03*g, adapters, grads)
    after = jnp.mean(model.predict(updated, state, 0.71, context)**2)
    assert float(after) < float(value)


def test_cfg_and_skipped_negative_vjp(model):
    state, context = fixture(model)
    negative = context[:, :2] * -0.25
    adapters = model.init_lora()
    kwargs = dict(cfg_scale=3.0, negative_context=negative, negative_skip_layers=(1,))
    positive = model.predict(adapters, state, 0.3, context)
    negative_v = model.predict(adapters, state, 0.3, negative, skip_layers=(1,))
    combined, pullback = model.predict_with_pullback(adapters, state, 0.3, context, **kwargs)
    np.testing.assert_allclose(combined, negative_v + 3*(positive-negative_v), atol=1e-6)
    cotangent = jnp.ones_like(state) / state.size
    manual = pullback(cotangent)
    ordinary = jax.grad(lambda a: jnp.mean(model.predict(a, state, 0.3, context, **kwargs)))(adapters)
    for key in adapters:
        np.testing.assert_allclose(manual[key], ordinary[key], rtol=4e-5, atol=2e-6, err_msg=key)


def test_bfloat16_base_has_finite_fp32_adapter_gradients():
    model = MossModel.tiny(seed=11, dtype=jnp.bfloat16)
    state, context = fixture(model)
    adapters = model.init_lora()
    value, gradients = model.loss_and_grad(adapters, state, 0.6, context, lambda v: jnp.mean(v**2))
    assert np.isfinite(value)
    assert all(v.dtype == jnp.float32 and np.isfinite(v).all() for v in gradients.values())
    assert sum(float(jnp.linalg.norm(v)) for k,v in gradients.items() if k.endswith(".lora_up.weight")) > 0
    assert all(np.all(v == 0) for k,v in gradients.items() if k.endswith(".lora_down.weight"))


def test_errors_are_explicit(model):
    state, context = fixture(model)
    with pytest.raises(ValueError, match="negative_context"):
        model.predict(model.init_lora(), state, 0.5, context, cfg_scale=3)
    with pytest.raises(ValueError, match="skip_layers"):
        model.predict(model.init_lora(), state, 0.5, context, skip_layers=(22,))


@pytest.mark.parametrize("sigma", [-0.01, 1.01, float("nan")])
def test_invalid_sigma_rejected(model, sigma):
    state, context = fixture(model)
    with pytest.raises(ValueError, match="sigma must be finite"):
        model.predict(model.init_lora(), state, sigma, context)


def test_non_fp32_master_rejected(model):
    state, context = fixture(model)
    adapters = jax.tree.map(lambda x: x.astype(jnp.bfloat16), model.init_lora())
    with pytest.raises(ValueError, match="FP32 master"):
        model.predict(adapters, state, 0.5, context)
