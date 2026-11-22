"""Numerical contracts at the rollout/BF16 learner precision boundary."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.model import MossModel, tiny_base
from moss_art import mxfp8


@pytest.mark.parametrize("cfg_scale", [1.0, 3.0])
def test_reference_is_exact_learner_forward_with_padding_cfg_and_nonzero_lora(cfg_scale):
    model = MossModel.tiny(seed=17, dtype=jnp.bfloat16,
        config={"num_layers": 2}, checkpoint_interval=2, offload_activations=False)
    rng = np.random.default_rng(18)
    adapters = {key: value + jnp.asarray(rng.normal(0, .03, value.shape), jnp.float32)
                for key, value in model.init_lora().items()}
    state = jnp.asarray(rng.normal(size=(1, 8, 2, 2)), jnp.float32)
    positive = jnp.asarray(rng.normal(size=(1, 5, 24)), jnp.bfloat16)
    negative = jnp.asarray(rng.normal(size=(1, 3, 24)), jnp.bfloat16)
    kwargs = dict(cfg_scale=cfg_scale, negative_context=negative,
                  negative_skip_layers=(1,), context_lengths=np.array([4], np.int32),
                  negative_context_lengths=np.array([2], np.int32))
    reference = model.predict_reference(adapters, state, .61, positive, **kwargs)
    learner, pullback = model.predict_with_pullback(adapters, state, .61, positive, **kwargs)
    sampler = model.predict(adapters, state, .61, positive, **kwargs)
    np.testing.assert_array_equal(reference, learner)
    np.testing.assert_array_equal(reference, sampler)
    assert reference.dtype == jnp.float32
    gradient = pullback(jnp.ones_like(learner) / learner.size)
    assert all(np.isfinite(np.asarray(x)).all() for x in gradient.values())
    assert any(np.any(np.asarray(x) != 0) for x in gradient.values())


def test_phase_reload_restores_original_bf16_base_and_preserves_lora():
    config = dict(dim=128, ffn_dim=256, num_heads=4, num_layers=1, text_dim=128,
                  freq_dim=16, latent_channels=8, rope_base=10000.)
    host = tiny_base(config, seed=12, dtype=jnp.bfloat16)
    learner = MossModel(host, config, rank=2, offload_activations=False)
    lora = {k: v + jnp.float32(.005) for k, v in learner.init_lora().items()}
    state = jnp.ones((1, 8, 2, 2)); context = jnp.ones((1, 3, 128)) * .1
    expected = np.asarray(learner.predict_reference(lora, state, .4, context)).copy()
    borrowed = learner.base["layers.0.proj.q.weight"]
    learner.release_base()
    assert borrowed.is_deleted()
    assert learner.base == {} and learner._blocks == []
    assert learner._time == {} and learner._final == {}
    assert learner.base_storage_bytes == 0
    # Releasing owned GPU/CPU base buffers must not invalidate caller inputs.
    assert all(not v.is_deleted() for v in host.values())
    assert all(not v.is_deleted() for v in lora.values())
    sampler = MossModel(host, config, rank=2, offload_activations=False, base_precision="mxfp8")
    assert sampler.mxfp8_modules == 6
    quantized = np.asarray(sampler.predict(lora, state, .4, context)).copy()
    assert np.isfinite(quantized).all()
    assert not np.array_equal(quantized, expected)
    with pytest.raises(ValueError, match="BF16 learner"):
        sampler.predict_reference(lora, state, .4, context)
    assert sampler.precision_contract()["quantization"]["history"] is False
    sampler.release_base()
    restored = MossModel(host, config, rank=2, offload_activations=False)
    actual = restored.predict_reference(lora, state, .4, context)
    np.testing.assert_array_equal(actual, expected)
    assert restored.mxfp8_modules == 0


def test_release_is_idempotent_and_rejects_old_forward_or_pullback():
    model = MossModel.tiny()
    adapters = model.init_lora()
    state = jnp.ones((1, 8, 2, 2)); context = jnp.ones((1, 3, 24))
    value, pullback = model.predict_with_pullback(adapters, state, .5, context)
    model.release_base()
    model.release_base()
    with pytest.raises(RuntimeError, match="released"):
        model.predict(adapters, state, .5, context)
    with pytest.raises(RuntimeError, match="outlive"):
        pullback(jnp.ones_like(value))


def test_quantization_has_no_delayed_scale_history():
    x = jnp.asarray(np.random.default_rng(2).normal(size=(128, 128)), jnp.bfloat16)
    q0, s0 = mxfp8.quantize(x)
    # An unrelated extreme batch must not change the next identical input.
    jax.block_until_ready(mxfp8.quantize(x * 1000))
    q1, s1 = mxfp8.quantize(x)
    np.testing.assert_array_equal(np.asarray(q0).view(np.uint8), np.asarray(q1).view(np.uint8))
    np.testing.assert_array_equal(np.asarray(s0).view(np.uint8), np.asarray(s1).view(np.uint8))
    contract = mxfp8.quantization_contract()
    assert contract["block_size"] == 32 and contract["history"] is False
    assert contract["activation_scales"] == "per_invocation"
