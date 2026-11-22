"""Small real CUDA/cuDNN check; two GPUs, synthetic tensors, no model assets."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.mesh_runtime import _loss_vjp
from moss_art.model import MossModel

pytestmark = pytest.mark.skipif(
    len(jax.local_devices()) != 2 or jax.devices()[0].platform != "gpu",
    reason="requires two selected CUDA devices")


def test_bf16_cudnn_base_and_distributed_regularizer_match_serial():
    config = dict(dim=128, ffn_dim=256, num_heads=2, num_layers=2,
                  text_dim=128, freq_dim=16, latent_channels=8, rope_base=10000.)
    options = dict(seed=17, config=config, rank=2, alpha=2., dtype=jnp.bfloat16,
                   offload_activations=False, attention_backend="cudnn")
    model = MossModel.tiny(data_parallel_size=2, **options)
    serial = MossModel.tiny(**options)
    rng = np.random.default_rng(42)
    params = {k: np.asarray(v)+rng.normal(0, .02, v.shape).astype(np.float32)
              for k, v in serial.init_lora().items()}
    adapters = {k: jax.device_put(v, model.parallel.replicated) for k, v in params.items()}
    state = rng.normal(size=(2, 8, 2, 2)).astype(np.float32)
    positive = rng.normal(size=(2, 8, 128)).astype(np.float32)
    negative = rng.normal(size=(2, 8, 128)).astype(np.float32)
    lengths = np.array([5, 7], np.int32)
    sigmas = np.array([.4, .7], np.float32)
    def conditions(rows):
        return dict(cfg_scale=2., negative_context=jnp.asarray(negative[rows], jnp.bfloat16),
                    negative_skip_layers=(1,), context_lengths=lengths[rows],
                    negative_context_lengths=np.array([3, 4], np.int32)[rows])
    base = model.predict_base(state, sigmas, jnp.asarray(positive, jnp.bfloat16), **conditions(slice(None)))
    prediction, pullback = model.predict_with_pullback(adapters, state, sigmas,
        jnp.asarray(positive, jnp.bfloat16), **conditions(slice(None)))
    valid = model.parallel.put(np.ones(2, bool))
    zeros = jnp.zeros_like(prediction)
    (loss, counts), cotangent, details = _loss_vjp(prediction, zeros,
        model.parallel.put(sigmas), zeros, model.parallel.put(np.zeros(2, np.float32)),
        model.parallel.put(np.full(2, -.05, np.float32)), model.parallel.put(np.zeros(2, bool)), valid,
        noise=.8, kl=1e-5, eps=1e-5, reference_prediction=base, reference_beta=.01, return_details=True)
    assert counts == (0, 2) and details["policy_active"] == 0
    assert details["policy_loss_sum"] == 0 and details["reference_mse_sum"] > 0
    np.testing.assert_allclose(loss, .01*details["reference_mse_sum"], rtol=2e-6)
    expected_cotangent = .02*(np.asarray(prediction)-np.asarray(base))/np.prod(state.shape[1:])
    np.testing.assert_allclose(cotangent, expected_cotangent, rtol=2e-6, atol=1e-9)
    actual_gradient = pullback(cotangent/2)
    expected_gradient = {k: np.zeros_like(v) for k, v in params.items()}
    for i in range(2):
        rows = slice(i, i+1)
        context = jnp.asarray(positive[rows], jnp.bfloat16)
        reference = serial.predict_base(state[rows], sigmas[rows], context, **conditions(rows))
        value, pb = serial.predict_with_pullback(params, state[rows], sigmas[rows], context, **conditions(rows))
        np.testing.assert_allclose(base[rows], reference, rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(prediction[rows], value, rtol=2e-3, atol=2e-3)
        gradient = pb(jnp.asarray(expected_cotangent[rows])/2)
        for key in params:
            expected_gradient[key] += np.asarray(gradient[key])
    for key, value in actual_gradient.items():
        assert np.isfinite(np.asarray(value)).all()
        np.testing.assert_allclose(value, expected_gradient[key], rtol=5e-3, atol=2e-5)
    assert any(np.any(np.asarray(v) != 0) for v in actual_gradient.values())
    # Disabling adapters produces the same result as their zero-delta initialization.
    np.testing.assert_array_equal(base, model.predict(model.init_lora(), state, sigmas,
        jnp.asarray(positive, jnp.bfloat16), **conditions(slice(None))))
