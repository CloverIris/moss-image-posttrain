"""CPU contract tests for explicit cuDNN dispatch; GPU proof lives in reports."""
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.attention import attention
from moss_art.model import MossModel
from vendor.moss_jax import dit


def inputs(length, batch=2, dtype=jnp.bfloat16):
    heads, head_dim = 2, 128
    rng = np.random.default_rng(923 + length)
    shape = (batch, length, heads * head_dim)
    q, k, v = [jnp.asarray(rng.normal(0, .3, shape), dtype) for _ in range(3)]
    rope = dit.rope_embeddings(1, length - 5, 5, head_dim, dtype=dtype)
    return q, k, v, rope, heads


@pytest.mark.parametrize("length", [17, 18])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bf16", "fp16"])
def test_cudnn_uses_rotated_unscaled_inputs_and_masks_only_padding(monkeypatch, length, dtype):
    q, k, v, rope, heads = inputs(length, dtype=dtype)
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    expected_q = dit.rotate(q.reshape(shape), rope).astype(dtype)
    expected_k = dit.rotate(k.reshape(shape), rope).astype(dtype)
    expected_v = v.reshape(shape)
    calls = []

    def fused(query, key, value, **kwargs):
        calls.append(kwargs)
        padded = length + length % 2
        assert query.shape == key.shape == value.shape == (shape[0], padded, heads, shape[-1])
        assert query.dtype == key.dtype == value.dtype == dtype
        np.testing.assert_array_equal(query[:, :length], expected_q)
        np.testing.assert_array_equal(key[:, :length], expected_k)
        np.testing.assert_array_equal(value[:, :length], expected_v)
        assert kwargs["implementation"] == "cudnn"
        assert kwargs["is_causal"] is False
        assert kwargs["scale"] == shape[-1] ** -0.5
        if length % 2:
            for x in (query, key, value):
                np.testing.assert_array_equal(x[:, length:], 0)
            for name in ("query_seq_lengths", "key_value_seq_lengths"):
                np.testing.assert_array_equal(kwargs[name], [length] * shape[0])
                assert kwargs[name].dtype == jnp.int32
        else:
            assert "query_seq_lengths" not in kwargs
            assert "key_value_seq_lengths" not in kwargs
        # Dummy output must be cropped regardless of the kernel's pad contents.
        return query.at[:, length:].set(jnp.asarray(42, dtype))

    monkeypatch.setattr(jax.nn, "dot_product_attention", fused)
    actual = attention(q, k, v, rope, heads, backend="cudnn")
    assert len(calls) == 1
    assert actual.shape == q.shape
    np.testing.assert_array_equal(actual, expected_q.reshape(q.shape))


def test_explicit_native_does_not_use_dispatcher(monkeypatch):
    q, k, v, rope, heads = inputs(17, dtype=jnp.float32)
    fused = Mock(side_effect=AssertionError("native must retain reference math"))
    monkeypatch.setattr(jax.nn, "dot_product_attention", fused)
    actual = attention(q, k, v, rope, heads, backend="native")
    expected = dit.attention(q, k, v, rope, heads)
    np.testing.assert_array_equal(actual, expected)
    fused.assert_not_called()


def test_cudnn_errors_never_silently_fall_back(monkeypatch):
    q, k, v, rope, heads = inputs(17)
    fused = Mock(side_effect=RuntimeError("requested cuDNN kernel unavailable"))
    monkeypatch.setattr(jax.nn, "dot_product_attention", fused)
    with pytest.raises(RuntimeError, match="requested cuDNN kernel unavailable"):
        attention(q, k, v, rope, heads, backend="cudnn")
    assert fused.call_count == 1


def test_cudnn_rejects_float32_or_mixed_qkv(monkeypatch):
    fused = Mock(side_effect=AssertionError("invalid dtype must fail before dispatch"))
    monkeypatch.setattr(jax.nn, "dot_product_attention", fused)
    q, k, v, rope, heads = inputs(17, dtype=jnp.float32)
    with pytest.raises(ValueError, match="same BF16 or FP16"):
        attention(q, k, v, rope, heads, backend="cudnn")
    with pytest.raises(ValueError, match="same BF16 or FP16"):
        attention(q, k, v.astype(jnp.bfloat16), rope, heads, backend="cudnn")
    fused.assert_not_called()


def test_unknown_backend_rejected_before_model_creation():
    q, k, v, rope, heads = inputs(17)
    with pytest.raises(ValueError, match="Unknown attention backend"):
        attention(q, k, v, rope, heads, backend="auto")
    with pytest.raises(ValueError, match="Unknown attention backend"):
        MossModel({}, attention_backend="auto")
    with pytest.raises(ValueError, match="Unknown attention backend"):
        MossModel.tiny(attention_backend="auto")


def test_cudnn_model_cpu_failure_precedes_weight_loading(monkeypatch):
    import moss_art.model as module
    load = Mock(side_effect=AssertionError("must reject CPU before loading weights"))
    monkeypatch.setattr(module, "load_config", load)
    monkeypatch.setattr(module, "load_host_params", load)
    cpu = jax.devices("cpu")[0]
    with pytest.raises(ValueError, match="requires a CUDA GPU"):
        MossModel.from_pretrained("/nonexistent", device=cpu, attention_backend="cudnn")
    with pytest.raises(ValueError, match="requires a CUDA GPU"):
        MossModel.tiny(device=cpu, dtype=jnp.bfloat16, attention_backend="cudnn")
    with pytest.raises(ValueError, match="requires a CUDA GPU"):
        MossModel({}, device=cpu, attention_backend="cudnn")
    load.assert_not_called()


def test_cudnn_model_rejects_fp32_base_before_transfer():
    # A sentinel device verifies dtype validation happens before device_put.
    gpu = SimpleNamespace(platform="gpu")
    with pytest.raises(ValueError, match="requires BF16 or FP16 base parameters"):
        MossModel({"video_embedder.weight": np.ones((2, 2), np.float32)},
                  device=gpu, attention_backend="cudnn")


def test_block_forward_and_manual_vjp_share_static_backend(monkeypatch):
    import moss_art.model as module
    calls = []
    reference_attention = module.attention

    def traced_attention(q, k, v, rope, heads, backend="native"):
        calls.append(backend)
        assert backend == "cudnn"
        # Exercise model plumbing on CPU, retaining real differentiation.
        return reference_attention(q, k, v, rope, heads, backend="native")

    model = MossModel.tiny(seed=37)
    adapters = module._view(model.init_lora(), "layers.0")
    base = model._blocks[0]
    x = jnp.ones((1, 7, model.config["dim"]), dtype=jnp.float32)
    rope = dit.rope_embeddings(2, 2, 3, model.config["dim"] // model.config["num_heads"])
    modulation = jnp.ones((1, 6, model.config["dim"]), dtype=jnp.float32) * .1
    module._block_forward.clear_cache()
    module._block_vjp.clear_cache()
    monkeypatch.setattr(module, "attention", traced_attention)
    try:
        actual = module._block_forward(base, adapters, x, rope, modulation,
                                       model.config["num_heads"], model.scale, "cudnn")
        gradients, dx = module._block_vjp(base, adapters, x, rope, modulation,
                                         jnp.ones_like(actual), model.config["num_heads"],
                                         model.scale, "cudnn")
        assert calls == ["cudnn", "cudnn"]
        assert np.isfinite(actual).all() and np.isfinite(dx).all()
        assert all(np.isfinite(value).all() for value in gradients.values())
        assert all(value.dtype == jnp.float32 for value in gradients.values())
    finally:
        module._block_forward.clear_cache()
        module._block_vjp.clear_cache()
