"""Per-block compiler options leave process-wide XLA policy unchanged."""
import os
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import moss_art.model as module


@pytest.mark.parametrize("level", [-1, 1, 3, 5, True, False, 4.0, "4", None])
def test_invalid_level_rejected_before_model_loading(monkeypatch, level):
    load = Mock(side_effect=AssertionError("invalid level must fail before loading"))
    monkeypatch.setattr(module, "load_config", load)
    monkeypatch.setattr(module, "load_host_params", load)
    with pytest.raises(ValueError, match="dit_autotune_level must be the integer 0 or 4"):
        module.MossModel.from_pretrained("/nonexistent", dit_autotune_level=level)
    with pytest.raises(ValueError, match="dit_autotune_level must be the integer 0 or 4"):
        module.MossModel.tiny(dit_autotune_level=level)
    with pytest.raises(ValueError, match="dit_autotune_level must be the integer 0 or 4"):
        module.MossModel({}, dit_autotune_level=level)
    load.assert_not_called()


def test_cpu_level_four_fails_before_allocating_weights(monkeypatch):
    load = Mock(side_effect=AssertionError("CPU must fail before loading"))
    monkeypatch.setattr(module, "load_config", load)
    monkeypatch.setattr(module, "load_host_params", load)
    monkeypatch.setattr(module, "tiny_base", load)
    cpu = jax.devices("cpu")[0]
    with pytest.raises(ValueError, match="dit_autotune_level=4 requires a CUDA GPU"):
        module.MossModel.from_pretrained("/nonexistent", device=cpu, dit_autotune_level=4)
    with pytest.raises(ValueError, match="dit_autotune_level=4 requires a CUDA GPU"):
        module.MossModel.tiny(device=cpu, dit_autotune_level=4)
    with pytest.raises(ValueError, match="dit_autotune_level=4 requires a CUDA GPU"):
        module.MossModel({}, device=cpu, dit_autotune_level=4)
    load.assert_not_called()
    module._validate_dit_autotune_level(4, SimpleNamespace(platform="gpu"))


def test_level_zero_reuses_reference_jits():
    forward, backward = module._get_block_kernels(0)
    assert forward is module._block_forward
    assert backward is module._block_vjp
    model = module.MossModel.tiny()
    assert model.dit_autotune_level == 0
    assert model._block_forward is forward
    assert model._block_vjp is backward


@pytest.mark.parametrize("flags", [
    None, "",
    "  --xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true  ",
    "--xla_gpu_autotune_level=0",
])
def test_factory_reuses_raw_top_level_jits_without_global_changes(monkeypatch, flags):
    calls = []
    if flags is None:
        monkeypatch.delenv("XLA_FLAGS", raising=False)
    else:
        monkeypatch.setenv("XLA_FLAGS", flags)
    original_environment = dict(os.environ)
    original_config = dict(jax.config.values)

    def fake_jit(function, **kwargs):
        # Neither argument may already be a compiled JIT wrapper.
        assert not hasattr(function, "lower")
        result = Mock(name=function.__name__)
        calls.append((function, kwargs, result))
        return result

    module._get_block_kernels.cache_clear()
    monkeypatch.setattr(jax, "jit", fake_jit)
    try:
        first = module._get_block_kernels(4)
        second = module._get_block_kernels(4)
        assert first is second
        assert len(calls) == 2
        assert calls[0][:2] == (
            module._dit_block,
            {"static_argnums": (5, 6, 7), "compiler_options": {"xla_gpu_autotune_level": 4}},
        )
        assert calls[1][:2] == (
            module._block_vjp_impl,
            {"static_argnums": (6, 7, 8), "compiler_options": {"xla_gpu_autotune_level": 4}},
        )
        assert module._get_block_kernels(0) == (module._block_forward, module._block_vjp)
        assert len(calls) == 2
        assert dict(os.environ) == original_environment
        assert dict(jax.config.values) == original_config
    finally:
        module._get_block_kernels.cache_clear()


@pytest.mark.parametrize("level", [0, 4])
def test_model_calls_selected_forward_and_vjp_at_top_level(monkeypatch, level):
    calls = []
    real_validate = module._validate_dit_autotune_level
    # GPU execution is covered separately. Permit CPU fixtures to verify which
    # selected callable the model actually uses for forward and manual replay.
    monkeypatch.setattr(module, "_validate_dit_autotune_level",
                        lambda value, device=None: real_validate(value))

    def forward(*args):
        assert not isinstance(args[2], jax.core.Tracer)
        calls.append("forward")
        return module._block_forward(*args)

    def backward(*args):
        assert not isinstance(args[2], jax.core.Tracer)
        calls.append("backward")
        return module._block_vjp(*args)

    factory = Mock(return_value=(forward, backward))
    monkeypatch.setattr(module, "_get_block_kernels", factory)
    model = module.MossModel.tiny(seed=15, dit_autotune_level=level)
    factory.assert_called_once_with(level)
    assert model.dit_autotune_level == level
    assert model._block_forward is forward and model._block_vjp is backward
    state = jnp.ones((1, model.latent_channels, 2, 2), jnp.float32)
    context = jnp.ones((1, 3, model.text_dim), jnp.float32)
    prediction, pullback = model.predict_with_pullback(model.init_lora(), state, .4, context)
    gradients = pullback(jnp.ones_like(prediction))
    assert calls == ["forward"] * model.config["num_layers"] + ["backward"] * model.config["num_layers"]
    assert np.isfinite(prediction).all()
    assert all(np.isfinite(value).all() and value.dtype == jnp.float32 for value in gradients.values())
