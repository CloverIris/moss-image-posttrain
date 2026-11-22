"""ART config must reach the actual segmented activation checkpoint path."""
import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.config import DEFAULTS, load_config, validate
from moss_art.model import MossModel
from moss_art.worker import NativeWorker


def configured(storage="cpu", interval=1):
    c = copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny", dtype="float32", activation_storage=storage,
                      checkpoint_interval=interval)
    c["runtime"].update(platform="cpu", devices=[0])
    c["lora"].update(rank=2, alpha=2.)
    c["sampling"].update(height=32, width=32, prompts_per_rollout=1, steps=2,
                         train_timesteps=[0, 1], guidance_scale=1., skip_layers=[])
    c["training"].update(max_updates=2)
    return validate(c)


def test_legacy_config_keeps_cpu_every_layer_defaults(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({}))
    c = load_config(path)
    assert c["model"]["activation_storage"] == "cpu"
    assert c["model"]["checkpoint_interval"] == 1


@pytest.mark.parametrize("storage,interval", [("cpu", 1), ("gpu", 6)])
def test_worker_passes_options_to_real_tiny_model(storage, interval):
    worker = NativeWorker(configured(storage, interval), 0)
    assert worker.model.offload_activations is (storage == "cpu")
    assert worker.model.checkpoint_interval == interval
    assert worker.model.base_precision == "bf16"
    info = worker.info()
    assert info["activation_storage"] == storage
    assert info["checkpoint_interval"] == interval
    assert info["base_precision"] == "bf16"


def test_pretrained_worker_passes_same_checkpoint_options(monkeypatch):
    captured = {}
    def load(path, **kwargs):
        captured.update(path=path, **kwargs)
        kwargs.pop("host_params", None)  # pretrained-only shared host map
        return MossModel.tiny(**kwargs)
    monkeypatch.setattr(MossModel, "from_pretrained", load)
    c = configured("gpu", 6)
    c["model"]["kind"] = "moss"
    worker = NativeWorker(c, 0)
    assert captured["offload_activations"] is False
    assert captured["checkpoint_interval"] == 6
    assert captured["base_precision"] == "bf16"
    assert captured["path"] == c["model"]["path"]
    assert worker.info()["activation_storage"] == "gpu"


@pytest.mark.parametrize("storage", ["cuda", "host", "", 1, True])
def test_invalid_activation_storage_rejected(storage):
    c = configured()
    c["model"]["activation_storage"] = storage
    with pytest.raises(ValueError, match="activation_storage"):
        validate(c)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, "6"])
def test_invalid_checkpoint_interval_rejected(interval):
    c = configured()
    c["model"]["checkpoint_interval"] = interval
    with pytest.raises(ValueError, match="checkpoint_interval"):
        validate(c)


def test_cpu_offload_and_device_sparse_checkpoints_preserve_prediction_and_gradient():
    # Both paths run on the CPU backend, so the test needs no CUDA device.
    # "gpu" selects device-resident JAX checkpoints rather than NumPy offload.
    cpu = NativeWorker(configured("cpu", 1), 0)
    resident = NativeWorker(configured("gpu", 6), 0)
    assert cpu.device.platform == resident.device.platform == "cpu"
    rng = np.random.default_rng(841)
    adapters = {k: v + jnp.asarray(rng.normal(0, .01, v.shape), jnp.float32)
                for k, v in cpu.lora.items()}
    x = jnp.asarray(rng.normal(size=(1, cpu.model.latent_channels, 2, 2)), jnp.float32)
    ctx = jnp.asarray(rng.normal(size=(1, 3, cpu.model.text_dim)), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=x.shape), jnp.float32)
    outputs, gradients = [], []
    for worker, offload in ((cpu, True), (resident, False)):
        prediction, tape = worker.model._forward(adapters, x, .45, ctx, record=True)
        assert all(isinstance(v, np.ndarray if offload else jax.Array)
                   for v in tape["checkpoints"].values())
        assert len(tape["checkpoints"]) == (2 if offload else 1)
        output, pullback = worker.model.predict_with_pullback(adapters, x, .45, ctx)
        np.testing.assert_array_equal(output, prediction)
        outputs.append(np.asarray(output))
        gradients.append(pullback(cotangent))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-6, atol=1e-6)
    for key in adapters:
        np.testing.assert_allclose(gradients[0][key], gradients[1][key],
                                   rtol=3e-5, atol=2e-6, err_msg=key)
