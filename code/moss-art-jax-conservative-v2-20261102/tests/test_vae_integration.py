import copy
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.config import load_config, validate, fingerprint


def test_vae_precision_configuration_is_explicit_and_identity_bound():
    base = load_config("configs/smoke_cpu.yaml")
    assert base["model"]["vae_precision"] == "fp32"
    assert base["model"]["vae_low_precision_stages"] == []
    tuned = copy.deepcopy(base)
    tuned["model"].update(vae_precision="float16", vae_low_precision_stages=["up1", "up2", "up3"])
    validate(tuned)
    assert fingerprint(base) != fingerprint(tuned)
    bf16 = copy.deepcopy(tuned)
    bf16["model"]["vae_precision"] = "bfloat16"
    assert fingerprint(tuned) != fingerprint(bf16)
    for precision, stages in [("fp32", ["up1"]), ("float16", []), ("auto", ["up1"]),
                              ("float16", ["up4"]), ("float16", ["up1", "up1"]),
                              ("float16", [True]), ("float16", [["up1"]]),
                              ("float16", "up1")]:
        bad = copy.deepcopy(base)
        bad["model"].update(vae_precision=precision, vae_low_precision_stages=stages)
        with pytest.raises(ValueError):
            validate(bad)


def test_worker_and_cpu_reward_decode_forward_the_same_precision_policy(monkeypatch):
    import moss_art.evaluate as evaluate
    import moss_art.rewards as rewards
    from moss_art.worker import NativeWorker
    from moss_art.trainer import reward_round

    calls = []
    images = np.zeros((2, 2, 2, 3), np.uint8)
    def decode(model_dir, endpoints, **kwargs):
        calls.append(kwargs)
        return images
    monkeypatch.setattr(evaluate, "decode_latents", decode)
    monkeypatch.setattr(rewards, "score_images", lambda *_: np.array([.1, .2], np.float32))
    c = load_config("configs/smoke_cpu.yaml")
    c["model"].update(vae_precision="float16", vae_low_precision_stages=["up1", "up2", "up3"])
    worker = object.__new__(NativeWorker)
    worker.c, worker.device = c, object()
    endpoints = np.zeros((2, 128, 1, 1), np.float32)
    assert worker.decode(endpoints)["images"] is images
    assert calls[-1] == dict(device=worker.device, precision="float16",
                            low_precision_stages=["up1", "up2", "up3"],
                            conv_backend="xla", conv_workspace_mib=256)
    c["reward"].update(kind="toy", cpu_decode=True, decode=True)
    scores, decoded = reward_round(c, None, [dict(endpoints=endpoints, prompts=["a", "b"])])
    assert decoded is images and scores.shape == (2,)
    assert calls[-1] == dict(device="cpu:0", precision="float16",
                            low_precision_stages=["up1", "up2", "up3"],
                            conv_backend="xla", conv_workspace_mib=256)


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_raw_nonfinite_rejected_before_rgb_clipping(monkeypatch, value):
    import moss_art.vae_precision as module
    import vendor.moss_jax.weights as weights
    from moss_art.evaluate import decode_latents
    monkeypatch.setattr(weights, "load_config", lambda *_: {})
    monkeypatch.setattr(weights, "load_host_params", lambda *_: {})
    def fake_decode(params, latents, config, **kwargs):
        assert kwargs["clamp"] is False
        return jnp.full((1, 3, 2, 2), value, dtype=jnp.float32)
    monkeypatch.setattr(module, "decode_latents", fake_decode)
    with pytest.raises(ValueError, match="nonfinite raw pixels before clipping"):
        decode_latents("/unused", np.zeros((1, 128, 1, 1), np.float32), device="cpu:0")


def test_finite_raw_rgb_clips_after_validation(monkeypatch):
    import moss_art.vae_precision as module
    import vendor.moss_jax.weights as weights
    from moss_art.evaluate import decode_latents
    monkeypatch.setattr(weights, "load_config", lambda *_: {})
    monkeypatch.setattr(weights, "load_host_params", lambda *_: {})
    def fake_decode(params, latents, config, **kwargs):
        assert kwargs["clamp"] is False
        return jnp.array([-2., 0., 2.], dtype=jnp.float32).reshape(1, 3, 1, 1)
    monkeypatch.setattr(module, "decode_latents", fake_decode)
    pixels = decode_latents("/unused", np.zeros((1, 128, 1, 1), np.float32), device="cpu:0")
    np.testing.assert_array_equal(pixels, np.array([[[[0, 128, 255]]]], np.uint8))


@pytest.mark.parametrize("precision,stages", [("float16", ""), ("fp32", "up1")])
def test_standalone_cli_rejects_ambiguous_precision_before_loading(precision, stages, tmp_path):
    from moss_art.evaluate import main
    with pytest.raises(SystemExit) as error:
        import json
        path=tmp_path/"decode.json"
        path.write_text(json.dumps(dict(model_dir="/unused",latents="/unused.npy",
            output_dir="/unused",vae_precision=precision,vae_stages=[stages] if stages else [])))
        main(["--config",str(path)])
    assert error.value.code == 2
