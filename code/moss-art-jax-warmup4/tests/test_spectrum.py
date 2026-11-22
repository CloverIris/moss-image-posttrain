"""Upstream math, exact head re-entry, rollout replay and sharding contracts."""
import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moss_art.config import DEFAULTS as CONFIG_DEFAULTS, fingerprint, validate
from moss_art.model import MossModel
from moss_art.spectrum import (DEFAULTS, UPSTREAM_REVISION, FeatureHistory,
    SpectrumRollout, actual_step_plan, forecast_weights, validate_options)

REFERENCE = Path(__file__).resolve().parents[1] / "inputs/spectrum_reference_20261025"


def upstream_schedule(total, options):
    """Execute the actual upstream decision methods, without importing ComfyUI."""
    if not REFERENCE.exists():
        pytest.skip("Pinned upstream source not installed")
    tree = ast.parse((REFERENCE / "spectrum.py").read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpectrumState")
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "should_cache")
    advance = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_advance_spectrum_state")
    namespace = {"math": math}
    exec("from __future__ import annotations\n" + ast.unparse(method) + "\n" + ast.unparse(advance), namespace)
    observations = [0]
    state = SimpleNamespace(active=True, step_idx=-1, last_sigma=None, one_sampler_only=False,
        steps_seen=0, mode="actual", fwd_count=0, curr_ws=options["window_size"],
        flex_window=options["flex_window"], consec_cached=0, warmup_steps=options["warmup_steps"],
        tail_actual_steps=options["tail_actual_steps"], num_steps=total, fsg_steps=frozenset(),
        schedule="window", delta=None, forecasters_ready=lambda keys: observations[0] >= options["degree"]+2)
    plan = []
    for i in range(total):
        namespace["_advance_spectrum_state"](state, 1.0 - i / total)
        actual = not namespace["should_cache"](state, ["conditional"])
        state.mode = "actual" if actual else "cached"
        observations[0] += int(actual)
        plan.append(actual)
    return tuple(plan)


@pytest.mark.parametrize("steps", [4, 16, 48, 50])
@pytest.mark.parametrize("flex", [0.0, 0.25, 0.5, 1.0])
def test_schedule_matches_upstream(steps, flex):
    options = dict(DEFAULTS, enabled=True, flex_window=flex)
    assert actual_step_plan(steps, options) == upstream_schedule(steps, options)
    if steps == 50 and flex == .5:
        assert sum(actual_step_plan(steps, options)) == 19


def test_weights_match_pinned_upstream_torch_oracle():
    oracle = json.loads((Path(__file__).parent / "spectrum_upstream_oracle.json").read_text())
    assert oracle["revision"] == UPSTREAM_REVISION
    for row in oracle["cases"]:
        result = forecast_weights(row["times"], row["target"], total_steps=row["total_steps"],
            degree=row["degree"], ridge_lambda=row["ridge_lambda"], blend_weight=row["blend_weight"])
        np.testing.assert_allclose(result, row["weights"], atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_fixed_ring_history_matches_explicit_sum_after_wrap(dtype):
    options = dict(DEFAULTS, enabled=True, history_size=5)
    history = FeatureHistory(5, 48, options)
    rng = np.random.default_rng(7)
    samples, times = [], []
    for i in [0, 1, 2, 3, 4, 7, 10, 13]:
        sample = jnp.asarray(rng.normal(size=(2, 4, 8)), dtype)
        history.observe(i, sample)
        samples.append(np.asarray(sample, np.float32)); times.append(i)
    assert history.buffer.shape == (5, 2, 4, 8)
    c = forecast_weights(times[-5:], 15, total_steps=48, degree=3, ridge_lambda=.1, blend_weight=.3)
    expected = np.zeros((2, 4, 8), np.float32)
    for coeff, sample in zip(c, samples[-5:]):
        expected = expected + np.float32(coeff) * sample
    expected = np.asarray(jnp.asarray(expected, dtype), np.float32)
    np.testing.assert_allclose(np.asarray(history.predict(15), np.float32), expected,
                               atol=3e-6 if dtype == jnp.float32 else 0.016, rtol=3e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("cfg_scale", [1.0, 3.0])
def test_head_reentry_is_bitwise_exact_with_padding_and_nonzero_lora(dtype, cfg_scale):
    model = MossModel.tiny(seed=17, dtype=dtype, offload_activations=False)
    rng = np.random.default_rng(42)
    adapters = {k: v + jnp.asarray(rng.normal(0, .01, v.shape), jnp.float32)
                for k, v in model.init_lora().items()}
    state = jnp.asarray(rng.normal(size=(2, 8, 2, 2)), jnp.float32)
    positive = jnp.asarray(rng.normal(size=(2, 5, 24)), dtype)
    kwargs = dict(cfg_scale=cfg_scale, negative_context=jnp.zeros((2, 4, 24), dtype),
        negative_skip_layers=(1,), context_lengths=np.array([3, 5], np.int32),
        negative_context_lengths=np.array([2, 3], np.int32))
    baseline = model.predict(adapters, state, .61, positive, **kwargs)
    actual, features = model.predict_with_features(adapters, state, .61, positive, **kwargs)
    head = model.predict_from_features(features, .61, state.shape, cfg_scale=cfg_scale)
    np.testing.assert_array_equal(actual, baseline)
    np.testing.assert_array_equal(head, baseline)
    model.release_base()
    with pytest.raises(RuntimeError, match="released"):
        model.predict_from_features(features, .61, state.shape, cfg_scale=cfg_scale)


def configuration():
    c = copy.deepcopy(CONFIG_DEFAULTS)
    c["model"].update(kind="tiny", dtype="float32", activation_storage="gpu")
    c["runtime"].update(platform="cpu", devices=[0])
    c["sampling"].update(height=32, width=32, steps=16, group_size=2,
        prompts_per_rollout=1, sampler="res_multistep", scheduler="beta",
        guidance_scale=1.5, skip_layers=[1], train_timesteps=[0, 6, 10, 13])
    c["data"].update(context_length=8, short_context_length=8)
    return c


def test_disabled_rollout_is_identical_and_training_points_are_actual():
    from moss_art.worker import NativeWorker
    c = configuration()
    worker = NativeWorker(c, 0)
    record = dict(sample_id="spectrum-fixture", prompt="synthetic fixture", negative_prompt="")
    baseline = worker.rollout([record], [3, 7])
    baseline_cache = {key: worker.cache[key].copy() for key in ("states", "old")}
    # Merely changing disabled forecast parameters cannot change sampling.
    c["sampling"]["spectrum"]["flex_window"] = 1.
    repeated = worker.rollout([record], [3, 7])
    np.testing.assert_array_equal(baseline["endpoints"], repeated["endpoints"])
    for key in baseline_cache:
        np.testing.assert_array_equal(baseline_cache[key], worker.cache[key])
    c["sampling"]["spectrum"]["enabled"] = True
    accelerated = worker.rollout([record], [3, 7])
    summary = accelerated["spectrum"]
    assert summary["forecast_steps"] > 0
    assert set(c["sampling"]["train_timesteps"]) <= set(summary["actual_step_indices"])
    positive, negative = worker._conditioning(record)
    from moss_art.conditioning import pad_context
    pos, pl = pad_context(positive, 8); neg, nl = pad_context(negative, 8)
    for row in range(2):
        for point, step in enumerate(worker.cache["selected"]):
            expected = worker.model.predict(worker.lora, worker.cache["states"][row:row+1, point],
                worker.cache["sigmas"][step], pos, **worker._kwargs(neg, (pl, nl)))
            np.testing.assert_array_equal(expected[0], worker.cache["old"][row, point])
    # A new trajectory batch starts warmup and cannot reuse another seed's history.
    again = worker.rollout([record], [3, 7])
    np.testing.assert_array_equal(again["endpoints"], accelerated["endpoints"])


def test_schema_validation_and_disabled_fingerprint_compatibility():
    c = configuration(); validate(c)
    without = copy.deepcopy(c); without["sampling"].pop("spectrum")
    assert fingerprint(c) == fingerprint(without)
    c["sampling"]["spectrum"]["enabled"] = True
    assert fingerprint(c) != fingerprint(without)
    for patch in ({"history_size": 4}, {"degree": 9}, {"blend_weight": 1.1},
                  {"ridge_lambda": 0.0}, {"window_size": .9}, {"flex_window": -1}):
        with pytest.raises(ValueError, match="spectrum"):
            validate_options(dict(DEFAULTS, **patch))


@pytest.mark.skipif(len(jax.local_devices()) < 2, reason="needs logical multi-device JAX")
def test_forecast_ring_stays_sharded_over_batch():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    size = len(jax.local_devices())
    mesh = Mesh(np.asarray(jax.local_devices()), ("dp",))
    sharding = NamedSharding(mesh, P("dp", None, None))
    h = FeatureHistory(5, 48, dict(DEFAULTS, enabled=True, history_size=5))
    for i in range(5):
        h.observe(i, jax.device_put(np.full((size, 4, 8), i, np.float32), sharding))
    result = h.predict(6)
    assert not result.is_fully_replicated and len(result.addressable_shards) == size
    assert h.buffer.sharding.spec[1] == "dp"
    for shard in h.buffer.addressable_shards:
        assert shard.data.shape[1] == 1


@pytest.mark.skipif(len(jax.local_devices()) != 4, reason="needs four logical JAX devices")
def test_mesh_rollout_matches_serial_with_forecast_and_actual_replay():
    from moss_art.worker import NativeWorker
    c = configuration()
    c["sampling"].update(group_size=4, rollout_batch_size=4)
    c["sampling"]["spectrum"]["enabled"] = True
    serial = NativeWorker(c, 0)
    mesh_config = copy.deepcopy(c)
    mesh_config["_mesh_dp_size"] = 4
    mesh_config["sampling"]["rollout_batch_size"] = 1
    mesh = NativeWorker(mesh_config, 0)
    record = dict(sample_id="mesh-spectrum", prompt="synthetic fixture", negative_prompt="")
    expected = serial.rollout([record], [10, 20, 30, 40])
    actual = mesh.rollout([record], [10, 20, 30, 40])
    np.testing.assert_allclose(actual["endpoints"], expected["endpoints"], rtol=6e-5, atol=1e-5)
    for key in ("states", "old"):
        np.testing.assert_allclose(mesh.cache[key], serial.cache[key], rtol=6e-5, atol=1e-5)
    assert actual["spectrum"]["actual_step_indices"] == expected["spectrum"]["actual_step_indices"]
