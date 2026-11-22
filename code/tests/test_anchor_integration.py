"""Numerical and scheduling contracts for native Mesh ART anchor integration.

Run with JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4. No GPU or external assets.
The calibrated Mesh fixture deliberately has no quantization mismatch: BF16-mode
native tiny replay provides a controlled oracle for target/reference alignment.
"""
import copy
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from moss_art.art import covariance, prompt_windows
from moss_art.batch_plan import following, select
from moss_art.config import DEFAULTS, validate
from moss_art.mesh_runtime import MeshPool
from moss_art.resolution_mixing import ResolutionPlan
from moss_art.trainer import (aggregate_gradients, make_payloads,
                              merge_rollout_rounds, paired_replay_payloads)


def configuration():
    c = copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny", dtype="float32", activation_storage="gpu",
                      checkpoint_interval=2)
    c["runtime"].update(platform="cpu", devices=[0, 1, 2, 3],
                        execution_backend="mesh", cpu_threads=2)
    c["data"].update(context_length=16, short_context_length=8)
    c["sampling"].update(height=32, width=32, steps=4, train_timesteps=[0, 3],
        group_size=4, prompts_per_rollout=1, split_group_across_workers=True,
        pair_cached_points=True, rollout_batch_size=1, guidance_scale=1.5,
        skip_layers=[1], sampler="euler", scheduler="moss", rollout_mode="bf16")
    c["training"].update(prompts_per_update=2, rollouts_per_prompt=4,
        training_points_per_trajectory=2, microbatch_size=1, max_updates=4,
        updates_per_rollout=2, evaluate_every=2, checkpoint_every=2)
    c["reward"]["kind"] = "toy_latent"
    return c


def records(count=60):
    return [dict(sample_id=str(i), prompt="anchor synthetic fixture " + str(i),
                 negative_prompt="") for i in range(count)]


def assert_tree_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_logical_collection_keeps_two_windows_and_original_time_sampling():
    c = configuration()
    c["sampling"]["steps"] = 48
    c["training"].update(prompts_per_update=8, rollouts_per_prompt=8,
                         training_points_per_trajectory=8)
    validate(c)
    assert c["training"]["rollouts_per_update"] == 8
    assert c["training"]["updates_per_rollout"] == 2
    assert c["sampling"]["train_timesteps"] == [0, 6, 13, 20, 26, 33, 40, 47]
    groups = np.repeat(np.arange(8), 8)
    windows = prompt_windows(groups, 8, seed=24)
    assert [len(window) for window in windows] == [256, 256]
    seen = set()
    for window in windows:
        for group in range(8):
            rows = np.unique(window[groups[window[:, 0]] == group, 0])
            assert len(rows) == 4
            for row in rows:
                assert set(window[window[:, 0] == row, 1]) == set(range(8))
        entries = set(map(tuple, window.tolist()))
        assert not entries & seen
        seen |= entries
    assert seen == {(row, slot) for row in range(64) for slot in range(8)}


def test_mixed_resolution_prefetch_advances_by_two_windows():
    c = configuration()
    c["resolution_mixing"] = {"enabled": True, "profiles": [
        {"height": 32, "width": 32, "weight": 2},
        {"height": 48, "width": 48, "weight": 1}]}
    validate(c)
    rows = records()
    plan = ResolutionPlan(c, rows)
    cursors = {}
    for update in range(0, 20, 2):
        chosen = select(c, rows, update, update, plan, cursors)
        future = following(c, rows, update, update, chosen, len(chosen)-1,
                           plan, cursors)
        next_cursors = dict(cursors)
        label = plan.labels[plan.index(update)]
        next_cursors[label] = next_cursors.get(label, 0) + len(chosen)
        expected = select(c, rows, update+2, update+len(chosen), plan,
                          next_cursors)[:1]
        assert future == expected
        assert len({plan.assignment(row) for row in chosen}) == 1
        cursors = next_cursors


def test_empty_window_preserves_nonzero_adamw_momentum_but_zero_gradient_kept_does_not():
    from moss_art.trainer import apply_window_update
    tx = optax.adamw(learning_rate=.02, weight_decay=.1)
    params = {"x": jnp.array([1., -2.], jnp.float32)}
    state = tx.init(params)
    delta, state = tx.update({"x": jnp.array([.3, -.2], jnp.float32)}, state, params)
    params = optax.apply_updates(params, delta)
    zero = jax.tree.map(jnp.zeros_like, params)
    result, next_state, delta, applied = apply_window_update(tx, zero, state, params, 0)
    assert not applied
    assert_tree_equal(result, params)
    assert_tree_equal(next_state, state)
    assert_tree_equal(delta, zero)
    result, next_state, delta, applied = apply_window_update(tx, zero, state, params, 1)
    assert applied
    assert not np.array_equal(result["x"], params["x"])
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(next_state),
                                                       jax.tree.leaves(state)))


def _collection(pool, c):
    outputs, caches = [], []
    for i, record in enumerate(records(2)):
        outputs.append(pool.call("rollout", make_payloads(c, [record], i)))
        caches.append(pool.call("export_replay"))
    merged, replay, groups = merge_rollout_rounds(outputs, caches, 1, True)
    return outputs, caches, merged, replay, groups


def _hash_anchors(pool):
    return [dict((key, hashlib.sha256(np.asarray(value).tobytes()).hexdigest())
                 for key, value in mapping.items()) for mapping in pool.anchor_cache]


@pytest.mark.skipif(len(jax.local_devices()) != 4, reason="requires four CPU devices")
def test_replay_identity_survives_merge_pair_shards_and_rejects_mixed_versions():
    c = configuration()
    validate(c)
    with MeshPool(c) as pool:
        outputs, caches, merged, replay, groups = _collection(pool, c)
        from moss_art.replay import replay_identity
        identity = replay_identity(caches[0])
        assert set(identity) == {"parameter_version", "rollout_precision", "precision_contract"}
        assert replay_identity(replay) == identity
        endpoints = np.concatenate([piece["endpoints"] for piece in merged])
        stats = {k: np.asarray(v) for k, v in covariance(endpoints,
            np.linspace(.1, .9, len(endpoints), dtype=np.float32), groups,
            group_size=4).items()}
        windows = prompt_windows(groups, 2, seed=7)
        paired_replay_payloads(pool, groups, stats, windows, caches=replay)
        assert replay_identity(pool.shards) == identity
        bad = copy.deepcopy(caches)
        bad[1][0]["parameter_version"] += 1
        with pytest.raises((ValueError, RuntimeError), match="(?i)(identity|version|replay)"):
            merge_rollout_rounds(outputs, bad, 1, True)


def _run_no_mismatch(mode, sampler, scheduler):
    c = configuration()
    c["sampling"].update(sampler=sampler, scheduler=scheduler)
    validate(c)
    with MeshPool(c) as pool:
        _, _, outputs, replay, groups = _collection(pool, c)
        endpoints = np.concatenate([piece["endpoints"] for piece in outputs])
        # One full group is inactive. It MUST remain in each window denominator.
        rewards = np.where(groups == 0,
            np.linspace(.1, .9, len(groups), dtype=np.float32), np.float32(.5))
        stats = {key: np.asarray(value) for key, value in covariance(
            endpoints, rewards, groups, group_size=4).items()}
        windows = prompt_windows(groups, 2, seed=7)
        payloads = paired_replay_payloads(pool, groups, stats, windows, caches=replay)
        # Calibration on no-mismatch replay is an intentional CPU oracle.
        pool.config["sampling"]["rollout_mode"] = mode
        pool.call("prepare_update", payloads)
        if mode == "mxfp8_calibrated":
            anchor_hash = _hash_anchors(pool)
            assert any(anchor_hash)
            anchor_version = pool.anchor_version
            assert anchor_version == pool.parameter_version
        tx = optax.sgd(.002)
        params = pool.worker.lora
        state = tx.init(params)
        results = []
        from moss_art.trainer import apply_window_update
        for window in range(2):
            reply = pool.call("gradient", [{"window": window} for _ in pool.workers])
            gradients, metrics = aggregate_gradients(reply)
            assert metrics["items"] == 8
            assert 0 < metrics["active_fraction"] <= .5
            assert float(optax.global_norm(gradients)) > 0
            results.append(({k: np.asarray(v).copy() for k, v in gradients.items()}, metrics))
            kept = sum(item["active"] for item in reply)
            params, state, delta, applied = apply_window_update(tx, gradients, state, params, kept)
            assert applied
            pool.call("set_lora", [{"lora": params} for _ in pool.workers])
            if mode == "mxfp8_calibrated":
                assert _hash_anchors(pool) == anchor_hash
                assert pool.anchor_version == anchor_version
                assert pool.parameter_version == anchor_version + window + 1
        return {k: np.asarray(v).copy() for k, v in params.items()}, results


@pytest.mark.skipif(len(jax.local_devices()) != 4, reason="requires four CPU devices")
@pytest.mark.parametrize("sampler,scheduler", [("euler", "moss"), ("res_multistep", "beta")])
def test_no_mismatch_raw_and_calibrated_two_windows_agree(sampler, scheduler):
    raw, raw_steps = _run_no_mismatch("bf16", sampler, scheduler)
    calibrated, calibrated_steps = _run_no_mismatch("mxfp8_calibrated", sampler, scheduler)
    for key in raw:
        np.testing.assert_allclose(raw[key], calibrated[key], rtol=3e-4, atol=3e-6)
    for (raw_grad, raw_metrics), (cal_grad, cal_metrics) in zip(raw_steps, calibrated_steps):
        assert raw_metrics["items"] == cal_metrics["items"]
        assert raw_metrics["active_fraction"] == cal_metrics["active_fraction"]
        np.testing.assert_allclose(raw_metrics["loss"], cal_metrics["loss"], rtol=3e-5, atol=3e-6)
        for key in raw_grad:
            np.testing.assert_allclose(raw_grad[key], cal_grad[key], rtol=3e-4, atol=3e-6)


@pytest.mark.skipif(len(jax.local_devices()) != 4, reason="requires four CPU devices")
def test_all_no_signal_collections_are_bounded_and_checkpoint_attempts_separately(tmp_path, monkeypatch):
    import moss_art.trainer as trainer
    from moss_art.checkpoint import load_checkpoint
    c = configuration()
    c["training"]["evaluate_every"] = 0
    c["runtime"]["output_dir"] = str(tmp_path / "no_signal")
    data = tmp_path / "synthetic.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in records(6)))
    c["data"]["prompts"] = str(data)
    validate(c)
    def constant_rewards(config, pool, outputs, **kwargs):
        count = sum(len(piece["endpoints"]) for piece in outputs)
        return np.full(count, .5, np.float32), None
    monkeypatch.setattr(trainer, "reward_round", constant_rewards)
    output = trainer.train(c)
    events = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [item["update"] for item in events] == [2, 4]
    assert all(item["event"] == "no_signal" and item["optimizer_updates"] == 0
               and not item["optimizer_step_applied"] for item in events)
    latest = json.loads((output / "latest.json").read_text())
    _, _, metadata = load_checkpoint(latest["checkpoint"])
    assert metadata["update"] == 4
    assert metadata["optimizer_updates"] == 0
    assert metadata["prompt_cursor"] == 4
    complete = json.loads((output / "complete.json").read_text())
    assert complete["update"] == 4 and complete["optimizer_updates"] == 0
    trainer.train(c, resume=latest["checkpoint"])
