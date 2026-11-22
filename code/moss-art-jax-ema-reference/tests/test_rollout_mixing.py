"""Prompt sampler quotas, immutable groups, replay metadata and ART integration."""
import copy
import json
from collections import Counter

import jax
import numpy as np
import pytest

from moss_art.config import DEFAULTS, fingerprint, validate
from moss_art.batch_plan import select, unique_pool
from moss_art.rollout_mixing import (MODES, mode_for_record, options_for_record,
    prompt_key, assignment_manifest, collection_stats, select_balanced, ensure_pool_capacity)


def config():
    c = copy.deepcopy(DEFAULTS)
    c["model"].update(kind="tiny", dtype="float32", activation_storage="gpu", checkpoint_interval=2)
    c["sampling"].update(height=32, width=32, steps=16, sampler="res_multistep", scheduler="beta",
        guidance_scale=1.5, skip_layers=[1], rollout_batch_size=1)
    c["sampling"]["rollout_mixing"]["enabled"] = True
    c["data"].update(context_length=8, short_context_length=8)
    c["runtime"].update(platform="cpu", devices=[0], execution_backend="workers", cpu_threads=2)
    c["training"].update(prompts_per_update=8, rollouts_per_prompt=4,
        training_points_per_trajectory=4, max_updates=2, updates_per_rollout=2,
        evaluate_every=0, checkpoint_every=2, microbatch_size=1)
    c["reward"]["kind"] = "toy_latent"
    validate(c)
    return c


def records(n=256):
    return [dict(sample_id=f"row-{i}", prompt=f"synthetic fixture {i}", negative_prompt="") for i in range(n)]


def test_prompt_assignment_survives_duplicate_ids_order_resolution_and_resume():
    c = config(); rows = records(); manifest = assignment_manifest(c, rows)
    assert set(mode_for_record(c, row) for row in rows) == set(MODES)
    for row in reversed(rows):
        mode = mode_for_record(c, row)
        duplicate = dict(row, sample_id="different-id")
        restored = json.loads(json.dumps(c))
        restored["sampling"].update(height=768, width=1024)
        assert mode == mode_for_record(c, duplicate) == mode_for_record(restored, row)
    assert assignment_manifest(c, rows) == manifest
    assert prompt_key(rows[0]) != prompt_key(dict(rows[0], negative_prompt="different condition"))


@pytest.mark.parametrize("full,spectrum,count", [(1, 1, 8), (1, 3, 8), (3, 1, 8), (2, 1, 24)])
def test_logical_collection_has_exact_prompt_quotas(full, spectrum, count):
    c = config(); c["training"]["prompts_per_update"] = count
    c["sampling"]["rollout_mixing"].update(full_dit_weight=full, spectrum_weight=spectrum)
    validate(c)
    rows = records(512)
    for cursor in (0, count, 3*count, 10*count):
        chosen = select(c, rows, 2, cursor)
        assert len({prompt_key(row) for row in chosen}) == count
        counts = Counter(mode_for_record(c, row) for row in chosen)
        assert counts["full_dit"] == count*full//(full+spectrum)
        assert counts["spectrum_default"] == count*spectrum//(full+spectrum)
        assert chosen == select(json.loads(json.dumps(c)), rows, 2, cursor)


def test_fractional_quota_rounding_and_single_prompt_bucket():
    c = config(); rows = records()
    full_counts = []
    for offset in range(100):
        chosen = select_balanced(c, rows, 3, 42, offset)
        full_counts.append(sum(mode_for_record(c, row) == "full_dit" for row in chosen))
    assert set(full_counts) == {1, 2}
    assert abs(np.mean(full_counts) - 1.5) < .15
    one = select_balanced(c, rows[:1], 1, 42, 0)
    assert one == rows[:1]


def test_resolution_and_sampler_buckets_compose_and_resume():
    from moss_art.resolution_mixing import ResolutionPlan
    c = config(); c["resolution_mixing"] = dict(enabled=True, profiles=[
        dict(height=32, width=32, weight=2), dict(height=48, width=48, weight=1)])
    validate(c); rows = records(512); plan = ResolutionPlan(c, rows)
    cursors = {}
    for update in range(0, 12, 2):
        chosen = select(c, rows, update, 0, plan, cursors)
        assert Counter(mode_for_record(c, row) for row in chosen) == dict(full_dit=4, spectrum_default=4)
        assert all(plan.assignment(row) == plan.index(update) for row in chosen)
        restored = ResolutionPlan(c, rows)
        assert chosen == select(c, rows, update, 0, restored, json.loads(json.dumps(cursors)))
        label = plan.labels[plan.index(update)]
        cursors[label] = cursors.get(label, 0) + 8


@pytest.mark.parametrize("full,spectrum,expected", [(1, 0, "full_dit"), (0, 1, "spectrum_default")])
def test_zero_weight_means_one_sampler_only(full, spectrum, expected):
    c = config(); c["sampling"]["rollout_mixing"].update(full_dit_weight=full, spectrum_weight=spectrum)
    validate(c)
    assert {mode_for_record(c, row) for row in records()} == {expected}


def test_invalid_weights_blend_and_disabled_fingerprint():
    c = config(); original = copy.deepcopy(c)
    for patch in (dict(full_dit_weight=-1), dict(full_dit_weight=0, spectrum_weight=0), dict(spectrum_weight=float("inf"))):
        bad = copy.deepcopy(original); bad["sampling"]["rollout_mixing"].update(patch)
        with pytest.raises(ValueError): validate(bad)
    c["sampling"]["spectrum"]["blend_weight"] = 1.0
    with pytest.raises(ValueError, match="blend"): validate(c)
    c = copy.deepcopy(original); c["sampling"]["rollout_mixing"]["enabled"] = False
    old = copy.deepcopy(c); old["sampling"].pop("rollout_mixing")
    assert fingerprint(c) == fingerprint(old)
    assert fingerprint(original) != fingerprint(old)
    short_pool = [next(r for r in records() if mode_for_record(original, r) == "full_dit")]
    short_pool += [r for r in records() if mode_for_record(original, r) == "spectrum_default"][:12]
    with pytest.raises(ValueError, match="full_dit"): ensure_pool_capacity(original, short_pool, 8)


def test_worker_group_is_one_method_and_matches_uniform_reference():
    from moss_art.worker import NativeWorker
    c = config(); worker = NativeWorker(c, 0)
    seeds = [10, 20, 30, 40]
    for mode in MODES:
        row = next(r for r in records() if mode_for_record(c, r) == mode)
        actual = worker.rollout([row], seeds)
        states, old = worker.cache["states"].copy(), worker.cache["old"].copy()
        assert actual["sampling_methods"] == [mode]*4
        assert len(set(actual["sampling_prompt_keys"])) == 1
        if mode == "full_dit": assert actual["dit_forecast_steps"] == [0]*4
        else:
            assert min(actual["dit_forecast_steps"]) > 0
            assert set(c["sampling"]["train_timesteps"]) <= set(actual["spectrum"]["actual_step_indices"])
        reference_config = copy.deepcopy(c)
        reference_config["sampling"]["rollout_mixing"]["enabled"] = False
        reference_config["sampling"]["spectrum"]["enabled"] = mode != "full_dit"
        reference = NativeWorker(reference_config, 0)
        expected = reference.rollout([row], seeds)
        np.testing.assert_array_equal(actual["endpoints"], expected["endpoints"])
        np.testing.assert_array_equal(states, reference.cache["states"])
        np.testing.assert_array_equal(old, reference.cache["old"])
        exported = worker.export_replay()
        assert exported["sampling_methods"] == [mode]*4


def test_reward_stats_do_not_mix_prompt_modes():
    outputs = [dict(sampling_methods=["full_dit"]*4+["spectrum_default"]*4,
                    sampling_prompt_keys=["a"]*4+["b"]*4)]
    result = collection_stats(outputs, np.array([1, 2, 3, 4, 10, 20, 30, 40]))
    assert result["full_dit"]["reward_mean"] == 2.5
    assert result["spectrum_default"]["reward_mean"] == 25
    assert all(v["images"] == 4 and v["prompts"] == 1 for v in result.values())


def test_saved_images_keep_the_matching_sampler_metadata(tmp_path):
    from moss_art.trainer import reward_round
    class DecodePool:
        def pause(self): pass
        def resume(self): pass
        def decode(self, endpoints):
            return [dict(images=np.zeros((len(part), 8, 8, 3), np.uint8)) for part in endpoints]
    c = config()
    row = dict(endpoints=np.zeros((8, 8, 2, 2), np.float32),
        prompts=["synthetic"]*8, sample_ids=["a"]*4+["b"]*4, seeds=list(range(8)),
        sampling_methods=["full_dit"]*4+["spectrum_default"]*4,
        dit_full_steps=[48]*4+[20]*4, dit_forecast_steps=[0]*4+[28]*4)
    rewards, images = reward_round(c, DecodePool(), [row], image_dir=tmp_path, force_decode=True)
    metadata = json.loads((tmp_path/"samples.json").read_text())
    assert len(rewards) == len(images) == len(metadata) == 8
    for i, item in enumerate(metadata):
        assert item["sampling_method"] == row["sampling_methods"][i]
        assert item["dit_full_steps"] == row["dit_full_steps"][i]
        assert item["dit_forecast_steps"] == row["dit_forecast_steps"][i]
        assert (tmp_path/f"{i:04d}.png").exists()


@pytest.mark.skipif(len(jax.local_devices()) != 4, reason="needs four logical devices")
@pytest.mark.parametrize("mixed_resolution", [False, True])
def test_mesh_art_two_updates_resume_and_sampler_reward_logs(tmp_path, mixed_resolution):
    from moss_art.trainer import train
    from moss_art.checkpoint import load_checkpoint
    c = config(); c["runtime"].update(devices=[0, 1, 2, 3], execution_backend="mesh")
    if mixed_resolution:
        c["resolution_mixing"] = dict(enabled=True, profiles=[
            dict(height=32, width=32, weight=2), dict(height=48, width=48, weight=1)])
    source = tmp_path/"data.jsonl"
    source.write_text("".join(json.dumps(row)+"\n" for row in records()))
    c["data"]["prompts"] = str(source)
    c["runtime"]["output_dir"] = str(tmp_path/"train")
    c["logging"].update(tensorboard=True, log_dir=str(tmp_path/"tb"))
    validate(c)
    output = train(c)
    assignments = json.loads((output/"sampling_assignments.json").read_text())
    rows = [json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(np.isfinite(row["grad_norm"]) for row in rows)
    assert all(row["items"] == 64 for row in rows)
    checkpoint = json.loads((output/"latest.json").read_text())["checkpoint"]
    _, _, metadata = load_checkpoint(checkpoint)
    assert metadata["update"] == 2
    c["training"]["max_updates"] = 4
    train(c, resume=checkpoint)
    assert json.loads((output/"sampling_assignments.json").read_text()) == assignments
    counts = []
    for path in sorted((output/"sampling").glob("rollout_*.json")):
        collection = json.loads(path.read_text()); methods = collection["methods"]
        assert all(methods[mode]["prompts"] == 4 and methods[mode]["images"] == 16 for mode in MODES)
        ids = {}
        for row in collection["rows"]:
            ids.setdefault(row["sample_id"], set()).add(row["sampling_method"])
        assert all(len(group) == 1 for group in ids.values())
        counts.append(len(collection["rows"]))
    assert counts == [32, 32]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_files = list((tmp_path/"tb").rglob("events.out.tfevents.*"))
    assert event_files
    tags = set()
    for file in event_files:
        accumulator = EventAccumulator(str(file)); accumulator.Reload()
        tags.update(accumulator.Tags()["scalars"])
    assert "rollout/full_dit/reward_mean" in tags
    assert "rollout/spectrum_default/reward_mean" in tags
