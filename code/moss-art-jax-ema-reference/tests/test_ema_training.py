"""CPU integration contracts for the moving EMA LoRA reference.

Run with JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4. No pretrained models or GPUs.
"""
import copy
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from moss_art.checkpoint import load_checkpoint
from moss_art.config import validate
from moss_art.mesh_runtime import MeshPool

pytestmark = pytest.mark.skipif(len(jax.local_devices()) != 4,
                               reason="requires four logical CPU devices")


def configuration(tmp_path):
    from test_staged import configuration as staged_fixture
    c = staged_fixture(tmp_path)
    c["art"].update(reference_mode="ema", reference_ema_decay=.6,
                    reference_mse_beta=1.)
    c["optimizer"]["learning_rate"] = .001
    c["training"].update(max_updates=4, checkpoint_every=2,
                         keep_checkpoints=3, evaluate_every=0)
    return validate(c)


def latest(output):
    return Path(json.loads((Path(output)/"latest.json").read_text())["checkpoint"])


def read_state(path):
    return load_checkpoint(path, include_reference_state=True)


def assert_tree_close(a, b, *, exact=False):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        if exact:
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, rtol=3e-5, atol=3e-7)


def assert_training_state_close(a, b):
    p, opt, _, ref = a
    q, other_opt, _, other_ref = b
    assert_tree_close(p, q)
    assert_tree_close(opt, other_opt)
    assert ref["mode"] == other_ref["mode"] == "ema"
    assert ref["decay"] == other_ref["decay"] == .6
    assert ref["updates"] == other_ref["updates"]
    assert_tree_close(ref["lora"], other_ref["lora"])


def test_staged_two_collections_match_monolithic_and_publication_retry_is_idempotent(
        tmp_path, monkeypatch):
    from moss_art.trainer import train
    from moss_art.staged import run
    import moss_art.prefetch as prefetch
    import moss_art.trainer as trainer

    c = configuration(tmp_path)
    mono = copy.deepcopy(c)
    mono["runtime"]["output_dir"] = str(tmp_path/"mono")
    mono["logging"]["log_dir"] = str(tmp_path/"mono_tb")
    train(mono)

    stage = copy.deepcopy(c)
    stage["runtime"]["output_dir"] = str(tmp_path/"stage")
    stage["logging"]["log_dir"] = str(tmp_path/"stage_tb")
    stage["staging"].update(enabled=True, phase="collect")
    for collection in range(2):
        stage["staging"]["phase"] = "collect"
        if collection:
            stage["run"]["resume"] = str(latest(tmp_path/"stage"))
        bundle = run(stage)
        _, _, learner_meta, handoff_ref = read_state(bundle/"learner")
        assert learner_meta["update"] == 2*collection
        assert handoff_ref["updates"] == 2*collection
        stage["staging"].update(phase="train", bundle=str(bundle))
        with monkeypatch.context() as patch:
            patch.setattr(prefetch, "DoubleBuffer",
                          lambda *a, **kw: pytest.fail("training loaded Qwen"))
            patch.setattr(trainer, "read_records",
                          lambda *a, **kw: pytest.fail("training reread prompts"))
            checkpoint = run(stage)
        state = read_state(checkpoint)
        assert state[2]["update"] == state[2]["optimizer_updates"] == 2*(collection+1)
        assert state[3]["updates"] == 2*(collection+1)
        before_rows = (tmp_path/"stage/metrics.jsonl").read_bytes()
        before_commit = (checkpoint/"COMMITTED").read_bytes()
        before_payload_hash = hashlib.sha256((checkpoint/"arrays.npz").read_bytes()).hexdigest()
        run(stage)
        assert (tmp_path/"stage/metrics.jsonl").read_bytes() == before_rows
        assert (checkpoint/"COMMITTED").read_bytes() == before_commit
        assert hashlib.sha256((checkpoint/"arrays.npz").read_bytes()).hexdigest() == before_payload_hash

    actual = read_state(latest(tmp_path/"stage"))
    expected = read_state(latest(tmp_path/"mono"))
    assert actual[3]["updates"] == expected[3]["updates"] == 4
    assert_training_state_close(actual, expected)
    events = [json.loads(x) for x in (tmp_path/"stage/metrics.jsonl").read_text().splitlines()]
    assert [x["reference_ema_updates_used"] for x in events] == [0, 1, 2, 3]
    assert [x["reference_ema_updates"] for x in events] == [1, 2, 3, 4]
    assert all(x["reference_mode_ema"] == 1 for x in events)
    assert all(not any(k.startswith("base_reference_") for k in x) for x in events)


def test_ema_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    from moss_art.trainer import train

    c = configuration(tmp_path)
    full = copy.deepcopy(c)
    full["runtime"]["output_dir"] = str(tmp_path/"full")
    full["logging"]["log_dir"] = str(tmp_path/"full_tb")
    train(full)

    resumed = copy.deepcopy(c)
    resumed["runtime"]["output_dir"] = str(tmp_path/"resumed")
    resumed["logging"]["log_dir"] = str(tmp_path/"resumed_tb")
    resumed["training"]["max_updates"] = 2
    train(resumed)
    before = read_state(latest(tmp_path/"resumed"))
    assert before[3]["updates"] == 2
    resumed["training"]["max_updates"] = 4
    train(resumed, resume=str(latest(tmp_path/"resumed")))
    after = read_state(latest(tmp_path/"resumed"))
    assert after[3]["updates"] == 4
    assert_training_state_close(after, read_state(latest(tmp_path/"full")))


def test_each_gradient_window_freezes_teacher_and_b_uses_post_a_ema(tmp_path, monkeypatch):
    from moss_art.trainer import train

    c = configuration(tmp_path)
    c["training"]["max_updates"] = 2
    seen = []
    original = MeshPool._gradient

    def observe(self, window):
        teacher = {k: np.asarray(v).copy() for k, v in self.worker.reference_lora.items()}
        learner = {k: np.asarray(v).copy() for k, v in self.worker.lora.items()}
        result = original(self, window)
        for key, value in teacher.items():
            np.testing.assert_array_equal(self.worker.reference_lora[key], value)
        seen.append((window, teacher, learner))
        return result

    monkeypatch.setattr(MeshPool, "_gradient", observe)
    train(c)
    assert [v[0] for v in seen] == [0, 1]
    first_teacher, first_learner = seen[0][1:]
    second_teacher, second_learner = seen[1][1:]
    assert_tree_close(first_teacher, first_learner, exact=True)
    assert any(not np.array_equal(first_learner[k], second_learner[k])
               for k in first_learner)
    for key in first_teacher:
        np.testing.assert_allclose(
            second_teacher[key],
            np.float32(.6)*first_teacher[key] + np.float32(.4)*second_learner[key],
            rtol=2e-6, atol=1e-8)
    assert any(not np.array_equal(second_teacher[k], second_learner[k])
               for k in second_teacher)
    state = read_state(latest(c["runtime"]["output_dir"]))
    assert state[3]["updates"] == 2


def test_no_signal_collections_do_not_advance_ema(tmp_path, monkeypatch):
    import moss_art.trainer as trainer

    c = configuration(tmp_path)
    c["art"]["reference_mse_beta"] = 0.
    def constant_rewards(config, pool, outputs, **kwargs):
        return np.full(sum(len(piece["endpoints"]) for piece in outputs),
                       .5, np.float32), None
    monkeypatch.setattr(trainer, "reward_round", constant_rewards)
    trainer.train(c)
    learner, _, meta, ref = read_state(latest(c["runtime"]["output_dir"]))
    assert meta["update"] == 4
    assert meta["optimizer_updates"] == ref["updates"] == 0
    assert_tree_close(learner, ref["lora"], exact=True)
    rows = [json.loads(x) for x in (Path(c["runtime"]["output_dir"])/"metrics.jsonl").read_text().splitlines()]
    assert [x["update"] for x in rows] == [2, 4]
    assert all(not x["optimizer_step_applied"] for x in rows)
