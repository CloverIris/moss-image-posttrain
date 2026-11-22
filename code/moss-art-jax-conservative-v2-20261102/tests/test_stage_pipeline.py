"""Synthetic end-to-end checks for tmpfs handoff and separate evaluator."""
import copy
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

import jax
import numpy as np
import pytest

from moss_art.stage_cache import save_bundle, load_bundle, load_attachment
from moss_art.staged import cache_root, retire_collection, _ready
from test_staged import configuration


def test_eval_attachment_can_be_read_without_touching_replay_arrays(tmp_path):
    path = save_bundle(tmp_path/"batch", {"state": np.ones((4, 8), np.float32)}, {}, (), {},
                       attachments={"eval_request.json": {"items": [{"image": "/saved/example.png"}]}})
    array = next(path.glob("array_*.npy"))
    array.write_bytes(b"intentionally invalid replay")
    assert load_attachment(path, "eval_request.json")["items"][0]["image"] == "/saved/example.png"
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(path)
    (path/"eval_request.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_attachment(path, "eval_request.json")


def test_external_scratch_namespaces_and_consumed_only_cleanup(tmp_path):
    c = configuration(tmp_path)
    c["staging"].update(cache_dir=str(tmp_path/"scratch"), keep_consumed_cache=False)
    first = cache_root(c)
    other = copy.deepcopy(c)
    other["runtime"]["output_dir"] += "_other"
    assert first != cache_root(other)
    bundle = first/"update_00000000"
    bundle.mkdir(parents=True)
    ready = Path(c["runtime"]["output_dir"])/"staging/ready.json"
    ready.parent.mkdir(parents=True)
    ready.write_text(json.dumps({"bundle": str(bundle)}))
    (bundle/"COMMITTED").write_text("fixture")
    assert _ready(c["runtime"]["output_dir"]) == bundle
    with pytest.raises(RuntimeError, match="unverified"):
        retire_collection(c, bundle)
    (bundle/"consumed.json").write_text("{}")
    with pytest.raises(RuntimeError, match="unverified"):
        retire_collection(other, bundle)
    retire_collection(c, bundle)
    assert not bundle.exists()
    assert _ready(c["runtime"]["output_dir"]) is None  # tmpfs vanished: recollect


def test_command_evaluator_reuses_image_paths_without_decoding_or_replay(tmp_path):
    from moss_art.stage_pipeline import _score_request
    script = tmp_path/"fake_reward.py"
    script.write_text('''import argparse,json
p=argparse.ArgumentParser()
p.add_argument('--request');p.add_argument('--output')
a=p.parse_args()
r=json.load(open(a.request))
assert [x['image'] for x in r['items']]==['/saved/first.png','/saved/second.png']
json.dump({'scores':[float(x['index']) for x in r['items']]},open(a.output,'w'))
''')
    c = configuration(tmp_path)
    c["reward"].update(kind="command", command=[sys.executable, str(script)])
    rows = [dict(index=i, prompt="synthetic", image=p) for i, p in enumerate(("/saved/first.png", "/saved/second.png"))]
    scores, details = _score_request(c, {"items": rows}, tmp_path)
    np.testing.assert_array_equal(scores, [0., 1.])
    assert details is None


@pytest.mark.skipif(len(jax.devices()) != 4, reason="requires four CPU devices")
def test_three_stages_preserve_parameters_optimizer_and_no_encoder_training(tmp_path, monkeypatch):
    from moss_art.trainer import train
    from moss_art.staged import run
    from moss_art.checkpoint import load_checkpoint
    import moss_art.trainer as trainer
    import moss_art.prefetch as prefetch
    mono = configuration(tmp_path)
    mono["runtime"]["output_dir"] = str(tmp_path/"mono")
    train(mono)
    c = copy.deepcopy(mono)
    c["runtime"]["output_dir"] = str(tmp_path/"three")
    c["logging"]["log_dir"] = str(tmp_path/"three_tb")
    c["staging"].update(enabled=True, phase="infer", cache_dir=str(tmp_path/"ram"))
    reward_round = trainer.reward_round
    def only_decode(*args, **kwargs):
        assert kwargs.get("defer_reward") is True
        return reward_round(*args, **kwargs)
    monkeypatch.setattr(trainer, "reward_round", only_decode)
    bundle = run(c)
    payload, _ = load_bundle(bundle)
    assert payload["phase"] == "inferred" and "prepared" not in payload and "rewards" not in payload
    assert not (bundle/"evaluated.json").exists()
    c["staging"].update(phase="train", bundle=str(bundle))
    with pytest.raises(FileNotFoundError):
        run(c)  # Never initialize/apply the learner before an evaluation commits.
    assert not (tmp_path/"three/latest.json").exists()
    c["staging"].update(phase="eval", bundle=str(bundle))
    monkeypatch.setattr(prefetch, "DoubleBuffer", lambda *a, **kw: pytest.fail("eval/train loaded Qwen"))
    monkeypatch.setattr(trainer, "read_records", lambda *a, **kw: pytest.fail("eval/train reread prompt text"))
    run(c)
    scores = (bundle/"evaluated.json").read_bytes()
    run(c)
    assert (bundle/"evaluated.json").read_bytes() == scores
    c["staging"]["phase"] = "train"
    checkpoint = run(c)
    expected = json.loads((tmp_path/"mono/latest.json").read_text())["checkpoint"]
    p, state, saved = load_checkpoint(checkpoint)
    q, other, _ = load_checkpoint(expected)
    assert saved["update"] == 2 and saved["optimizer_updates"] == 2
    for k in p:
        np.testing.assert_allclose(p[k], q[k], rtol=2e-5, atol=2e-7)
    for x, y in zip(jax.tree.leaves(state), jax.tree.leaves(other)):
        np.testing.assert_allclose(x, y, rtol=2e-5, atol=2e-7)
    before = (tmp_path/"three/metrics.jsonl").read_bytes()
    run(c)
    assert (tmp_path/"three/metrics.jsonl").read_bytes() == before


@pytest.mark.skipif(len(jax.devices()) != 4, reason="requires four CPU devices")
def test_missing_tmpfs_batch_recollects_same_seeds_from_durable_checkpoint(tmp_path):
    from moss_art.stage_pipeline import stage_main
    c = configuration(tmp_path)
    c["training"]["max_updates"] = 4
    c["staging"].update(enabled=True, cache_dir=str(tmp_path/"ram"), keep_consumed_cache=False)
    config = tmp_path/"config.json"
    config.write_text(json.dumps(c))
    for phase in ("infer", "eval", "train"):
        stage_main(phase, ["--config", str(config)])
    latest = Path(c["runtime"]["output_dir"])/"latest.json"
    checkpoint = latest.read_bytes()
    assert json.loads(checkpoint)["update"] == 2
    first = stage_main("infer", ["--config", str(config)])
    request = load_attachment(first, "eval_request.json")
    assert first.parent == cache_root(c) and not first.is_symlink()
    shutil.rmtree(first)  # Simulate volatile data lost; durable files remain.
    assert _ready(c["runtime"]["output_dir"]) is None
    second = stage_main("infer", ["--config", str(config)])
    after = load_attachment(second, "eval_request.json")
    assert [(x["sample_id"], x["seed"]) for x in request["items"]] == [(x["sample_id"], x["seed"]) for x in after["items"]]
    assert latest.read_bytes() == checkpoint


@pytest.mark.skipif(len(jax.devices()) != 4 or not Path("/dev/shm").is_dir(), reason="requires four Linux CPU devices")
def test_shell_runs_three_serial_processes_on_tmpfs_and_resumes(tmp_path):
    c = configuration(tmp_path)
    c["training"]["max_updates"] = 4
    c["staging"].update(enabled=True, keep_consumed_cache=False)
    # Exercise a collected resolution different from the base config, too.
    c["resolution_mixing"].update(enabled=True, profiles=[
        dict(height=48, width=48, weight=1, group_size=4, rollouts_per_update=2, microbatch_size=1),
        dict(height=32, width=32, weight=1, group_size=4, rollouts_per_update=2, microbatch_size=1)])
    with tempfile.TemporaryDirectory(prefix="moss-art-stage-test-", dir="/dev/shm") as scratch:
        c["staging"]["cache_dir"] = scratch
        config = tmp_path/"config.json"
        config.write_text(json.dumps(c))
        environment = dict(os.environ, JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="4", MOSS_PYTHON=sys.executable)
        argv = ["bash", "scripts/run_stages.sh", "--config", str(config)]
        process = subprocess.run(argv, cwd=Path(__file__).resolve().parents[1], env=environment,
                                 text=True, capture_output=True, timeout=240)
        assert process.returncode == 0, process.stdout+process.stderr
        output = Path(c["runtime"]["output_dir"])
        phases = [json.loads(x)["phase"] for x in (output/"staging/process_times.jsonl").read_text().splitlines()]
        assert phases == ["infer", "eval", "train"]*2
        assert not list(Path(scratch).glob("*/update_*"))
        assert json.loads((output/"complete.json").read_text())["update"] == 4
        assert (output/"adapter.safetensors").is_file()
        before = (output/"metrics.jsonl").read_bytes()
        again = subprocess.run(argv, cwd=Path(__file__).resolve().parents[1], env=environment,
                               text=True, capture_output=True, timeout=60)
        assert again.returncode == 0, again.stdout+again.stderr
        assert (output/"metrics.jsonl").read_bytes() == before
