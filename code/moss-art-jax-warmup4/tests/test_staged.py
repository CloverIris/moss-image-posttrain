"""Two-process phase parity, cache integrity, and restart-safe publication."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from moss_art.config import DEFAULTS, validate, fingerprint
from moss_art.stage_cache import save_bundle, load_bundle


def test_cache_deduplicates_repeated_contexts_and_distinct_equal_copies(tmp_path):
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    payload = dict(contexts=[array, array, array.copy()], shape=(1, 2), scalar=np.array(.5, np.float32),
                   bf16=np.array([1., 2.], dtype=jnp.bfloat16))
    path = save_bundle(tmp_path/"bundle", payload, {"x": np.ones(2, np.float32)}, (), {"update": 0})
    manifest = json.loads((path/"manifest.json").read_text())
    assert len(manifest["arrays"]) == 3
    decoded, metadata = load_bundle(path)
    assert metadata == {"update": 0}
    assert decoded["contexts"][0] is decoded["contexts"][1] is decoded["contexts"][2]
    np.testing.assert_array_equal(decoded["contexts"][0], array)
    assert decoded["scalar"].shape == () and decoded["shape"] == (1, 2)
    assert str(decoded["bf16"].dtype) == "bfloat16"
    with pytest.raises(FileExistsError):
        save_bundle(path, payload, {}, (), {})
    file = next(path.glob("array_*.npy"))
    with file.open("r+b") as stream:
        stream.seek(-1, 2); stream.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(path)


def configuration(tmp_path):
    from test_anchor_integration import configuration as fixture_config
    c = fixture_config()
    c["training"].update(max_updates=2, evaluate_every=0)
    c["art"]["reference_mse_beta"] = .01
    c["runtime"]["output_dir"] = str(tmp_path/"output")
    c["logging"].update(tensorboard=True, log_dir=str(tmp_path/"tb"))
    data = tmp_path/"synthetic.jsonl"
    data.write_text("".join(json.dumps(dict(id=str(i), prompt="synthetic staged fixture "+str(i)))+"\n" for i in range(6)))
    c["data"]["prompts"] = str(data)
    return validate(c)


def test_staging_execution_fields_do_not_change_algorithm_fingerprint():
    c = copy.deepcopy(DEFAULTS)
    original = fingerprint(c)
    c["staging"].update(enabled=True, phase="train", bundle="/cache/example")
    assert fingerprint(c) == original


@pytest.mark.skipif(len(jax.devices()) != 4, reason="requires four CPU devices")
def test_staged_cached_training_matches_monolithic_and_does_not_load_encoders(tmp_path, monkeypatch):
    from moss_art.trainer import train
    from moss_art.staged import run
    from moss_art.checkpoint import load_checkpoint
    import moss_art.prefetch as prefetch
    import moss_art.trainer as trainer
    mono = configuration(tmp_path)
    mono["runtime"]["output_dir"] = str(tmp_path/"monolithic")
    train(mono)
    stage = copy.deepcopy(mono)
    stage["runtime"]["output_dir"] = str(tmp_path/"staged")
    stage["logging"]["log_dir"] = str(tmp_path/"staged_tb")
    stage["staging"].update(enabled=True, phase="collect")
    cache = run(stage)
    assert (cache/"COMMITTED").is_file()
    assert not (tmp_path/"staged/latest.json").exists()
    assert not (tmp_path/"staged/metrics.jsonl").exists()
    stage["staging"].update(phase="train", bundle=str(cache))
    monkeypatch.setattr(prefetch, "DoubleBuffer", lambda *a, **kw: pytest.fail("training loaded Qwen"))
    monkeypatch.setattr(trainer, "read_records", lambda *a, **kw: pytest.fail("training reread prompt contents"))
    checkpoint = run(stage)
    expected = json.loads((tmp_path/"monolithic/latest.json").read_text())["checkpoint"]
    p, state, metadata = load_checkpoint(checkpoint)
    q, other, _ = load_checkpoint(expected)
    assert metadata["update"] == 2 and metadata["optimizer_updates"] == 2
    for k in p:
        np.testing.assert_allclose(p[k], q[k], rtol=2e-5, atol=2e-7)
    for x, y in zip(jax.tree.leaves(state), jax.tree.leaves(other)):
        np.testing.assert_allclose(x, y, rtol=2e-5, atol=2e-7)
    events_before = (tmp_path/"staged/metrics.jsonl").read_bytes()
    run(stage)  # Replay-safe: a completed collection cannot update twice.
    assert (tmp_path/"staged/metrics.jsonl").read_bytes() == events_before


@pytest.mark.skipif(len(jax.devices()) != 4, reason="requires four CPU devices")
@pytest.mark.parametrize("async_prefetch", [False, True])
def test_auto_uses_separate_processes_and_resumes_without_duplicate_steps(tmp_path, async_prefetch):
    c = configuration(tmp_path)
    c["training"]["max_updates"] = 4
    c["preparation"]["async_prefetch"] = async_prefetch
    c["staging"].update(enabled=True, phase="auto", keep_consumed_cache=not async_prefetch)
    config = tmp_path/"config.json"
    config.write_text(json.dumps(c))
    environment = os.environ.copy()
    environment.update(JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="4")
    process = subprocess.run([sys.executable, "run.py", "train", "--config", str(config)],
        cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout+process.stderr
    output = Path(c["runtime"]["output_dir"])
    phases = [json.loads(line) for line in (output/"staging/process_times.jsonl").read_text().splitlines()]
    assert [p["phase"] for p in phases] == ["collect", "train", "collect", "train"]
    assert all(p["returncode"] == 0 for p in phases)
    assert (output/"adapter.safetensors").is_file()
    assert json.loads((output/"complete.json").read_text())["update"] == 4
    if async_prefetch:
        assert not list((output/"stage_cache").glob("update_*"))
    rows = (output/"metrics.jsonl").read_bytes()
    again = subprocess.run([sys.executable, "run.py", "train", "--config", str(config)],
        cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=60)
    assert again.returncode == 0, again.stdout+again.stderr
    assert (output/"metrics.jsonl").read_bytes() == rows
    # A completed output must not silently accept a different RL objective.
    from moss_art.staged import run_auto
    from moss_art.checkpoint import CheckpointError
    c["art"]["reference_mse_beta"] = .02
    with pytest.raises(CheckpointError, match="config_fingerprint"):
        run_auto(c)
