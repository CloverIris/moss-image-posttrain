"""Explicit CPU-only migration preserves optimizer/learner and initializes one teacher."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import optax
import pytest

from moss_art.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from moss_art.config import DEFAULTS, load_config
from moss_art.reference import update_reference, validate_reference_state
from moss_art.trainer import identity_for
from scripts.initialize_ema_reference import migrate, _source_info

REPO = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2))
    return path


def hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.iterdir() if p.is_file()}


def trees_equal(a, b):
    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    assert ta == tb
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(x, y)


@pytest.fixture
def fixture(tmp_path):
    data = dump(tmp_path / "prompts.jsonl", {"id": "fixture", "prompt": "synthetic migration fixture"})
    c = copy.deepcopy(DEFAULTS)
    c["name"] = "migration-fixture"
    c["model"]["kind"] = "tiny"
    c["sampling"]["skip_layers"] = [1]
    c["runtime"].update(platform="cpu", devices=[0], output_dir=str(tmp_path / "old-output"))
    c["reward"].update(kind="toy_latent", decode=False)
    c["data"]["prompts"] = str(data)
    c["art"].update(reference_mode="base", reference_mse_beta=1.)
    source_config = dump(tmp_path / "source.json", c)
    target = copy.deepcopy(c)
    target["name"] = "ema-migration-fixture"
    target["art"].update(reference_mode="ema", reference_ema_decay=.8)
    target["runtime"]["output_dir"] = str(tmp_path / "new-output")
    target["run"]["resume"] = str(tmp_path / "new-checkpoint")
    target_config = dump(tmp_path / "target.json", target)
    learner = {"A": np.arange(6, dtype=np.float32).reshape(2, 3) / 100,
               "B": np.full((3, 2), .03, np.float32)}
    tx = optax.adamw(.001)
    optimizer = tx.init(learner)
    metadata = {**identity_for(load_config(source_config)), "update": 6, "optimizer_updates": 6,
                "rollout": 72, "prompt_cursor": 72, "seed": 42, "resolution_cursors": {"0": 72},
                "stage_cache_id": "/old/scratch/update4",
                "staged_events": [{"update": 5}, {"update": 6}],
                "staged_timings": {"total": .5}}
    source = save_checkpoint(tmp_path / "source-checkpoint", learner, optimizer, metadata)
    settings = {"source_repo": str(REPO), "source_config": str(source_config),
                "target_config": str(target_config), "source_checkpoint": str(source),
                "target_checkpoint": str(tmp_path / "new-checkpoint"),
                "report": str(tmp_path / "migration-report.json")}
    return settings, learner, optimizer, metadata


def test_migration_preserves_all_numeric_state_and_source_and_resumes_ema(fixture, tmp_path):
    settings, learner, optimizer, metadata = fixture
    source = Path(settings["source_checkpoint"])
    before = hashes(source)
    record = migrate(settings)
    assert record["status"] == "passed"
    assert record["source_unchanged"]
    assert before == hashes(source)
    target_config = load_config(settings["target_config"])
    got, got_opt, got_meta, teacher = load_checkpoint(settings["target_checkpoint"],
        expected_metadata=identity_for(target_config), include_reference_state=True)
    trees_equal(learner, got)
    trees_equal(optimizer, got_opt)
    trees_equal(learner, teacher["lora"])
    assert teacher["updates"] == 0
    for name in ("update", "optimizer_updates", "rollout", "prompt_cursor", "seed", "resolution_cursors"):
        assert got_meta[name] == metadata[name]
    for name in ("stage_cache_id", "staged_events", "staged_timings"):
        assert name not in got_meta
        assert got_meta["reference_initialization"]["stripped_transient_metadata"][name] == metadata[name]
    assert got_meta["reference_initialization"]["origin_optimizer_updates"] == 6
    next_learner = {k: v + np.float32(.1) for k, v in got.items()}
    teacher = update_reference(target_config, teacher, next_learner, True)
    assert teacher["updates"] == 1
    path = save_checkpoint(tmp_path / "resume", next_learner, got_opt, got_meta, reference_state=teacher)
    _, _, _, restored_teacher = load_checkpoint(path, include_reference_state=True)
    validate_reference_state(target_config, restored_teacher, next_learner)
    trees_equal(teacher["lora"], restored_teacher["lora"])
    assert len(load_checkpoint(settings["target_checkpoint"])) == 3
    report = json.loads(Path(settings["report"]).read_text())
    assert report["preserved_arrays"] == report["target_arrays"]
    assert report["reference_updates"] == 0


@pytest.mark.parametrize("path,value", [
    (("optimizer", "learning_rate"), .1),
    (("art", "reference_mse_beta"), .01),
    (("training", "max_updates"), 22),
    (("sampling", "steps"), 48),
])
def test_forbids_other_algorithm_changes_and_retains_failed_report(fixture, path, value):
    settings, _, _, _ = fixture
    config_file = Path(settings["target_config"])
    c = json.loads(config_file.read_text())
    c[path[0]][path[1]] = value
    dump(config_file, c)
    with pytest.raises(ValueError, match="Forbidden numerical"):
        migrate(settings)
    assert not Path(settings["target_checkpoint"]).exists()
    report = json.loads(Path(settings["report"]).read_text())
    assert report["status"] == "failed"
    assert report["source_unchanged"]


@pytest.mark.parametrize("which,mode", [("source", "ema"), ("target", "base")])
def test_requires_explicit_base_to_ema_direction(fixture, which, mode):
    settings, _, _, _ = fixture
    config_file = Path(settings[which + "_config"])
    c = json.loads(config_file.read_text())
    c["art"]["reference_mode"] = mode
    dump(config_file, c)
    with pytest.raises(ValueError, match="configuration must use"):
        migrate(settings)
    assert not Path(settings["target_checkpoint"]).exists()


def test_source_identity_mismatch_fails_before_publish(fixture):
    settings, _, _, _ = fixture
    c = json.loads(Path(settings["source_config"]).read_text())
    c["optimizer"]["learning_rate"] = .0007
    dump(Path(settings["source_config"]), c)
    target = json.loads(Path(settings["target_config"]).read_text())
    target["optimizer"]["learning_rate"] = .0007
    dump(Path(settings["target_config"]), target)
    with pytest.raises(CheckpointError, match="incompatible resume field"):
        migrate(settings)
    assert not Path(settings["target_checkpoint"]).exists()
    assert json.loads(Path(settings["report"]).read_text())["source_unchanged"]


def test_target_resume_identity_is_strict_and_outputs_cannot_be_reused(fixture):
    settings, _, _, _ = fixture
    migrate(settings)
    c = json.loads(Path(settings["target_config"]).read_text())
    c["art"]["reference_ema_decay"] = .7
    dump(Path(settings["target_config"]), c)
    with pytest.raises(CheckpointError, match="incompatible resume field"):
        load_checkpoint(settings["target_checkpoint"],
                        expected_metadata=identity_for(load_config(settings["target_config"])),
                        include_reference_state=True)
    with pytest.raises(FileExistsError):
        migrate(settings)


def test_cli_requires_config_and_rejects_extra_options():
    result = subprocess.run([sys.executable, str(REPO / "scripts/initialize_ema_reference.py"),
                             "--source_checkpoint", "anything"], text=True, capture_output=True)
    assert result.returncode != 0
    assert "--config" in result.stderr


def test_report_cannot_be_added_inside_immutable_target(fixture):
    settings, _, _, _ = fixture
    settings["report"] = str(Path(settings["target_checkpoint"]) / "report.json")
    with pytest.raises(ValueError, match="outside the target"):
        migrate(settings)
    assert not Path(settings["target_checkpoint"]).exists()


def test_legacy_base_source_computes_its_own_identity(fixture, tmp_path):
    snapshot = REPO / "EMA_SOURCE_SNAPSHOT.json"
    if not snapshot.exists():
        pytest.skip("No external legacy snapshot available")
    legacy = Path(json.loads(snapshot.read_text())["source"])
    if not (legacy / "moss_art/trainer.py").is_file():
        pytest.skip("External legacy snapshot unavailable")
    settings, learner, optimizer, metadata = fixture
    old = json.loads(Path(settings["source_config"]).read_text())
    old["art"].pop("reference_mode")
    old["art"].pop("reference_ema_decay")
    dump(Path(settings["source_config"]), old)
    settings["source_repo"] = str(legacy)
    source_identity = _source_info(legacy, Path(settings["source_config"]))["identity"]
    old_metadata = dict(metadata, **source_identity)
    source = save_checkpoint(tmp_path / "legacy-checkpoint", learner, optimizer, old_metadata)
    settings["source_checkpoint"] = str(source)
    record = migrate(settings)
    assert record["status"] == "passed"
    assert record["source_identity"] == source_identity
    assert record["target_identity"]["experiment"]["source_files"] != source_identity["experiment"]["source_files"]
    assert record["source_unchanged"]


def test_config_only_cli_runs_cpu_migration(fixture, tmp_path):
    settings, _, _, _ = fixture
    config_file = dump(tmp_path / "migration.json", settings)
    result = subprocess.run([sys.executable, str(REPO / "scripts/initialize_ema_reference.py"),
                             "--config", str(config_file)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["update"] == payload["optimizer_updates"] == 6
    assert payload["reference_updates"] == 0
    assert json.loads(Path(settings["report"]).read_text())["source_unchanged"]
