#!/usr/bin/env python3
"""Explicit, CPU-only Base-reference -> EMA-reference checkpoint migration.

All arguments are supplied by --config JSON:
source_repo, source_config, target_config, source_checkpoint,
target_checkpoint, report.

The source is never rewritten. The new learner and optimizer are byte-identical
to the source, and the teacher starts at that learner with EMA updates=0.
This utility neither resumes training nor launches a model.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
KEYS = {"source_repo", "source_config", "target_config", "source_checkpoint",
        "target_checkpoint", "report"}
TRANSIENT = ("stage_cache_id", "staged_events", "staged_timings")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _source_files(path):
    result = {}
    for name in ("COMMITTED", "manifest.json", "arrays.npz"):
        p = Path(path) / name
        if not p.is_file() or p.is_symlink():
            raise ValueError(f"Missing or symlinked source checkpoint file: {name}")
        st = p.stat()
        result[name] = {"sha256": _sha(p), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    return result


def _settings(value):
    if not isinstance(value, dict) or set(value) != KEYS:
        raise ValueError("Migration config must contain exactly: " + ", ".join(sorted(KEYS)))
    result = {}
    for key in sorted(KEYS):
        text = value[key]
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise ValueError(f"{key} must be a nonempty path string")
        result[key] = Path(text).expanduser().resolve()
    source = result["source_checkpoint"]
    for name in ("target_checkpoint", "report"):
        target = result[name]
        if target == source or source in target.parents:
            raise ValueError(f"{name} cannot be inside the source checkpoint")
    target = result["target_checkpoint"]
    report = result["report"]
    if report == target or target in report.parents:
        raise ValueError("report must be outside the target checkpoint")
    if not (result["source_repo"] / "moss_art/trainer.py").is_file():
        raise ValueError("source_repo must contain moss_art/trainer.py")
    return result


def _source_info(repo, config):
    # The source implementation, not the target's rewritten defaults/source list,
    # computes the source resume identity. Importing identity_for constructs no model.
    code = """import json, pathlib, sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from moss_art.config import load_config
from moss_art.trainer import identity_for
c = load_config(sys.argv[2])
print(json.dumps({"config": c, "identity": identity_for(c)}, sort_keys=True, allow_nan=False))
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.run([sys.executable, "-B", "-c", code, str(repo), str(config)],
                             cwd=repo, env=env, text=True, capture_output=True)
    if process.returncode:
        raise ValueError("Cannot compute source identity: " + process.stderr[-4000:])
    try:
        value = json.loads(process.stdout)
        assert isinstance(value["config"], dict) and isinstance(value["identity"], dict)
        return value
    except (ValueError, KeyError, AssertionError) as error:
        raise ValueError("Source identity subprocess returned an invalid result") from error


def _algorithm_view(config):
    c = copy.deepcopy(config)
    for name in ("name", "run", "logging", "staging"):
        c.pop(name, None)
    for name in ("output_dir", "allow_busy"):
        c["runtime"].pop(name, None)
    for name in ("reference_mode", "reference_ema_decay"):
        c["art"].pop(name, None)
    return c


def _differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            name = prefix + "." + key if prefix else key
            if key not in a or key not in b:
                out.append({"path": name, "source": a.get(key), "target": b.get(key)})
            else:
                out.extend(_differences(a[key], b[key], name))
        return out
    if a != b:
        return [{"path": prefix, "source": a, "target": b}]
    return []


def _tree_audit(value):
    import jax
    import numpy as np
    leaves, treedef = jax.tree.flatten(value)
    records = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jax.Array) and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            array = np.asarray(jax.random.key_data(leaf))
            dtype = str(leaf.dtype)
        else:
            array = np.asarray(leaf)
            dtype = str(array.dtype)
        if array.dtype.hasobject:
            raise ValueError("Object arrays cannot be audited")
        records.append({"leaf": i, "dtype": dtype, "shape": list(array.shape),
                        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                        "bytes": int(array.nbytes)})
    body = {"structure": str(treedef), "leaves": records}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body


def _report(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix=".ema-migration-report-",
                                         dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        # Atomic publication without ever replacing an existing report.
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def migrate(value):
    settings = _settings(value)
    target = settings["target_checkpoint"]
    report = settings["report"]
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Target checkpoint already exists: {target}")
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"Migration report already exists: {report}")
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    if any(d.platform != "cpu" for d in jax.devices()):
        raise RuntimeError("Migration must run in a fresh CPU-only process")
    from moss_art.config import load_config
    from moss_art.trainer import identity_for
    from moss_art.checkpoint import save_checkpoint, load_checkpoint
    from moss_art.reference import initialize_reference, validate_reference_state
    record = {"version": 1, "status": "started",
              "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "settings": {k: str(v) for k, v in settings.items()}}
    source_before = None
    try:
        source_before = _source_files(settings["source_checkpoint"])
        info = _source_info(settings["source_repo"], settings["source_config"])
        source_config, source_identity = info["config"], info["identity"]
        target_config = load_config(settings["target_config"])
        if source_config["art"].get("reference_mode", "base") != "base":
            raise ValueError("Source configuration must use Base reference mode")
        if target_config["art"].get("reference_mode", "base") != "ema":
            raise ValueError("Target configuration must use EMA reference mode")
        differences = _differences(_algorithm_view(source_config), _algorithm_view(target_config))
        if differences:
            raise ValueError("Forbidden numerical/configuration changes: " +
                             ", ".join(x["path"] for x in differences))
        record["allowed_config_changes"] = _differences(source_config, target_config)
        target_identity = identity_for(target_config)
        learner, optimizer, metadata, old_reference = load_checkpoint(
            settings["source_checkpoint"], expected_metadata=source_identity,
            include_reference_state=True)
        if old_reference is not None:
            raise ValueError("Source checkpoint already contains an EMA reference")
        if type(metadata.get("update")) is not int or type(metadata.get("optimizer_updates")) is not int:
            raise ValueError("Source checkpoint must record update and optimizer_updates")
        if metadata["update"] < 0 or metadata["optimizer_updates"] < 0:
            raise ValueError("Source checkpoint counters must be nonnegative")
        before = {"learner": _tree_audit(learner), "optimizer": _tree_audit(optimizer)}
        reference = initialize_reference(target_config, learner)
        validate_reference_state(target_config, reference, learner)
        new_metadata = copy.deepcopy(metadata)
        stripped = {k: new_metadata.pop(k) for k in TRANSIENT if k in new_metadata}
        new_metadata.update(target_identity)
        provenance = {
            "method": "initialize_ema_from_current_student",
            "source_repo": str(settings["source_repo"]),
            "source_checkpoint": str(settings["source_checkpoint"]),
            "source_config": str(settings["source_config"]),
            "source_identity": source_identity,
            "source_checkpoint_files": source_before,
            "origin_update": metadata["update"],
            "origin_optimizer_updates": metadata["optimizer_updates"],
            "ema_updates_at_initialization": 0,
            "decay": reference["decay"],
            "stripped_transient_metadata": stripped,
        }
        new_metadata["reference_initialization"] = provenance
        # Code edits during migration invalidate the target identity rather than
        # publishing a checkpoint tied to a mixed source snapshot.
        if identity_for(target_config) != target_identity:
            raise RuntimeError("Target numerical source changed during migration")
        save_checkpoint(target, learner, optimizer, new_metadata, reference_state=reference)
        got, got_optimizer, got_metadata, got_reference = load_checkpoint(
            target, expected_metadata=target_identity, include_reference_state=True)
        validate_reference_state(target_config, got_reference, got)
        after = {"learner": _tree_audit(got), "optimizer": _tree_audit(got_optimizer)}
        if before != after:
            raise RuntimeError("Migration changed learner or optimizer bytes")
        if _tree_audit(reference["lora"]) != _tree_audit(got_reference["lora"]):
            raise RuntimeError("Migrated EMA teacher did not round-trip exactly")
        if got_reference["updates"] != 0 or got_metadata != new_metadata:
            raise RuntimeError("Migrated state metadata did not round-trip exactly")
        source_after = _source_files(settings["source_checkpoint"])
        if source_after != source_before:
            raise RuntimeError("Source checkpoint changed during migration")
        record.update(status="passed", source_identity=source_identity,
                      target_identity=target_identity, source_before=source_before,
                      source_after=source_after, source_unchanged=True,
                      preserved_arrays=before, target_arrays=after,
                      reference_arrays=_tree_audit(got_reference["lora"]),
                      update=got_metadata["update"],
                      optimizer_updates=got_metadata["optimizer_updates"],
                      reference_updates=got_reference["updates"],
                      stripped_transient_metadata=stripped,
                      target_checkpoint_files=_source_files(target))
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        if source_before is not None:
            record["source_before"] = source_before
            try:
                record["source_after"] = _source_files(settings["source_checkpoint"])
                record["source_unchanged"] = record["source_after"] == source_before
            except Exception as proof_error:
                record["source_audit_error"] = f"{type(proof_error).__name__}: {proof_error}"
        record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _report(report, record)
        raise
    record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _report(report, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    value = json.loads(Path(args.config).read_text())
    record = migrate(value)
    print(json.dumps({"status": record["status"],
                      "update": record["update"],
                      "optimizer_updates": record["optimizer_updates"],
                      "reference_updates": record["reference_updates"],
                      "checkpoint": record["settings"]["target_checkpoint"],
                      "report": record["settings"]["report"]}, sort_keys=True))


if __name__ == "__main__":
    main()
