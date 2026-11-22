"""Atomic EMA checkpoint persistence, old-reader compatibility and strict integrity."""
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from moss_art.checkpoint import CheckpointError, save_checkpoint, load_checkpoint
from moss_art.reference import initialize_reference, update_reference, validate_reference_state


def config():
    return {"art": {"reference_mode": "ema", "reference_ema_decay": .75}}


def arrays():
    return {"A": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
            "B": jnp.zeros((3, 2), jnp.float32)}


def assert_tree_equal(a, b):
    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    assert ta == tb
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(x, y)


def transition(params, opt_state, teacher, tx, step):
    gradients = jax.grad(lambda p: sum(jnp.mean((x - (step + 1)) ** 2) for x in p.values()))(params)
    delta, opt_state = tx.update(gradients, opt_state, params)
    params = optax.apply_updates(params, delta)
    teacher = update_reference(config(), teacher, params, True)
    return params, opt_state, teacher


def test_save_resume_reproduces_learner_optimizer_and_teacher_exactly(tmp_path):
    params = arrays()
    tx = optax.adamw(.01)
    opt_state = tx.init(params)
    teacher = initialize_reference(config(), params)
    for i in range(3):
        params, opt_state, teacher = transition(params, opt_state, teacher, tx, i)
    path = save_checkpoint(tmp_path / "step3", params, opt_state, {"update": 3}, reference_state=teacher)
    got, got_opt, metadata, got_teacher = load_checkpoint(path, include_reference_state=True)
    assert metadata == {"update": 3}
    assert_tree_equal(params, got)
    assert_tree_equal(opt_state, got_opt)
    assert_tree_equal(teacher["lora"], got_teacher["lora"])
    assert teacher["updates"] == got_teacher["updates"] == 3
    assert validate_reference_state(config(), got_teacher, got) is got_teacher
    assert len(load_checkpoint(path)) == 3  # Existing export/evaluation clients.
    for i in range(3, 6):
        params, opt_state, teacher = transition(params, opt_state, teacher, tx, i)
        got, got_opt, got_teacher = transition(got, got_opt, got_teacher, tx, i)
    assert_tree_equal(params, got)
    assert_tree_equal(opt_state, got_opt)
    assert_tree_equal(teacher["lora"], got_teacher["lora"])
    assert teacher["updates"] == got_teacher["updates"] == 6
    with np.load(path / "arrays.npz", allow_pickle=False) as data:
        assert all(not data[k].dtype.hasobject for k in data.files)


def test_legacy_checkpoint_loads_without_inventing_a_teacher(tmp_path):
    params = arrays()
    path = save_checkpoint(tmp_path / "legacy", params, (), {"update": 2})
    assert "reference_state" not in json.loads((path / "manifest.json").read_text())
    assert len(load_checkpoint(path)) == 3
    loaded, state, metadata, teacher = load_checkpoint(path, include_reference_state=True)
    assert teacher is None
    with pytest.raises(ValueError, match="missing"):
        validate_reference_state(config(), teacher, loaded)


def recommit(path, manifest):
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    (path / "manifest.json").write_bytes(body)
    (path / "COMMITTED").write_text(hashlib.sha256(body).hexdigest() + "\n")


@pytest.mark.parametrize("include", [False, True])
def test_reference_state_is_checked_even_for_legacy_callers(tmp_path, include):
    params = arrays()
    path = save_checkpoint(tmp_path / "ema", params, (), {}, reference_state=initialize_reference(config(), params))
    m = json.loads((path / "manifest.json").read_text())
    for pair in m["reference_state"]["items"]:
        if pair[0] == "updates":
            pair[1]["value"] = -1
    recommit(path, m)
    with pytest.raises(CheckpointError, match="nonnegative"):
        load_checkpoint(path, include_reference_state=include)


def test_reference_payload_hash_and_array_usage_are_integrity_bound(tmp_path):
    params = arrays()
    path = save_checkpoint(tmp_path / "ema", params, (), {}, reference_state=initialize_reference(config(), params))
    m = json.loads((path / "manifest.json").read_text())
    m.pop("reference_state")
    recommit(path, m)
    with pytest.raises(CheckpointError, match="unreferenced"):
        load_checkpoint(path)
