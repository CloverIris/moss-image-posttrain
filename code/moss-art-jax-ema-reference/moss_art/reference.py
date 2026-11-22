"""Lagged reference state over the trainable FP32 LoRA factors.

Frozen base weights are shared by learner and reference. This is an EMA of the
optimized LoRA factors, not an EMA of the merged, bilinear LoRA weight delta.
The state is updated only after a successful optimizer transition and is saved
alongside that learner transition. No reference state is inferred on resume.
"""
from __future__ import annotations
import math
import numpy as np


def _mode(c):
    mode = c["art"].get("reference_mode", "base")
    if mode not in ("base", "ema"):
        raise ValueError("art.reference_mode must be base or ema")
    return mode


def _decay(c):
    value = c["art"].get("reference_ema_decay", .99)
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("art.reference_ema_decay must be finite and lie in [0, 1)")
    return float(value)


def _array(value, name, *, require_fp32=False):
    import jax
    if not isinstance(value, (np.ndarray, jax.Array)):
        raise ValueError(f"{name} must be an array")
    dtype = value.dtype
    if str(dtype) not in ("float16", "float32", "float64", "bfloat16"):
        raise ValueError(f"{name} must have a real floating dtype")
    if require_fp32 and str(dtype) != "float32":
        raise ValueError(f"{name} must use float32")
    if isinstance(value, jax.Array):
        finite = bool(np.asarray(jax.numpy.all(jax.numpy.isfinite(value))))
    else:
        finite = bool(np.isfinite(value).all())
    if not finite:
        raise ValueError(f"{name} must be finite")
    return value


def _parameters(lora, name, *, require_fp32=False):
    if not isinstance(lora, dict) or not lora or not all(isinstance(k, str) for k in lora):
        raise ValueError(f"{name} must be a nonempty string-keyed LoRA dictionary")
    for key, value in lora.items():
        _array(value, f"{name}.{key}", require_fp32=require_fp32)
    return lora


def initialize_reference(c, lora):
    """Create an independent FP32 teacher from the current learner, or None.

    Call only for a fresh run or an explicitly recorded checkpoint migration.
    A missing EMA state in an EMA resume is an error, never a request to reset it.
    """
    if _mode(c) == "base":
        return None
    import jax
    import jax.numpy as jnp
    _parameters(lora, "learner")
    teacher = {}
    for key, value in lora.items():
        teacher[key] = (jax.lax.stop_gradient(jnp.array(value, dtype=jnp.float32, copy=True))
                        if isinstance(value, jax.Array) else np.array(value, dtype=np.float32, copy=True))
    return {"mode": "ema", "decay": _decay(c), "updates": 0, "lora": teacher}


def validate_reference_state(c, state, lora):
    """Strict eager validation at restore/installation boundaries; return state."""
    if _mode(c) == "base":
        if state is not None:
            raise ValueError("Base reference mode cannot restore an EMA reference state")
        return None
    if state is None:
        raise ValueError("EMA reference state is missing; use an explicit initialization migration")
    if not isinstance(state, dict) or set(state) != {"mode", "decay", "updates", "lora"}:
        raise ValueError("Invalid EMA reference state fields")
    if state["mode"] != "ema":
        raise ValueError("EMA reference state mode mismatch")
    decay = state["decay"]
    if type(decay) not in (int, float) or not math.isfinite(decay) or float(decay) != _decay(c):
        raise ValueError("EMA reference decay mismatch")
    if type(state["updates"]) is not int or state["updates"] < 0:
        raise ValueError("EMA reference updates must be a nonnegative integer")
    _parameters(lora, "learner")
    teacher = _parameters(state["lora"], "EMA reference", require_fp32=True)
    if set(teacher) != set(lora):
        raise ValueError("EMA reference LoRA keys mismatch")
    for key, value in teacher.items():
        if value.shape != lora[key].shape:
            raise ValueError(f"EMA reference LoRA shape mismatch: {key}")
    return state


def ema_parameters(reference, learner, decay):
    """Pure numerical recurrence with a stopped teacher result; JAX AD/JIT safe."""
    import jax
    import jax.numpy as jnp
    result = {}
    for key, old in reference.items():
        new = learner[key]
        if isinstance(old, (jax.Array, jax.core.Tracer)) or isinstance(new, (jax.Array, jax.core.Tracer)):
            result[key] = jax.lax.stop_gradient(
                jnp.asarray(old, jnp.float32) * jnp.float32(decay)
                + jnp.asarray(new, jnp.float32) * jnp.float32(1 - decay))
        else:
            result[key] = (np.asarray(old, np.float32) * np.float32(decay)
                           + np.asarray(new, np.float32) * np.float32(1 - decay))
    return result


def update_reference(c, state, lora, applied):
    """Advance EMA exactly once after an applied learner update; skips are no-op."""
    validate_reference_state(c, state, lora)
    if state is None:
        return None
    if not isinstance(applied, (bool, np.bool_)):
        raise ValueError("applied must be a boolean optimizer-transition flag")
    if not applied:
        return state
    teacher = ema_parameters(state["lora"], lora, state["decay"])
    result = dict(state, lora=teacher, updates=state["updates"] + 1)
    validate_reference_state(c, result, lora)
    return result
