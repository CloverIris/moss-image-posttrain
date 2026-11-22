"""Frozen JAX ART kernels; provenance: DiffusionART commit 58e2052.

Statistics MUST be computed over complete prompt groups before microbatching.
Only current predictions receive gradients. Eager data checks are skipped under
JIT; validate rollout caches on the host first.
"""
from __future__ import annotations
import math
import jax
import jax.numpy as jnp
import numpy as np


def _scalar(v, name, positive=False):
    v = float(v)
    if not math.isfinite(v) or (v <= 0 if positive else v < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return v


def _check(c, message):
    if not isinstance(c, jax.core.Tracer) and not bool(np.asarray(c)):
        raise ValueError(message)


def _frozen(v, dtype=jnp.float32):
    return jax.lax.stop_gradient(jnp.asarray(v, dtype))


def _batch(v):
    if v.ndim < 2 or not v.shape[0] or not math.prod(v.shape):
        raise ValueError("nonempty batch and feature dimensions required")


def _vector(v, ref, name, dtype=jnp.float32):
    v = jnp.asarray(v)
    if v.ndim == 0 or v.size != ref.shape[0] or v.shape[0] != ref.shape[0]:
        raise ValueError(f"{name} requires one value per batch item")
    return _frozen(v.reshape(-1), dtype)


def _expand(v, ref):
    return v.reshape((-1,) + (1,) * (ref.ndim - 1))


def _mean(v):
    return jnp.mean(v.reshape(v.shape[0], -1), axis=1)


def covariance(endpoints, rewards, groups, group_count=None, *, mode="scalar",
               group_size=None, seed=0, round_index=0, group_offset=0,
               any_preference_active=None, reduction="unbiased",
               direction_mode="historical", scale_mode="unit_normalized_covariance",
               guidance_gain=1., zero_signal_mode="exact", zero_signal_atol=0., eps=1e-5):
    """K-1 covariance; historical MAD/RMS floor; exact constant-reward guard.

    Scalar mode is JIT-compatible with explicit group_count. Pairwise is a host
    API using NumPy-seeded disjoint pairs, with ties at 0.5 (RNG permutations need
    not match Torch). Result includes upstream group_* and convenient row_* keys.
    """
    endpoints = jnp.asarray(endpoints)
    _batch(endpoints)
    eps = _scalar(eps, "eps", True)
    gain = _scalar(guidance_gain, "guidance_gain")
    atol = _scalar(zero_signal_atol, "zero_signal_atol")
    if reduction not in ("unbiased", "population"):
        raise ValueError("reduction must be unbiased or population")
    if direction_mode not in ("historical", "normalized_aggregate"):
        raise ValueError("unknown direction_mode")
    if zero_signal_mode not in ("historical", "exact", "threshold"):
        raise ValueError("unknown zero_signal_mode")
    if (zero_signal_mode in ("historical", "exact") and atol != 0) or (zero_signal_mode == "threshold" and atol <= 0):
        raise ValueError("threshold requires positive atol; exact/historical require zero")
    if scale_mode not in ("reward_mad", "raw_covariance", "reward_std_normalized_covariance", "unit_normalized_covariance"):
        raise ValueError("unknown scale_mode")
    if mode not in ("scalar", "pairwise"):
        raise ValueError("mode must be scalar or pairwise")
    groups = jnp.asarray(groups)
    if groups.ndim != 1 or not groups.size or not jnp.issubdtype(groups.dtype, jnp.integer):
        raise ValueError("groups must be a nonempty one-dimensional integer vector")
    if group_count is None:
        if isinstance(groups, jax.core.Tracer):
            raise ValueError("group_count is required under JIT")
        group_count = int(np.asarray(groups).max()) + 1
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count < 1:
        raise ValueError("group_count must be a positive integer")
    x = _frozen(endpoints).reshape(endpoints.shape[0], -1)
    r = _frozen(rewards).reshape(-1)
    g = _frozen(groups, jnp.int32)
    if r.size != len(x) or g.size != len(x):
        raise ValueError("rewards and groups must match endpoint count")
    _check(jnp.all((g >= 0) & (g < group_count)), "group index out of range")
    _check(jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(r)), "endpoints and rewards must be finite")
    pair_rows = None
    if mode == "pairwise":
        hg, hs = np.asarray(groups), np.asarray(r)
        labels, pairs = np.full(len(x), .5, np.float32), []
        for group in range(group_count):
            members = np.flatnonzero(hg == group)
            if len(members) not in (2, 4, 8) or (group_size is not None and len(members) != group_size):
                raise ValueError("pairwise requires complete K=2/4/8 groups")
            rng = np.random.default_rng((int(seed) + 1_000_003 * (int(round_index) + int(group_offset) + group)) % (2**63-1))
            pair = rng.permutation(members).reshape(-1, 2)
            left, right = pair[:, 0], pair[:, 1]
            label = np.where(hs[left] > hs[right], 1., np.where(hs[left] < hs[right], 0., .5))
            labels[left], labels[right] = label, 1.-label
            pairs.append(pair)
        pair_rows, r = jnp.asarray(np.concatenate(pairs), jnp.int32), _frozen(labels)
    counts = jnp.zeros(group_count, jnp.float32).at[g].add(1.)
    sums = jnp.zeros(group_count, jnp.float32).at[g].add(r)
    squares = jnp.zeros(group_count, jnp.float32).at[g].add(r*r)
    means = sums / jnp.maximum(counts, 1.)
    stds = jnp.sqrt(jnp.maximum(squares / jnp.maximum(counts, 1.) - means**2, 0.))
    centered = r-means[g]
    maxima = jnp.full(group_count, -jnp.inf).at[g].max(r)
    minima = jnp.full(group_count, jnp.inf).at[g].min(r)
    if zero_signal_mode == "exact":
        constant = maxima == minima
        centered = jnp.where(constant[g], 0., centered)
        means = jnp.where(constant, maxima, means)
        stds = jnp.where(constant, 0., stds)
    mad = jnp.zeros(group_count, jnp.float32).at[g].add(jnp.abs(centered)) / jnp.maximum(counts, 1.)
    numerator = jnp.zeros((group_count, x.shape[1]), jnp.float32).at[g].add(centered[:, None]*x)
    cov = numerator / jnp.maximum(counts-int(reduction == "unbiased"), 1.)[:, None]
    cov = jnp.where((counts > 1)[:, None], cov, 0.)
    rms = jnp.sqrt(jnp.mean(cov**2, axis=1))
    magnitude = {"reward_mad": mad, "raw_covariance": rms,
                 "reward_std_normalized_covariance": rms/(stds+eps),
                 "unit_normalized_covariance": jnp.ones_like(rms)}[scale_mode]*gain
    signal = cov / jnp.where(mad > 0, mad, 1.)[:, None]
    signal_rms = jnp.sqrt(jnp.mean(signal**2, axis=1))
    active = (counts > 1) & (mad > 0) & (signal_rms > 0) & (magnitude > 0)
    if zero_signal_mode == "threshold":
        active &= (maxima-minima > atol) & (rms > atol)
    if zero_signal_mode in ("exact", "threshold"):
        magnitude = jnp.where(active, magnitude, 0.)
    if direction_mode == "historical":
        direction = signal / jnp.maximum(signal_rms, eps)[:, None]
    else:
        # Scale before squaring: safe tiny-covariance norm without global x64.
        maximum = jnp.max(jnp.abs(cov), axis=1)
        scaled = cov / jnp.where(maximum > 0, maximum, 1.)[:, None]
        norm = jnp.sqrt(jnp.mean(scaled**2, axis=1))
        direction = scaled / jnp.where(norm > 0, norm, 1.)[:, None]
    direction = jnp.where(active[:, None], direction, 0.)
    shape = (group_count, *endpoints.shape[1:])
    result = dict(group_covariance=cov.reshape(shape), group_directions=direction.reshape(shape),
                  group_guidance_magnitudes=magnitude, group_active=active, group_counts=counts,
                  group_reward_means=means, group_reward_stds=stds, group_reward_abs_means=mad,
                  group_covariance_rms=rms, row_directions=direction.reshape(shape)[g],
                  row_guidance_magnitudes=magnitude[g], row_active=active[g],
                  row_restore_anchor=jnp.zeros(len(x), jnp.bool_), row_trainable=active[g])
    if mode == "pairwise":
        preference = r != .5
        any_active = bool(np.asarray(jnp.any(preference))) if any_preference_active is None else bool(any_preference_active)
        row_active, anchor = active[g] & preference, ~preference & any_active
        result.update(binary_rewards=r, pair_rows=pair_rows, row_preference_active=preference,
                      row_active=row_active, row_guidance_magnitudes=magnitude[g]*preference,
                      row_restore_anchor=anchor, row_trainable=row_active | anchor)
    return jax.tree.map(jax.lax.stop_gradient, result)


def art_loss(current, old, sigma, direction, magnitude, dt=None, *, state=None,
             ode_velocity_coefficient=None, current_ode_mean=None, old_ode_mean=None,
             loss_space="velocity_linear_sigma", direction_normalization="rms",
             mask_space="velocity", velocity_mask_metric="scheduler_reference_metric",
             corrective_mask_enabled=True, movement_mask_threshold=1e-5,
             noise_normalized_v_radius=.02, q2_reference_sde_noise_level=.8,
             q2_reference_kl_threshold=1e-5, restore_anchor_mask=None,
             target_projection_radius=0., reference_prediction=None,
             reference_mse_beta=0., eps=1e-5, anchor_mode="cached", valid_mask=None):
    """Return feature-mean per-item loss against frozen old-G.

    Sum and divide by ORIGINAL window size, including masked and inactive rows.
    Scheduler-reference movement is a KL proxy, not a separate-model true KL.

    ``cached`` preserves the historical rollout-reference path. ``self`` uses
    stop_gradient(current), and is valid ONLY while accumulating window A at
    the unchanged rollout source theta0. ``calibrated`` consumes a supplied
    learner-compute reference cached at theta0 before either update (window B).
    Both calibrated modes use this same reference for target AND movement,
    ignoring supplied ODE means when deriving movement. No additional model
    evaluation happens here, and the caller must guarantee version matching.

    valid_mask excludes finite padding but not inactive/masked genuine rows
    from valid_count. The caller sums FP32 loss/gradient numerators globally,
    divides by global valid_count, and skips parameters AND optimizer state
    iff global kept_count == 0. effective_keep_mask includes a configured
    reference regularizer, which must not be discarded with inactive ART rows.
    """
    v = jnp.asarray(current, jnp.float32)
    _batch(v)
    if anchor_mode not in ("cached", "self", "calibrated"):
        raise ValueError("anchor_mode must be cached, self or calibrated")
    valid = (jnp.ones(v.shape[0], jnp.bool_) if valid_mask is None else
             _vector(valid_mask, v, "valid_mask", jnp.bool_))
    if np.shape(old) != v.shape or np.shape(direction) != v.shape:
        raise ValueError("old and direction must match current shape")
    if loss_space not in ("x0", "velocity_uniform", "velocity_linear_sigma"):
        raise ValueError("unsupported loss_space")
    if direction_normalization not in ("rms", "none"):
        raise ValueError("direction_normalization must be rms or none")
    if mask_space not in ("velocity", "ode_mean"):
        raise ValueError("mask_space must be velocity or ode_mean")
    if velocity_mask_metric not in ("fixed_mse", "noise_normalized_v", "q2_sde_kl", "scheduler_reference_metric"):
        raise ValueError("unsupported velocity_mask_metric")
    if mask_space == "ode_mean" and velocity_mask_metric != "fixed_mse":
        raise ValueError("ode_mean requires fixed_mse metric")
    eps = _scalar(eps, "eps", True)
    threshold = _scalar(movement_mask_threshold, "movement_mask_threshold")
    radius = _scalar(noise_normalized_v_radius, "noise_normalized_v_radius", True)
    noise = _scalar(q2_reference_sde_noise_level, "q2_reference_sde_noise_level", True)
    kl_threshold = _scalar(q2_reference_kl_threshold, "q2_reference_kl_threshold")
    projection = _scalar(target_projection_radius, "target_projection_radius")
    reference_beta = _scalar(reference_mse_beta, "reference_mse_beta")
    if projection and loss_space != "x0":
        raise ValueError("projection requires x0 space")
    old = _frozen(v if anchor_mode == "self" else old)
    s = _vector(sigma, v, "sigma")
    magnitude = _vector(magnitude, v, "magnitude")
    _check(jnp.all(jnp.isfinite(s) & (s >= 0) & (s <= 1)), "sigma must be finite in [0,1]")
    _check(jnp.all(jnp.isfinite(magnitude) & (magnitude >= 0)), "magnitude must be finite and nonnegative")
    raw = _frozen(direction)
    raw_rms = jnp.sqrt(_mean(raw**2))
    normalized = raw / _expand(jnp.maximum(raw_rms, eps), v) if direction_normalization == "rms" else raw
    displacement = _expand(magnitude, v)*normalized
    weight = s if loss_space == "velocity_linear_sigma" else jnp.ones_like(s)
    if loss_space == "x0":
        if state is None or np.shape(state) != v.shape:
            raise ValueError("x0 requires state matching current")
        x = _frozen(state)
        if projection:
            displacement *= _expand(jnp.minimum(projection/jnp.maximum(jnp.sqrt(_mean(displacement**2)), eps), 1.), v)
        target = jax.lax.stop_gradient(x-_expand(s, v)*old+displacement)
        residual = (target-(x-_expand(s, v)*v) if anchor_mode == "cached" else
                    _expand(s, v)*(v-old)+displacement)
        desired = -_expand(s, v)*jax.lax.stop_gradient(residual)
    else:
        target = jax.lax.stop_gradient(old-displacement)
        # Form the calibrated residual before adding/subtracting a potentially
        # large old velocity. In particular self-anchor retains arbitrarily
        # small FP32 G and its gradient even when old-G rounds back to old.
        residual = target-v if anchor_mode == "cached" else (old-v)-displacement
        desired = jax.lax.stop_gradient(residual)
    unmasked = weight*_mean(residual**2)
    movement = v-old
    energy, alignment = _mean(movement**2), _mean(movement*desired)
    if dt is not None and ode_velocity_coefficient is not None:
        raise ValueError("supply dt or ode_velocity_coefficient, not both")
    delta_input = dt if dt is not None else ode_velocity_coefficient
    if mask_space == "ode_mean" or velocity_mask_metric in ("q2_sde_kl", "scheduler_reference_metric"):
        if delta_input is None:
            raise ValueError("this mask requires negative dt / ode_velocity_coefficient")
        delta = _vector(delta_input, v, "dt")
        _check(jnp.all(jnp.isfinite(delta) & (delta < 0)), "ODE coefficients must be finite and negative")
        if mask_space == "ode_mean":
            if anchor_mode == "cached":
                if current_ode_mean is None or old_ode_mean is None or np.shape(current_ode_mean) != v.shape or np.shape(old_ode_mean) != v.shape:
                    raise ValueError("ode_mean requires matching current and old ODE means")
                move = jnp.asarray(current_ode_mean, jnp.float32)-_frozen(old_ode_mean)
            else:
                # The common state/history term cancels for the fixed replay
                # transition. Never let stale rollout old_ode_mean override
                # the learner-compute reference used by the regression target.
                move = _expand(delta, v)*movement
            energy, alignment = _mean(move**2), _mean(move*_expand(delta, v)*desired)
        else:
            initial = jnp.isclose(s, 1., atol=eps, rtol=0.)
            std = jnp.where(initial, noise, jnp.sqrt(jnp.maximum(noise**2*s/jnp.maximum(1.-s, eps)*(-delta), 0.)))
            coefficient = jnp.where(initial, delta, delta*(1.+.5*noise**2))
            energy = coefficient**2*energy/(2.*jnp.maximum(std**2, eps**2))
            threshold = kl_threshold
    elif velocity_mask_metric == "noise_normalized_v":
        energy = jnp.where(s > 0, energy/jnp.where(s > 0, s, 1.), 0.)
        threshold = radius**2
    active = (magnitude > 0) & (raw_rms > 0) & (s > 0)
    requested = jnp.zeros_like(active) if restore_anchor_mask is None else _vector(restore_anchor_mask, v, "restore_anchor_mask", jnp.bool_)
    anchor = requested & ~active & (s > 0)
    masked = jax.lax.stop_gradient(valid & active & corrective_mask_enabled & (energy >= threshold) & (alignment > 0))
    keep = jax.lax.stop_gradient(valid & (anchor | (active & ~masked)))
    policy = jnp.where(keep, unmasked, jnp.zeros_like(unmasked))
    reference = jnp.zeros_like(policy)
    if reference_prediction is not None:
        if np.shape(reference_prediction) != v.shape:
            raise ValueError("reference_prediction must match current")
        reference = jnp.where(valid, _mean((v-_frozen(reference_prediction))**2), 0.)
    elif reference_beta:
        raise ValueError("reference_mse_beta requires reference_prediction")
    effective_keep = jax.lax.stop_gradient(keep | (valid & (reference_beta > 0)))
    return dict(per_item_loss=policy+reference_beta*reference, per_item_policy_loss=policy,
                per_item_reference_mse=reference, per_item_unmasked_loss=jnp.where(valid, unmasked, 0.),
                target=target, keep_mask=keep, masked=masked,
                valid_mask=valid, effective_keep_mask=effective_keep,
                valid_count=jnp.sum(valid, dtype=jnp.int32),
                policy_kept_count=jnp.sum(keep, dtype=jnp.int32),
                kept_count=jnp.sum(effective_keep, dtype=jnp.int32),
                movement_score=jax.lax.stop_gradient(energy),
                outward_alignment=jax.lax.stop_gradient(alignment))


def prompt_windows(groups, selected_timesteps, seed=0):
    """Two prompt-balanced disjoint windows of (row, selected timestep SLOT).

    Every row carries all selected slots. Input selected_timesteps is a positive
    count or a 1-D sequence (only its length is used). Inactive rows remain.
    """
    ids = np.asarray(groups)
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu" or (ids < 0).any():
        raise ValueError("groups require nonempty nonnegative integer vector")
    if isinstance(selected_timesteps, (int, np.integer)) and not isinstance(selected_timesteps, bool):
        count = int(selected_timesteps)
    else:
        selected = np.asarray(selected_timesteps)
        if selected.ndim != 1:
            raise ValueError("selected_timesteps requires positive count or 1-D sequence")
        count = len(selected)
    if count < 1:
        raise ValueError("at least one selected timestep required")
    order = np.random.default_rng(int(seed)).permutation(len(ids))
    assignment = np.zeros(len(ids), np.int8)
    for group in np.unique(ids):
        members = order[ids[order] == group]
        if len(members) % 2:
            raise ValueError("each complete prompt group must divide evenly into two windows")
        assignment[members[len(members)//2:]] = 1
    return tuple(np.asarray([(int(row), slot) for row in order if assignment[row] == window
                             for slot in range(count)], np.int64).reshape(-1, 2) for window in range(2))


def reduce_window_loss(per_item_loss, original_item_count):
    """Microbatch contribution divided by full ORIGINAL update-window size."""
    if isinstance(original_item_count, bool) or not isinstance(original_item_count, (int, np.integer)) or original_item_count <= 0:
        raise ValueError("original_item_count must be positive integer")
    return jnp.sum(jnp.asarray(per_item_loss, jnp.float32))/int(original_item_count)


art_scalar_covariance = covariance
