"""Optional Spectrum/Chebyshev head-feature forecasting for JAX rollouts.

Adapted from ComfyUI-Spectrum-KSampler, moss.py::_Forecaster and the window
schedule in spectrum.py, commit 1baeab6f9ee94a75011353c6ef41963888cc8979.
Copyright (c) 2026 sorryhyun, MIT; see docs/licenses/Spectrum-MIT.txt.
Algorithm: Han et al., Adaptive Spectral Feature Forecasting for Diffusion
Sampling Acceleration, https://arxiv.org/abs/2603.01623.

The small ridge solve is FP64 NumPy on the controller (no JAX global flags).
Feature storage, FP32 weighted sums and the original MOSS head run in JAX.
Importing this module never initializes JAX or a GPU.
"""
from functools import lru_cache
import math
import numpy as np


DEFAULTS = dict(enabled=False, degree=3, ridge_lambda=0.1, blend_weight=0.3,
                window_size=2.0, flex_window=0.5, warmup_steps=6,
                tail_actual_steps=3, history_size=100)
UPSTREAM_REVISION = "1baeab6f9ee94a75011353c6ef41963888cc8979"


def validate_options(options):
    """Configuration types are also checked by the shared strict schema."""
    if set(options) != set(DEFAULTS):
        raise ValueError("sampling.spectrum has missing or unknown fields")
    if type(options["enabled"]) is not bool:
        raise ValueError("sampling.spectrum.enabled must be boolean")
    for key in ("degree", "warmup_steps", "tail_actual_steps", "history_size"):
        if type(options[key]) is not int or options[key] < 0:
            raise ValueError(f"sampling.spectrum.{key} must be a nonnegative integer")
    if options["degree"] > 8 or options["history_size"] < options["degree"] + 2:
        raise ValueError("sampling.spectrum needs degree <= 8 and history_size >= degree + 2")
    for key in ("ridge_lambda", "blend_weight", "window_size", "flex_window"):
        value = options[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"sampling.spectrum.{key} must be finite")
    if options["ridge_lambda"] <= 0 or not 0 <= options["blend_weight"] <= 1:
        raise ValueError("sampling.spectrum requires positive ridge_lambda and blend_weight in [0,1]")
    if options["window_size"] < 1 or options["flex_window"] < 0:
        raise ValueError("sampling.spectrum requires window_size >= 1 and flex_window >= 0")
    return options


def actual_step_plan(total_steps, options, forced_steps=()):
    """Upstream growing-window schedule, using sampler indices, not sigma values.

    ART replay points are mandatory actual evaluations. The plan is rebuilt
    after scheduler deduplication; feature histories never cross trajectories.
    """
    validate_options(options)
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    forced = frozenset(forced_steps)
    if any(type(i) is not int or not 0 <= i < total_steps for i in forced):
        raise ValueError("Forced Spectrum steps must be valid sampler indices")
    observations = consecutive_cached = 0
    window = options["window_size"]
    plan = []
    for i in range(total_steps):
        if i:
            if plan[-1]:
                if i - 1 >= options["warmup_steps"]:
                    window = round(window + options["flex_window"], 3)
                consecutive_cached = 0
            else:
                consecutive_cached += 1
        actual = (not options["enabled"] or i in forced or
                  i < options["warmup_steps"] or
                  i >= total_steps - options["tail_actual_steps"] or
                  observations < max(2, options["degree"] + 2) or
                  (consecutive_cached + 1) % max(1, math.floor(window)) == 0)
        plan.append(actual)
        observations = min(options["history_size"], observations + int(actual))
    return tuple(plan)


def forecast_weights(times, target, *, total_steps, degree, ridge_lambda, blend_weight):
    """Exact upstream linear-form ridge/Taylor weights in chronological order."""
    times = np.asarray(times, np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Forecast observations must be finite, strictly increasing indices")
    def design(t):
        tau = (2.0 * np.asarray(t, np.float64) / total_steps - 1.0).reshape(-1, 1)
        cols = [np.ones_like(tau), tau]
        for _ in range(2, degree + 1):
            cols.append(2 * tau * cols[-1] - cols[-2])
        return np.concatenate(cols[:degree + 1], axis=1)
    x = design(times)
    a = x.T @ x + ridge_lambda * np.eye(x.shape[1], dtype=np.float64)
    c = x @ np.linalg.solve(a, design([target])[0])
    if blend_weight < 1:
        dt = max(times[-1] - times[-2], 1e-8)
        k = (target - times[-1]) / dt
        taylor = np.zeros_like(c)
        taylor[-1], taylor[-2] = 1 + k, -k
        c = (1 - blend_weight) * taylor + blend_weight * c
    return c


@lru_cache(maxsize=1)
def _kernels():
    import jax
    import jax.numpy as jnp
    @jax.jit(donate_argnums=(0,))
    def update(history, feature, slot):
        return jax.lax.dynamic_update_index_in_dim(history, feature, slot, axis=0)
    @jax.jit
    def predict(history, weights):
        # Sequential FP32 accumulation, like upstream out.add_(h, alpha=c).
        # Never cast the whole K-feature history to FP32 or stack a second copy.
        def add(i, acc):
            return acc + history[i].astype(jnp.float32) * weights[i]
        result = jax.lax.fori_loop(0, history.shape[0], add,
                                  jnp.zeros(history.shape[1:], jnp.float32))
        return result.astype(history.dtype)
    return update, predict


class FeatureHistory:
    """Fixed-shape donated ring buffer; preserve DP/TP feature sharding."""
    def __init__(self, capacity, total_steps, options):
        self.capacity, self.total, self.options = capacity, total_steps, options
        self.times, self.slots = [], []
        self.next_slot = 0
        self.buffer = None

    def observe(self, index, feature):
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        if self.times and index <= self.times[-1]:
            raise ValueError("Spectrum observations must advance monotonically")
        if self.buffer is None:
            sharding = feature.sharding
            if isinstance(sharding, NamedSharding):
                sharding = NamedSharding(sharding.mesh, P(None, *sharding.spec))
            self.buffer = jnp.zeros((self.capacity, *feature.shape), feature.dtype, device=sharding)
        if feature.shape != self.buffer.shape[1:] or feature.dtype != self.buffer.dtype:
            raise ValueError("Spectrum feature shape/dtype changed inside a trajectory")
        slot = self.next_slot
        self.buffer = _kernels()[0](self.buffer, jax.lax.stop_gradient(feature), jnp.int32(slot))
        self.next_slot = (slot + 1) % self.capacity
        self.times.append(float(index)); self.slots.append(slot)
        if len(self.times) > self.capacity:
            self.times.pop(0); self.slots.pop(0)

    def predict(self, index):
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P
        if self.buffer is None or len(self.times) < 2:
            raise RuntimeError("Spectrum forecast requested without sufficient observations")
        c = forecast_weights(self.times, index, total_steps=self.total,
            degree=self.options["degree"], ridge_lambda=self.options["ridge_lambda"],
            blend_weight=self.options["blend_weight"])
        weights = np.zeros(self.capacity, np.float32)
        weights[self.slots] = c
        sharding = self.buffer.sharding
        if isinstance(sharding, NamedSharding):
            sharding = NamedSharding(sharding.mesh, P())
        return _kernels()[1](self.buffer, jax.device_put(weights, sharding))


class SpectrumRollout:
    """One immutable-weight batch of trajectories; instantiate anew every batch."""
    def __init__(self, total_steps, options, *, forced_steps=()):
        self.options = dict(options)
        self.plan = actual_step_plan(total_steps, options, forced_steps)
        self.total = total_steps
        self.forced_steps = sorted(set(forced_steps))
        self.capacity = min(options["history_size"], sum(self.plan))
        self.histories = []
        self.steps_seen = 0

    def predict(self, model, adapters, state, sigma, context, *, index, **kwargs):
        if index != self.steps_seen or index >= self.total:
            raise ValueError("Spectrum calls must follow the sampler indices exactly once")
        if self.plan[index]:
            velocity, features = model.predict_with_features(adapters, state, sigma, context, **kwargs)
            if not self.histories:
                self.histories = [FeatureHistory(self.capacity, self.total, self.options) for _ in features]
            if len(features) != len(self.histories):
                raise ValueError("Spectrum CFG branch count changed inside a trajectory")
            for history, feature in zip(self.histories, features):
                history.observe(index, feature)
        else:
            velocity = model.predict_from_features(tuple(h.predict(index) for h in self.histories),
                sigma, state.shape, cfg_scale=kwargs.get("cfg_scale", 1.0))
        self.steps_seen += 1
        return velocity

    def stats(self):
        return dict(enabled=True, method="chebyshev", actual_steps=sum(self.plan),
                    forecast_steps=self.total - sum(self.plan),
                    actual_step_indices=[i for i, actual in enumerate(self.plan) if actual],
                    forced_training_steps=self.forced_steps, history_capacity=self.capacity,
                    feature_history_bytes=sum(h.buffer.nbytes for h in self.histories if h.buffer is not None))
