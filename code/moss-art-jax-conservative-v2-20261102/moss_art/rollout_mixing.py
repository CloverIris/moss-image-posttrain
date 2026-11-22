"""Stable per-prompt sampler assignment and prompt-balanced collection quotas.

Imports no accelerator framework. Prompt identity uses unmodified conditioning
text, never row ID, sample seed, device, optimizer update or resolution.
"""
import hashlib
import json
import math
import numpy as np

DEFAULTS = dict(enabled=False, full_dit_weight=1.0, spectrum_weight=1.0)
MODES = ("full_dit", "spectrum_default")
ROW_FIELDS = ("sampling_methods", "sampling_prompt_keys", "dit_full_steps", "dit_forecast_steps")


def enabled(c):
    return c["sampling"].get("rollout_mixing", {}).get("enabled", False)


def validate_mixing(c):
    options = c["sampling"]["rollout_mixing"]
    weights = [options[key] for key in ("full_dit_weight", "spectrum_weight")]
    if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("sampling.rollout_mixing weights must be finite and nonnegative")
    if not math.isfinite(sum(weights)) or sum(weights) <= 0:
        raise ValueError("sampling.rollout_mixing weights must have a finite positive sum")
    if options["enabled"] and c["sampling"]["spectrum"]["blend_weight"] != 0.3:
        raise ValueError("rollout_mixing uses the default blend: spectrum.blend_weight must be 0.3")


def prompt_key(record):
    body = json.dumps([record["prompt"], record.get("negative_prompt", "")],
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _uniform(key):
    return (int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") >> 11) / 2**53


def mode_for_record(c, record):
    if enabled(c):
        o = c["sampling"]["rollout_mixing"]
        fraction = o["full_dit_weight"] / (o["full_dit_weight"] + o["spectrum_weight"])
        value = _uniform(f'prompt-sampler:{c["training"]["seed"]}:' + prompt_key(record))
        return MODES[0] if value < fraction else MODES[1]
    spectrum = c["sampling"].get("spectrum", {})
    if not spectrum.get("enabled", False): return "full_dit"
    blend = spectrum.get("blend_weight", 0.3)
    return "spectrum_default" if blend == .3 else "chebyshev_pure" if blend == 1.0 else "spectrum_custom"


def options_for_record(c, record):
    from .spectrum import DEFAULTS as SPECTRUM_DEFAULTS
    options = dict(c["sampling"].get("spectrum", SPECTRUM_DEFAULTS))
    mode = mode_for_record(c, record)
    options["enabled"] = mode != "full_dit"
    return mode, options


def _buckets(c, records):
    buckets = {mode: [] for mode in MODES}
    for row in records: buckets[mode_for_record(c, row)].append(row)
    o = c["sampling"]["rollout_mixing"]
    requested = dict(zip(MODES, (o["full_dit_weight"], o["spectrum_weight"])))
    active = [mode for mode in MODES if buckets[mode] and requested[mode] > 0]
    total = sum(requested[mode] for mode in active)
    return buckets, active, {mode: requested[mode]/total for mode in active}


def ensure_pool_capacity(c, records, count):
    buckets, active, weights = _buckets(c, records)
    if not active or len(records) < count:
        raise ValueError("Not enough distinct prompts for rollout mixing")
    for mode in active:
        needed = math.ceil(count * weights[mode] - 1e-12)
        if len(buckets[mode]) < needed:
            raise ValueError(f"rollout_mixing needs at least {needed} distinct {mode} prompts in each active resolution bucket; got {len(buckets[mode])}")


def select_balanced(c, records, count, seed, offset):
    """Exact integer quotas when divisible; unbiased systematic rounding otherwise.

    Fixed prompt buckets keep a repeated prompt on the same sampler. Selection
    is deterministic at a resumable prompt cursor, with no additional RNG state.
    Empty buckets are excluded (a one-prompt overfit run has only one sampler).
    """
    ensure_pool_capacity(c, records, count)
    buckets, active, weights = _buckets(c, records)
    shift = _uniform(f"prompt-sampler-quota:{seed}:{offset}")
    positions = (np.arange(count, dtype=np.float64) + shift) / count
    cumulative = np.cumsum([weights[mode] for mode in active])
    assignments = np.searchsorted(cumulative, positions, side="right")
    rng = np.random.default_rng((seed + offset) % 2**32)
    selected = []
    for i, mode in enumerate(active):
        n = int(np.count_nonzero(assignments == i))
        selected.extend(buckets[mode][int(j)] for j in rng.choice(len(buckets[mode]), size=n, replace=False))
    return [selected[int(j)] for j in rng.permutation(count)]


def assignment_manifest(c, records):
    rows = [dict(sample_id=r["sample_id"], prompt_key=prompt_key(r),
                 sampling_method=mode_for_record(c, r)) for r in records]
    unique = {r["prompt_key"]: r["sampling_method"] for r in rows}
    return dict(seed=c["training"]["seed"], weights=dict(c["sampling"]["rollout_mixing"]),
        unique_prompt_counts={mode: sum(value == mode for value in unique.values()) for mode in MODES},
        records=rows)


def collection_stats(outputs, rewards):
    """Per-method reward diagnostics in the same row order as reward_round."""
    methods = sum([list(o["sampling_methods"]) for o in outputs], [])
    keys = sum([list(o["sampling_prompt_keys"]) for o in outputs], [])
    if len(methods) != len(rewards) or len(keys) != len(rewards):
        raise ValueError("Sampler row metadata does not match reward order")
    result = {}
    rewards = np.asarray(rewards, np.float32)
    for mode in sorted(set(methods)):
        mask = np.asarray([m == mode for m in methods])
        values = rewards[mask]
        result[mode] = dict(images=int(mask.sum()),
            prompts=len({k for k, keep in zip(keys, mask) if keep}),
            reward_mean=float(values.mean()), reward_std=float(values.std()),
            reward_min=float(values.min()), reward_max=float(values.max()))
    return result
