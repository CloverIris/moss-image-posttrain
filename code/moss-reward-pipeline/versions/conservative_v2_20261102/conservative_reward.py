"""Conservative anime reward prototype; no models, image IO, or training jobs.

This is a fusion kernel, NOT a trained style/defect detector. All thresholds,
coverage weights, and rule reliabilities require frozen, held-out calibration.
Unknown guards abstain. Two biased aesthetic heads can still agree incorrectly.

Integration requirements for the audited MOSS trainer:
  1. Keep original prompts and image-ID joins unchanged. Store rules separately.
  2. Use native PixAI IDs/names from the exact pinned catalog. Do not invent aliases.
     Supply unfiltered target-tag probabilities; missing is unknown, never zero.
  3. Replace BOTH rank stages. Do not rank/whiten these rewards afterwards.
  4. Thread art_covariance_kwargs() through BOTH stage_pipeline.prepare_training
     and worker.prepare_update, including config validation. Setting JSON alone
     does not change those call sites in the currently audited trainer.
  5. Enable/calibrate the existing frozen-Base reference-MSE loss independently.
     This is not an image-space similarity penalty or true policy KL.
  6. Keep all historical runs immutable; training launch needs separate approval.

Run tests: python -m unittest discover -s moss_reward_v2 -v
The toy values in tests are not production calibration recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import copysign, fsum, isfinite, tanh
from typing import Mapping, Sequence


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _unit(value: float, name: str) -> float:
    value = _finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class HeadCalibration:
    """Raw-score units; deadband from ties/repeats, temperature from held-out pairs."""

    deadband: float
    temperature: float

    def __post_init__(self) -> None:
        if _finite(self.deadband, "deadband") < 0:
            raise ValueError("deadband must be nonnegative")
        if _finite(self.temperature, "temperature") <= 0:
            raise ValueError("temperature must be positive")

    def preference(self, delta: float) -> float:
        delta = _finite(delta, "score difference")
        magnitude = max(abs(delta) - self.deadband, 0.0)
        return copysign(tanh(magnitude / self.temperature), delta)


@dataclass(frozen=True)
class Guard:
    """Empirically validated constraint result, not a tagger's raw probability."""

    status: str
    reliability: float
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail", "unknown"}:
            raise ValueError("invalid guard status")
        _unit(self.reliability, "guard reliability")
        if not self.reason:
            raise ValueError("guard evidence/reason is required")


@dataclass(frozen=True)
class TagRule:
    """One explicit target, with two calibrated thresholds and an uncertainty band.

    The same native tag can be required present OR absent depending on the prompt.
    There is intentionally no global blacklist for gloss, detail, or oil painting.
    Unsupported requests and artist/style nuances need other evidence or abstain.
    """

    category: str
    tag: str
    desired_present: bool
    absent_at: float
    present_at: float
    reliability: float

    def __post_init__(self) -> None:
        if not self.category or not self.tag:
            raise ValueError("native category and tag are required")
        if type(self.desired_present) is not bool:
            raise ValueError("desired_present must be bool")
        if _unit(self.absent_at, "absent_at") >= _unit(self.present_at, "present_at"):
            raise ValueError("absent_at must be below present_at")
        _unit(self.reliability, "rule reliability")


def catalog_from_native_config(
    tags: Sequence[str], tags_split: Sequence[tuple[str, int]],
) -> dict[str, set[str]]:
    """Mirror pinned PixAI config.tags/config.tags_split without renaming labels."""
    catalog: dict[str, set[str]] = {}
    offset = 0
    for category, count in tags_split:
        if not isinstance(category, str) or not category or category in catalog:
            raise ValueError("missing or duplicate category")
        if type(count) is not int or count < 0 or offset + count > len(tags):
            raise ValueError("invalid native category length")
        names = tags[offset:offset + count]
        if any(not isinstance(tag, str) or not tag for tag in names):
            raise ValueError("invalid native tag name")
        if len(set(names)) != count:
            raise ValueError("duplicate native tag")
        catalog[category] = set(names)
        offset += count
    if offset != len(tags):
        raise ValueError("category lengths do not cover the native vocabulary")
    return catalog


def guard_from_native_tags(
    probabilities: Mapping[str, Mapping[str, float]],
    rules: Sequence[TagRule],
    catalog: Mapping[str, set[str]],
    *,
    min_reliability: float,
) -> Guard:
    """Check a separately curated rule sidecar; never parse or rewrite prompts.

    All rules must pass before aesthetics can act. One reliable violation is
    sufficient to fail. Missing/filtered scores, unknown labels, low reliability,
    and probabilities inside the uncertainty band are not interpreted as failure.
    Category presence alone does NOT certify all important prompt constraints.
    """
    threshold = _unit(min_reliability, "min_reliability")
    if threshold == 0:
        raise ValueError("min_reliability must be positive")
    if not rules:
        return Guard("unknown", 0.0, "no audited target rules")
    seen: set[tuple[str, str]] = set()
    outcomes: list[Guard] = []
    for rule in rules:
        key = (rule.category, rule.tag)
        if key in seen:
            raise ValueError("duplicate or contradictory tag rule")
        seen.add(key)
        reason = f"{rule.category}:{rule.tag}"
        if rule.tag not in catalog.get(rule.category, set()):
            outcomes.append(Guard("unknown", 0.0, "unsupported tag " + reason))
            continue
        probability = probabilities.get(rule.category, {}).get(rule.tag)
        if probability is None:
            outcomes.append(Guard("unknown", 0.0, "missing probability " + reason))
            continue
        probability = _unit(probability, reason)
        if rule.reliability < threshold:
            outcomes.append(Guard("unknown", 0.0, "uncalibrated rule " + reason))
            continue
        if probability <= rule.absent_at:
            present = False
        elif probability >= rule.present_at:
            present = True
        else:
            outcomes.append(Guard("unknown", 0.0, "uncertain probability " + reason))
            continue
        status = "pass" if present == rule.desired_present else "fail"
        outcomes.append(Guard(status, rule.reliability, reason))
    failures = [outcome for outcome in outcomes if outcome.status == "fail"]
    if failures:
        return max(failures, key=lambda outcome: outcome.reliability)
    unknown = next((outcome for outcome in outcomes if outcome.status == "unknown"), None)
    if unknown is not None:
        return unknown
    return Guard("pass", min(outcome.reliability for outcome in outcomes), "all audited targets pass")


@dataclass(frozen=True)
class Candidate:
    id: str
    prompt_id: str
    kawai: float
    deepghs: float
    guard: Guard

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not isinstance(self.prompt_id, str) or not self.id or not self.prompt_id:
            raise ValueError("image ID and prompt ID are required")
        _finite(self.kawai, "Kawai raw score")
        _finite(self.deepghs, "DeepGHS percentile")
        if not -1 <= self.kawai <= 1 or not 0 <= self.deepghs <= 1:
            raise ValueError("expected Kawai weighted [-1,1] and DeepGHS percentile [0,1]")


@dataclass(frozen=True)
class RewardPolicy:
    calibration_id: str
    kawai: HeadCalibration
    deepghs: HeadCalibration
    aesthetic_coverage: float
    min_guard_reliability: float
    aesthetic_cap: float
    expected_group_size: int

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("frozen calibration/coverage revision is required")
        _unit(self.aesthetic_coverage, "aesthetic coverage")
        reliability = _unit(self.min_guard_reliability, "min_guard_reliability")
        cap = _unit(self.aesthetic_cap, "aesthetic_cap")
        if reliability == 0 or cap > reliability:
            raise ValueError("require 0 < min_guard_reliability and aesthetic_cap <= it")
        if type(self.expected_group_size) is not int or self.expected_group_size < 2:
            raise ValueError("a complete group needs at least two images")


@dataclass(frozen=True)
class RewardResult:
    ids: tuple[str, ...]
    rewards: tuple[float, ...]
    reward_mad: float
    pair_votes: tuple[tuple[float, ...], ...]
    aesthetic_pairs: int
    constraint_pairs: int
    abstained_pairs: int
    signal_active: bool
    calibration_id: str


def fuse_group(candidates: Sequence[Candidate], policy: RewardPolicy) -> RewardResult:
    """Bounded antisymmetric pair votes -> zero-mean scalar rewards, without rank.

    Reliable pass beats reliable fail regardless of aesthetic score. Aesthetics
    operates only between passes, only when both heads agree outside deadbands.
    Unknown pairs and fail-vs-fail abstain. All-abstain groups return exact zeros.
    Coverage=0 disables aesthetics, NOT validated prompt-constraint supervision.

    The aesthetic cap <= reliability floor guarantees every clear pass has a
    higher scalar reward than every clear fail in a mixed group, even if that
    pass loses all other aesthetic comparisons. No such guarantee is made for
    truthfulness of the supplied guards; that requires real calibration.
    """
    count = len(candidates)
    if count != policy.expected_group_size:
        raise ValueError("partial or incorrectly sized prompt group")
    if len({candidate.id for candidate in candidates}) != count:
        raise ValueError("duplicate image ID")
    if len({candidate.prompt_id for candidate in candidates}) != 1:
        raise ValueError("mixed prompt groups")
    statuses = [candidate.guard.status if candidate.guard.reliability >= policy.min_guard_reliability
                else "unknown" for candidate in candidates]
    matrix = [[0.0] * count for _ in candidates]
    aesthetic_pairs = constraint_pairs = abstained_pairs = 0
    for i in range(count):
        for j in range(i + 1, count):
            left, right = candidates[i], candidates[j]
            a, b = statuses[i], statuses[j]
            confidence = min(left.guard.reliability, right.guard.reliability)
            vote = 0.0
            if {a, b} == {"pass", "fail"}:
                vote = confidence if a == "pass" else -confidence
                constraint_pairs += 1
            elif a == b == "pass" and policy.aesthetic_coverage > 0:
                k = policy.kawai.preference(left.kawai - right.kawai)
                d = policy.deepghs.preference(left.deepghs - right.deepghs)
                if (k > 0 and d > 0) or (k < 0 and d < 0):
                    vote = (policy.aesthetic_cap * policy.aesthetic_coverage * confidence
                            * copysign(min(abs(k), abs(d)), k))
                    if vote != 0:aesthetic_pairs += 1
            if vote == 0:
                abstained_pairs += 1
            matrix[i][j], matrix[j][i] = vote, -vote
    rewards = tuple(fsum(row) / (count - 1) for row in matrix)
    center = fsum(rewards) / count
    mad = fsum(abs(reward - center) for reward in rewards) / count
    return RewardResult(
        ids=tuple(candidate.id for candidate in candidates), rewards=rewards,
        reward_mad=mad, pair_votes=tuple(tuple(row) for row in matrix),
        aesthetic_pairs=aesthetic_pairs, constraint_pairs=constraint_pairs,
        abstained_pairs=abstained_pairs, signal_active=mad > 0,
        calibration_id=policy.calibration_id,
    )


def art_covariance_kwargs(*, guidance_gain: float) -> dict[str, object]:
    """Explicit experimental ART variant: retain reward-MAD scale, not unit scale.

    Reward amplitude matters only if the live learner actually uses these kwargs.
    Group-active in ART also depends on nonzero endpoint covariance, not MAD alone.
    This changes the historical ART guidance-scale rule and requires an ablation.
    """
    gain = _finite(guidance_gain, "guidance_gain")
    if gain <= 0:
        raise ValueError("guidance_gain must be positive")
    return dict(mode="scalar", scale_mode="reward_mad", guidance_gain=gain,
                zero_signal_mode="exact", zero_signal_atol=0.0)


def validate_training_contract(
    actual_covariance_options: Mapping[str, object], *, reference_mse_beta: float,
) -> None:
    """Call on ACTUAL passed arguments, not merely intended JSON configuration."""
    if actual_covariance_options.get("scale_mode") != "reward_mad":
        raise ValueError("unit normalization would erase reward confidence")
    if actual_covariance_options.get("mode") != "scalar":
        raise ValueError("do not convert these scalar votes to hard pairwise labels")
    if actual_covariance_options.get("zero_signal_mode") != "exact":
        raise ValueError("exact zero-signal handling is required")
    if _finite(actual_covariance_options.get("zero_signal_atol"), "zero_signal_atol") != 0:
        raise ValueError("exact zero-signal mode requires zero atol")
    if _finite(actual_covariance_options.get("guidance_gain"), "guidance_gain") <= 0:
        raise ValueError("guidance_gain must be positive")
    if _finite(reference_mse_beta, "reference_mse_beta") <= 0:
        raise ValueError("a calibrated frozen-Base regularizer is required")
