"""CPU-only fixed-scale or group-rank reward fusion and strict ID joins."""
import math
from collections.abc import Mapping

MODELS = ('kawai', 'deepghs', 'hps')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def index_items(rows, expected_ids, name):
    output = {}
    for row in rows:
        key = row.get('id')
        if key in output:
            raise ValueError(f'{name}: duplicate image id {key}')
        output[key] = row
    if set(output) != set(expected_ids):
        raise ValueError(f'{name}: image IDs do not match the request')
    return output


def aesthetic_models(kawai_enabled=True, enabled_models=None):
    """Resolve explicit active heads, with unchanged legacy Kawai-switch defaults.

    An explicit model list is authoritative. Callers with both a manifest and
    legacy flags validate their consistency at the config/metadata boundary.
    """
    if type(kawai_enabled) is not bool:
        raise ValueError('kawai_enabled must be boolean')
    if enabled_models is None:
        return MODELS if kawai_enabled else ('deepghs', 'hps')
    if (not isinstance(enabled_models, (list, tuple)) or not enabled_models or
            any(type(name) is not str or name not in MODELS for name in enabled_models) or
            len(set(enabled_models)) != len(enabled_models)):
        raise ValueError('enabled_models must contain unique known aesthetic model names')
    return tuple(name for name in MODELS if name in enabled_models)


def resolve_weights(calibration, *, weights=None, kawai_enabled=True, enabled_models=None):
    """Keep frozen scales; explicit weights must exactly match the active heads."""
    models = aesthetic_models(kawai_enabled, enabled_models)
    if weights is None:
        source = calibration['weights']
        if not isinstance(source, Mapping):
            raise ValueError('Calibration weights must be a mapping')
        if enabled_models is None and kawai_enabled:
            # Preserve legacy full-model validation, including sum-to-one.
            weights = source
        else:
            try:
                subset = {name: source[name] for name in models}
            except KeyError as exc:
                raise ValueError('Missing enabled aesthetic weight') from exc
            if any(not finite(x) or x < 0 for x in subset.values()):
                raise ValueError('Enabled aesthetic weights must be finite and nonnegative')
            total = math.fsum(subset.values())
            if total <= 0:
                raise ValueError('Enabled aesthetic weights must have a positive sum')
            weights = {name: value / total for name, value in subset.items()}
    if (not isinstance(weights, Mapping) or set(weights) != set(models) or
            any(not finite(x) or x < 0 for x in weights.values())):
        raise ValueError('Require finite nonnegative weights for exactly the enabled aesthetic models')
    if not math.isclose(math.fsum(weights.values()), 1., abs_tol=1e-9):
        raise ValueError('Aesthetic weights must sum to one')
    return dict(weights)


def fuse(raw_scores, calibration, artifact_weight=1., *, weights=None,
         sdg_enabled=True, kawai_enabled=True, enabled_models=None):
    models = aesthetic_models(kawai_enabled, enabled_models)
    weights = resolve_weights(calibration, weights=weights, kawai_enabled=kawai_enabled,
                              enabled_models=enabled_models)
    if not finite(artifact_weight) or artifact_weight < 0:
        raise ValueError('artifact_weight must be finite and nonnegative')
    if type(sdg_enabled) is not bool:
        raise ValueError('sdg_enabled must be boolean')
    if not sdg_enabled and artifact_weight != 0:
        raise ValueError('Disabled SDG requires artifact_weight=0')
    required = (*models, 'sdg_artifact') if sdg_enabled else models
    if any(not finite(raw_scores.get(k)) for k in required):
        raise ValueError('All enabled raw reward components must be finite')
    if not sdg_enabled and 'sdg_artifact' in raw_scores:
        raise ValueError('Disabled SDG must not report an artifact score')
    if (set(raw_scores) & set(MODELS)) - set(models):
        raise ValueError('Disabled aesthetic heads must not report scores')
    gain, offset = calibration['output_gain'], calibration['output_offset']
    if not finite(gain) or gain <= 0 or not finite(offset):
        raise ValueError('Invalid frozen output gain/offset')
    normalized, contributions = {}, {}
    for name in models:
        params = calibration['parameters'][name]
        if not finite(params['scale']) or params['scale'] <= 0 or not finite(params['mean']):
            raise ValueError(f'Invalid fixed scale for {name}')
        normalized[name] = (raw_scores[name]-params['mean'])/params['scale']
        contributions[name] = gain*weights[name]*normalized[name]
    aesthetic = offset + math.fsum(contributions.values())
    if sdg_enabled:
        contributions['sdg_artifact'] = artifact_weight*raw_scores['sdg_artifact']
    reward = aesthetic + contributions.get('sdg_artifact', 0.)
    if not math.isfinite(reward):
        raise ValueError('Nonfinite fused reward')
    result = dict(reward=reward, aesthetic_reward=aesthetic,
                normalized_scores=normalized, contributions=contributions,
                fusion_weights=dict(weights), artifact_weight=artifact_weight,
                output_offset=offset, output_gain=gain,
                enabled_models=list(models), kawai_enabled="kawai" in models)
    if sdg_enabled:
        result['artifact_penalty'] = raw_scores['sdg_artifact']
    else:
        result['sdg_enabled'] = False
    return result


def rank_percentiles(values):
    """Ascending average-tie ranks mapped to [0, 1], preserving input order.

    Exact equal scores share their average rank; an all-equal group maps to .5.
    No epsilon is applied to raw scores, so this does not invent a confidence
    threshold or silently merge legitimate small score differences.
    """
    values = list(values)
    if len(values) < 2 or any(not finite(value) for value in values):
        raise ValueError('Ranking requires at least two finite scores')
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [None] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        percentile = (start + end - 1) / (2 * (len(values) - 1))
        for index in order[start:end]:
            output[index] = percentile
        start = end
    return output


def fuse_group(rows, calibration, *, gamma=.5, weights=None,
               kawai_enabled=True, expected_group_size=8, enabled_models=None):
    """Rank each model, apply a common power, fuse, then rank the fusion.

    This API intentionally consumes exactly ONE complete prompt group. The
    caller must separate training and validation domains and pass the correct
    expected_group_size for each (e.g. train K8, fixed validation K4). It must
    not use unprefixed/original prompt IDs to combine distinct evaluation jobs.

    `reward` and `aesthetic_reward` become the final linear rank reward.
    `rank_reward` retains that value if a validation caller restores the frozen
    absolute reward for monitoring. `normalized_scores` and `contributions`
    remain the original frozen-calibration diagnostics; they are explicitly NOT
    an additive decomposition of the rank reward. Rank contributions add to
    rank_fused_score only, before the final rank operation.

    Only aesthetic heads are supported. SDG is deliberately rejected rather
    than silently changing where an artifact penalty enters this new objective.
    No input row or calibration is modified.
    """
    if type(expected_group_size) is not int or expected_group_size < 2:
        raise ValueError('expected_group_size must be an integer >= 2')
    if not finite(gamma) or not 0 < gamma <= 1:
        raise ValueError('rank power gamma must satisfy 0 < gamma <= 1')
    rows = list(rows)
    if len(rows) != expected_group_size:
        raise ValueError('Rank fusion requires exactly one complete prompt group')
    models = aesthetic_models(kawai_enabled, enabled_models)
    weights = resolve_weights(calibration, weights=weights,
                              kawai_enabled=kawai_enabled, enabled_models=enabled_models)
    ids, prompt_ids, domains = set(), set(), set()
    output = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError('Rank fusion rows must be mappings')
        image_id, prompt_id = row.get('id'), row.get('prompt_id')
        if (type(image_id) not in (str, int) or image_id == '' or
                type(prompt_id) not in (str, int) or prompt_id == ''):
            raise ValueError('Rank fusion rows require image and prompt IDs')
        if image_id in ids:
            raise ValueError('Rank fusion has duplicate image IDs')
        ids.add(image_id)
        prompt_ids.add((type(prompt_id), prompt_id))
        # Presence is part of the key: a missing domain must not merge with an
        # explicitly tagged domain, even if the caller reused a prompt ID.
        domain = []
        for key in ('domain', 'evaluation_domain', 'split'):
            value = row.get(key)
            if value is not None and type(value) not in (str, int):
                raise ValueError('Rank fusion domain labels must be scalar IDs')
            domain.append((key, key in row, type(value), value))
        domains.add(tuple(domain))
        raw = row.get('raw_scores')
        if not isinstance(raw, Mapping) or set(raw) != set(models):
            raise ValueError('Rank fusion requires exactly the enabled raw aesthetic scores; SDG is unsupported')
        frozen = fuse(raw, calibration, 0., weights=weights,
                      sdg_enabled=False, kawai_enabled=kawai_enabled, enabled_models=enabled_models)
        item = dict(row)
        item['raw_scores'] = dict(raw)
        item.update(frozen)
        item.update(frozen_reward=frozen['reward'],
                    frozen_aesthetic_reward=frozen['aesthetic_reward'],
                    fusion_mode='rank_power_rank', rank_power_gamma=float(gamma),
                    rank_group_size=expected_group_size,
                    contributions_space='frozen_calibration',
                    model_rank_percentiles={}, rank_component_scores={},
                    weighted_rank_contributions={})
        output.append(item)
    if len(prompt_ids) != 1 or len(domains) != 1:
        raise ValueError('Rank fusion must not mix prompt groups or evaluation domains')
    for model in models:
        percentiles = rank_percentiles([row['raw_scores'][model] for row in output])
        for item, percentile in zip(output, percentiles):
            powered = percentile ** gamma
            item['model_rank_percentiles'][model] = percentile
            item['rank_component_scores'][model] = powered
            item['weighted_rank_contributions'][model] = weights[model] * powered
    mixed = [math.fsum(item['weighted_rank_contributions'].values()) for item in output]
    final = rank_percentiles(mixed)
    for item, mixed_score, rank_score in zip(output, mixed, final):
        item.update(rank_fused_score=mixed_score, final_rank_score=rank_score,
                    rank_reward=rank_score, reward=rank_score,
                    aesthetic_reward=rank_score)
    return output
