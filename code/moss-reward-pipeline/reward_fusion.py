"""CPU-only fixed-scale reward fusion and strict ID joins."""
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


def aesthetic_models(kawai_enabled=True):
    """The active aesthetic heads; SDG remains an independent defect term."""
    if type(kawai_enabled) is not bool:
        raise ValueError('kawai_enabled must be boolean')
    return MODELS if kawai_enabled else ('deepghs', 'hps')


def resolve_weights(calibration, *, weights=None, kawai_enabled=True):
    """Keep frozen scales while optionally renormalizing enabled head weights.

    Explicit weights must contain precisely the active aesthetic model names.
    When Kawai is disabled without an override, preserve the frozen D:H ratio.
    No Kawai calibration parameter or weight value is needed in that mode.
    """
    models = aesthetic_models(kawai_enabled)
    if weights is None:
        source = calibration['weights']
        if not isinstance(source, Mapping):
            raise ValueError('Calibration weights must be a mapping')
        if kawai_enabled:
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
         sdg_enabled=True, kawai_enabled=True):
    models = aesthetic_models(kawai_enabled)
    weights = resolve_weights(calibration, weights=weights, kawai_enabled=kawai_enabled)
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
    if not kawai_enabled and 'kawai' in raw_scores:
        raise ValueError('Disabled Kawai must not report an aesthetic score')
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
                enabled_models=list(models), kawai_enabled=kawai_enabled)
    if sdg_enabled:
        result['artifact_penalty'] = raw_scores['sdg_artifact']
    else:
        result['sdg_enabled'] = False
    return result
