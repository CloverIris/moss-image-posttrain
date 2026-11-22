"""CPU-only fixed-scale reward fusion and strict ID joins."""
import math

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


def fuse(raw_scores, calibration, artifact_weight=1., *, weights=None):
    weights = calibration['weights'] if weights is None else weights
    if set(weights) != set(MODELS) or any(not finite(x) or x < 0 for x in weights.values()):
        raise ValueError('Require finite nonnegative Kawai/DeepGHS/HPS weights')
    if not math.isclose(math.fsum(weights.values()), 1., abs_tol=1e-9):
        raise ValueError('Aesthetic weights must sum to one')
    if not finite(artifact_weight) or artifact_weight < 0:
        raise ValueError('artifact_weight must be finite and nonnegative')
    if any(not finite(raw_scores.get(k)) for k in (*MODELS, 'sdg_artifact')):
        raise ValueError('All four raw reward components must be finite')
    gain, offset = calibration['output_gain'], calibration['output_offset']
    if not finite(gain) or gain <= 0 or not finite(offset):
        raise ValueError('Invalid frozen output gain/offset')
    normalized, contributions = {}, {}
    for name in MODELS:
        params = calibration['parameters'][name]
        if not finite(params['scale']) or params['scale'] <= 0 or not finite(params['mean']):
            raise ValueError(f'Invalid fixed scale for {name}')
        normalized[name] = (raw_scores[name]-params['mean'])/params['scale']
        contributions[name] = gain*weights[name]*normalized[name]
    aesthetic = offset + math.fsum(contributions.values())
    contributions['sdg_artifact'] = artifact_weight*raw_scores['sdg_artifact']
    reward = aesthetic + contributions['sdg_artifact']
    if not math.isfinite(reward):
        raise ValueError('Nonfinite fused reward')
    return dict(reward=reward, aesthetic_reward=aesthetic,
                artifact_penalty=raw_scores['sdg_artifact'],
                normalized_scores=normalized, contributions=contributions,
                fusion_weights=dict(weights), artifact_weight=artifact_weight,
                output_offset=offset, output_gain=gain)
