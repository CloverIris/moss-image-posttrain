"""Artifact-only SDG penalty, using the official pixel reward implementation.

No model is loaded. misalignment is retained for logging but never scored.
The returned penalty is the mean of the official spatial penalty map.
"""
from functools import lru_cache
import importlib.util
import math
from pathlib import Path

from sdg_output import parse_prediction, validate_boxes


@lru_cache(maxsize=4)
def _official_map(repo):
    path = Path(repo).resolve() / 'boxflow_grpo/src/flow_factory/utils/pixel_reward_map.py'
    spec = importlib.util.spec_from_file_location('sdg_official_artifact_pixel_map', path)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot load the official SDG pixel reward map')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_pixel_reward_map


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


class ArtifactScorer:
    def __init__(self, repo, *, alpha_artifact=.5, latent_size=64, penalty_weight=1.):
        if not _number(alpha_artifact) or alpha_artifact < 0:
            raise ValueError('alpha_artifact must be finite and nonnegative')
        if type(latent_size) is not int or latent_size <= 0:
            raise ValueError('latent_size must be a positive integer')
        if not _number(penalty_weight) or penalty_weight < 0:
            raise ValueError('penalty_weight must be finite and nonnegative')
        self.repo = str(Path(repo).resolve())
        self.alpha_artifact = float(alpha_artifact)
        self.latent_size = latent_size
        self.penalty_weight = float(penalty_weight)

    def score_boxes(self, boxes, *, base_reward=0., image_size=1024):
        boxes = validate_boxes(boxes)
        if not _number(base_reward):
            raise ValueError('base_reward must be finite')
        if type(image_size) is not int or image_size <= 0:
            raise ValueError('image_size must be a positive integer')
        artifacts = [box for box in boxes if box['label'] == 'artifact']
        # Filter before calling upstream, and also pass alpha_misalignment=0.
        # An empty valid answer and a misalignment-only answer both score zero.
        if artifacts:
            spatial_map = _official_map(self.repo)(
                bboxes=artifacts, scalar_reward=0., latent_size=self.latent_size,
                image_size=image_size, alpha_artifact=self.alpha_artifact,
                alpha_misalignment=0., importance_weighting=True,
                min_penalty=0., stack_same_label=False,
                area_normalize_artifact=False, area_normalize_misalignment=False,
            )
            penalty = float(spatial_map.mean())
        else:
            penalty = 0.
        weighted = self.penalty_weight * penalty
        reward = float(base_reward) + weighted
        if not math.isfinite(reward):
            raise ValueError('nonfinite_artifact_reward')
        return dict(reward=reward, base_reward=float(base_reward), artifact_penalty=penalty,
                    weighted_artifact_penalty=weighted, artifact_count=len(artifacts),
                    ignored_misalignment_count=len(boxes)-len(artifacts),
                    mode='artifact_only', alpha_artifact=self.alpha_artifact,
                    alpha_misalignment=0., latent_size=self.latent_size,
                    penalty_weight=self.penalty_weight)

    def score_prediction(self, prediction, *, base_reward=0., image_size=None):
        if prediction.get('status', 'ok') != 'ok':
            raise ValueError('invalid_source_prediction')
        if prediction.get('finish_reason', 'stop') != 'stop':
            raise ValueError('truncated_or_unfinished_generation')
        if 'raw_response' in prediction:
            if 'finish_reason' not in prediction:
                raise ValueError('missing_generation_finish_reason')
            boxes = parse_prediction(prediction['raw_response'], prediction['finish_reason'])
        elif 'boxes' in prediction:
            boxes = validate_boxes(prediction['boxes'])
        else:
            raise ValueError('missing_sdg_prediction')
        result = self.score_boxes(boxes, base_reward=base_reward,
                                  image_size=prediction.get('resolution', 1024) if image_size is None else image_size)
        result['sdg_boxes'] = boxes  # Preserve ignored diagnoses for audit.
        return result
