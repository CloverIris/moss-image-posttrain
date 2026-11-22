import copy
from unittest.mock import patch

import pytest

from moss_art.resolution_mixing import apply_profile
from moss_art.worker import NativeWorker


def config():
    return {
        "sampling": {"height": 768, "width": 768, "group_size": 16,
                     "rollout_batch_size": 1, "steps": 48},
        "training": {"microbatch_size": 1, "rollouts_per_update": 8},
    }


class Pool:
    def __init__(self, c, shared=True):
        self.config = copy.deepcopy(c)
        self.workers = [0, 1, 2, 3]
        self.worker = NativeWorker.__new__(NativeWorker)
        self.worker.c = self.config if shared else copy.deepcopy(self.config)
        self.worker.cache = {"old_replay": True}
        self.result = None

    def call(self, operation, payloads):
        assert operation == "set_sampling_profile"
        self.result = self.worker.set_sampling_profile(**payloads[0])
        return [self.result] * len(self.workers)


@pytest.mark.parametrize("shared", [True, False])
def test_resolution_change_is_detected_before_supervisor_config_commit(shared):
    c = config()
    pool = Pool(c, shared=shared)
    profile = {"height": 1024, "width": 1024, "weight": 1,
               "group_size": 16, "rollout_batch_size": 2,
               "microbatch_size": 2, "rollouts_per_update": 12}
    with patch("jax.clear_caches") as clear:
        label = apply_profile(c, pool, profile)
    assert label == "1024x1024"
    assert pool.result["shape_changed"] is True
    clear.assert_called_once_with()
    assert pool.worker.cache is None
    assert c == pool.config
    assert c["sampling"]["height"] == pool.worker.c["sampling"]["height"] == 1024
    assert c["sampling"]["steps"] == 48
    assert pool.worker.c["training"]["microbatch_size"] == 2
    assert c["training"]["rollouts_per_update"] == 12


def test_same_shape_preserves_compiled_executables():
    c = config()
    pool = Pool(c)
    with patch("jax.clear_caches") as clear:
        assert apply_profile(c, pool, {"height": 768, "width": 768}) == "768x768"
    assert pool.result["shape_changed"] is False
    clear.assert_not_called()
    assert pool.worker.cache is None


def test_failed_worker_call_does_not_commit_supervisor_configuration():
    c = config()
    pool = Pool(c, shared=False)
    before_c, before_pool = copy.deepcopy(c), copy.deepcopy(pool.config)
    with patch.object(pool, "call", side_effect=RuntimeError("worker unavailable")):
        with pytest.raises(RuntimeError, match="worker unavailable"):
            apply_profile(c, pool, {"height": 1024, "width": 1024})
    assert c == before_c
    assert pool.config == before_pool
