"""Regression tests for host embedding-cache bounds and active context lifetime."""
from collections import OrderedDict
from types import SimpleNamespace
import jax
import numpy as np

from moss_art.worker import NativeWorker


def bare_worker(kind):
    worker = NativeWorker.__new__(NativeWorker)
    worker.c = {"model": {"kind": kind, "path": "/unused"},
                "data": {"cache": "/unused"}}
    worker.model = SimpleNamespace(config={"text_dim": 24})
    worker.device = jax.devices("cpu")[0]
    worker.text_cache = OrderedDict()
    worker._prompt_cache = None
    worker._compiled_text_lengths = set()
    return worker


def record(index):
    return {"sample_id": str(index), "prompt": "prompt " + str(index)}


def test_embedding_cache_is_lru_and_preserves_active_context():
    worker = bare_worker("tiny")
    active, _ = worker._conditioning(record(0))
    original = np.asarray(active).copy()
    for i in range(1, 32):
        worker._conditioning(record(i))
    worker._conditioning(record(0))  # refresh recency
    worker._conditioning(record(32))
    assert len(worker.text_cache) == 32
    assert "0" in worker.text_cache and "1" not in worker.text_cache
    for i in range(33, 70):
        worker._conditioning(record(i))
    assert len(worker.text_cache) == 32 and "0" not in worker.text_cache
    np.testing.assert_array_equal(active, original)


def test_native_prompt_cache_identity_is_checked_once(monkeypatch):
    instances = []
    class FakePromptCache:
        def __init__(self, *args, **kwargs):
            instances.append(self)
            self.loads = 0
        def load(self, record):
            self.loads += 1
            return {"positive": np.ones((1, 2, 24), np.float32),
                    "negative": np.zeros((1, 1, 24), np.float32)}
    monkeypatch.setattr("moss_art.data.PromptCache", FakePromptCache)
    worker = bare_worker("moss")
    worker._conditioning(record(0))
    worker._conditioning(record(1))
    worker._conditioning(record(0))
    assert len(instances) == 1 and instances[0].loads == 2


def test_compilation_cache_clears_only_when_boundary_shape_budget_is_exceeded(monkeypatch):
    worker = bare_worker("tiny")
    cleared = []
    monkeypatch.setattr(jax, "clear_caches", lambda: cleared.append(True))
    def contexts(lengths):
        return [(np.zeros((1, length, 24), np.float32),
                 np.zeros((1, 1, 24), np.float32)) for length in lengths]
    first = contexts(range(1, 17))
    before = [[value.copy() for value in pair] for pair in first]
    assert worker._manage_compilation_cache(first) is False
    assert cleared == [] and worker._compiled_text_lengths == set(range(1, 17))
    assert worker._manage_compilation_cache(contexts([16])) is False
    assert worker._manage_compilation_cache(contexts([17])) is True
    assert cleared == [True] and worker._compiled_text_lengths == {1, 17}
    assert worker._manage_compilation_cache(contexts([18])) is False
    for pair, original in zip(first, before):
        for array, expected in zip(pair, original):
            np.testing.assert_array_equal(array, expected)
