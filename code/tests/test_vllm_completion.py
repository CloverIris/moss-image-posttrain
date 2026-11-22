"""Failures during final reader validation must never publish a ready cache."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moss_art.data import atomic_json
from moss_art.vllm_prepare import prepare_cache


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = {"complete": True, "entries": {}}
        atomic_json(self.root / "manifest.json", self.manifest)
        self.config = {"model": {"path": "unused-model"},
                       "preparation": {"vllm": {"data_parallel_size": 1, "devices": [0, 1]}}}

    def tearDown(self):
        self.temporary.cleanup()

    def current(self):
        return json.loads((self.root / "manifest.json").read_text())

    def test_cached_resume_validation_failure_clears_previous_complete(self):
        plan = (self.root, copy.deepcopy(self.manifest), [], ["selected-record"], [], 1)
        with patch("moss_art.vllm_prepare.build_plan", return_value=plan), \
                patch("moss_art.vllm_prepare.PromptCache") as cache:
            cache.return_value.validate_prompts.side_effect = ValueError("corrupt cached features")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                prepare_cache(self.config)
        self.assertIs(self.current()["complete"], False)
        self.assertFalse((self.root / "preparation_report.json").exists())

    def test_single_replica_cannot_publish_complete_before_final_validation(self):
        jobs = [{"key": "synthetic", "input_ids": [1]}]
        records = ["selected-record"]
        plan = (self.root, copy.deepcopy(self.manifest), jobs, records, [], 1)
        observed = []

        def encode(config, jobs, directory, manifest, *, mark_complete):
            self.assertFalse(mark_complete)
            self.assertIs(self.current()["complete"], False)
            observed.append("encoded")

        def validate(kept):
            self.assertEqual(kept, records)
            self.assertIs(self.current()["complete"], False)
            observed.append("validated")

        with patch("moss_art.vllm_prepare.build_plan", return_value=plan), \
                patch("moss_art.vllm_prepare.select_devices"), \
                patch("moss_art.vllm_prepare.encode_jobs", side_effect=encode), \
                patch("moss_art.vllm_prepare.PromptCache") as cache:
            cache.return_value.validate_prompts.side_effect = validate
            prepare_cache(self.config)
        self.assertEqual(observed, ["encoded", "validated"])
        self.assertIs(self.current()["complete"], True)

    def test_parallel_producer_failure_leaves_root_incomplete(self):
        self.config["preparation"]["vllm"]["data_parallel_size"] = 2
        plan = (self.root, copy.deepcopy(self.manifest), [{"key": "synthetic", "input_ids": [1]}], [], [], 1)
        with patch("moss_art.vllm_prepare.build_plan", return_value=plan), \
                patch("moss_art.vllm_parallel.run_parallel", side_effect=RuntimeError("rank failed")), \
                patch("moss_art.vllm_prepare.PromptCache") as cache:
            with self.assertRaisesRegex(RuntimeError, "rank failed"):
                prepare_cache(self.config)
        cache.assert_not_called()
        self.assertIs(self.current()["complete"], False)
        self.assertFalse((self.root / "preparation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
