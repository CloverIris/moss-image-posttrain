"""CPU-only final-checkpoint drain tests; no CUDA/model dependencies."""
from pathlib import Path
import unittest

import flush_validation
import run_reward_with_validation as wrapper
import test_run_reward_with_validation as fixtures


class FakeTime:
    def __init__(self, callback=None):
        self.now = 0.
        self.callback = callback

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.callback:
            self.callback()


class FlushTests(fixtures.WrapperTests):
    # Reuse only fixtures/helpers, not inherited test methods.
    def config(self, expected):
        self.cfg.update(sdg_enabled=False, artifact_weight=0)
        self.cfg.pop("sdg_config", None)
        path = self.root / "pipeline.json"
        wrapper.save(path, self.cfg)
        return dict(pipeline_config=str(path), service_root=str(self.service),
                    expected_updates=expected, archive_root=str(self.root / "flush"),
                    timeout_seconds=30, poll_interval_seconds=10)

    def invoke_flush(self, config, fake_time=None):
        fake_time = fake_time or FakeTime()
        return flush_validation.run(config, pipeline=self.pipeline,
                                    splitter=wrapper.split_groups,
                                    clock=fake_time.clock, sleep=fake_time.sleep)

    def test_final_ready_scores_eval_only_and_restart_is_idempotent(self):
        ready, _ = self.ready("checkpoint_00000020", 20)
        cfg = self.config([20])
        result = self.invoke_flush(cfg)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_updates"], [20])
        self.assertEqual(len(self.pipeline.calls), 1)
        self.assertEqual(len(self.pipeline.calls[0]["items"]), 16)
        self.assertTrue(all(row["id"].startswith(wrapper.EVAL_PREFIX) for row in self.pipeline.calls[0]["items"]))
        ack = wrapper.read_json(self.outbox / (ready["job_id"] + ".ack.json"))
        domains = wrapper.read_json(Path(ack["evaluation_result"]).parents[0] / "result.json")
        self.assertFalse(domains["sdg_enabled"])
        collection = Path(domains["combined_archive"])
        domain_map = wrapper.read_json(collection / "evaluation_domains.json")
        self.assertEqual((domain_map["training_count"], domain_map["validation_count"]), (0, 16))
        self.assertEqual(wrapper.read_json(collection / "training_summary.json")["reward"], None)
        self.invoke_flush(cfg)
        self.assertEqual(len(self.pipeline.calls), 1)

    def test_waits_for_expected_ready_and_commits_each_one(self):
        self.ready("checkpoint_00000016", 16)
        added = []
        def generate_remaining():
            if not added:
                self.ready("checkpoint_00000018", 18)
                self.ready("checkpoint_00000020", 20)
                added.append(True)
        timer = FakeTime(generate_remaining)
        result = self.invoke_flush(self.config([16, 18, 20]), timer)
        self.assertEqual(result["completed_updates"], [16, 18, 20])
        self.assertEqual(len(self.pipeline.calls), 3)
        self.assertTrue(all(len(req["items"]) == 16 for req in self.pipeline.calls))

    def test_failed_scoring_keeps_manifest_and_retries_same_job(self):
        self.ready("checkpoint_00000020", 20)
        self.pipeline.fail_once = True
        result = self.invoke_flush(self.config([20]))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(self.pipeline.calls[0], self.pipeline.calls[1])

    def test_unavailable_expected_checkpoint_times_out_without_gpu_calls(self):
        with self.assertRaises(TimeoutError):
            self.invoke_flush(self.config([20]))
        self.assertFalse(self.pipeline.calls)
        self.assertEqual(wrapper.read_json(self.root / "flush" / "flush_summary.json")["status"], "timeout")


# Inherited WrapperTests methods are covered by their own module; do not run
# them a second time for this fixture subclass.
for _name in dir(fixtures.WrapperTests):
    if _name.startswith("test_") and _name not in FlushTests.__dict__:
        setattr(FlushTests, _name, None)


if __name__ == "__main__":
    unittest.main()
