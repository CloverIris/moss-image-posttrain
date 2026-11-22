import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("report_validation", Path(__file__).with_name("report_validation.py"))
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class ReportTests(unittest.TestCase):
    def only_deepghs_hps(self,job):
        result=reporter.read_json(job['evaluation_result'])
        manifest=dict(enabled_models=['deepghs','hps'],kawai_enabled=False,sdg_enabled=False)
        result.update(manifest)
        for row in result['details']:
            row.update(manifest)
            for key in ('raw_scores','normalized_scores','contributions'):row[key].pop('kawai')
        reporter.atomic_json(job['evaluation_result'],result)
        ack=reporter.read_json(job['ack']);ack.update(manifest);reporter.atomic_json(job['ack'],ack)
        summary=reporter.read_json(job['summary']);summary['ack_sha256']=reporter.file_digest(job['ack'])
        reporter.atomic_json(job['summary'],summary)
        return job

    def fixture(self, root):
        root = Path(root)
        items, details = [], []
        for p in range(4):
            for r in range(4):
                index = p*4+r
                key = f"eval:checkpoint14:p{p}:r{r}"
                item = {"id": key, "image": str(root / f"{index}.png"),
                        "prompt_id": f"original-{p}", "prompt_index": p,
                        "rollout_id": r, "seed": 300+index}
                items.append(item)
                details.append({"id": "checkpoint-eval::checkpoint14::"+key,
                    "original_id": key, "image": item["image"], "reward": float(index),
                    "aesthetic_reward": float(index),
                    "raw_scores": {"kawai": index+0.2, "deepghs": index+0.3, "hps": index+0.4},
                    "normalized_scores": {name: index*0.1 for name in reporter.MODELS},
                    "contributions": {name: index/3 for name in reporter.MODELS}})
        request = root / "request.json"
        reporter.atomic_json(request, {"version": 1, "items": items})
        identity = {"checkpoint_id": "checkpoint14", "source_step": 14,
                    "request_sha256": reporter.file_digest(request), "checkpoint_manifest_sha256": "manifest"}
        result = root / "result.json"
        reporter.atomic_json(result, {"version": 1, "status": "complete", **identity,
            "scores": [row["reward"] for row in details], "details": details, "scored_at": 5000.0})
        ack = root / "checkpoint14.ack.json"
        reporter.atomic_json(ack, {"version": 1, "status": "complete", **identity})
        summary = root / "summary.json"
        reporter.atomic_json(summary, {**identity, "update": 14, "request": str(request),
            "ack_sha256": reporter.file_digest(ack), "fixed_seeds": [row["seed"] for row in items]})
        return {"ack": str(ack), "evaluation_result": str(result), "scores": str(result),
                "summary": str(summary), "source_step": 14, "logging_log_dir": str(root / "tb")}

    def test_actual_components_groups_and_no_sdg(self):
        with tempfile.TemporaryDirectory() as root:
            report, paths = reporter.validate_and_summarize(self.fixture(root))
            self.assertEqual(report["overall"]["total"]["mean"], 7.5)
            self.assertEqual([p["total"]["mean"] for p in report["per_prompt"]], [1.5, 5.5, 9.5, 13.5])
            self.assertEqual(report["per_prompt"][0]["ranking"][0]["rollout_id"], 3)
            self.assertAlmostEqual(report["overall"]["raw"]["kawai"]["mean"], 7.7)
            values = reporter.scalar_values(report)
            for key in ("validation/total", "validation/aesthetic", "validation/kawai",
                        "validation/deepghs", "validation/hps", "validation/per_prompt_0",
                        "validation/per_prompt_3", "validation/contribution/hps"):
                self.assertIn(key, values)
            self.assertFalse(any("sdg" in key for key in values))
            self.assertFalse(any(name in sys.modules for name in ("torch", "jax", "tensorflow")))

    def test_two_models_have_no_kawai_placeholder_or_scalar(self):
        with tempfile.TemporaryDirectory() as root:
            job=self.only_deepghs_hps(self.fixture(root));report,_=reporter.validate_and_summarize(job)
            self.assertEqual(report['enabled_models'],['deepghs','hps'])
            self.assertFalse(report['kawai_enabled']);self.assertFalse(report['sdg_enabled'])
            self.assertEqual(set(report['overall']['raw']),{'deepghs','hps'})
            values=reporter.scalar_values(report)
            self.assertIn('validation/deepghs',values);self.assertIn('validation/hps',values)
            self.assertEqual(values['validation/per_prompt_0'],1.5)
            self.assertFalse(any('kawai' in key or 'sdg' in key for key in values))

    def test_two_model_manifest_does_not_hide_missing_active_score(self):
        with tempfile.TemporaryDirectory() as root:
            job=self.only_deepghs_hps(self.fixture(root))
            result=reporter.read_json(job['evaluation_result']);del result['details'][0]['raw_scores']['hps']
            reporter.atomic_json(job['evaluation_result'],result)
            with self.assertRaisesRegex(ValueError,'enabled models'):reporter.validate_and_summarize(job)

    def test_nonfinite_missing_component_or_order_fails(self):
        for change in ("nonfinite", "component", "order", "seed"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root:
                job = self.fixture(root)
                path = Path(job["evaluation_result"])
                result = reporter.read_json(path)
                if change == "nonfinite":
                    # JSON permits NaN as input, but report must reject it.
                    result["scores"][0] = float("nan")
                elif change == "component":
                    del result["details"][0]["raw_scores"]["hps"]
                elif change == "order":
                    result["details"][0], result["details"][1] = result["details"][1], result["details"][0]
                else:
                    summary = reporter.read_json(job["summary"])
                    summary["fixed_seeds"][0] = 900
                    reporter.atomic_json(job["summary"], summary)
                path.write_text(json.dumps(result), encoding="utf-8")
                with self.assertRaises(ValueError):
                    reporter.validate_and_summarize(job)

    def test_daemon_four_base_seed_plan_expands_strictly(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.fixture(root)
            summary = reporter.read_json(job["summary"])
            summary["fixed_seeds"] = [300, 301, 302, 303]
            reporter.atomic_json(job["summary"], summary)
            report, _ = reporter.validate_and_summarize(job)
            self.assertEqual([row["seed"] for row in report["images"]], list(range(300, 316)))
            summary["fixed_seeds"][2] += 100
            reporter.atomic_json(job["summary"], summary)
            with self.assertRaisesRegex(ValueError, "seeds differ"):
                reporter.validate_and_summarize(job)

    def test_report_and_event_retry_are_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.fixture(root)
            summary_before = Path(job["summary"]).read_bytes()
            fake_event = lambda report: json.dumps(reporter.scalar_values(report), sort_keys=True).encode()
            with mock.patch.object(reporter, "event_bytes", side_effect=fake_event) as writer:
                first = reporter.run(job)
                again = reporter.run(job)
                self.assertEqual(writer.call_count, 1)
                self.assertEqual(first, again)
            self.assertEqual(Path(job["summary"]).read_bytes(), summary_before)
            self.assertEqual(len(list(Path(job["logging_log_dir"]).glob("events.out.tfevents.*"))), 1)

    def test_retry_after_event_before_marker_has_one_event(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.fixture(root)
            original = reporter.atomic_json
            def crash_marker(path, value):
                if Path(path).name == "validation_report.complete.json":
                    raise InterruptedError("simulate power loss before success marker")
                original(path, value)
            fake_event = lambda report: json.dumps(reporter.scalar_values(report), sort_keys=True).encode()
            with mock.patch.object(reporter, "event_bytes", side_effect=fake_event), \
                    mock.patch.object(reporter, "atomic_json", side_effect=crash_marker):
                with self.assertRaises(InterruptedError):
                    reporter.run(job)
            with mock.patch.object(reporter, "event_bytes", side_effect=fake_event):
                reporter.run(job)
            self.assertEqual(len(list(Path(job["logging_log_dir"]).glob("events.out.tfevents.*"))), 1)

    @unittest.skipUnless(importlib.util.find_spec("tensorboard"), "TensorBoard not installed in local interpreter")
    def test_real_event_has_source_step_and_one_set_of_tags(self):
        with tempfile.TemporaryDirectory() as root:
            job = self.fixture(root)
            done = reporter.run(job)
            from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
            events = list(EventFileLoader(done["event"]).Load())
            point = [event for event in events if event.HasField("summary")]
            self.assertEqual(len(point), 1)
            self.assertEqual(point[0].step, 14)
            self.assertEqual(len(point[0].summary.value), len(reporter.scalar_values(
                reporter.read_json(done["report"]))))
            self.assertFalse(any(name in sys.modules for name in ("torch", "jax", "tensorflow")))


if __name__ == "__main__":
    unittest.main()
