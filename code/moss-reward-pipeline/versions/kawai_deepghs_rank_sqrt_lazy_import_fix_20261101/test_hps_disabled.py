"""CPU-only active Kawai/DeepGHS regressions; HPS must not be a hidden dependency."""
import copy
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import reward_fusion as fusion
import run_reward_pipeline as pipeline
from test_kawai_disabled import archive_lock_test_support

CHECKPOINT = Path(__file__).resolve().parent / "checkpoint_eval"
if str(CHECKPOINT) not in sys.path:
    sys.path.insert(0, str(CHECKPOINT))
import run_reward_with_validation as wrapper
import report_validation as reporter
import test_run_reward_with_validation as wrapper_fixtures
import test_report_validation as report_fixtures

KD = ["kawai", "deepghs"]
WEIGHTS = dict(kawai=.5, deepghs=.5)


def calibration():
    return dict(weights=dict(kawai=.4, deepghs=.4, hps=.2),
                output_offset=.5, output_gain=.08,
                parameters={name: dict(mean=1., scale=2.)
                            for name in ("kawai", "deepghs", "hps")})


class NoHPSReads(dict):
    def __getitem__(self, key):
        if key == "hps":
            raise AssertionError("Inactive HPS calibration was accessed")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "hps":
            raise AssertionError("Inactive HPS calibration was accessed")
        return super().get(key, default)


class KDFusionTests(unittest.TestCase):
    def test_explicit_active_set_is_canonical_and_old_defaults_still_work(self):
        self.assertEqual(fusion.aesthetic_models(enabled_models=["deepghs", "kawai"]),
                         tuple(KD))
        self.assertEqual(fusion.aesthetic_models(), ("kawai", "deepghs", "hps"))
        self.assertEqual(fusion.aesthetic_models(False), ("deepghs", "hps"))
        for names in ([], ["kawai", "kawai"], ["deepghs", "unknown"], "kawai"):
            with self.subTest(names=names), self.assertRaises(ValueError):
                fusion.aesthetic_models(enabled_models=names)

    def test_subset_uses_only_active_scales_and_renormalizes_frozen_weights(self):
        cal = calibration()
        cal["weights"] = NoHPSReads(cal["weights"])
        cal["parameters"] = NoHPSReads(cal["parameters"])
        result = fusion.fuse(dict(kawai=3., deepghs=1.), cal, 0.,
                             sdg_enabled=False, enabled_models=KD)
        self.assertEqual(result["enabled_models"], KD)
        self.assertTrue(result["kawai_enabled"])
        self.assertFalse(result["sdg_enabled"])
        self.assertEqual(result["fusion_weights"], WEIGHTS)
        self.assertAlmostEqual(result["reward"], .54)
        self.assertEqual(result["reward"], .5 + math.fsum(result["contributions"].values()))
        for field in ("normalized_scores", "contributions"):
            self.assertEqual(set(result[field]), set(KD))

    def test_explicit_active_weights_need_no_hps_calibration(self):
        cal = calibration()
        del cal["weights"]["hps"]
        del cal["parameters"]["hps"]
        result = fusion.fuse(dict(kawai=3., deepghs=1.), cal, 0.,
                             sdg_enabled=False, enabled_models=KD, weights=WEIGHTS)
        self.assertAlmostEqual(result["reward"], .54)

    def test_active_scores_and_weight_keys_fail_closed(self):
        for name in KD:
            for bad in (None, True, float("nan"), float("inf")):
                raw = dict(kawai=3., deepghs=1.)
                raw[name] = bad
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    fusion.fuse(raw, calibration(), 0., sdg_enabled=False,
                                enabled_models=KD, weights=WEIGHTS)
            raw = dict(kawai=3., deepghs=1.)
            del raw[name]
            with self.assertRaises(ValueError):
                fusion.fuse(raw, calibration(), 0., sdg_enabled=False,
                            enabled_models=KD, weights=WEIGHTS)
        for bad in (dict(kawai=1.), dict(kawai=.5, deepghs=.5, hps=0.),
                    dict(kawai=.5, deepghs=.4), dict(kawai=-1., deepghs=2.)):
            with self.subTest(weights=bad), self.assertRaises(ValueError):
                fusion.fuse(dict(kawai=3., deepghs=1.), calibration(), 0.,
                            sdg_enabled=False, enabled_models=KD, weights=bad)
        with self.assertRaises(ValueError):
            fusion.fuse(dict(kawai=3., deepghs=1., hps=0.), calibration(), 0.,
                        sdg_enabled=False, enabled_models=KD, weights=WEIGHTS)

    def test_k4_hand_calculated_sqrt_then_rerank_with_ties(self):
        rows = [dict(id=str(i), prompt_id="same", raw_scores=dict(kawai=float(i), deepghs=float(3-i)))
                for i in range(4)]
        got = fusion.fuse_group(rows, calibration(), enabled_models=KD,
                               weights=WEIGHTS, gamma=.5, expected_group_size=4)
        expected = [1/6, 5/6, 5/6, 1/6]
        for row, rank in zip(got, expected):
            self.assertAlmostEqual(row["reward"], rank)
            self.assertEqual(row["enabled_models"], KD)
            self.assertEqual(set(row["model_rank_percentiles"]), set(KD))
            self.assertEqual(set(row["weighted_rank_contributions"]), set(KD))
            frozen = fusion.fuse(row["raw_scores"], calibration(), 0.,
                                sdg_enabled=False, enabled_models=KD, weights=WEIGHTS)
            self.assertEqual(row["frozen_reward"], frozen["reward"])

    def test_k8_monotone_heads_and_old_group_results_are_preserved(self):
        rows = [dict(id=str(i), prompt_id="same", raw_scores=dict(kawai=float(i), deepghs=2.*i))
                for i in range(8)]
        got = fusion.fuse_group(rows, calibration(), enabled_models=KD,
                               weights=WEIGHTS, expected_group_size=8)
        self.assertEqual([r["reward"] for r in got], [i/7 for i in range(8)])
        dh = [dict(id=str(i), prompt_id="same", raw_scores=dict(deepghs=float(i), hps=3.-i))
              for i in range(4)]
        old = fusion.fuse_group(dh, calibration(), kawai_enabled=False, expected_group_size=4)
        new = fusion.fuse_group(dh, calibration(), enabled_models=["deepghs", "hps"],
                                expected_group_size=4)
        self.assertEqual(old, new)
        allraw = dict(kawai=3., deepghs=1., hps=-1., sdg_artifact=-.05)
        self.assertEqual(fusion.fuse(allraw, calibration()),
                         fusion.fuse(allraw, calibration(), enabled_models=list(fusion.MODELS)))


class KDPipelineTests(unittest.TestCase):
    def fixture(self, root):
        for name in ("run.py", "worker.py"):
            (root/name).write_text("# CPU-only source fixture\n", encoding="utf-8")
        pipeline.save(root/"batch.json", dict(stage_order=["deepghs"], deepghs={}))
        cal = calibration()
        cal["weights"] = dict(WEIGHTS)
        cal["parameters"].pop("hps")
        pipeline.save(root/"calibration.json", cal)
        image = root/"sample.png"
        image.write_bytes(b"CPU fixture: hashing only, never decoded")
        cfg = dict(enabled_models=KD, kawai_enabled=True, sdg_enabled=False, artifact_weight=0.,
                   fusion_weights=WEIGHTS, fusion_mode="frozen_calibration",
                   calibration=str(root/"calibration.json"), batch_config=str(root/"batch.json"),
                   batch_root=str(root), allowed_archive_root=str(root), archive_root=str(root/"archives"),
                   controller_python="not executed", stage_timeout=1)
        request = dict(version=1, items=[
            dict(id=key, sample_id="p"+key, prompt="unchanged", image=str(image))
            for key in ("a", "b")])
        calls = []
        def score_batch(argv, log, timeout):
            calls.append(list(argv))
            stage_start = argv.index("--stages") + 1
            stages = [x for x in argv[stage_start:] if not x.startswith("--")]
            self.assertEqual(stages, ["deepghs"])
            pipeline.save(Path(argv[argv.index("--output")+1]), dict(status="complete", items=[
                dict(id="b", deepghs=dict(status="ok", percentile=1.)),
                dict(id="a", deepghs=dict(status="ok", percentile=3.))]))
        kawai = dict(items=[dict(id="b", kawai=1.), dict(id="a", kawai=3.)])
        return cfg, request, score_batch, kawai, calls

    def test_hps_config_and_provenance_files_are_truly_optional(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg, request, score, kawai, calls = self.fixture(root)
            # There is deliberately no hps section, native_config or HPS project.
            provenance = pipeline.provenance(cfg)
            self.assertFalse(any("hpsv3" in p or Path(p).name == "native.json" for p in provenance))
            with archive_lock_test_support(), patch.object(pipeline, "run_program", side_effect=score), \
                 patch.object(pipeline, "run_kawai_parallel", return_value=kawai), \
                 patch.object(pipeline, "run_sdg_parallel", side_effect=AssertionError("SDG started")):
                result = pipeline.run(cfg, request, root/"output.json")
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["enabled_models"], KD)
            self.assertEqual(result["scores"], [.58, .5])
            for row in result["details"]:
                for field in ("raw_scores", "normalized_scores", "contributions", "model_details"):
                    self.assertEqual(set(row[field]), set(KD))
            archive = Path(result["archive"])
            manifest = json.loads((archive/"manifest.json").read_text())
            self.assertEqual(manifest["stage_order"], ["deepghs", "kawai"])
            self.assertTrue((archive/"deepghs.json").exists())
            self.assertFalse((archive/"deepghs_hps.json").exists())
            self.assertFalse((archive/"sdg.json").exists())
            with (archive/"scores.csv").open(encoding="utf-8-sig", newline="") as f:
                fields = csv.DictReader(f).fieldnames
            self.assertEqual(fields, ["id", "prompt_id", "image", "kawai", "deepghs",
                                      "aesthetic_reward", "reward"])

    def test_k8_training_and_k4_validation_keep_different_reward_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg, request, _, _, _ = self.fixture(root)
            cfg.update(fusion_mode="rank_power_rank", rank_power_gamma=.5,
                       rank_training_group_size=8, rank_eval_group_size=4,
                       checkpoint_eval_reward_mode="frozen_calibration")
            image = request["items"][0]["image"]
            request["items"] = []
            for domain, n in (("train", 8), ("eval", 4)):
                for i in range(n):
                    row = dict(id=domain+str(i), sample_id="same-prompt", prompt="unchanged", image=image)
                    if domain == "eval":
                        row["checkpoint_eval"] = dict(job_id="fixed")
                    request["items"].append(row)
            vals = {r["id"]: float(i) for i, r in enumerate(request["items"])}
            def score(argv, log, timeout):
                self.assertEqual(argv[argv.index("--stages")+1:argv.index("--resume")], ["deepghs"])
                pipeline.save(Path(argv[argv.index("--output")+1]), dict(status="complete", items=[
                    dict(id=k, deepghs=dict(status="ok", percentile=v)) for k, v in vals.items()]))
            kawai = dict(items=[dict(id=k, kawai=v) for k, v in reversed(list(vals.items()))])
            with archive_lock_test_support(), patch.object(pipeline, "run_program", side_effect=score), \
                 patch.object(pipeline, "run_kawai_parallel", return_value=kawai):
                result = pipeline.run(cfg, request, root/"output.json")
            train, valid = result["details"][:8], result["details"][8:]
            self.assertEqual([x["reward"] for x in train], [i/7 for i in range(8)])
            self.assertTrue(all(x["reward_semantics"] == "rank_power_rank_training" for x in train))
            self.assertTrue(all(x["reward"] == x["frozen_reward"] for x in valid))
            self.assertTrue(all(x["reward_semantics"] == "frozen_calibration_validation" for x in valid))
            self.assertTrue(any(x["reward"] != x["rank_reward"] for x in valid))
            self.assertEqual([x["rank_reward"] for x in valid], [i/3 for i in range(4)])


class KDFakePipeline(wrapper_fixtures.FakePipeline):
    def run(self, cfg, request, output):
        self.calls.append(copy.deepcopy(request))
        fingerprint, archive = wrapper.pipeline_identity(self, cfg, request)
        rows = []
        for i, item in enumerate(self.canonical_items(request)):
            models = KD
            rows.append(dict(id=item["id"], image=item["image"], prompt_id=item["prompt_id"],
                             enabled_models=KD, kawai_enabled=True,
                             reward=i/100, aesthetic_reward=i/100,
                             raw_scores={name: i/100 for name in models},
                             normalized_scores={name: i/100 for name in models},
                             contributions={name: i/100 for name in models}))
        result = dict(scores=[x["reward"] for x in rows], details=rows, archive=str(archive),
                      fingerprint=fingerprint, enabled_models=KD, kawai_enabled=True, sdg_enabled=False)
        wrapper.save(output, result)
        return result


class KDCheckpointTests(unittest.TestCase):
    def test_wrapper_claim_publish_summary_skip_hps_without_hps_config(self):
        helper = wrapper_fixtures.WrapperTests()
        helper.setUp()
        try:
            helper.cfg.update(enabled_models=KD, kawai_enabled=True, sdg_enabled=False)
            wrapper.save(helper.root/"batch.json", dict(stage_order=["deepghs"], deepghs={}))
            helper.pipeline = KDFakePipeline()
            ready, _ = helper.ready()
            with patch.object(wrapper, "hps_plan", side_effect=AssertionError("HPS planner called")):
                result = helper.invoke()
            self.assertEqual(len(result["scores"]), 192)
            ack = wrapper.read_json(helper.outbox/(ready["job_id"]+".ack.json"))
            evaluation = wrapper.read_json(Path(ready["result_dir"])/"result.json")
            summary = wrapper.read_json(Path(ready["result_dir"])/"summary.json")
            for owner in (ack, evaluation, summary):
                self.assertEqual(owner["enabled_models"], KD)
            for detail in evaluation["details"]:
                self.assertEqual(set(detail["raw_scores"]), set(KD))
            self.assertEqual(set(summary["raw_component_mean"]), set(KD))
        finally:
            helper.tearDown()

    def test_report_and_scalars_only_active_kawai_deepghs(self):
        helper = report_fixtures.ReportTests()
        with tempfile.TemporaryDirectory() as folder:
            job = helper.fixture(folder)
            manifest = dict(enabled_models=KD, kawai_enabled=True, sdg_enabled=False)
            result = reporter.read_json(job["evaluation_result"])
            result.update(manifest)
            for row in result["details"]:
                row.update(manifest)
                for field in ("raw_scores", "normalized_scores", "contributions"):
                    row[field].pop("hps")
            reporter.atomic_json(job["evaluation_result"], result)
            ack = reporter.read_json(job["ack"])
            ack.update(manifest)
            reporter.atomic_json(job["ack"], ack)
            summary = reporter.read_json(job["summary"])
            summary["ack_sha256"] = reporter.file_digest(job["ack"])
            reporter.atomic_json(job["summary"], summary)
            report, _ = reporter.validate_and_summarize(job)
            self.assertEqual(report["enabled_models"], KD)
            self.assertEqual(set(report["overall"]["raw"]), set(KD))
            values = reporter.scalar_values(report)
            self.assertIn("validation/kawai", values)
            self.assertIn("validation/deepghs", values)
            self.assertFalse(any("hps" in key or "sdg" in key for key in values))

    def test_missing_kd_score_and_undeclared_hps_are_rejected_by_summary(self):
        metadata = dict(enabled_models=KD, kawai_enabled=True, sdg_enabled=False)
        row = dict(id="x", prompt_id="p", image="x.png", reward=.5, aesthetic_reward=.5,
                   enabled_models=KD, kawai_enabled=True,
                   raw_scores=dict(kawai=1., deepghs=1.),
                   normalized_scores=dict(kawai=0., deepghs=0.),
                   contributions=dict(kawai=0., deepghs=0.))
        self.assertEqual(wrapper.enabled_aesthetic_models(metadata, [row]), tuple(KD))
        for field in ("raw_scores", "normalized_scores", "contributions"):
            for kind in ("missing", "extra"):
                changed = copy.deepcopy(row)
                if kind == "missing":
                    changed[field].pop("kawai")
                else:
                    changed[field]["hps"] = 0.
                with self.subTest(field=field, kind=kind), self.assertRaises(ValueError):
                    wrapper.summary_for([changed], metadata)


if __name__ == "__main__":
    unittest.main()
