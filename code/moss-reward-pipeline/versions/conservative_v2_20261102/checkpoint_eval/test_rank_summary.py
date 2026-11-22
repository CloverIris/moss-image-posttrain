"""CPU-only score-semantics regression tests for ordinal training feedback."""
import tempfile
import unittest

import run_reward_with_validation as wrapper
import report_validation as reporter
import test_report_validation as report_fixtures


class RankSummaryTests(unittest.TestCase):
    def rows(self, meaning="rank_power_rank_training"):
        result = []
        for index, (absolute, ordinal) in enumerate(((0.2, 0.0), (0.8, 1.0))):
            value = ordinal if meaning == "rank_power_rank_training" else absolute
            result.append(dict(id=str(index), prompt_id="one-prompt", image="image.png",
                enabled_models=["deepghs", "hps"], kawai_enabled=False,
                fusion_mode="rank_power_rank", reward_semantics=meaning,
                contributions_space="frozen_calibration",
                raw_scores=dict(deepghs=absolute, hps=absolute * 10),
                normalized_scores=dict(deepghs=absolute, hps=absolute),
                contributions=dict(deepghs=absolute / 2, hps=absolute / 2),
                frozen_reward=absolute, rank_reward=ordinal,
                reward=value, aesthetic_reward=value))
        return result

    @staticmethod
    def metadata(**extra):
        return dict(enabled_models=["deepghs", "hps"], kawai_enabled=False,
                    sdg_enabled=False, **extra)

    def test_training_rank_and_absolute_diagnostics_are_distinct(self):
        rows = self.rows()
        rows[1]["frozen_reward"] = 0.4
        summary = wrapper.summary_for(rows, self.metadata(domain="training"))
        self.assertEqual(summary["reward"]["mean"], 0.5)
        self.assertAlmostEqual(summary["frozen_reward"]["mean"], 0.3)
        self.assertEqual(summary["rank_reward"]["mean"], 0.5)
        self.assertEqual(summary["fusion_mode"], "rank_power_rank")
        self.assertEqual(summary["reward_semantics"], "rank_power_rank_training")
        self.assertEqual(summary["contribution_semantics"], "frozen_calibration_diagnostic")

    def test_validation_retains_absolute_quality_mean(self):
        rows = self.rows("frozen_calibration_validation")
        rows[1]["reward"] = rows[1]["aesthetic_reward"] = rows[1]["frozen_reward"] = 0.4
        summary = wrapper.summary_for(rows, self.metadata(domain="checkpoint_validation"))
        self.assertAlmostEqual(summary["reward"]["mean"], 0.3)
        self.assertEqual(summary["rank_reward"]["mean"], 0.5)
        self.assertEqual(summary["reward_semantics"], "frozen_calibration_validation")

    def test_mixed_domains_or_fusion_rows_are_rejected(self):
        rows = self.rows()
        rows[1] = self.rows("frozen_calibration_validation")[1]
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            wrapper.summary_for(rows, self.metadata())
        rows = self.rows()
        rows[1].pop("fusion_mode")
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            wrapper.summary_for(rows, self.metadata())

    def test_declared_semantics_must_match_values_and_domain(self):
        rows = self.rows()
        rows[0]["reward"] = rows[0]["frozen_reward"]
        with self.assertRaisesRegex(ValueError, "declared"):
            wrapper.summary_for(rows, self.metadata())
        with self.assertRaisesRegex(ValueError, "Validation summary"):
            wrapper.summary_for(self.rows(), self.metadata(domain="checkpoint_validation"))
        with self.assertRaisesRegex(ValueError, "Training summary"):
            wrapper.summary_for(self.rows("frozen_calibration_validation"), self.metadata(domain="training"))
        with self.assertRaisesRegex(ValueError, "Summary reward semantics"):
            wrapper.summary_for(self.rows(), self.metadata(reward_semantics="frozen_calibration_validation"))

    def test_missing_nonfinite_or_out_of_bounds_scores_are_rejected(self):
        for key, value in (("frozen_reward", float("nan")), ("rank_reward", float("inf")),
                           ("rank_reward", 1.1), ("reward", False),
                           ("reward_semantics", None), ("contributions_space", "rank")):
            with self.subTest(key=key, value=value):
                rows = self.rows()
                rows[0][key] = value
                with self.assertRaises(ValueError):
                    wrapper.summary_for(rows, self.metadata())
        rows = self.rows()
        del rows[0]["frozen_reward"]
        with self.assertRaisesRegex(ValueError, "finite frozen_reward"):
            wrapper.summary_for(rows, self.metadata())

    def test_legacy_and_empty_summaries_do_not_acquire_rank_fields(self):
        rows = self.rows()
        for row in rows:
            for key in ("fusion_mode", "reward_semantics", "frozen_reward", "rank_reward", "contributions_space"):
                row.pop(key)
        for summary in (wrapper.summary_for(rows, self.metadata()), wrapper.summary_for([], self.metadata())):
            for key in ("fusion_mode", "reward_semantics", "frozen_reward", "rank_reward", "contribution_semantics"):
                self.assertNotIn(key, summary)

    def test_actual_checkpoint_report_and_scalars_use_absolute_not_rank(self):
        helper = report_fixtures.ReportTests()
        with tempfile.TemporaryDirectory() as root:
            job = helper.only_deepghs_hps(helper.fixture(root))
            result = reporter.read_json(job["evaluation_result"])
            for index, row in enumerate(result["details"]):
                row.update(fusion_mode="rank_power_rank", reward_semantics="frozen_calibration_validation",
                           frozen_reward=row["reward"], rank_reward=(index % 4) / 3,
                           contributions_space="frozen_calibration")
            reporter.atomic_json(job["evaluation_result"], result)
            report, _ = reporter.validate_and_summarize(job)
            self.assertEqual(report["overall"]["total"]["mean"], 7.5)
            self.assertEqual(reporter.scalar_values(report)["validation/total"], 7.5)
            self.assertEqual([p["total"]["mean"] for p in report["per_prompt"]], [1.5, 5.5, 9.5, 13.5])


if __name__ == "__main__":
    unittest.main()
