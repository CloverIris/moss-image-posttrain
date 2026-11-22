"""Deterministic CPU tests. Synthetic labels/thresholds are NOT PixAI calibration."""

from dataclasses import replace
from math import fsum
import random
import unittest

from conservative_reward import (
    Candidate, Guard, HeadCalibration, RewardPolicy, TagRule,
    art_covariance_kwargs, catalog_from_native_config, fuse_group, guard_from_native_tags,
    validate_training_contract,
)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.policy = RewardPolicy(
            calibration_id="synthetic-test-only", kawai=HeadCalibration(0.01, 0.2),
            deepghs=HeadCalibration(0.01, 0.1), aesthetic_coverage=1.0,
            min_guard_reliability=0.8, aesthetic_cap=0.25, expected_group_size=2,
        )
        self.passed = Guard("pass", 0.95, "synthetic audited style pass")
        self.failed = Guard("fail", 0.95, "synthetic audited style violation")
        self.unknown = Guard("unknown", 0.0, "style not covered")

    def candidate(self, name, k, d, guard=None, prompt="p"):
        return Candidate(name, prompt, k, d, guard if guard is not None else self.passed)

    def test_heads_agree(self):
        result = fuse_group([self.candidate("a", 0.7, 0.8), self.candidate("b", 0.2, 0.3)], self.policy)
        self.assertGreater(result.rewards[0], 0)
        self.assertLess(result.rewards[1], 0)
        self.assertAlmostEqual(fsum(result.rewards), 0)

    def test_disagreement_abstains(self):
        result = fuse_group([self.candidate("a", 0.7, 0.3), self.candidate("b", 0.2, 0.8)], self.policy)
        self.assertEqual(result.rewards, (0.0, 0.0))
        self.assertFalse(result.signal_active)

    def test_small_gap_abstains(self):
        result = fuse_group([self.candidate("a", 0.205, 0.305), self.candidate("b", 0.2, 0.3)], self.policy)
        self.assertEqual(result.rewards, (0.0, 0.0))

    def test_one_head_tie_abstains(self):
        result = fuse_group([self.candidate("a", 0.7, 0.3), self.candidate("b", 0.2, 0.3)], self.policy)
        self.assertEqual(result.rewards, (0.0, 0.0))

    def test_high_aesthetic_cannot_buy_style_failure(self):
        result = fuse_group([self.candidate("flat", -0.4, 0.25), self.candidate("oily", 0.7, 0.8, self.failed)], self.policy)
        self.assertGreater(result.rewards[0], result.rewards[1])

    def test_unknown_not_promoted_or_penalized(self):
        policy = replace(self.policy, expected_group_size=3)
        result = fuse_group([self.candidate("a", 0.7, 0.8), self.candidate("b", 0.2, 0.3),
                             self.candidate("u", 1.0, 1.0, self.unknown)], policy)
        self.assertEqual(result.rewards[2], 0.0)
        self.assertEqual(result.abstained_pairs, 2)

    def test_unknown_group_does_not_learn(self):
        result = fuse_group([self.candidate("a", 0.7, 0.8, self.unknown), self.candidate("b", 0.2, 0.3, self.unknown)], self.policy)
        self.assertEqual(result.reward_mad, 0)

    def test_all_fail_abstains(self):
        result = fuse_group([self.candidate("a", 0.7, 0.8, self.failed), self.candidate("b", 0.2, 0.3, self.failed)], self.policy)
        self.assertEqual(result.rewards, (0.0, 0.0))

    def test_lower_coverage_reduces_mad(self):
        rows = [self.candidate("a", 0.7, 0.8), self.candidate("b", 0.2, 0.3)]
        strong = fuse_group(rows, self.policy)
        weak = fuse_group(rows, replace(self.policy, aesthetic_coverage=0.1))
        self.assertAlmostEqual(weak.reward_mad, 0.1 * strong.reward_mad)

    def test_zero_coverage_still_trains_validated_constraints(self):
        rows = [self.candidate("a", -0.4, 0.25), self.candidate("b", 0.7, 0.8, self.failed)]
        result = fuse_group(rows, replace(self.policy, aesthetic_coverage=0.0))
        self.assertGreater(result.reward_mad, 0)
        self.assertEqual(result.aesthetic_pairs, 0)

    def test_compressing_raw_scores_no_longer_invariant(self):
        rows = [self.candidate("a", 0.7, 0.8), self.candidate("b", 0.2, 0.3)]
        original = fuse_group(rows, self.policy)
        tiny = fuse_group([replace(row, kawai=row.kawai * 0.001, deepghs=row.deepghs * 0.001) for row in rows], self.policy)
        self.assertGreater(original.reward_mad, 0)
        self.assertEqual(tiny.reward_mad, 0)

    def test_order_and_offset_invariance(self):
        rows = [self.candidate("a", 0.5, 0.6), self.candidate("b", 0.2, 0.3)]
        original = fuse_group(rows, self.policy)
        reversed_result = fuse_group(list(reversed(rows)), self.policy)
        shifted = fuse_group([replace(row, kawai=row.kawai + 0.1, deepghs=row.deepghs + 0.1) for row in rows], self.policy)
        self.assertEqual(original.rewards, tuple(reversed(reversed_result.rewards)))
        for x, y in zip(original.rewards, shifted.rewards):
            self.assertAlmostEqual(x, y)

    def test_pass_fail_global_order_and_bounds_property(self):
        rng = random.Random(42)
        policy = replace(self.policy, expected_group_size=8)
        for _ in range(1000):
            rows = [self.candidate(str(i), rng.uniform(-1, 1), rng.random(),
                    Guard(rng.choice(["pass", "fail", "unknown"]), rng.uniform(0.8, 1.0), "synthetic")) for i in range(8)]
            result = fuse_group(rows, policy)
            self.assertAlmostEqual(fsum(result.rewards), 0, places=12)
            self.assertTrue(all(-1 <= value <= 1 for value in result.rewards))
            self.assertEqual(result.aesthetic_pairs + result.constraint_pairs + result.abstained_pairs, 28)
            for i, left in enumerate(rows):
                if left.guard.status == "pass":
                    for j, right in enumerate(rows):
                        if right.guard.status == "fail":
                            self.assertGreater(result.rewards[i], result.rewards[j])
                if left.guard.status == "unknown":
                    self.assertEqual(result.rewards[i], 0.0)
                for j in range(8):
                    self.assertEqual(result.pair_votes[i][j], -result.pair_votes[j][i])

    def test_invalid_group_and_invalid_scores_fail_closed(self):
        a = self.candidate("a", 0.5, 0.6)
        for rows in [[a], [a, a], [a, self.candidate("b", 0.2, 0.3, prompt="other")]]:
            with self.assertRaises(ValueError):
                fuse_group(rows, self.policy)
        for value in [float("nan"), float("inf"), True, "0.3"]:
            with self.assertRaises(ValueError):
                self.candidate("bad", value, 0.5)
        with self.assertRaises(ValueError):
            self.candidate("bad", 0.3, 5.0)  # DeepGHS raw score is NOT percentile.

    def test_low_reliability_is_unknown(self):
        result = fuse_group([self.candidate("a", 0.7, 0.8, Guard("pass", 0.5, "weak")),
                             self.candidate("b", 0.2, 0.3)], self.policy)
        self.assertEqual(result.rewards, (0.0, 0.0))

    def rule(self, desired=True, tag="toy_flat"):
        return TagRule("style", tag, desired, 0.2, 0.8, 0.95)

    def tag_guard(self, scores, rules=None, catalog=None):
        return guard_from_native_tags(scores, rules if rules is not None else [self.rule()],
            catalog if catalog is not None else {"style": {"toy_flat"}}, min_reliability=0.8)

    def test_native_tag_guards_use_uncertainty_band(self):
        self.assertEqual(self.tag_guard({"style": {"toy_flat": 0.9}}).status, "pass")
        self.assertEqual(self.tag_guard({"style": {"toy_flat": 0.1}}).status, "fail")
        self.assertEqual(self.tag_guard({"style": {"toy_flat": 0.5}}).status, "unknown")

    def test_native_catalog_keeps_exact_names_and_validates_lengths(self):
        self.assertEqual(catalog_from_native_config(["toy_flat", "toy name"], [("style", 1), ("general", 1)]),
                         {"style": {"toy_flat"}, "general": {"toy name"}})
        for tags, splits in [(["a"], [("style", 2)]), (["a", "a"], [("style", 2)]),
                             (["a"], []), (["a"], [("style", 1), ("style", 0)])]:
            with self.assertRaises(ValueError):
                catalog_from_native_config(tags, splits)

    def test_missing_and_unsupported_tags_not_false_negative(self):
        self.assertEqual(self.tag_guard({}).status, "unknown")
        self.assertEqual(self.tag_guard({"style": {"toy_flat": 0.9}}, catalog={}).status, "unknown")
        self.assertEqual(self.tag_guard({}, rules=[]).status, "unknown")

    def test_no_global_gloss_blacklist(self):
        scores = {"style": {"toy_gloss": 0.95}}
        catalog = {"style": {"toy_gloss"}}
        self.assertEqual(self.tag_guard(scores, [self.rule(True, "toy_gloss")], catalog).status, "pass")
        self.assertEqual(self.tag_guard(scores, [self.rule(False, "toy_gloss")], catalog).status, "fail")

    def test_duplicate_rule_and_bad_probability_raise(self):
        with self.assertRaises(ValueError):
            self.tag_guard({}, [self.rule(), self.rule(False)])
        with self.assertRaises(ValueError):
            self.tag_guard({"style": {"toy_flat": float("nan")}})

    def test_actual_learner_options_and_base_anchor_required(self):
        options = art_covariance_kwargs(guidance_gain=0.2)
        validate_training_contract(options, reference_mse_beta=0.1)
        for changes, beta in [({"scale_mode": "unit_normalized_covariance"}, 0.1),
                               ({"mode": "pairwise"}, 0.1), ({}, 0.0)]:
            with self.assertRaises(ValueError):
                validate_training_contract(options | changes, reference_mse_beta=beta)

    def test_invalid_calibration_and_unsafe_policy_raise(self):
        with self.assertRaises(ValueError):
            HeadCalibration(0.0, 0.0)
        with self.assertRaises(ValueError):
            replace(self.policy, aesthetic_cap=0.9)


if __name__ == "__main__":
    unittest.main()
