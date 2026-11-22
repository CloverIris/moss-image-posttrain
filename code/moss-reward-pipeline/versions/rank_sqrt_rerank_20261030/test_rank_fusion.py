"""CPU-only tests of model rank -> power -> fusion -> final linear rank."""
import copy
import math
import unittest

import reward_fusion as fusion


def calibration():
    return dict(weights=dict(kawai=.4, deepghs=.4, hps=.2), output_offset=.5,
                output_gain=.08, parameters={name: dict(mean=1., scale=2.)
                                             for name in fusion.MODELS})


def rows(deepghs=(0., 1., 2., 3.), hps=(3., 2., 1., 0.), *, prompt='p'):
    return [dict(id=f'i{i}', prompt_id=prompt, image=f'{i}.png',
                 raw_scores=dict(deepghs=d, hps=h))
            for i, (d, h) in enumerate(zip(deepghs, hps))]


def grouped(items=None, **kwargs):
    return fusion.fuse_group(rows() if items is None else items, calibration(),
                             kawai_enabled=False, expected_group_size=4, **kwargs)


class RankFusionTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for x, y in zip(actual, expected):
            self.assertAlmostEqual(x, y, places=12)

    def test_percentile_ranks_average_exact_ties(self):
        self.assertVectorClose(fusion.rank_percentiles([7., 2., 2., 9.]),
                               [2/3, 1/6, 1/6, 1.])
        self.assertEqual(fusion.rank_percentiles([3., 3., 3., 3.]), [.5] * 4)
        self.assertVectorClose(fusion.rank_percentiles([1., 1. + 1e-13]), [0., 1.])

    def test_hand_calculated_power_before_fusion_then_linear_rank(self):
        result = grouped()
        # D ranks=[0,1/3,2/3,1], H reverses them. Concavity changes the
        # weighted compromise: image2 beats image3 despite D's 2:1 weight.
        expected_mixed = [1/3, (2*math.sqrt(1/3)+math.sqrt(2/3))/3,
                          (2*math.sqrt(2/3)+math.sqrt(1/3))/3, 2/3]
        self.assertVectorClose([r['rank_fused_score'] for r in result], expected_mixed)
        self.assertVectorClose([r['reward'] for r in result], [0., 1/3, 1., 2/3])
        self.assertEqual([r['rank_reward'] for r in result], [r['reward'] for r in result])
        self.assertEqual([r['final_rank_score'] for r in result], [r['reward'] for r in result])
        self.assertVectorClose([r['model_rank_percentiles']['deepghs'] for r in result],
                               [0., 1/3, 2/3, 1.])
        for r in result:
            self.assertEqual(r['reward'], r['aesthetic_reward'])
            self.assertEqual(r['rank_fused_score'], math.fsum(r['weighted_rank_contributions'].values()))

    def test_gamma_changes_order_not_final_rank_spacing(self):
        linear, sqrt = grouped(gamma=1.), grouped(gamma=.5)
        self.assertVectorClose([r['reward'] for r in linear], [0., 1/3, 2/3, 1.])
        self.assertNotEqual([r['reward'] for r in linear], [r['reward'] for r in sqrt])
        self.assertVectorClose(sorted(r['reward'] for r in sqrt), [0., 1/3, 2/3, 1.])

    def test_frozen_absolute_diagnostics_preserved_and_labelled(self):
        source = rows()
        original = copy.deepcopy(source)
        result = grouped(source)
        for before, after in zip(source, result):
            old = fusion.fuse(before['raw_scores'], calibration(), 0.,
                              kawai_enabled=False, sdg_enabled=False)
            self.assertEqual(after['frozen_reward'], old['reward'])
            self.assertEqual(after['frozen_aesthetic_reward'], old['aesthetic_reward'])
            self.assertEqual(after['contributions'], old['contributions'])
            self.assertEqual(after['normalized_scores'], old['normalized_scores'])
            self.assertEqual(after['contributions_space'], 'frozen_calibration')
            self.assertEqual(after['fusion_mode'], 'rank_power_rank')
            self.assertEqual(after['rank_power_gamma'], .5)
            self.assertEqual(after['rank_group_size'], 4)
            self.assertEqual(after['image'], before['image'])
            self.assertIsNot(after, before)
            self.assertIsNot(after['raw_scores'], before['raw_scores'])
        self.assertEqual(source, original)

    def test_all_constant_models_produce_no_art_signal(self):
        result = grouped(rows((1.,) * 4, (2.,) * 4))
        self.assertEqual([r['reward'] for r in result], [.5] * 4)
        for r in result:
            self.assertEqual(r['model_rank_percentiles'], dict(deepghs=.5, hps=.5))
        center = [r['reward'] - .5 for r in result]
        self.assertEqual(center, [0.] * 4)

    def test_one_constant_head_does_not_invent_a_preference(self):
        result = grouped(rows((0., 1., 2., 3.), (2.,) * 4))
        self.assertVectorClose([r['reward'] for r in result], [0., 1/3, 2/3, 1.])
        self.assertEqual([r['model_rank_percentiles']['hps'] for r in result], [.5] * 4)

    def test_final_average_ties_and_cancelled_preferences(self):
        weights = dict(deepghs=.5, hps=.5)
        linear = grouped(gamma=1., weights=weights)
        self.assertEqual([r['reward'] for r in linear], [.5] * 4)
        sqrt = grouped(gamma=.5, weights=weights)
        self.assertVectorClose([r['reward'] for r in sqrt], [1/6, 5/6, 5/6, 1/6])

    def test_monotone_score_transform_does_not_change_rank_objective(self):
        base = rows()
        changed = copy.deepcopy(base)
        for r in changed:
            r['raw_scores']['deepghs'] = 7. + math.exp(r['raw_scores']['deepghs'])
            r['raw_scores']['hps'] = 1000. + .001 * r['raw_scores']['hps']
        a, b = grouped(base), grouped(changed)
        for name in ('model_rank_percentiles', 'rank_component_scores',
                     'weighted_rank_contributions', 'rank_fused_score', 'reward'):
            self.assertEqual([r[name] for r in a], [r[name] for r in b])

    def test_permutation_equivariant_including_ties(self):
        source = rows((1., 1., 2., 3.), (2., 3., 1., 1.))
        original = {r['id']: r for r in grouped(source)}
        permuted = grouped([source[i] for i in [2, 0, 3, 1]])
        for r in permuted:
            self.assertEqual(r, original[r['id']])

    def test_training_eight_and_validation_four_are_explicit(self):
        eight = rows(range(8), reversed(range(8)))
        r = fusion.fuse_group(eight, calibration(), kawai_enabled=False)
        self.assertEqual(len(r), 8)
        self.assertEqual([x['rank_group_size'] for x in r], [8] * 8)
        with self.assertRaises(ValueError):
            fusion.fuse_group(rows(), calibration(), kawai_enabled=False)

    def test_optional_kawai_uses_identical_rank_procedure(self):
        source = rows()
        for i, r in enumerate(source):
            r['raw_scores']['kawai'] = float(i)
        result = fusion.fuse_group(source, calibration(), expected_group_size=4)
        for r in result:
            self.assertEqual(set(r['rank_component_scores']), set(fusion.MODELS))
            self.assertEqual(r['enabled_models'], list(fusion.MODELS))

    def test_group_size_gamma_and_ids_are_strict(self):
        for gamma in (0., -.1, 1.01, float('nan'), float('inf'), True, '0.5'):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                grouped(gamma=gamma)
        for size in (True, 0, 1, 4., None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                fusion.fuse_group(rows(), calibration(), expected_group_size=size)
        for source in (rows()[:-1], rows() + [rows()[0]], []):
            with self.assertRaises(ValueError):
                grouped(source)
        for field, value in [('id', 'i0'), ('id', None), ('id', ''),
                             ('prompt_id', 'other'), ('prompt_id', None)]:
            source = rows()
            source[1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                grouped(source)

    def test_domains_cannot_be_mixed_even_when_prompt_id_matches(self):
        for field in ('domain', 'evaluation_domain', 'split'):
            source = rows()
            source[0][field] = 'checkpoint_eval'
            with self.subTest(field=field), self.assertRaises(ValueError):
                grouped(source)
        source = rows()
        for r in source:
            r['domain'] = 'checkpoint_eval'
        self.assertEqual(len(grouped(source)), 4)

    def test_invalid_raw_scores_and_weights_fail_closed(self):
        for model in ('deepghs', 'hps'):
            for value in (None, True, float('nan'), float('inf')):
                source = rows()
                source[1]['raw_scores'][model] = value
                with self.subTest(model=model, value=value), self.assertRaises(ValueError):
                    grouped(source)
        for extra in ('sdg_artifact', 'kawai', 'unknown'):
            source = rows()
            source[0]['raw_scores'][extra] = .1
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                grouped(source)
        for weights in (dict(deepghs=1.), dict(deepghs=.5, hps=.4),
                        dict(deepghs=-1., hps=2.)):
            with self.assertRaises(ValueError):
                grouped(weights=weights)

    def test_legacy_fuse_default_is_unchanged(self):
        raw = dict(kawai=3., deepghs=1., hps=-1., sdg_artifact=-.05)
        result = fusion.fuse(raw, calibration())
        self.assertAlmostEqual(result['reward'], .466)
        self.assertNotIn('fusion_mode', result)
        self.assertNotIn('rank_reward', result)


if __name__ == '__main__':
    unittest.main()
