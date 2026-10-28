import itertools
import unittest

import numpy as np

from policy_optimization import (
    BEST_RESPONSE_TOLERANCE, DEFAULT_VALUE_TOLERANCE, evaluate_policy,
    evaluate_potential_policy, optimize_policy, select_potential_policy,
)


def two_defense_supremum(lower, upper, ac, dc, potential):
    """Enumerate affine breakpoints and one-sided values on a line segment.

    This check uses no linear-programming solver. Crossings of mission values
    and attacker-payoff endpoints partition the segment into regions with a
    constant response and an affine defender value.
    """
    endpoints = np.concatenate((lower, upper), axis=1)
    attack_costs = np.tile(ac, 2)
    lines = [(endpoints[0, j] + attack_costs[j],
              endpoints[1, j] - endpoints[0, j]) for j in range(len(attack_costs))]
    lines += [(lower[0, j], lower[1, j] - lower[0, j]) for j in range(len(ac))]
    points = {0., 1.}
    for (a, b), (c, d) in itertools.combinations(lines, 2):
        if b != d:
            t = (c - a) / (b - d)
            if 0. < t < 1.:
                points.add(float(t))
    points = sorted(points)

    def response(t):
        x = np.array([1. - t, t])
        lo, hi = x @ lower, x @ upper
        plus, minus = 1. - lo - ac, 1. - hi - ac
        threshold = max(minus) if potential else max(plus)
        indices = np.flatnonzero(plus >= threshold - 1e-12)
        return min(indices, key=lambda j: (lo[j], j))

    def value(t, j):
        return (1. - t) * (lower[0, j] - dc[0]) + t * (lower[1, j] - dc[1])

    values = [value(t, response(t)) for t in points]
    for left, right in zip(points[:-1], points[1:]):
        if right - left > 1e-12:
            j = response((left + right) / 2.)
            values.extend((value(left, j), value(right, j)))
    return max(values)


class PolicyOptimizationTests(unittest.TestCase):
    def assert_result(self, result, lower, upper, ac, dc, budget, potential=False):
        self.assertTrue(np.all(result.policy >= 0))
        self.assertAlmostEqual(float(np.sum(result.policy)), 1., places=12)
        self.assertTrue(np.all(result.policy[dc > budget + 1e-12] == 0))
        if potential:
            ev = evaluate_potential_policy(result.policy, lower, upper, ac, dc)
        else:
            ev = evaluate_policy(result.policy, lower, ac, dc)
        self.assertEqual(result.attacker_index, ev.attacker_index)
        self.assertAlmostEqual(result.defender_value, ev.defender_value, places=12)
        self.assertGreaterEqual(result.optimality_gap, 0)
        self.assertLessEqual(result.optimality_gap, DEFAULT_VALUE_TOLERANCE)
        self.assertAlmostEqual(result.value_supremum - ev.defender_value,
                               result.optimality_gap, places=9)

    def test_unattained_supremum(self):
        lower = np.array([[.9, .2], [.9, .8]])
        ac, dc = np.array([0., .2]), np.array([0., .05])
        threshold = (.9 - .2 - .2) / (.8 - .2)
        expected = .9 - .05 * threshold
        for potential in (False, True):
            result = (select_potential_policy(lower, lower, ac, dc, .05) if potential
                      else optimize_policy(lower, ac, dc, .05))
            self.assertAlmostEqual(result.value_supremum, expected, places=10)
            self.assertGreater(result.policy[1], threshold)
            self.assertEqual(result.attacker_index, 0)
            self.assert_result(result, lower, lower, ac, dc, .05, potential)
        boundary = evaluate_policy(np.array([1. - threshold, threshold]), lower, ac, dc)
        self.assertEqual(boundary.attacker_index, 1)
        self.assertLess(boundary.defender_value, expected - .1)

    def test_narrow_response_region(self):
        left, right, slope = .4, .40002, .3
        lower = np.array([[.8, .6 + slope * right, .5 - slope * left],
                          [.8, .6 - slope * (1. - right), .5 + slope * (1. - left)]])
        ac, dc = np.array([0., .2, .3]), np.array([0., .1])
        for potential in (False, True):
            result = (select_potential_policy(lower, lower, ac, dc, .1) if potential
                      else optimize_policy(lower, ac, dc, .1))
            self.assertAlmostEqual(result.value_supremum, .8 - .1 * left, places=10)
            self.assertGreater(result.policy[1], left)
            self.assertLess(result.policy[1], right)
            self.assertEqual(result.attacker_index, 0)
            self.assert_result(result, lower, lower, ac, dc, .1, potential)

    def test_persistent_pessimistic_tie(self):
        lower = np.array([[.75, .5], [.75, .5]])
        ac, dc = np.array([0., .25]), np.array([0., .125])
        for potential in (False, True):
            result = (select_potential_policy(lower, lower, ac, dc, .125) if potential
                      else optimize_policy(lower, ac, dc, .125))
            self.assertEqual(result.attacker_index, 1)
            self.assertAlmostEqual(result.value_supremum, .5, places=12)
            self.assert_result(result, lower, lower, ac, dc, .125, potential)

    def test_identical_attacks(self):
        lower = np.array([[.25, .25], [.875, .875]])
        ac, dc = np.array([.125, .125]), np.array([0., .125])
        result = optimize_policy(lower, ac, dc, .125)
        self.assertEqual(result.attacker_index, 0)
        self.assertAlmostEqual(result.defender_value, .75, places=12)
        self.assert_result(result, lower, lower, ac, dc, .125)

    def test_empty_strict_region_at_a_shared_boundary(self):
        lower = np.array([[.75, .75, .375], [.75, .5, .625]])
        ac, dc = np.array([0., .125, .25]), np.array([0., .125])
        for potential in (False, True):
            result = (select_potential_policy(lower, lower, ac, dc, .125) if potential
                      else optimize_policy(lower, ac, dc, .125))
            self.assertAlmostEqual(result.value_supremum, .5625, places=12)
            self.assertEqual(result.attacker_index, 1)
            self.assertGreater(result.policy[1], .5)
            self.assert_result(result, lower, lower, ac, dc, .125, potential)

    def test_potential_responses_include_endpoint_ties(self):
        lower = np.array([[.2, .8], [.8, .2]])
        upper, ac, dc = np.ones_like(lower), np.zeros(2), np.zeros(2)
        result = select_potential_policy(lower, upper, ac, dc, 0.)
        self.assertAlmostEqual(result.value_supremum, .5, places=12)
        np.testing.assert_allclose(result.policy, [.5, .5], atol=1e-12)
        ev = evaluate_potential_policy(np.array([1., 0.]), lower,
                                       np.array([[.8, .8], [.8, .8]]), ac, dc)
        self.assertEqual(ev.best_response_indices, (0, 1))
        self.assert_result(result, lower, upper, ac, dc, 0., True)

    def test_hard_budget_filters_support(self):
        lower = np.array([[.3], [.9]])
        ac, dc = np.array([0.]), np.array([0., .2])
        for potential in (False, True):
            result = (select_potential_policy(lower, lower, ac, dc, .1) if potential
                      else optimize_policy(lower, ac, dc, .1))
            np.testing.assert_array_equal(result.policy, [1., 0.])
            self.assertAlmostEqual(result.defender_value, .3, places=12)
            self.assert_result(result, lower, lower, ac, dc, .1, potential)

    def test_cost_coefficients_and_zero_attack_weight(self):
        lower = np.array([[.3, .8], [.8, .3]])
        ac, dc = np.array([.1, .2]), np.array([.02, .06])
        for la, ld in ((0., 2.), (.7, 1.8)):
            result = optimize_policy(lower, ac, dc, .06, la, ld)
            expected = two_defense_supremum(lower, lower, la * ac, ld * dc, False)
            self.assertAlmostEqual(result.value_supremum, expected, places=9)
            ev = evaluate_policy(result.policy, lower, ac, dc, la, ld)
            self.assertLessEqual(expected - ev.defender_value, DEFAULT_VALUE_TOLERANCE)

    def test_independent_piecewise_enumeration(self):
        rng = np.random.default_rng(2026)
        for index in range(60):
            lower = rng.uniform(.03, .87, size=(2, 3))
            upper = lower + (1. - lower) * rng.uniform(0., 1., size=(2, 3))
            ac, dc = rng.uniform(0., .3, size=3), rng.uniform(0., .15, size=2)
            for potential in (False, True):
                with self.subTest(index=index, potential=potential):
                    result = (select_potential_policy(lower, upper, ac, dc, .15) if potential
                              else optimize_policy(lower, ac, dc, .15))
                    expected = two_defense_supremum(lower, upper, ac, dc, potential)
                    self.assertAlmostEqual(result.value_supremum, expected, places=8)
                    self.assert_result(result, lower, upper, ac, dc, .15, potential)

    def test_three_defense_simplex_grid(self):
        rng = np.random.default_rng(37)
        for index in range(8):
            lower = rng.uniform(.1, .8, (3, 3))
            upper = lower + (1. - lower) * rng.random((3, 3))
            ac, dc = np.array([.05, .12, .2]), np.array([0., .04, .09])
            grid = np.array([[i / 25., j / 25., (25 - i - j) / 25.]
                             for i in range(26) for j in range(26 - i)])
            for potential in (False, True):
                with self.subTest(index=index, potential=potential):
                    result = (select_potential_policy(lower, upper, ac, dc, .09) if potential
                              else optimize_policy(lower, ac, dc, .09))
                    if potential:
                        values = [evaluate_potential_policy(x, lower, upper, ac, dc).defender_value
                                  for x in grid]
                    else:
                        values = [evaluate_policy(x, lower, ac, dc).defender_value for x in grid]
                    self.assertGreaterEqual(result.defender_value + DEFAULT_VALUE_TOLERANCE, max(values))
                    self.assert_result(result, lower, upper, ac, dc, .09, potential)

    def test_zero_width_potential_game(self):
        rng = np.random.default_rng(93)
        for _ in range(15):
            lower = rng.random((3, 3))
            upper_result = optimize_policy(lower)
            potential = select_potential_policy(lower, lower)
            self.assertAlmostEqual(upper_result.value_supremum, potential.value_supremum, places=8)
            self.assertLessEqual(abs(upper_result.defender_value - potential.defender_value),
                                 DEFAULT_VALUE_TOLERANCE)

    def test_unresolved_numerical_response_raises(self):
        # The exact response advantage is below the evaluator's tie tolerance.
        # A material value gap must produce an error, not an accepted policy.
        delta = BEST_RESPONSE_TOLERANCE / 4.
        lower = np.array([[.75, .5 + delta]])
        ac, dc = np.array([0., .25]), np.array([0.])
        with self.assertRaises(RuntimeError):
            optimize_policy(lower, ac, dc, 0.)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            optimize_policy(np.ones((3, 3)), budget=.01)
        with self.assertRaises(ValueError):
            optimize_policy(np.ones((3, 3)), value_tolerance=float('nan'))
        with self.assertRaises(ValueError):
            select_potential_policy(np.ones((3, 3)), np.zeros((3, 3)))


if __name__ == '__main__':
    unittest.main()
