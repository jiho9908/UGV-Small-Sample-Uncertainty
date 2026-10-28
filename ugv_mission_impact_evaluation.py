from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm, t as student_t

from policy_optimization import (
    ATTACK_COSTS, DEFENSE_COSTS, BUDGET, LAMBDA_ATTACK, LAMBDA_DEFENSE,
    DEFAULT_VALUE_TOLERANCE, BEST_RESPONSE_TOLERANCE,
    PolicyEvaluation, evaluate_policy, optimize_policy,
    evaluate_potential_policy, select_potential_policy,
)



NAVIGATION_TRUTH = np.array(
    [
        [0.84, 0.48, 0.42],
        [0.50, 0.86, 0.46],
        [0.44, 0.52, 0.88],
    ],
    dtype=float,
)

MISSION_NORMAL_TRUTH = 0.95

MISSION_DEGRADED_TRUTH = 0.10

PRIMARY_SAMPLE_SIZE = 15

DEFAULT_REPETITIONS = 1000

DEFAULT_SEED = 1

DEFAULT_SAMPLE_SIZES = (5, 10, 15, 30, 60, 120, 300, 1000)

PRIMARY_INTERVAL_SETTINGS = (
    ("clopper_pearson", 0.90),
    ("clopper_pearson", 0.95),
    ("clopper_pearson", 0.99),
    ("wilson", 0.95),
    ("jeffreys", 0.95),
)

SWEEP_PRIMARY_SETTING = ("clopper_pearson", 0.95)

REAL_WORLD_SUCCESS_COUNTS = np.array([[15, 6], [15, 13]], dtype=int)

REAL_WORLD_SAMPLE_SIZE = 15

REAL_WORLD_PRIMARY_ATTACK_COST = 0.30

REAL_WORLD_PRIMARY_DEFENSE_COST = 0.20

REAL_WORLD_PRIMARY_BUDGET = 0.20

REAL_WORLD_ATTACK_COST_SWEEP = (0.10, 0.30, 0.70)

REAL_WORLD_DEFENSE_COST_SWEEP = (0.10, 0.20, 0.50)

REAL_WORLD_BUDGET_SWEEP = (0.10, 0.20, 0.50)

REAL_WORLD_THRESHOLD_INCREMENT = 0.001

SCENARIO_NAVIGATION = {
    "table_ii": NAVIGATION_TRUTH.copy(),
    "small_defense_contrast": np.array([[.70, .68, .66], [.69, .71, .67], [.68, .69, .72]]),
    "broadly_effective_defense": np.array([[.90, .89, .88], [.60, .68, .61], [.59, .60, .69]]),
    "stable_first_attack": np.array([[.30, .82, .86], [.35, .87, .84], [.33, .84, .90]]),
    "near_uniform_policy_boundary": np.array([[.76, .605, .571], [.64, .725, .571], [.64, .605, .691]]),
    "far_uniform_policy_boundary": np.array([[.88, .46, .86], [.79, .54, .85], [.82, .43, .91]]),
}

DEFAULT_TRAJECTORY_BUDGETS = (15, 60, 300)


@dataclass(frozen=True)
class RealWorldPolicyResult:

    policy: np.ndarray
    attacker_index: int
    guaranteed_mission: float
    expected_defense_cost: float
    defender_value: float
    deterrence_threshold: float


def mission_truth_matrix() -> np.ndarray:

    return (
        NAVIGATION_TRUTH * MISSION_NORMAL_TRUTH
        + (1.0 - NAVIGATION_TRUTH) * MISSION_DEGRADED_TRUTH
    )


def generate_counts(
    repetitions: int, sample_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    navigation = np.empty((repetitions, 3, 3), dtype=int)
    mission_normal = np.empty_like(navigation)
    mission_degraded = np.empty_like(navigation)
    for repetition in range(repetitions):
        navigation[repetition] = rng.binomial(sample_size, NAVIGATION_TRUTH)
        mission_normal[repetition] = rng.binomial(
            sample_size, MISSION_NORMAL_TRUTH, size=(3, 3)
        )
        mission_degraded[repetition] = rng.binomial(
            sample_size, MISSION_DEGRADED_TRUTH, size=(3, 3)
        )
    return navigation, mission_normal, mission_degraded


def _binomial_inputs(successes, sample_size, confidence):
    y, n = np.broadcast_arrays(np.asarray(successes, dtype=float),
                                np.asarray(sample_size, dtype=float))
    if not (0 < confidence < 1):
        raise ValueError("Confidence must lie strictly between zero and one.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(n)):
        raise ValueError("Binomial counts must be finite.")
    if np.any(n < 0) or np.any(y < 0) or np.any(y > n):
        raise ValueError("Counts must satisfy 0 <= successes <= trials.")
    if np.any(y != np.floor(y)) or np.any(n != np.floor(n)):
        raise ValueError("Binomial counts must be integers.")
    return y, n


def clopper_pearson_interval(successes, sample_size, confidence):

    y, n = _binomial_inputs(successes, sample_size, confidence)
    alpha = 1. - confidence
    lower, upper = np.zeros_like(y), np.ones_like(y)
    positive, incomplete = y > 0, y < n
    lower[positive] = beta.ppf(alpha / 2., y[positive], n[positive] - y[positive] + 1.)
    upper[incomplete] = beta.ppf(1. - alpha / 2., y[incomplete] + 1., n[incomplete] - y[incomplete])
    return lower, upper


def wilson_interval(successes, sample_size, confidence):

    y, n = _binomial_inputs(successes, sample_size, confidence)
    lower, upper = np.zeros_like(y), np.ones_like(y)
    active = n > 0
    trials, phat = n[active], y[active] / n[active]
    z = norm.ppf(.5 + confidence / 2.)
    denominator = 1. + z * z / trials
    center = (phat + z * z / (2. * trials)) / denominator
    half = z * np.sqrt(phat * (1. - phat) / trials + z * z / (4. * trials * trials)) / denominator
    lower[active], upper[active] = np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)
    return lower, upper


def jeffreys_interval(successes, sample_size, confidence):

    y, n = _binomial_inputs(successes, sample_size, confidence)
    alpha = 1. - confidence
    lower = beta.ppf(alpha / 2., y + .5, n - y + .5)
    upper = beta.ppf(1. - alpha / 2., y + .5, n - y + .5)
    return np.where(n == 0, 0., lower), np.where(n == 0, 1., upper)


def binomial_interval(
    successes: np.ndarray,
    sample_size: int,
    confidence: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "clopper_pearson":
        return clopper_pearson_interval(successes, sample_size, confidence)
    if method == "wilson":
        return wilson_interval(successes, sample_size, confidence)
    if method == "jeffreys":
        return jeffreys_interval(successes, sample_size, confidence)
    raise ValueError(f"Unknown interval method: {method}")


def propagate_mission_intervals(
    navigation_interval: tuple[np.ndarray, np.ndarray],
    normal_interval: tuple[np.ndarray, np.ndarray],
    degraded_interval: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:

    nav_lower, nav_upper = navigation_interval
    normal_lower, normal_upper = normal_interval
    degraded_lower, degraded_upper = degraded_interval
    vertices = []
    for navigation in (nav_lower, nav_upper):
        for normal in (normal_lower, normal_upper):
            for degraded in (degraded_lower, degraded_upper):
                vertices.append(
                    navigation * normal + (1.0 - navigation) * degraded
                )
    stacked = np.stack(vertices, axis=-1)
    return np.min(stacked, axis=-1), np.max(stacked, axis=-1)


def point_mission_estimate(
    navigation_successes: np.ndarray,
    normal_successes: np.ndarray,
    degraded_successes: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    navigation = navigation_successes / sample_size
    normal = normal_successes / sample_size
    degraded = degraded_successes / sample_size
    return navigation * normal + (1.0 - navigation) * degraded


def solve_real_world_policy(
    mission_lower: np.ndarray,
    attack_cost: float,
    defense_cost: float,
    budget: float,
    threshold_increment: float = REAL_WORLD_THRESHOLD_INCREMENT,
    best_response_tolerance: float = BEST_RESPONSE_TOLERANCE,
) -> RealWorldPolicyResult:

    if not math.isfinite(threshold_increment) or threshold_increment <= 0.0:
        raise ValueError("threshold_increment must be a finite positive value.")

    lower = np.asarray(mission_lower, dtype=float)
    if lower.shape != (2, 2):
        raise ValueError("The real-world reduced model must have shape (2, 2).")
    if not math.isclose(float(lower[0, 0]), float(lower[1, 0]), abs_tol=1e-12):
        raise ValueError("The two no-attack lower utilities must be equal.")
    l0 = float(lower[0, 0])
    l1 = float(lower[0, 1])
    l2 = float(lower[1, 1])
    improvement = l2 - l1
    if improvement <= 0.0:
        raise ValueError("The enhanced defense must improve the attack-cell lower value.")
    threshold = (l0 - l1 - attack_cost) / improvement

    x2_candidates = [0.0]
    if defense_cost <= budget + 1.0e-12:
        x2_candidates.append(1.0)

        if 0.0 <= threshold < 1.0:
            x2_candidates.append(min(1.0, threshold + threshold_increment))
    x2_candidates = sorted(set(x2_candidates))

    attack_costs = np.asarray([0.0, attack_cost], dtype=float)
    defense_costs = np.asarray([0.0, defense_cost], dtype=float)
    evaluated: list[tuple[float, PolicyEvaluation]] = []
    for x2 in x2_candidates:
        policy = np.asarray([1.0 - x2, x2], dtype=float)
        evaluation = evaluate_policy(
            policy,
            lower,
            attack_costs=attack_costs,
            defense_costs=defense_costs,
            best_response_tolerance=best_response_tolerance,
        )
        evaluated.append((x2, evaluation))
    x2, evaluation = max(
        evaluated,
        key=lambda item: (
            item[1].defender_value,
            -item[1].expected_defense_cost,
            -item[0],
        ),
    )
    return RealWorldPolicyResult(
        policy=np.asarray([1.0 - x2, x2], dtype=float),
        attacker_index=evaluation.attacker_index,
        guaranteed_mission=evaluation.guaranteed_mission,
        expected_defense_cost=evaluation.expected_defense_cost,
        defender_value=evaluation.defender_value,
        deterrence_threshold=threshold,
    )


def _envelope_evaluation(policy, vertices, attack_costs, defense_costs):
    mission = np.min(np.einsum("i,sij->sj", policy, vertices), axis=0)
    payoffs = 1. - mission - attack_costs
    responses = tuple(int(j) for j in np.flatnonzero(payoffs >= np.max(payoffs) - BEST_RESPONSE_TOLERANCE))
    response = min(responses, key=lambda j: (float(mission[j]), j))
    cost = float(policy @ defense_costs)
    return PolicyEvaluation(response, responses, float(mission[response]), cost,
                            float(mission[response] - cost))


def _add_crossing(points, intercept, slope):
    if abs(slope) > 1e-14:
        root = -intercept / slope
        if 0. < root < 1.:
            points.add(float(root))


def optimize_two_defense_envelope(vertices, attack_costs, defense_costs, budget,
                                  value_tolerance=DEFAULT_VALUE_TOLERANCE):

    vertices = np.asarray(vertices, dtype=float)
    ac, dc = np.asarray(attack_costs), np.asarray(defense_costs)
    if vertices.ndim != 3 or vertices.shape[1] != 2:
        raise ValueError("The envelope solver requires two defenses.")
    feasible = dc <= budget + 1e-12
    if not np.any(feasible):
        raise ValueError("No feasible defense.")
    evaluate = lambda t: _envelope_evaluation(np.array([1. - t, t]), vertices, ac, dc)
    if not np.all(feasible):
        t = float(feasible[1])
        ev = evaluate(t)
        return np.array([1. - t, t]), ev, ev.defender_value
    intercept = vertices[:, 0, :]
    slope = vertices[:, 1, :] - vertices[:, 0, :]
    points = {0., 1.}
    lines = [(float(intercept[s, j] + ac[j]), float(slope[s, j]))
             for s in range(len(vertices)) for j in range(len(ac))]
    for (a, b), (c, d) in itertools.combinations(lines, 2):
        _add_crossing(points, a - c, b - d)
    points = sorted(points)
    candidates = [(t, evaluate(t)) for t in points]
    supremum = max(ev.defender_value for _, ev in candidates)
    for left, right in zip(points[:-1], points[1:]):
        if right - left < 1e-13:
            continue
        middle = (left + right) / 2.
        middle_evaluation = evaluate(middle)
        j = middle_evaluation.attacker_index
        state = int(np.argmin(intercept[:, j] + slope[:, j] * middle))
        net_intercept = float(intercept[state, j] - dc[0])
        net_slope = float(slope[state, j] - (dc[1] - dc[0]))
        for endpoint in (left, right):
            limit = net_intercept + net_slope * endpoint
            supremum = max(supremum, limit)
            distance = abs(limit - (net_intercept + net_slope * middle))
            fraction = min(.5, value_tolerance / (2. * distance)) if distance else .5
            for _ in range(12):
                t = endpoint + fraction * (middle - endpoint)
                ev = evaluate(t)
                if abs(ev.defender_value - (net_intercept + net_slope * t)) < 1e-8:
                    candidates.append((t, ev))
                    break
                fraction = min(1., fraction * 4.)
    t, evaluation = max(candidates, key=lambda item: (item[1].defender_value,
                                                     -item[1].expected_defense_cost, -item[0]))
    if supremum - evaluation.defender_value > value_tolerance + 1e-8:
        raise RuntimeError("The piecewise envelope could not resolve the requested tolerance.")
    return np.array([1. - t, t]), evaluation, supremum


def derived_seed(seed, family, case, sample_size):
    return int(np.random.SeedSequence([seed, family, case, sample_size]).generate_state(1)[0])


def generate_scenario_counts(repetitions, sample_size, seed, navigation_truth):
    rng = np.random.default_rng(seed)
    shape = (repetitions,) + navigation_truth.shape
    navigation = np.empty(shape, dtype=int)
    normal = np.empty(shape, dtype=int)
    degraded = np.empty(shape, dtype=int)
    for r in range(repetitions):
        navigation[r] = rng.binomial(sample_size, navigation_truth)
        normal[r] = rng.binomial(sample_size, MISSION_NORMAL_TRUTH, size=navigation_truth.shape)
        degraded[r] = rng.binomial(sample_size, MISSION_DEGRADED_TRUTH, size=navigation_truth.shape)
    return navigation, normal, degraded


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_interval(values):
    data = np.asarray(values, dtype=float)
    mean = float(np.mean(data))
    error = float(student_t.ppf(.975, data.size - 1) * np.std(data, ddof=1)
                  / math.sqrt(data.size))
    return mean, mean - error, mean + error


def text_result(records, section, quantity, value, low=None, high=None):
    records.append({'section': section, 'quantity': quantity, 'value': value,
                    'mc95_lower': '' if low is None else low,
                    'mc95_upper': '' if high is None else high})


def policy_regret(policy, truth, supremum):
    regret = supremum - evaluate_policy(policy, truth).defender_value
    if regret < -1e-7:
        raise RuntimeError('A policy exceeded the generating-model supremum.')
    return max(0., regret)


def run_illustrative(output, seed, text):
    n = PRIMARY_SAMPLE_SIZE
    counts = generate_counts(1, n, seed)
    intervals = [binomial_interval(y, n, .95, 'clopper_pearson') for y in counts]
    lower, upper = (v[0] for v in propagate_mission_intervals(*intervals))
    point = point_mission_estimate(*counts, n)[0]
    truth = mission_truth_matrix()
    worked = {'defense': 'd2', 'attack': 'a2', 'trials_per_local_row': n}
    for name, y, bounds in zip(('navigation_normal', 'mission_success_given_normal',
                               'mission_success_given_degraded'), counts, intervals):
        worked.update({name + '_successes': int(y[0, 1, 1]),
                       name + '_lower': float(bounds[0][0, 1, 1]),
                       name + '_upper': float(bounds[1][0, 1, 1])})
    worked.update({'point_mission_estimate': float(point[1, 1]),
                   'mission_interval_lower': float(lower[1, 1]),
                   'mission_interval_upper': float(upper[1, 1]),
                   'ground_truth_mission': float(truth[1, 1]),
                   'covered': bool(lower[1, 1] <= truth[1, 1] <= upper[1, 1])})
    write_csv(output / 'table_03_worked_cell.csv', [worked])
    write_csv(output / 'table_04a_mission_intervals.csv', [
        {'defense': f'd{i + 1}', 'attack': f'a{j + 1}',
         'mission_interval_lower': float(lower[i, j]),
         'mission_interval_upper': float(upper[i, j])}
        for i in range(3) for j in range(3)])
    rows = []
    for name, matrix in [('Oracle', truth), ('Point', point), ('CP95', lower)]:
        selected = optimize_policy(matrix)
        rows.append({'policy_source': name,
                     **{f'policy_d{i + 1}': float(v) for i, v in enumerate(selected.policy)},
                     'model_conditional_defender_value': selected.defender_value})
        text_result(text, 'IV-A2', name + '_objective_loss', selected.optimality_gap)
    write_csv(output / 'table_04b_defense_policies.csv', rows)


def evaluate_samples(matrices, truth, oracle_policy, supremum):
    policies, responses, distances, regrets = [], [], [], []
    for matrix in matrices:
        result = optimize_policy(matrix)
        policies.append(result.policy)
        responses.append(result.attacker_index)
        distances.append(float(np.sum(np.abs(result.policy - oracle_policy))))
        regrets.append(policy_regret(result.policy, truth, supremum))
    return {'policies': np.asarray(policies), 'responses': np.asarray(responses),
            'distances': np.asarray(distances), 'regrets': np.asarray(regrets)}


def analyze_samples(n, repetitions, seed, settings, oracle_policy, supremum):
    truth = mission_truth_matrix()
    counts = generate_counts(repetitions, n, seed)
    point = evaluate_samples(point_mission_estimate(*counts, n), truth, oracle_policy, supremum)
    models = {}
    for method, confidence in settings:
        local = [binomial_interval(y, n, confidence, method) for y in counts]
        lower, upper = propagate_mission_intervals(*local)
        model = evaluate_samples(lower, truth, oracle_policy, supremum)
        covered = (lower <= truth) & (truth <= upper)
        successes = int(np.sum(np.all(covered, axis=(1, 2))))
        joint_lo, joint_hi = wilson_interval(np.array([successes]), repetitions, .95)
        widths = np.mean(upper - lower, axis=(1, 2))
        local_widths = np.mean(np.concatenate([
            (hi - lo).reshape(repetitions, -1) for lo, hi in local], axis=1), axis=1)
        model.update({'cell_coverage': float(np.mean(np.mean(covered, axis=(1, 2)))),
                      'joint_coverage': successes / repetitions,
                      'joint_coverage_mc95_low': float(joint_lo[0]),
                      'joint_coverage_mc95_high': float(joint_hi[0]),
                      'mission_width': mean_interval(widths),
                      'local_width': mean_interval(local_widths)})
        models[(method, confidence)] = model
    return point, models


def run_repeated(output, repetitions, seed, text):
    truth = mission_truth_matrix()
    oracle = optimize_policy(truth)
    oracle_policy, supremum = oracle.policy, oracle.value_supremum
    primary = analyze_samples(PRIMARY_SAMPLE_SIZE, repetitions, seed,
                              PRIMARY_INTERVAL_SETTINGS, oracle_policy, supremum)
    point, models = primary
    cp = models[SWEEP_PRIMARY_SETTING]
    text_result(text, 'IV-A3a', 'Point_minus_CP_regret_N15',
                *mean_interval(point['regrets'] - cp['regrets']))
    interval_rows = []
    for (method, confidence), model in models.items():
        interval_rows.append({'interval_method': method, 'confidence_level': confidence,
                              'joint_coverage': model['joint_coverage'],
                              'joint_coverage_mc95_low': model['joint_coverage_mc95_low'],
                              'joint_coverage_mc95_high': model['joint_coverage_mc95_high'],
                              'mean_mission_interval_width': model['mission_width'][0],
                              'oracle_policy_l1': float(np.mean(model['distances'])),
                              'policy_regret': float(np.mean(model['regrets']))})
        if method in ('wilson', 'jeffreys'):
            text_result(text, 'IV-A3b', method + '_CP_policy_l1',
                        float(np.mean(np.sum(np.abs(model['policies'] - cp['policies']), axis=1))))
            text_result(text, 'IV-A3b', method + '_CP_response_agreement',
                        float(np.mean(model['responses'] == cp['responses'])))
    text_result(text, 'IV-A3b', 'minimum_cell_coverage',
                min(m['cell_coverage'] for m in models.values()))
    text_result(text, 'IV-A3b', 'maximum_cell_coverage',
                max(m['cell_coverage'] for m in models.values()))
    write_csv(output / 'table_06_interval_constructions.csv', interval_rows)
    sweep = []
    for n in DEFAULT_SAMPLE_SIZES:
        print(f'  Sample size N={n}', flush=True)
        point, models = primary if n == PRIMARY_SAMPLE_SIZE else analyze_samples(
            n, repetitions, seed, [SWEEP_PRIMARY_SETTING], oracle_policy, supremum)
        cp = models[SWEEP_PRIMARY_SETTING]
        distance = mean_interval(np.sum(np.abs(cp['policies'] - point['policies']), axis=1))
        row = {'sample_size': n, 'cp_cell_coverage': cp['cell_coverage'],
               'cp_joint_coverage': cp['joint_coverage'],
               'cp_mission_width': cp['mission_width'][0],
               'cp_local_width': cp['local_width'][0],
               'cp_oracle_l1': float(np.mean(cp['distances'])),
               'cp_regret': float(np.mean(cp['regrets'])),
               'point_oracle_l1': float(np.mean(point['distances'])),
               'point_regret': float(np.mean(point['regrets'])),
               'cp_point_l1': distance[0]}
        for name, bounds in [('cp_mission_width', cp['mission_width']),
                              ('cp_local_width', cp['local_width']),
                              ('cp_point_l1', distance)]:
            row[name + '_mc95_lower'], row[name + '_mc95_upper'] = bounds[1:]
        sweep.append(row)
    write_csv(output / 'table_05_sample_size.csv', sweep)
    create_figure(sweep, output)


def run_scenarios(output, repetitions, seed, text):
    rows = []
    n = PRIMARY_SAMPLE_SIZE
    for index, (name, navigation) in enumerate(SCENARIO_NAVIGATION.items()):
        print(f'  Generating matrix: {name}', flush=True)
        truth = navigation * MISSION_NORMAL_TRUTH + (1. - navigation) * MISSION_DEGRADED_TRUTH
        supremum = optimize_policy(truth).value_supremum
        if name in ('near_uniform_policy_boundary', 'far_uniform_policy_boundary'):
            payoffs = np.sort(1. - np.full(3, 1. / 3.) @ truth - ATTACK_COSTS)
            text_result(text, 'IV-A4', name + '_payoff_gap', float(payoffs[-1] - payoffs[-2]))
        stream_seed = seed if name == 'table_ii' else derived_seed(seed, 200, index, n)
        counts = generate_scenario_counts(repetitions, n, stream_seed, navigation)
        lower, upper = propagate_mission_intervals(*[
            binomial_interval(y, n, .95, 'clopper_pearson') for y in counts])
        point = point_mission_estimate(*counts, n)
        regrets = np.empty((repetitions, 3))
        for r in range(repetitions):
            selected = (optimize_policy(point[r]), optimize_policy(lower[r]),
                        select_potential_policy(lower[r], upper[r]))
            for j, result in enumerate(selected):
                regrets[r, j] = policy_regret(result.policy, truth, supremum)
        rows.append({'scenario': name, 'Point_regret': float(np.mean(regrets[:, 0])),
                     'CP_U_regret': float(np.mean(regrets[:, 1])),
                     'CP_P_regret': float(np.mean(regrets[:, 2]))})
        if name == 'broadly_effective_defense':
            text_result(text, 'IV-A4', 'broadly_effective_CP_U_minus_CP_P_regret',
                        *mean_interval(regrets[:, 1] - regrets[:, 2]))
    write_csv(output / 'table_07_generating_matrices.csv', rows)


def run_propagation(output, repetitions, seed, text):
    navigation_given_localization = np.array([.25, .95])
    mission_given_navigation = np.array([.05, .95])
    navigation_truth = .25 + (.95 - .25) * NAVIGATION_TRUTH
    truth = .05 + (.95 - .05) * navigation_truth
    supremum = optimize_policy(truth).value_supremum
    rows = []
    for budget in DEFAULT_TRAJECTORY_BUDGETS:
        print(f'  Trajectories per cell B={budget}', flush=True)
        rng = np.random.default_rng(derived_seed(seed, 400, 0, budget))
        successes = np.zeros((repetitions, 3, 3, 5), dtype=int)
        trials = np.zeros_like(successes)
        endpoints = np.zeros((repetitions, 3, 3), dtype=int)
        shape = (3, 3, budget)
        for r in range(repetitions):
            localization = rng.random(shape) < NAVIGATION_TRUTH[:, :, None]
            navigation = rng.random(shape) < navigation_given_localization[localization.astype(int)]
            mission = rng.random(shape) < mission_given_navigation[navigation.astype(int)]
            successes[r, :, :, 0] = localization.sum(axis=-1)
            trials[r, :, :, 0] = budget
            successes[r, :, :, 1] = (localization & navigation).sum(axis=-1)
            trials[r, :, :, 1] = localization.sum(axis=-1)
            successes[r, :, :, 2] = (~localization & navigation).sum(axis=-1)
            trials[r, :, :, 2] = (~localization).sum(axis=-1)
            successes[r, :, :, 3] = (navigation & mission).sum(axis=-1)
            trials[r, :, :, 3] = navigation.sum(axis=-1)
            successes[r, :, :, 4] = (~navigation & mission).sum(axis=-1)
            trials[r, :, :, 4] = (~navigation).sum(axis=-1)
            endpoints[r] = mission.sum(axis=-1)
        local_lo, local_hi = binomial_interval(successes, trials, .95, 'clopper_pearson')
        graph_lo, graph_hi = np.ones((repetitions, 3, 3)), np.zeros((repetitions, 3, 3))
        for bits in itertools.product([False, True], repeat=5):
            p, q1, q0, r1, r0 = [local_hi[..., j] if bit else local_lo[..., j]
                                 for j, bit in enumerate(bits)]
            nav = p * q1 + (1. - p) * q0
            value = nav * r1 + (1. - nav) * r0
            graph_lo = np.minimum(graph_lo, value)
            graph_hi = np.maximum(graph_hi, value)
        endpoint_lo, endpoint_hi = binomial_interval(endpoints, budget, .95, 'clopper_pearson')
        regrets = np.empty((repetitions, 2))
        widths = np.empty((repetitions, 2))
        for r in range(repetitions):
            for j, (lo, hi) in enumerate(((graph_lo[r], graph_hi[r]),
                                          (endpoint_lo[r], endpoint_hi[r]))):
                policy = optimize_policy(lo).policy
                regrets[r, j] = policy_regret(policy, truth, supremum)
                widths[r, j] = float(np.mean(hi - lo))
        rows.append({'trajectories_per_cell': budget,
                     'graph_width': float(np.mean(widths[:, 0])),
                     'endpoint_width': float(np.mean(widths[:, 1])),
                     'graph_regret': float(np.mean(regrets[:, 0])),
                     'endpoint_regret': float(np.mean(regrets[:, 1]))})
        text_result(text, 'IV-A4', f'Endpoint_minus_Graph_regret_B{budget}',
                    *mean_interval(regrets[:, 1] - regrets[:, 0]))
    write_csv(output / 'table_08_propagation.csv', rows)


def run_assumptions(repetitions, seed, text):
    intercept = np.array([[.9, .2], [.9, .85]])
    slope = np.array([[0., .7], [0., -.6]])
    ac, dc, budget = np.array([0., .2]), np.array([0., .04]), .04
    vertices = np.stack([intercept + slope * theta for theta in (.3, .7)])
    lower, upper = np.min(vertices, axis=0), np.max(vertices, axis=0)
    rectangular = optimize_policy(lower, ac, dc, budget)
    coupled, _, _ = optimize_two_defense_envelope(vertices, ac, dc, budget)
    uniform = np.array([.5, .5])
    coupled_values = np.einsum('i,sij->sj', uniform, vertices)[:, 1]
    truth = intercept + .5 * slope
    supremum = optimize_policy(truth, ac, dc, budget).value_supremum
    results = {
        'rectangular_interval_lower': float(uniform @ lower[:, 1]),
        'rectangular_interval_upper': float(uniform @ upper[:, 1]),
        'coupled_interval_lower': float(np.min(coupled_values)),
        'coupled_interval_upper': float(np.max(coupled_values)),
        'rectangular_x2': float(rectangular.policy[1]),
        'coupled_x2': float(coupled[1]),
        'rectangular_regret': max(0., supremum - evaluate_policy(
            rectangular.policy, truth, ac, dc).defender_value),
        'coupled_regret': max(0., supremum - evaluate_policy(
            coupled, truth, ac, dc).defender_value),
    }
    for name, value in results.items():
        text_result(text, 'IV-A5', name, value)
    n = 300
    probabilities = np.stack([NAVIGATION_TRUTH,
                              np.full((3, 3), MISSION_NORMAL_TRUTH),
                              np.full((3, 3), MISSION_DEGRADED_TRUTH)])
    for rho, stream in ((0., 0), (.6, 2)):
        rng = np.random.default_rng(derived_seed(seed, 500, stream, n))
        state = rng.random((repetitions, 3, 3, 3)) < probabilities
        counts = state.astype(int)
        for _ in range(n - 1):
            transition = (1. - rho) * probabilities + rho * state
            state = rng.random(state.shape) < transition
            counts += state
        lo, hi = binomial_interval(counts, n, .95, 'clopper_pearson')
        coverage = np.mean((lo <= probabilities) & (probabilities <= hi), axis=(1, 2, 3))
        text_result(text, 'IV-A5', f'row_coverage_rho{rho:.1f}', float(np.mean(coverage)))


def run_real_world(output, text):
    point = REAL_WORLD_SUCCESS_COUNTS / REAL_WORLD_SAMPLE_SIZE
    ca, cd, budget = REAL_WORLD_PRIMARY_ATTACK_COST, REAL_WORLD_PRIMARY_DEFENSE_COST, REAL_WORLD_PRIMARY_BUDGET
    rows, primary_lower, primary_upper = [], None, None
    settings = list(PRIMARY_INTERVAL_SETTINGS) + [('point', None)]
    labels = {'clopper_pearson': 'CP', 'wilson': 'Wilson', 'jeffreys': 'Jeffreys'}
    for method, confidence in settings:
        lo, hi = (point, point) if method == 'point' else binomial_interval(
            REAL_WORLD_SUCCESS_COUNTS, REAL_WORLD_SAMPLE_SIZE, confidence, method)
        selected = solve_real_world_policy(lo, ca, cd, budget)
        label = 'Point' if method == 'point' else f'{labels[method]} {confidence:.0%}'
        rows.append({'setting': label,
                     'mean_interval_width': '' if method == 'point' else float(np.mean(hi - lo)),
                     'deterrence_threshold_tau': selected.deterrence_threshold,
                     'policy_d2_x2': float(selected.policy[1]),
                     'model_conditional_mission_lower_GM': selected.guaranteed_mission,
                     'defender_value_Phi': selected.defender_value})
        if (method, confidence) == SWEEP_PRIMARY_SETTING:
            primary_lower, primary_upper = lo, hi
            write_csv(output / 'table_09_observations.csv', [
                {'defense': f'd{i + 1}', 'attack': f'a{j + 1}',
                 'successes': int(REAL_WORLD_SUCCESS_COUNTS[i, j]), 'trials': REAL_WORLD_SAMPLE_SIZE,
                 'point_probability': float(point[i, j]),
                 'interval_lower': float(lo[i, j]), 'interval_upper': float(hi[i, j])}
                for i in range(2) for j in range(2)])
    write_csv(output / 'table_10_real_world_intervals.csv', rows)
    lo, hi = primary_lower, primary_upper
    grouped = {}
    for b, d, a in itertools.product(REAL_WORLD_BUDGET_SWEEP,
                                     REAL_WORLD_DEFENSE_COST_SWEEP, REAL_WORLD_ATTACK_COST_SWEEP):
        selected = solve_real_world_policy(lo, a, d, b)
        feasible = b >= d
        key = ('b>=c_D' if feasible else 'b<c_D', d if feasible else '', a)
        row = {'budget_condition': key[0], 'defense_cost': key[1], 'attack_cost': a,
               'policy_d2_x2': float(selected.policy[1]),
               'attacker_response': f'a{selected.attacker_index + 1}'}
        if key in grouped and (abs(grouped[key]['policy_d2_x2'] - row['policy_d2_x2']) > 1e-10
                               or grouped[key]['attacker_response'] != row['attacker_response']):
            raise RuntimeError('Cost-budget outcomes differ within a table group.')
        grouped[key] = row
    cost_rows = sorted(grouped.values(), key=lambda r: (r['budget_condition'] != 'b<c_D',
                       float(r['defense_cost'] or 0), r['attack_cost']))
    write_csv(output / 'table_11_cost_budget.csv', cost_rows)
    cp_row = next(r for r in rows if r['setting'] == 'CP 95%')
    text_result(text, 'IV-B3', 'CP_minus_Point_x2', cp_row['policy_d2_x2'] - rows[-1]['policy_d2_x2'])
    text_result(text, 'IV-B3', 'strict_deterrence_objective_loss',
                LAMBDA_DEFENSE * cd * REAL_WORLD_THRESHOLD_INCREMENT)
    for name, value in [('non_deterrable_upper_attack_cost', lo[0, 0] - lo[1, 1]),
                         ('zero_defense_deterrence_attack_cost', lo[0, 0] - lo[0, 1]),
                         ('attack_cell_utility_improvement', lo[1, 1] - lo[0, 1])]:
        text_result(text, 'IV-B3', name, float(value))
    ac, dc = np.array([0., ca]), np.array([0., cd])
    cp = solve_real_world_policy(lo, ca, cd, budget,
                                best_response_tolerance=BEST_RESPONSE_TOLERANCE)
    point_policy = solve_real_world_policy(point, ca, cd, budget,
                                          best_response_tolerance=BEST_RESPONSE_TOLERANCE)
    potential = select_potential_policy(lo, hi, ac, dc, budget)
    for name, policy in [('CP_U', cp.policy), ('Point', point_policy.policy), ('CP_P', potential.policy)]:
        upper_value = evaluate_policy(policy, lo, ac, dc).defender_value
        potential_value = evaluate_potential_policy(policy, lo, hi, ac, dc).defender_value
        for suffix, value in [('x2', float(policy[1])), ('Phi', upper_value), ('Phi_pot', potential_value)]:
            text_result(text, 'IV-B4', name + '_' + suffix, value)
    realization = lo.copy()
    realization[:, 0] = hi[:, 0]
    example = evaluate_policy(cp.policy, realization, ac, dc)
    payoffs = 1. - cp.policy @ realization - ac
    for name, value in [('upper_rule_response', f'a{cp.attacker_index + 1}'),
                         ('realization_attacker_payoff_a1', float(payoffs[0])),
                         ('realization_attacker_payoff_a2', float(payoffs[1])),
                         ('realization_response', f'a{example.attacker_index + 1}'),
                         ('realization_defender_value', example.defender_value),
                         ('potential_value_intercept', float(lo[0, 1])),
                         ('potential_value_slope', float(lo[1, 1] - lo[0, 1] - cd))]:
        text_result(text, 'IV-B4', name, value)


def create_figure(rows, output):
    n = np.array([r['sample_size'] for r in rows])
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 7,
                         'axes.titlesize': 7.5, 'axes.labelsize': 7,
                         'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'axes.linewidth': .6, 'lines.linewidth': 1,
                         'lines.markersize': 3, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.65), layout='constrained')
    for ax, key, label, color, marker in [
        (axes[0], 'cp_mission_width', 'Mission', '.15', 'o'),
        (axes[0], 'cp_local_width', 'Local', '.55', 's'),
        (axes[1], 'cp_point_l1', None, '.15', 'o')]:
        means = np.array([r[key] for r in rows])
        lo = np.array([r[key + '_mc95_lower'] for r in rows])
        hi = np.array([r[key + '_mc95_upper'] for r in rows])
        ax.errorbar(n, means, yerr=np.maximum(0, np.vstack([means - lo, hi - means])),
                    color=color, marker=marker, capsize=1.5, elinewidth=.6, label=label)
    axes[0].set_title('(a) Interval width')
    axes[0].set_ylabel('Mean width')
    axes[0].legend(fontsize=6, frameon=False)
    axes[1].set_title('(b) CP-point distance')
    axes[1].set_ylabel(r'Mean policy $\ell_1$')
    axes[0].set_yticks([0, .3, .6, .9])
    axes[1].set_yticks([0, .2, .4, .6])
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks([5, 15, 60, 300, 1000], ['5', '15', '60', '300', '1000'])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'Observations per row, $N$')
        ax.grid(alpha=.25, linewidth=.5)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axvline(15, color='.5', linestyle='--', linewidth=.6)
    fig.savefig(output / 'figure_04.pdf', dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Numerical evaluation for the UGV mission-impact paper.')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--repetitions', type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.repetitions < 2 or args.seed < 0:
        parser.error('Use at least two repetitions and a nonnegative seed.')
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('The output path must be a new or empty directory.')
    output.mkdir(parents=True, exist_ok=True)
    text = []
    print('Illustrative experiment', flush=True)
    run_illustrative(output, args.seed, text)
    print('Repeated sampling', flush=True)
    run_repeated(output, args.repetitions, args.seed, text)
    print('Generating-matrix comparison', flush=True)
    run_scenarios(output, args.repetitions, args.seed, text)
    print('Matched trajectories', flush=True)
    run_propagation(output, args.repetitions, args.seed, text)
    print('Assumption sensitivity', flush=True)
    run_assumptions(args.repetitions, args.seed, text)
    print('Real-world instantiation', flush=True)
    run_real_world(output, text)
    write_csv(output / 'text_results.csv', text)
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()

