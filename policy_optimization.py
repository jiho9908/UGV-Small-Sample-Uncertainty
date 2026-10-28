from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog


ATTACK_COSTS = np.array([0.12, 0.15, 0.18], dtype=float)
DEFENSE_COSTS = np.array([0.05, 0.08, 0.12], dtype=float)
BUDGET = 0.12
LAMBDA_ATTACK = 1.0
LAMBDA_DEFENSE = 1.0
DEFAULT_VALUE_TOLERANCE = 1.0e-6
BEST_RESPONSE_TOLERANCE = 1.0e-9
LP_TOLERANCE = 1.0e-10
FEASIBILITY_TOLERANCE = 1.0e-8
LP_OPTIONS = {
    "primal_feasibility_tolerance": LP_TOLERANCE,
    "dual_feasibility_tolerance": LP_TOLERANCE,
}

@dataclass(frozen=True)
class PolicyEvaluation:
    attacker_index: int
    best_response_indices: tuple[int, ...]
    guaranteed_mission: float
    expected_defense_cost: float
    defender_value: float


@dataclass(frozen=True)
class PolicyResult:
    policy: np.ndarray
    attacker_index: int
    guaranteed_mission: float
    expected_defense_cost: float
    defender_value: float
    value_supremum: float
    optimality_gap: float


def _game_inputs(mission, attack_costs, defense_costs, budget, lambda_attack,
                 lambda_defense):
    mission = np.asarray(mission, dtype=float)
    attack_costs = np.asarray(attack_costs, dtype=float)
    defense_costs = np.asarray(defense_costs, dtype=float)
    if mission.ndim != 2 or min(mission.shape) < 1:
        raise ValueError("Mission values must form a nonempty matrix.")
    if attack_costs.shape != (mission.shape[1],) or defense_costs.shape != (mission.shape[0],):
        raise ValueError("Cost vectors do not match the mission matrix.")
    if not all(np.all(np.isfinite(a)) for a in (mission, attack_costs, defense_costs)):
        raise ValueError("Mission values and costs must be finite.")
    if np.any(mission < 0) or np.any(mission > 1):
        raise ValueError("Mission values must lie in [0, 1].")
    if np.any(attack_costs < 0) or np.any(defense_costs < 0):
        raise ValueError("Costs must be nonnegative.")
    if not all(math.isfinite(v) and v >= 0 for v in (budget, lambda_attack, lambda_defense)):
        raise ValueError("Budget and cost coefficients must be finite and nonnegative.")
    feasible = defense_costs <= budget + 1.0e-12
    if not np.any(feasible):
        raise ValueError("No defense satisfies the hard budget.")
    return mission, lambda_attack * attack_costs, lambda_defense * defense_costs, feasible


def _value_tolerance(value_tolerance):
    if not math.isfinite(value_tolerance) or value_tolerance <= 10 * LP_TOLERANCE:
        raise ValueError("The value tolerance must be finite and exceed 10 * LP_TOLERANCE.")


def _policy_vector(policy, size):
    policy = np.asarray(policy, dtype=float)
    if (policy.shape != (size,) or not np.all(np.isfinite(policy))
            or np.min(policy) < -FEASIBILITY_TOLERANCE
            or abs(float(np.sum(policy)) - 1.) > FEASIBILITY_TOLERANCE):
        raise ValueError("A policy must be a finite probability vector.")
    return policy


def evaluate_policy(policy, mission_values, attack_costs=ATTACK_COSTS,
                    defense_costs=DEFENSE_COSTS, lambda_attack=LAMBDA_ATTACK,
                    lambda_defense=LAMBDA_DEFENSE,
                    best_response_tolerance=BEST_RESPONSE_TOLERANCE):
    """Evaluate upper attacker payoffs; ties minimize defender mission utility."""
    values = np.asarray(mission_values, dtype=float)
    policy = _policy_vector(policy, values.shape[0])
    mission = policy @ values
    payoffs = 1. - mission - lambda_attack * np.asarray(attack_costs)
    responses = tuple(int(j) for j in np.flatnonzero(
        payoffs >= np.max(payoffs) - best_response_tolerance))
    j = min(responses, key=lambda k: (float(mission[k]), k))
    cost = float(policy @ np.asarray(defense_costs))
    return PolicyEvaluation(j, responses, float(mission[j]), cost,
                            float(mission[j] - lambda_defense * cost))


def evaluate_potential_policy(policy, mission_lower, mission_upper,
                              attack_costs=ATTACK_COSTS, defense_costs=DEFENSE_COSTS,
                              lambda_attack=LAMBDA_ATTACK,
                              lambda_defense=LAMBDA_DEFENSE):
    """Evaluate all attacks that can be best responses in a rectangular game."""
    policy = _policy_vector(policy, np.asarray(mission_lower).shape[0])
    lower, upper = policy @ mission_lower, policy @ mission_upper
    ac = lambda_attack * np.asarray(attack_costs)
    plus, minus = 1. - lower - ac, 1. - upper - ac
    responses = tuple(int(j) for j in np.flatnonzero(
        plus >= np.max(minus) - BEST_RESPONSE_TOLERANCE))
    j = min(responses, key=lambda k: (float(lower[k]), k))
    cost = float(policy @ np.asarray(defense_costs))
    return PolicyEvaluation(j, responses, float(lower[j]), cost,
                            float(lower[j] - lambda_defense * cost))


def _lp(c, rows, limits, bounds, n_defenses):
    a_eq = np.zeros((1, len(c)))
    a_eq[0, :n_defenses] = 1.
    solved = linprog(
        c, A_ub=np.asarray(rows) if len(rows) else None,
        b_ub=np.asarray(limits) if len(limits) else None,
        A_eq=a_eq, b_eq=[1.], bounds=bounds,
        method="highs", options=LP_OPTIONS,
    )
    if solved.status == 2:
        return None
    if not solved.success:
        raise RuntimeError(f"Linear programming failed: {solved.message}")
    return solved


def _fraction_solve(rows, rhs):
    """Solve a small square system using exact rational arithmetic."""
    n = len(rhs)
    a = [list(row) + [b] for row, b in zip(rows, rhs)]
    for col in range(n):
        pivot = next((i for i in range(col, n) if a[i][col]), None)
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for i in range(n):
            if i != col:
                scale = a[i][col]
                a[i] = [u - scale * v for u, v in zip(a[i], a[col])]
    return [row[-1] for row in a]


def _exact_interior(weak_a, weak_b, strict_a, strict_b, feasible):
    """Resolve degenerate strict feasibility for the small experiment games.

    A bounded closed polytope has a point satisfying every strict inequality
    exactly when each one is strict at some vertex. Averaging these witnesses
    gives an interior point. Fractions represent the supplied floating-point
    coefficients exactly.
    """
    indices = np.flatnonzero(feasible)
    m = len(indices)
    if m > 3:
        raise RuntimeError("Strict feasibility is numerically unresolved for this game size.")
    rows = np.vstack((weak_a[:, indices], strict_a[:, indices], -np.eye(m)))
    rhs = np.r_[weak_b, strict_b, np.zeros(m)]
    rows = [[Fraction(float(v)) for v in row] for row in rows]
    rhs = [Fraction(float(v)) for v in rhs]
    first_strict = len(weak_b)
    witnesses = [None] * len(strict_b)
    for active in itertools.combinations(range(len(rows)), m - 1):
        x = _fraction_solve([[Fraction(1)] * m] + [rows[i] for i in active],
                            [Fraction(1)] + [rhs[i] for i in active])
        if x is None:
            continue
        slack = [b - sum(a * v for a, v in zip(row, x)) for row, b in zip(rows, rhs)]
        if any(v < 0 for v in slack):
            continue
        for j in range(len(strict_b)):
            if slack[first_strict + j] > 0:
                witnesses[j] = x
        if all(point is not None for point in witnesses):
            result = np.zeros(len(feasible))
            for k, index in enumerate(indices):
                result[index] = float(sum(point[k] for point in witnesses) / len(witnesses))
            return result
    return None


def _region_solution(pieces, weak_a, weak_b, strict_a, strict_b, feasible,
                     value_tolerance):
    """Bound a response region and construct policies within a value budget."""
    m = len(feasible)
    pieces = np.asarray(pieces, dtype=float).reshape(-1, m)
    weak_a = np.asarray(weak_a, dtype=float).reshape(-1, m)
    strict_a = np.asarray(strict_a, dtype=float).reshape(-1, m)
    weak_b, strict_b = np.asarray(weak_b), np.asarray(strict_b)
    if len(strict_b) and np.any(np.min(strict_a[:, feasible], axis=1) >= strict_b):
        return None
    bounds = [(0., 1.) if allowed else (0., 0.) for allowed in feasible]
    rows = [np.r_[row, 0.] for row in np.vstack((weak_a, strict_a))]
    rhs = list(weak_b) + list(strict_b)
    rows.extend(np.r_[-piece, 1.] for piece in pieces)
    rhs.extend([0.] * len(pieces))
    closed = _lp(np.r_[np.zeros(m), -1.], rows, rhs, bounds + [(None, None)], m)
    if closed is None:
        return None
    bound = float(-closed.fun)
    policies = [np.asarray(closed.x[:m])]
    if not len(strict_b):
        return bound, policies
    if np.min(strict_b - strict_a @ policies[0]) > 4 * BEST_RESPONSE_TOLERANCE:
        return bound, policies

    # Find a feasible interior point using row-scaled strict inequalities.
    scales = np.max(np.abs(strict_a[:, feasible] - strict_b[:, None]), axis=1)
    interior_rows = [np.r_[row, 0.] for row in weak_a]
    interior_rows.extend(np.r_[row / scale, 1.] for row, scale in zip(strict_a, scales))
    interior = _lp(np.r_[np.zeros(m), -1.], interior_rows,
                   list(weak_b) + list(strict_b / scales), bounds + [(0., 1.)], m)
    if interior is None or interior.x[-1] <= 10 * LP_TOLERANCE:
        inside = _exact_interior(weak_a, weak_b, strict_a, strict_b, feasible)
        if inside is None:
            return None
    else:
        inside = np.asarray(interior.x[:m])
    policies.append(inside)

    # Construct candidate policies within the allowed objective loss.
    for loss_fraction in (.5, .95):
        near_rows = [np.r_[row, 0.] for row in weak_a]
        near_rows.extend(np.r_[row, 1.] for row in strict_a)
        near_rows.extend(np.r_[-piece, 0.] for piece in pieces)
        near_rhs = (list(weak_b) + list(strict_b)
                    + [-bound + loss_fraction * value_tolerance] * len(pieces))
        near = _lp(np.r_[np.zeros(m), -1.], near_rows, near_rhs,
                   bounds + [(0., None)], m)
        if near is not None:
            policies.append(np.asarray(near.x[:m]))
            if near.x[-1] > 4 * BEST_RESPONSE_TOLERANCE:
                break
    return bound, policies


def _finish_policy(regions, evaluator, feasible, value_tolerance, rule):
    if not regions:
        raise RuntimeError("No feasible response region was found.")
    supremum = max(bound for bound, _ in regions)
    candidates = []
    for _, policies in regions:
        for raw_policy in policies:
            x = _policy_vector(raw_policy, len(feasible)).copy()
            if np.any(np.abs(x[~feasible]) > FEASIBILITY_TOLERANCE):
                raise RuntimeError("The policy uses a defense outside the hard budget.")
            x[~feasible] = 0.
            x = np.maximum(x, 0.)
            x /= np.sum(x)
            candidates.append((evaluator(x), x))
    ev, policy = max(candidates, key=lambda item: (
        item[0].defender_value, -item[0].expected_defense_cost,
        -item[0].attacker_index, *(-item[1])))
    gap = supremum - ev.defender_value
    if not math.isfinite(gap) or gap < -FEASIBILITY_TOLERANCE or gap > value_tolerance:
        raise RuntimeError(
            f"{rule} policy did not meet the numerical objective bound: "
            f"gap={gap:.12g}, tolerance={value_tolerance:.12g}.")
    return PolicyResult(
        policy, ev.attacker_index, ev.guaranteed_mission, ev.expected_defense_cost,
        ev.defender_value, supremum, max(0., gap))


def optimize_policy(mission_lower, attack_costs=ATTACK_COSTS,
                    defense_costs=DEFENSE_COSTS, budget=BUDGET,
                    lambda_attack=LAMBDA_ATTACK, lambda_defense=LAMBDA_DEFENSE,
                    value_tolerance=DEFAULT_VALUE_TOLERANCE):
    """Select an epsilon-optimal policy for the upper-payoff response rule.

    On an attacker-payoff tie, an attack with a higher scaled attack cost has
    lower mission utility. Such alternatives require a strict inequality;
    alternatives with equal or lower costs allow a weak inequality.
    """
    _value_tolerance(value_tolerance)
    values, ac, dc, feasible = _game_inputs(
        mission_lower, attack_costs, defense_costs, budget, lambda_attack, lambda_defense)
    regions = []
    for j in range(values.shape[1]):
        weak_a, weak_b, strict_a, strict_b = [], [], [], []
        for k in range(values.shape[1]):
            if k == j:
                continue
            row, rhs = values[:, j] - values[:, k], float(ac[k] - ac[j])
            if ac[k] > ac[j]:
                strict_a.append(row)
                strict_b.append(rhs)
            else:
                weak_a.append(row)
                weak_b.append(rhs)
        region = _region_solution([values[:, j] - dc], weak_a, weak_b,
                                  strict_a, strict_b, feasible, value_tolerance)
        if region is not None:
            regions.append(region)
    evaluator = lambda x: evaluate_policy(x, values, attack_costs, defense_costs,
                                          lambda_attack, lambda_defense)
    return _finish_policy(regions, evaluator, feasible, value_tolerance, "upper")


def select_potential_policy(mission_lower, mission_upper, attack_costs=ATTACK_COSTS,
                            defense_costs=DEFENSE_COSTS, budget=BUDGET,
                            lambda_attack=LAMBDA_ATTACK, lambda_defense=LAMBDA_DEFENSE,
                            value_tolerance=DEFAULT_VALUE_TOLERANCE):
    """Select an epsilon-optimal policy over potential best responses (CP-P).

    Attack j is retained when A_j^+ >= max_k A_k^- and excluded only when
    some witness k satisfies A_j^+ < A_k^-. Each nonempty response set and
    its possible exclusion witnesses are enumerated, including boundary ties.
    """
    _value_tolerance(value_tolerance)
    lower, ac, dc, feasible = _game_inputs(
        mission_lower, attack_costs, defense_costs, budget, lambda_attack, lambda_defense)
    upper = np.asarray(mission_upper, dtype=float)
    if (upper.shape != lower.shape or not np.all(np.isfinite(upper))
            or np.any(upper < lower) or np.any(upper > 1)):
        raise ValueError("Invalid mission-utility intervals.")
    n = lower.shape[1]
    choices = [(-1,) + tuple(k for k in range(n) if k != j) for j in range(n)]
    regions = []
    for pattern in itertools.product(*choices):
        retained = [j for j, witness in enumerate(pattern) if witness == -1]
        if not retained:
            continue
        weak_a, weak_b, strict_a, strict_b = [], [], [], []
        for j, witness in enumerate(pattern):
            if witness == -1:
                for k in range(n):
                    if k != j:
                        weak_a.append(lower[:, j] - upper[:, k])
                        weak_b.append(float(ac[k] - ac[j]))
            else:
                strict_a.append(upper[:, witness] - lower[:, j])
                strict_b.append(float(ac[j] - ac[witness]))
        region = _region_solution([lower[:, j] - dc for j in retained],
                                  weak_a, weak_b, strict_a, strict_b,
                                  feasible, value_tolerance)
        if region is not None:
            regions.append(region)
    evaluator = lambda x: evaluate_potential_policy(
        x, lower, upper, attack_costs, defense_costs, lambda_attack, lambda_defense)
    return _finish_policy(regions, evaluator, feasible, value_tolerance, "potential")
