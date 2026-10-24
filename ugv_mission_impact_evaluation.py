from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import linprog
    from scipy.stats import beta, norm, t as student_t
except ImportError as exc:
    raise SystemExit(
        "This script requires NumPy, SciPy, and Matplotlib. Install them with:\n"
        "    python -m pip install -r requirements.txt"
    ) from exc

ATTACK_NAMES = (
    "LiDAR spoofing",
    "GNSS spoofing",
    "CAN-message disruption",
)
DEFENSE_NAMES = (
    "Perception-consistency monitor",
    "GNSS-inertial cross-check",
    "CAN anomaly filter with fail-safe response",
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
ATTACK_COSTS = np.array([0.12, 0.15, 0.18], dtype=float)
DEFENSE_COSTS = np.array([0.05, 0.08, 0.12], dtype=float)
BUDGET = 0.12
LAMBDA_ATTACK = 1.0
LAMBDA_DEFENSE = 1.0
PRIMARY_SAMPLE_SIZE = 15
DEFAULT_REPETITIONS = 1000
DEFAULT_SEED = 1
DEFAULT_SAMPLE_SIZES = (5, 10, 15, 30, 60, 120, 300, 1000)
DEFAULT_RESPONSE_MARGIN = 1.0e-4
BEST_RESPONSE_TOLERANCE = 1.0e-8
PRIMARY_INTERVAL_SETTINGS = (
    ("clopper_pearson", 0.90),
    ("clopper_pearson", 0.95),
    ("clopper_pearson", 0.99),
    ("wilson", 0.95),
    ("jeffreys", 0.95),
)
SWEEP_PRIMARY_SETTING = ("clopper_pearson", 0.95)
REAL_WORLD_ATTACK_NAMES = (
    "No environmental manipulation",
    "Table-repositioning attack",
)
REAL_WORLD_DEFENSE_NAMES = (
    "Default configuration",
    "Monitoring-and-replanning defense",
)
REAL_WORLD_SUCCESS_COUNTS = np.array([[15, 6], [15, 13]], dtype=int)
REAL_WORLD_SAMPLE_SIZE = 15
REAL_WORLD_PRIMARY_ATTACK_COST = 0.30
REAL_WORLD_PRIMARY_DEFENSE_COST = 0.20
REAL_WORLD_PRIMARY_BUDGET = 0.20
REAL_WORLD_ATTACK_COST_SWEEP = (0.10, 0.30, 0.70)
REAL_WORLD_DEFENSE_COST_SWEEP = (0.10, 0.20, 0.50)
REAL_WORLD_BUDGET_SWEEP = (0.10, 0.20, 0.50)
REAL_WORLD_THRESHOLD_INCREMENT = 0.001


# Experiment settings

REFERENCE_BEST_RESPONSE_TOLERANCE = 1.0e-9
LP_TOLERANCE = 1.0e-10
STRICT_MARGIN_FLOOR = 5.0e-9
DEFAULT_VALUE_TOLERANCE = 1.0e-6
LP_OPTIONS = {
    "primal_feasibility_tolerance": LP_TOLERANCE,
    "dual_feasibility_tolerance": LP_TOLERANCE,
}
SCENARIO_NAVIGATION = {
    "table_ii": NAVIGATION_TRUTH.copy(),
    "small_defense_contrast": np.array([[.70, .68, .66], [.69, .71, .67], [.68, .69, .72]]),
    "broadly_effective_defense": np.array([[.90, .89, .88], [.60, .68, .61], [.59, .60, .69]]),
    "stable_first_attack": np.array([[.30, .82, .86], [.35, .87, .84], [.33, .84, .90]]),
    "near_uniform_policy_boundary": np.array([[.76, .605, .571], [.64, .725, .571], [.64, .605, .691]]),
    "far_uniform_policy_boundary": np.array([[.88, .46, .86], [.79, .54, .85], [.82, .43, .91]]),
}
DEFAULT_TRAJECTORY_BUDGETS = (15, 60, 300)
COUPLING_CASES = {'opposed_defenses': {'intercept': [[0.9, 0.2], [0.9, 0.85]],
                      'slope': [[0.0, 0.7], [0.0, -0.6]],
                      'attack_costs': [0.0, 0.2],
                      'defense_costs': [0.0, 0.04],
                      'budget': 0.04}}


@dataclass(frozen=True)
class PolicyResult:

    policy: np.ndarray
    attacker_index: int
    guaranteed_mission: float
    expected_defense_cost: float
    defender_value: float
    response_margin_used: float
    minimum_payoff_gap: float


@dataclass(frozen=True)
class PolicyEvaluation:

    attacker_index: int
    best_response_indices: tuple[int, ...]
    guaranteed_mission: float
    expected_defense_cost: float
    defender_value: float


@dataclass
class DatasetAnalysis:

    sample_size: int
    point_rows: list[dict[str, Any]]
    interval_rows: dict[tuple[str, float], list[dict[str, Any]]]
    interval_summaries: list[dict[str, Any]]


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


def clopper_pearson_interval(
    successes: np.ndarray, sample_size: int, confidence: float
) -> tuple[np.ndarray, np.ndarray]:

    y = np.asarray(successes, dtype=float)
    alpha = 1.0 - confidence
    lower = np.zeros_like(y)
    upper = np.ones_like(y)
    positive = y > 0
    incomplete = y < sample_size
    lower[positive] = beta.ppf(
        alpha / 2.0, y[positive], sample_size - y[positive] + 1.0
    )
    upper[incomplete] = beta.ppf(
        1.0 - alpha / 2.0,
        y[incomplete] + 1.0,
        sample_size - y[incomplete],
    )
    return lower, upper


def wilson_interval(
    successes: np.ndarray, sample_size: int, confidence: float
) -> tuple[np.ndarray, np.ndarray]:

    y = np.asarray(successes, dtype=float)
    phat = y / sample_size
    z = norm.ppf(0.5 + confidence / 2.0)
    z2 = z * z
    denominator = 1.0 + z2 / sample_size
    center = (phat + z2 / (2.0 * sample_size)) / denominator
    half_width = (
        z
        * np.sqrt(
            phat * (1.0 - phat) / sample_size
            + z2 / (4.0 * sample_size * sample_size)
        )
        / denominator
    )
    return (
        np.clip(center - half_width, 0.0, 1.0),
        np.clip(center + half_width, 0.0, 1.0),
    )


def jeffreys_interval(
    successes: np.ndarray, sample_size: int, confidence: float
) -> tuple[np.ndarray, np.ndarray]:

    y = np.asarray(successes, dtype=float)
    alpha = 1.0 - confidence
    posterior_a = y + 0.5
    posterior_b = sample_size - y + 0.5
    return (
        beta.ppf(alpha / 2.0, posterior_a, posterior_b),
        beta.ppf(1.0 - alpha / 2.0, posterior_a, posterior_b),
    )


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


def evaluate_policy(
    policy: np.ndarray,
    mission_values: np.ndarray,
    attack_costs: np.ndarray = ATTACK_COSTS,
    defense_costs: np.ndarray = DEFENSE_COSTS,
    lambda_attack: float = LAMBDA_ATTACK,
    lambda_defense: float = LAMBDA_DEFENSE,
    best_response_tolerance: float = BEST_RESPONSE_TOLERANCE,
) -> PolicyEvaluation:

    mission_by_attack = policy @ mission_values
    attacker_payoffs = 1.0 - mission_by_attack - lambda_attack * attack_costs
    maximum_payoff = float(np.max(attacker_payoffs))
    response_indices = tuple(
        int(index)
        for index in np.flatnonzero(
            attacker_payoffs >= maximum_payoff - best_response_tolerance
        )
    )
    attacker_index = min(
        response_indices,
        key=lambda index: (float(mission_by_attack[index]), index),
    )
    guaranteed_mission = float(mission_by_attack[attacker_index])
    expected_cost = float(policy @ defense_costs)
    return PolicyEvaluation(
        attacker_index=attacker_index,
        best_response_indices=response_indices,
        guaranteed_mission=guaranteed_mission,
        expected_defense_cost=expected_cost,
        defender_value=guaranteed_mission - lambda_defense * expected_cost,
    )


def optimize_policy(
    mission_lower: np.ndarray,
    attack_costs: np.ndarray = ATTACK_COSTS,
    defense_costs: np.ndarray = DEFENSE_COSTS,
    budget: float = BUDGET,
    lambda_attack: float = LAMBDA_ATTACK,
    lambda_defense: float = LAMBDA_DEFENSE,
    response_margin: float = DEFAULT_RESPONSE_MARGIN,
) -> PolicyResult:

    mission_lower = np.asarray(mission_lower, dtype=float)
    feasible = defense_costs <= budget + 1.0e-12
    if not np.any(feasible):
        raise ValueError("No defense is feasible under the stated budget.")
    bounds = [(0.0, None) if allowed else (0.0, 0.0) for allowed in feasible]
    best: PolicyResult | None = None

    for margin in (response_margin, 0.0):
        candidates: list[PolicyResult] = []
        for response in range(attack_costs.size):
            inequalities = []
            upper_bounds = []
            for alternative in range(attack_costs.size):
                if alternative == response:
                    continue
                inequalities.append(
                    mission_lower[:, response]
                    - mission_lower[:, alternative]
                )
                upper_bounds.append(
                    lambda_attack
                    * (attack_costs[alternative] - attack_costs[response])
                    - margin
                )

            objective = -(
                mission_lower[:, response] - lambda_defense * defense_costs
            )
            result = linprog(
                objective,
                A_ub=np.asarray(inequalities, dtype=float),
                b_ub=np.asarray(upper_bounds, dtype=float),
                A_eq=np.ones((1, defense_costs.size), dtype=float),
                b_eq=np.array([1.0]),
                bounds=bounds,
                method="highs",
            )
            if not result.success:
                continue

            policy = np.asarray(result.x, dtype=float)
            policy[np.abs(policy) < 1.0e-12] = 0.0
            policy = policy / np.sum(policy)
            model_evaluation = evaluate_policy(
                policy,
                mission_lower,
                attack_costs,
                defense_costs,
                lambda_attack,
                lambda_defense,
            )
            attacker_payoffs = (
                1.0 - policy @ mission_lower - lambda_attack * attack_costs
            )
            payoff_gaps = [
                attacker_payoffs[response] - attacker_payoffs[alternative]
                for alternative in range(attack_costs.size)
                if alternative != response
            ]
            minimum_gap = float(min(payoff_gaps))
            if margin > 0 and minimum_gap < margin - 1.0e-7:
                continue

            if margin > 0:
                guaranteed_mission = float(policy @ mission_lower[:, response])
                expected_cost = float(policy @ defense_costs)
                defender_value = guaranteed_mission - lambda_defense * expected_cost
                reported_response = response
            else:
                guaranteed_mission = model_evaluation.guaranteed_mission
                expected_cost = model_evaluation.expected_defense_cost
                defender_value = model_evaluation.defender_value
                reported_response = model_evaluation.attacker_index

            candidates.append(
                PolicyResult(
                    policy=policy,
                    attacker_index=reported_response,
                    guaranteed_mission=guaranteed_mission,
                    expected_defense_cost=expected_cost,
                    defender_value=defender_value,
                    response_margin_used=margin,
                    minimum_payoff_gap=minimum_gap,
                )
            )

        if candidates:

            best = max(
                candidates,
                key=lambda item: (
                    item.defender_value,
                    -item.attacker_index,
                    *(-item.policy),
                ),
            )
            break

    if best is None:
        raise RuntimeError("Policy optimization failed in every response region.")
    return best


def mean_mcse(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    mean = float(np.mean(data))
    if data.size < 2:
        return mean, float("nan")
    return mean, float(np.std(data, ddof=1) / math.sqrt(data.size))


def mean_t_interval(
    values: Sequence[float] | np.ndarray, confidence: float = 0.95
) -> dict[str, float | int]:

    data = np.asarray(values, dtype=float)
    mean, mcse = mean_mcse(data)
    sd = float(np.std(data, ddof=1)) if data.size > 1 else float("nan")
    if data.size > 1:
        critical = float(student_t.ppf(0.5 + confidence / 2.0, data.size - 1))
        lower = mean - critical * mcse
        upper = mean + critical * mcse
    else:
        lower = upper = float("nan")
    return {
        "observations": int(data.size),
        "mean": mean,
        "standard_deviation": sd,
        "mcse": mcse,
        "mc95_low": lower,
        "mc95_high": upper,
    }


def wilson_monte_carlo_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    lower, upper = wilson_interval(
        np.asarray([successes], dtype=float), trials, confidence
    )
    return float(lower[0]), float(upper[0])


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


def read_csv(path: Path) -> list[dict[str, str]]:

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def format_policy(policy: Iterable[float], digits: int = 4) -> str:
    return "(" + ",".join(f"{float(value):.{digits}f}" for value in policy) + ")"


def interval_label(method: str) -> str:
    return {
        "clopper_pearson": "Clopper--Pearson",
        "wilson": "Wilson",
        "jeffreys": "Jeffreys",
    }[method]


def summarize_policy_rows(
    rows: list[dict[str, Any]], prefix: str
) -> dict[str, Any]:
    l1 = np.asarray([float(row["l1_distance_to_oracle"]) for row in rows])
    regret = np.asarray([float(row["policy_regret"]) for row in rows])
    truth_value = np.asarray(
        [float(row["truth_defender_value"]) for row in rows]
    )
    l1_mean, l1_mcse = mean_mcse(l1)
    regret_mean, regret_mcse = mean_mcse(regret)
    truth_mean, truth_mcse = mean_mcse(truth_value)
    return {
        f"{prefix}_policy_l1_mean": l1_mean,
        f"{prefix}_policy_l1_mcse": l1_mcse,
        f"{prefix}_policy_regret_mean": regret_mean,
        f"{prefix}_policy_regret_mcse": regret_mcse,
        f"{prefix}_truth_value_mean": truth_mean,
        f"{prefix}_truth_value_mcse": truth_mcse,
    }


def paired_policy_summary(
    point_rows: list[dict[str, Any]],
    credal_rows: list[dict[str, Any]],
    sample_size: int,
) -> dict[str, Any]:

    if len(point_rows) != len(credal_rows):
        raise ValueError("Paired model results have different lengths.")
    point_by_rep = {int(row["repetition"]): row for row in point_rows}
    credal_by_rep = {int(row["repetition"]): row for row in credal_rows}
    if set(point_by_rep) != set(credal_by_rep):
        raise ValueError("Paired model results have different repetition IDs.")

    policy_distances = []
    regret_differences = []
    response_agreements = []
    for repetition in sorted(point_by_rep):
        point = point_by_rep[repetition]
        credal = credal_by_rep[repetition]
        point_policy = np.array(
            [point["policy_d1"], point["policy_d2"], point["policy_d3"]],
            dtype=float,
        )
        credal_policy = np.array(
            [credal["policy_d1"], credal["policy_d2"], credal["policy_d3"]],
            dtype=float,
        )
        policy_l1 = float(np.sum(np.abs(credal_policy - point_policy)))
        regret_difference = float(point["policy_regret"]) - float(
            credal["policy_regret"]
        )
        response_agreement = (
            point["model_attacker_best_response"]
            == credal["model_attacker_best_response"]
        )
        policy_distances.append(policy_l1)
        regret_differences.append(regret_difference)
        response_agreements.append(response_agreement)

    policy_stats = mean_t_interval(policy_distances)
    regret_stats = mean_t_interval(regret_differences)
    regret_array = np.asarray(regret_differences)
    summary = {
        "sample_size": sample_size,
        "repetitions": len(policy_distances),
        "cp_point_policy_l1_mean": policy_stats["mean"],
        "cp_point_policy_l1_standard_deviation": policy_stats[
            "standard_deviation"
        ],
        "cp_point_policy_l1_mcse": policy_stats["mcse"],
        "cp_point_policy_l1_mc95_low": policy_stats["mc95_low"],
        "cp_point_policy_l1_mc95_high": policy_stats["mc95_high"],
        "paired_regret_definition": "point_regret_minus_cp95_regret",
        "paired_regret_difference_mean": regret_stats["mean"],
        "paired_regret_difference_standard_deviation": regret_stats[
            "standard_deviation"
        ],
        "paired_regret_difference_mcse": regret_stats["mcse"],
        "paired_regret_difference_mc95_low": regret_stats["mc95_low"],
        "paired_regret_difference_mc95_high": regret_stats["mc95_high"],
        "paired_regret_positive_count": int(np.sum(regret_array > 0.0)),
        "paired_regret_negative_count": int(np.sum(regret_array < 0.0)),
        "paired_regret_equal_count": int(np.sum(regret_array == 0.0)),
        "model_attacker_response_agreement_rate": float(
            np.mean(response_agreements)
        ),
    }
    return summary


def paired_interval_policy_comparison(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    sample_size: int,
    candidate_method: str,
    candidate_confidence: float,
) -> dict[str, Any]:

    reference_by_rep = {int(row["repetition"]): row for row in reference_rows}
    candidate_by_rep = {int(row["repetition"]): row for row in candidate_rows}
    if set(reference_by_rep) != set(candidate_by_rep):
        raise ValueError("Paired interval models have different repetition IDs.")

    distances: list[float] = []
    agreements: list[bool] = []
    for repetition in sorted(reference_by_rep):
        reference = reference_by_rep[repetition]
        candidate = candidate_by_rep[repetition]
        reference_policy = np.asarray(
            [reference[f"policy_d{index}"] for index in range(1, 4)], dtype=float
        )
        candidate_policy = np.asarray(
            [candidate[f"policy_d{index}"] for index in range(1, 4)], dtype=float
        )
        distance = float(np.sum(np.abs(reference_policy - candidate_policy)))
        agreement = (
            reference["model_attacker_best_response"]
            == candidate["model_attacker_best_response"]
        )
        distances.append(distance)
        agreements.append(agreement)

    distance_stats = mean_t_interval(distances)
    summary = {
        "sample_size": sample_size,
        "repetitions": len(distances),
        "reference_method": "clopper_pearson",
        "reference_confidence_level": 0.95,
        "candidate_method": candidate_method,
        "candidate_confidence_level": candidate_confidence,
        "policy_l1_mean": distance_stats["mean"],
        "policy_l1_mcse": distance_stats["mcse"],
        "policy_l1_mc95_low": distance_stats["mc95_low"],
        "policy_l1_mc95_high": distance_stats["mc95_high"],
        "attacker_response_agreement_rate": float(np.mean(agreements)),
    }
    return summary


def illustrative_policy_row(source: str, result: PolicyResult) -> dict[str, Any]:
    return {
        "policy_source": source,
        "policy_d1": float(result.policy[0]),
        "policy_d2": float(result.policy[1]),
        "policy_d3": float(result.policy[2]),
        "policy_latex": format_policy(result.policy),
        "attacker_best_response": f"a{result.attacker_index + 1}",
        "model_conditional_mission_lower": result.guaranteed_mission,
        "expected_defense_cost": result.expected_defense_cost,
        "model_conditional_defender_value": result.defender_value,
        "attacker_payoff_margin": result.response_margin_used,
    }


def run_illustrative_full_pipeline(
    output: Path, seed: int
) -> None:

    directory = output / "illustrative_full_pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    sample_size = PRIMARY_SAMPLE_SIZE
    navigation, mission_normal, mission_degraded = generate_counts(
        1, sample_size, seed
    )
    point_mission = point_mission_estimate(
        navigation, mission_normal, mission_degraded, sample_size
    )[0]

    confidence = 0.95
    method = "clopper_pearson"
    navigation_interval = binomial_interval(
        navigation, sample_size, confidence, method
    )
    normal_interval = binomial_interval(
        mission_normal, sample_size, confidence, method
    )
    degraded_interval = binomial_interval(
        mission_degraded, sample_size, confidence, method
    )
    mission_lower, mission_upper = propagate_mission_intervals(
        navigation_interval, normal_interval, degraded_interval
    )
    mission_lower = mission_lower[0]
    mission_upper = mission_upper[0]
    truth_matrix = mission_truth_matrix()

    local_rows: list[dict[str, Any]] = []
    row_specs = (
        (
            "navigation_normal",
            navigation,
            navigation_interval,
            lambda defense, attack: NAVIGATION_TRUTH[defense, attack],
        ),
        (
            "mission_success_given_normal",
            mission_normal,
            normal_interval,
            lambda _defense, _attack: MISSION_NORMAL_TRUTH,
        ),
        (
            "mission_success_given_degraded",
            mission_degraded,
            degraded_interval,
            lambda _defense, _attack: MISSION_DEGRADED_TRUTH,
        ),
    )
    for defense in range(3):
        for attack in range(3):
            for row_name, count_array, interval, truth_getter in row_specs:
                successes = int(count_array[0, defense, attack])
                local_rows.append(
                    {
                        "defense": f"d{defense + 1}",
                        "defense_name": DEFENSE_NAMES[defense],
                        "attack": f"a{attack + 1}",
                        "attack_name": ATTACK_NAMES[attack],
                        "local_row": row_name,
                        "successes": successes,
                        "trials": sample_size,
                        "observed_proportion": successes / sample_size,
                        "ground_truth": truth_getter(defense, attack),
                        "interval_method": method,
                        "confidence_level": confidence,
                        "interval_lower": float(interval[0][0, defense, attack]),
                        "interval_upper": float(interval[1][0, defense, attack]),
                    }
                )

    mission_rows: list[dict[str, Any]] = []
    for defense in range(3):
        for attack in range(3):
            truth = float(truth_matrix[defense, attack])
            lower = float(mission_lower[defense, attack])
            upper = float(mission_upper[defense, attack])
            mission_rows.append(
                {
                    "defense": f"d{defense + 1}",
                    "defense_name": DEFENSE_NAMES[defense],
                    "attack": f"a{attack + 1}",
                    "attack_name": ATTACK_NAMES[attack],
                    "point_mission_estimate": float(
                        point_mission[defense, attack]
                    ),
                    "mission_interval_lower": lower,
                    "mission_interval_upper": upper,
                    "ground_truth_mission": truth,
                    "covered": lower <= truth <= upper,
                }
            )

    oracle_result = optimize_policy(truth_matrix)
    point_result = optimize_policy(point_mission)
    interval_result = optimize_policy(mission_lower)
    policy_rows = [
        illustrative_policy_row("ground_truth_oracle", oracle_result),
        illustrative_policy_row("point_probability_baseline", point_result),
        illustrative_policy_row("cp95_interval_model", interval_result),
    ]

    d2a2_local = [
        row
        for row in local_rows
        if row["defense"] == "d2" and row["attack"] == "a2"
    ]
    d2a2_mission = next(
        row
        for row in mission_rows
        if row["defense"] == "d2" and row["attack"] == "a2"
    )
    worked_row: dict[str, Any] = {
        "defense": "d2",
        "attack": "a2",
        "trials_per_local_row": sample_size,
    }
    for row in d2a2_local:
        prefix = str(row["local_row"])
        worked_row[f"{prefix}_successes"] = row["successes"]
        worked_row[f"{prefix}_lower"] = row["interval_lower"]
        worked_row[f"{prefix}_upper"] = row["interval_upper"]
    worked_row.update(
        {
            "point_mission_estimate": d2a2_mission["point_mission_estimate"],
            "mission_interval_lower": d2a2_mission["mission_interval_lower"],
            "mission_interval_upper": d2a2_mission["mission_interval_upper"],
            "ground_truth_mission": d2a2_mission["ground_truth_mission"],
            "covered": d2a2_mission["covered"],
        }
    )

    write_csv(directory / "mission_utility_intervals.csv", mission_rows)
    write_csv(directory / "defense_policies.csv", policy_rows)
    write_csv(directory / "worked_cell_d2_a2.csv", [worked_row])


def find_summary(
    summaries: list[dict[str, Any]], method: str, confidence: float
) -> dict[str, Any]:
    return next(
        row
        for row in summaries
        if row["interval_method"] == method
        and math.isclose(float(row["confidence_level"]), confidence)
    )


def real_world_setting_label(method: str, confidence: float | None) -> str:
    if method == "point":
        return "Point-probability baseline"
    short_name = "CP" if method == "clopper_pearson" else interval_label(method)
    return f"{short_name} {100.0 * float(confidence):.0f}%"


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


def run_real_world_instantiation(output: Path) -> None:

    directory = output / "real_world_instantiation"
    directory.mkdir(parents=True, exist_ok=True)
    point = REAL_WORLD_SUCCESS_COUNTS / REAL_WORLD_SAMPLE_SIZE
    cp95_observations: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    lower_by_setting: dict[tuple[str, float | None], np.ndarray] = {}

    for method, confidence in PRIMARY_INTERVAL_SETTINGS:
        lower, upper = binomial_interval(
            REAL_WORLD_SUCCESS_COUNTS,
            REAL_WORLD_SAMPLE_SIZE,
            confidence,
            method,
        )
        lower_by_setting[(method, confidence)] = lower
        label = real_world_setting_label(method, confidence)
        if (method, confidence) == SWEEP_PRIMARY_SETTING:
            for defense in range(2):
                for attack in range(2):
                    cp95_observations.append(
                        {
                            "defense": f"d{defense + 1}",
                            "defense_name": REAL_WORLD_DEFENSE_NAMES[defense],
                            "attack": f"a{attack + 1}",
                            "attack_name": REAL_WORLD_ATTACK_NAMES[attack],
                            "successes": int(
                                REAL_WORLD_SUCCESS_COUNTS[defense, attack]
                            ),
                            "trials": REAL_WORLD_SAMPLE_SIZE,
                            "point_probability": float(point[defense, attack]),
                            "interval_lower": float(lower[defense, attack]),
                            "interval_upper": float(upper[defense, attack]),
                        }
                    )
        policy = solve_real_world_policy(
            lower,
            REAL_WORLD_PRIMARY_ATTACK_COST,
            REAL_WORLD_PRIMARY_DEFENSE_COST,
            REAL_WORLD_PRIMARY_BUDGET,
        )
        sensitivity_rows.append(
            {
                "setting": label,
                "interval_method": method,
                "confidence_level": confidence,
                "mean_interval_width": float(np.mean(upper - lower)),
                "L0_no_attack": float(lower[0, 0]),
                "L1_attack_default": float(lower[0, 1]),
                "L2_attack_enhanced": float(lower[1, 1]),
                "deterrence_threshold_tau": policy.deterrence_threshold,
                "policy_d1": float(policy.policy[0]),
                "policy_d2_x2": float(policy.policy[1]),
                "attacker_best_response": f"a{policy.attacker_index + 1}",
                "model_conditional_mission_lower_GM": policy.guaranteed_mission,
                "expected_defense_cost": policy.expected_defense_cost,
                "defender_value_Phi": policy.defender_value,
            }
        )

    lower_by_setting[("point", None)] = point
    point_policy = solve_real_world_policy(
        point,
        REAL_WORLD_PRIMARY_ATTACK_COST,
        REAL_WORLD_PRIMARY_DEFENSE_COST,
        REAL_WORLD_PRIMARY_BUDGET,
    )
    sensitivity_rows.append(
        {
            "setting": real_world_setting_label("point", None),
            "interval_method": "point",
            "confidence_level": "",
            "mean_interval_width": "",
            "L0_no_attack": float(point[0, 0]),
            "L1_attack_default": float(point[0, 1]),
            "L2_attack_enhanced": float(point[1, 1]),
            "deterrence_threshold_tau": point_policy.deterrence_threshold,
            "policy_d1": float(point_policy.policy[0]),
            "policy_d2_x2": float(point_policy.policy[1]),
            "attacker_best_response": f"a{point_policy.attacker_index + 1}",
            "model_conditional_mission_lower_GM": point_policy.guaranteed_mission,
            "expected_defense_cost": point_policy.expected_defense_cost,
            "defender_value_Phi": point_policy.defender_value,
        }
    )

    cp95_lower = lower_by_setting[("clopper_pearson", 0.95)]
    cost_budget_rows: list[dict[str, Any]] = []
    for budget in REAL_WORLD_BUDGET_SWEEP:
        for defense_cost in REAL_WORLD_DEFENSE_COST_SWEEP:
            for attack_cost in REAL_WORLD_ATTACK_COST_SWEEP:
                policy = solve_real_world_policy(
                    cp95_lower, attack_cost, defense_cost, budget
                )
                cost_budget_rows.append(
                    {
                        "budget": budget,
                        "defense_cost_cD": defense_cost,
                        "attack_cost_cA": attack_cost,
                        "d2_budget_feasible": defense_cost <= budget + 1.0e-12,
                        "deterrence_threshold_tau": policy.deterrence_threshold,
                        "policy_d1": float(policy.policy[0]),
                        "policy_d2_x2": float(policy.policy[1]),
                        "attacker_best_response": f"a{policy.attacker_index + 1}",
                        "model_conditional_mission_lower_GM": policy.guaranteed_mission,
                        "expected_defense_cost": policy.expected_defense_cost,
                        "defender_value_Phi": policy.defender_value,
                    }
                )

    def result_text(attack_cost: float, defense_cost: float, budget: float) -> str:
        policy = solve_real_world_policy(
            cp95_lower, attack_cost, defense_cost, budget
        )
        return f"{policy.policy[1]:.4f}/a{policy.attacker_index + 1}"

    grouped_rows: list[dict[str, Any]] = [
        {
            "budget_condition": "b<c_D",
            "defense_cost_cD": "--",
            **{
                f"cA_{attack_cost:.2f}": result_text(attack_cost, 0.20, 0.10)
                for attack_cost in REAL_WORLD_ATTACK_COST_SWEEP
            },
        }
    ]
    for defense_cost in REAL_WORLD_DEFENSE_COST_SWEEP:
        grouped_rows.append(
            {
                "budget_condition": "b>=c_D",
                "defense_cost_cD": defense_cost,
                **{
                    f"cA_{attack_cost:.2f}": result_text(
                        attack_cost, defense_cost, defense_cost
                    )
                    for attack_cost in REAL_WORLD_ATTACK_COST_SWEEP
                },
            }
        )

    l0 = float(cp95_lower[0, 0])
    l1 = float(cp95_lower[0, 1])
    l2 = float(cp95_lower[1, 1])
    regime_rows = [
        {
            "quantity": "non_deterrable_upper_attack_cost",
            "value": l0 - l2,
            "interpretation": "tau>=1 when c_A is at or below this value",
        },
        {
            "quantity": "zero_defense_deterrence_attack_cost",
            "value": l0 - l1,
            "interpretation": (
                "tau=0 at this boundary; strict deterrence at x2=0 requires "
                "c_A above it under pessimistic tie-breaking"
            ),
        },
        {
            "quantity": "attack_cell_utility_improvement",
            "value": l2 - l1,
            "interpretation": "L2-L1 comparison used in the defense-cost sweep",
        },
    ]

    interval_sensitivity = [
        {
            "setting": row["setting"],
            "mean_interval_width": row["mean_interval_width"],
            "deterrence_threshold_tau": row["deterrence_threshold_tau"],
            "policy_d2_x2": row["policy_d2_x2"],
            "attacker_best_response": row["attacker_best_response"],
            "model_conditional_mission_lower_GM": row["model_conditional_mission_lower_GM"],
            "defender_value_Phi": row["defender_value_Phi"],
        }
        for row in sensitivity_rows
    ]
    write_csv(directory / "observations_cp95.csv", cp95_observations)
    write_csv(directory / "interval_sensitivity.csv", interval_sensitivity)
    write_csv(directory / "cost_budget_sensitivity.csv", cost_budget_rows)
    write_csv(directory / "cost_budget_table.csv", grouped_rows)
    write_csv(directory / "cost_budget_regime_boundaries.csv", regime_rows)


@dataclass(frozen=True)
class BoundedPolicyResult:

    policy: np.ndarray
    attacker_index: int
    guaranteed_mission: float
    expected_defense_cost: float
    defender_value: float
    response_margin_used: float
    minimum_payoff_gap: float
    value_supremum: float = float("nan")
    optimality_gap: float = float("nan")
    response_rule: str = "upper"


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


def conditional_clopper_pearson_interval(successes, sample_size, confidence):

    y, n = _binomial_inputs(successes, sample_size, confidence)
    alpha = 1. - confidence
    lower, upper = np.zeros_like(y), np.ones_like(y)
    positive, incomplete = y > 0, y < n
    lower[positive] = beta.ppf(alpha / 2., y[positive], n[positive] - y[positive] + 1.)
    upper[incomplete] = beta.ppf(1. - alpha / 2., y[incomplete] + 1., n[incomplete] - y[incomplete])
    return lower, upper


def conditional_wilson_interval(successes, sample_size, confidence):

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


def conditional_jeffreys_interval(successes, sample_size, confidence):

    y, n = _binomial_inputs(successes, sample_size, confidence)
    alpha = 1. - confidence
    lower = beta.ppf(alpha / 2., y + .5, n - y + .5)
    upper = beta.ppf(1. - alpha / 2., y + .5, n - y + .5)
    return np.where(n == 0, 0., lower), np.where(n == 0, 1., upper)


def conditional_binomial_interval(
    successes: np.ndarray,
    sample_size: int,
    confidence: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "clopper_pearson":
        return conditional_clopper_pearson_interval(successes, sample_size, confidence)
    if method == "wilson":
        return conditional_wilson_interval(successes, sample_size, confidence)
    if method == "jeffreys":
        return conditional_jeffreys_interval(successes, sample_size, confidence)
    raise ValueError(f"Unknown interval method: {method}")


def _game_inputs(mission, attack_costs, defense_costs, budget, lambda_attack,
                 lambda_defense):
    mission = np.asarray(mission, dtype=float)
    attack_costs = np.asarray(attack_costs, dtype=float)
    defense_costs = np.asarray(defense_costs, dtype=float)
    if mission.ndim != 2 or min(mission.shape) < 1:
        raise ValueError("Mission values must form a nonempty matrix.")
    if attack_costs.shape != (mission.shape[1],) or defense_costs.shape != (mission.shape[0],):
        raise ValueError("Cost vectors do not match the mission matrix.")
    arrays = (mission, attack_costs, defense_costs)
    if not all(np.all(np.isfinite(a)) for a in arrays):
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


def _lp(c, rows, limits, bounds, n_defenses):
    a_eq = np.zeros((1, len(c)))
    a_eq[0, :n_defenses] = 1.0
    result = linprog(
        c, A_ub=np.asarray(rows) if len(rows) else None,
        b_ub=np.asarray(limits) if len(limits) else None,
        A_eq=a_eq, b_eq=[1.0], bounds=bounds,
        method="highs", options=LP_OPTIONS,
    )
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(f"Linear programming failed: {result.message}")
    return result


def _region_solution(pieces, weak_a, weak_b, strict_a, strict_b, feasible,
                     response_margin, value_tolerance):

    m = len(feasible)
    pieces = np.asarray(pieces, dtype=float).reshape(-1, m)
    weak_a = np.asarray(weak_a, dtype=float).reshape(-1, m)
    strict_a = np.asarray(strict_a, dtype=float).reshape(-1, m)
    weak_b = np.asarray(weak_b, dtype=float)
    strict_b = np.asarray(strict_b, dtype=float)
    if len(strict_b) and np.any(np.min(strict_a[:, feasible], axis=1) >= strict_b - LP_TOLERANCE):
        return None
    bounds = [(0., 1.) if f else (0., 0.) for f in feasible]
    constraints = np.vstack((weak_a, strict_a))
    limits = np.concatenate((weak_b, strict_b))
    rows = [np.r_[row, 0.] for row in constraints]
    rhs = list(limits)
    rows.extend(np.r_[-row, 1.] for row in pieces)
    rhs.extend([0.] * len(pieces))
    closure = _lp(np.r_[np.zeros(m), -1.], rows, rhs,
                  bounds + [(None, None)], m)
    if closure is None:
        return None
    bound = float(-closure.fun)
    policy = np.asarray(closure.x[:m])
    if not len(strict_b):
        return bound, policy, 0.
    gaps = strict_b - strict_a @ policy
    if np.min(gaps) >= STRICT_MARGIN_FLOOR:
        return bound, policy, float(np.min(gaps))
    interior_rows = [np.r_[row, 0.] for row in weak_a]
    interior_rows.extend(np.r_[row, 1.] for row in strict_a)
    interior = _lp(np.r_[np.zeros(m), -1.], interior_rows,
                   list(weak_b) + list(strict_b), bounds + [(0., 1.)], m)
    if interior is None or interior.x[-1] <= LP_TOLERANCE:
        return None
    interior_policy = np.asarray(interior.x[:m])
    interior_gaps = strict_b - strict_a @ interior_policy
    if np.min(interior_gaps) <= STRICT_MARGIN_FLOOR:
        return bound, interior_policy, float(np.min(interior_gaps))
    target_margin = min(response_margin, 0.5 * float(np.min(interior_gaps)))
    shifted_rhs = rhs.copy()
    start = len(weak_b)
    for index in range(len(strict_b)):
        shifted_rhs[start + index] -= target_margin
    shifted = _lp(np.r_[np.zeros(m), -1.], rows, shifted_rhs,
                  bounds + [(None, None)], m)
    if shifted is not None:
        trial = np.asarray(shifted.x[:m])
        trial_gap = float(np.min(strict_b - strict_a @ trial))
        if trial_gap >= STRICT_MARGIN_FLOOR and bound + shifted.fun <= value_tolerance:
            return bound, trial, trial_gap
    loss = max(0., bound - float(np.min(pieces @ interior_policy)))
    maximum_mix = min(1., value_tolerance / (2. * loss)) if loss > 0 else 1.
    needs_margin = gaps < STRICT_MARGIN_FLOOR
    required_mix = float(np.max((STRICT_MARGIN_FLOOR - gaps[needs_margin]) /
                                (interior_gaps[needs_margin] - gaps[needs_margin])))
    fraction = max(required_mix * 1.1, min(maximum_mix, target_margin / float(np.min(interior_gaps))))
    fraction = min(1., fraction)
    candidate = (1. - fraction) * policy + fraction * interior_policy
    return bound, candidate, float(np.min(strict_b - strict_a @ candidate))


def evaluate_reference_policy(
    policy: np.ndarray,
    mission_values: np.ndarray,
    attack_costs: np.ndarray = ATTACK_COSTS,
    defense_costs: np.ndarray = DEFENSE_COSTS,
    lambda_attack: float = LAMBDA_ATTACK,
    lambda_defense: float = LAMBDA_DEFENSE,
    best_response_tolerance: float = REFERENCE_BEST_RESPONSE_TOLERANCE,
) -> PolicyEvaluation:

    policy = np.asarray(policy, dtype=float)
    mission_by_attack = policy @ np.asarray(mission_values, dtype=float)
    attacker_payoffs = 1. - mission_by_attack - lambda_attack * np.asarray(attack_costs)
    maximum = float(np.max(attacker_payoffs))
    responses = tuple(int(j) for j in np.flatnonzero(attacker_payoffs >= maximum - best_response_tolerance))
    response = min(responses, key=lambda j: (float(mission_by_attack[j]), j))
    mission = float(mission_by_attack[response])
    cost = float(policy @ defense_costs)
    return PolicyEvaluation(response, responses, mission, cost, mission - lambda_defense * cost)


def _finish_policy(candidates, evaluator, mission_lower, attack_costs,
                   lambda_attack, value_tolerance, rule):
    if not candidates:
        raise RuntimeError("No feasible response region was found.")
    supremum = max(item[0] for item in candidates)
    evaluated = [(evaluator(item[1]), item) for item in candidates]
    evaluation, selected = max(evaluated, key=lambda z: (
        z[0].defender_value, -z[0].expected_defense_cost,
        -z[0].attacker_index, *(-z[1][1])))
    _, policy, margin = selected
    gap = supremum - evaluation.defender_value
    if gap < -1.0e-8 or gap > value_tolerance + 1.0e-8:
        raise RuntimeError(f"Policy value did not meet its LP bound: gap={gap:.6g}")
    payoffs = 1. - policy @ mission_lower - lambda_attack * np.asarray(attack_costs)
    response = evaluation.attacker_index
    response_gaps = [payoffs[response] - payoffs[k] for k in range(len(payoffs)) if k != response]
    return BoundedPolicyResult(
        policy=policy, attacker_index=response,
        guaranteed_mission=evaluation.guaranteed_mission,
        expected_defense_cost=evaluation.expected_defense_cost,
        defender_value=evaluation.defender_value,
        response_margin_used=margin,
        minimum_payoff_gap=float(min(response_gaps)) if response_gaps else 0.,
        value_supremum=supremum, optimality_gap=max(0., gap), response_rule=rule,
    )


def reference_optimize_policy(
    mission_lower: np.ndarray,
    attack_costs: np.ndarray = ATTACK_COSTS,
    defense_costs: np.ndarray = DEFENSE_COSTS,
    budget: float = BUDGET,
    lambda_attack: float = LAMBDA_ATTACK,
    lambda_defense: float = LAMBDA_DEFENSE,
    response_margin: float = DEFAULT_RESPONSE_MARGIN,
    value_tolerance: float = DEFAULT_VALUE_TOLERANCE,
) -> BoundedPolicyResult:

    if response_margin < STRICT_MARGIN_FLOOR or value_tolerance <= 0:
        raise ValueError("Response margin or value tolerance is too small.")
    values, ac, dc, feasible = _game_inputs(mission_lower, attack_costs, defense_costs,
                                           budget, lambda_attack, lambda_defense)
    candidates = []
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
        result = _region_solution([values[:, j] - dc], weak_a, weak_b,
                                  strict_a, strict_b, feasible, response_margin,
                                  value_tolerance)
        if result is not None:
            candidates.append(result)
    evaluator = lambda x: evaluate_reference_policy(x, values, attack_costs, defense_costs,
                                          lambda_attack, lambda_defense)
    return _finish_policy(candidates, evaluator, values, attack_costs,
                          lambda_attack, value_tolerance, "upper")


def evaluate_potential_policy(policy, mission_lower, mission_upper,
                              attack_costs=ATTACK_COSTS, defense_costs=DEFENSE_COSTS,
                              lambda_attack=LAMBDA_ATTACK, lambda_defense=LAMBDA_DEFENSE):

    lower, upper = policy @ mission_lower, policy @ mission_upper
    ac = lambda_attack * np.asarray(attack_costs)
    plus, minus = 1. - lower - ac, 1. - upper - ac
    responses = tuple(int(j) for j in np.flatnonzero(plus >= np.max(minus) - REFERENCE_BEST_RESPONSE_TOLERANCE))
    response = min(responses, key=lambda j: (float(lower[j]), j))
    cost = float(policy @ defense_costs)
    mission = float(lower[response])
    return PolicyEvaluation(response, responses, mission, cost, mission - lambda_defense * cost)


def optimize_potential_policy(mission_lower, mission_upper, attack_costs=ATTACK_COSTS,
                              defense_costs=DEFENSE_COSTS, budget=BUDGET,
                              lambda_attack=LAMBDA_ATTACK, lambda_defense=LAMBDA_DEFENSE,
                              response_margin=DEFAULT_RESPONSE_MARGIN,
                              value_tolerance=DEFAULT_VALUE_TOLERANCE):

    lower, ac, dc, feasible = _game_inputs(mission_lower, attack_costs, defense_costs,
                                          budget, lambda_attack, lambda_defense)
    upper = np.asarray(mission_upper, dtype=float)
    if upper.shape != lower.shape or not np.all(np.isfinite(upper)):
        raise ValueError("Upper and lower mission bounds must have the same finite shape.")
    if np.any(upper < lower) or np.any(upper > 1):
        raise ValueError("Mission intervals must satisfy 0 <= lower <= upper <= 1.")
    if response_margin < STRICT_MARGIN_FLOOR or value_tolerance <= 0:
        raise ValueError("Response margin or value tolerance is too small.")
    n = lower.shape[1]
    choices = [(-1,) + tuple(k for k in range(n) if k != j) for j in range(n)]
    candidates = []
    for pattern in itertools.product(*choices):
        retained = [j for j, witness in enumerate(pattern) if witness == -1]
        if not retained:
            continue
        strict_a, strict_b = [], []
        for j, witness in enumerate(pattern):
            if witness != -1:
                strict_a.append(upper[:, witness] - lower[:, j])
                strict_b.append(float(ac[j] - ac[witness]))
        result = _region_solution([lower[:, j] - dc for j in retained], [], [],
                                  strict_a, strict_b, feasible, response_margin,
                                  value_tolerance)
        if result is not None:
            candidates.append(result)
    evaluator = lambda x: evaluate_potential_policy(x, lower, upper, attack_costs,
                                                    defense_costs, lambda_attack,
                                                    lambda_defense)
    return _finish_policy(candidates, evaluator, lower, attack_costs,
                          lambda_attack, value_tolerance, "potential")


def policy_row(
    sample_size: int,
    repetition: int,
    model: str,
    method: str,
    confidence: float | None,
    result: BoundedPolicyResult,
    oracle_policy: np.ndarray,
    oracle_regret_reference: float,
    truth_matrix: np.ndarray,
) -> dict[str, Any]:

    truth_evaluation = evaluate_policy(result.policy, truth_matrix)
    raw_regret = oracle_regret_reference - truth_evaluation.defender_value
    if raw_regret < -1.0e-7:
        raise RuntimeError(
            "A policy exceeded the computed oracle value beyond numerical "
            f"tolerance ({raw_regret=})."
        )
    return {
        "sample_size": sample_size,
        "repetition": repetition,
        "model": model,
        "interval_method": method,
        "confidence_level": "" if confidence is None else confidence,
        "policy_d1": float(result.policy[0]),
        "policy_d2": float(result.policy[1]),
        "policy_d3": float(result.policy[2]),
        "model_attacker_best_response": f"a{result.attacker_index + 1}",
        "model_conditional_mission_lower": result.guaranteed_mission,
        "model_expected_defense_cost": result.expected_defense_cost,
        "model_conditional_net_value": result.defender_value,
        "response_margin_used": result.response_margin_used,
        "model_value_supremum": result.value_supremum,
        "model_optimality_gap": result.optimality_gap,
        "minimum_attacker_payoff_gap": result.minimum_payoff_gap,
        "l1_distance_to_oracle": float(
            np.sum(np.abs(result.policy - oracle_policy))
        ),
        "truth_attacker_best_response": f"a{truth_evaluation.attacker_index + 1}",
        "truth_mission_value": truth_evaluation.guaranteed_mission,
        "truth_expected_defense_cost": truth_evaluation.expected_defense_cost,
        "truth_defender_value": truth_evaluation.defender_value,
        "policy_regret": max(0.0, raw_regret),
    }


def analyze_dataset(
    sample_size: int,
    repetitions: int,
    seed: int,
    interval_settings: Sequence[tuple[str, float]],
    oracle_result: BoundedPolicyResult,
    oracle_regret_reference: float,
) -> DatasetAnalysis:

    truth_matrix = mission_truth_matrix()
    navigation, mission_normal, mission_degraded = generate_counts(
        repetitions, sample_size, seed
    )
    point_mission = point_mission_estimate(
        navigation, mission_normal, mission_degraded, sample_size
    )

    point_rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        result = optimize_policy_with_bounds(point_mission[repetition])
        point_rows.append(
            policy_row(
                sample_size,
                repetition + 1,
                "point",
                "none",
                None,
                result,
                oracle_result.policy,
                oracle_regret_reference,
                truth_matrix,
            )
        )

    row_truths = {
        "navigation_normal": np.broadcast_to(NAVIGATION_TRUTH, navigation.shape),
        "mission_success_given_normal": np.full(
            mission_normal.shape, MISSION_NORMAL_TRUTH
        ),
        "mission_success_given_degraded": np.full(
            mission_degraded.shape, MISSION_DEGRADED_TRUTH
        ),
    }
    interval_rows: dict[tuple[str, float], list[dict[str, Any]]] = {}
    interval_summaries: list[dict[str, Any]] = []

    for method, confidence in interval_settings:
        setting = (method, confidence)
        navigation_interval = conditional_binomial_interval(
            navigation, sample_size, confidence, method
        )
        normal_interval = conditional_binomial_interval(
            mission_normal, sample_size, confidence, method
        )
        degraded_interval = conditional_binomial_interval(
            mission_degraded, sample_size, confidence, method
        )
        mission_lower, mission_upper = propagate_mission_intervals(
            navigation_interval, normal_interval, degraded_interval
        )
        mission_coverage = (mission_lower <= truth_matrix) & (
            truth_matrix <= mission_upper
        )

        local_intervals = {
            "navigation_normal": navigation_interval,
            "mission_success_given_normal": normal_interval,
            "mission_success_given_degraded": degraded_interval,
        }
        local_coverage_by_type: dict[str, np.ndarray] = {}
        local_width_by_type: dict[str, np.ndarray] = {}
        for row_type, (lower, upper) in local_intervals.items():
            local_coverage_by_type[row_type] = (
                (lower <= row_truths[row_type])
                & (row_truths[row_type] <= upper)
            )
            local_width_by_type[row_type] = upper - lower

        model_rows: list[dict[str, Any]] = []
        for repetition in range(repetitions):
            result = optimize_policy_with_bounds(mission_lower[repetition])
            model_rows.append(
                policy_row(
                    sample_size,
                    repetition + 1,
                    "credal",
                    method,
                    confidence,
                    result,
                    oracle_result.policy,
                    oracle_regret_reference,
                    truth_matrix,
                )
            )
        interval_rows[setting] = model_rows

        all_cells_covered = np.all(mission_coverage, axis=(1, 2))
        joint_successes = int(np.sum(all_cells_covered))
        joint_low, joint_high = wilson_monte_carlo_interval(
            joint_successes, repetitions
        )
        per_repetition_cell_coverage = np.mean(
            mission_coverage, axis=(1, 2)
        )
        per_repetition_width = np.mean(
            mission_upper - mission_lower, axis=(1, 2)
        )
        cell_mean, cell_mcse = mean_mcse(per_repetition_cell_coverage)
        width_mean, width_mcse = mean_mcse(per_repetition_width)

        local_coverage_all = np.concatenate(
            [
                values.reshape(repetitions, -1)
                for values in local_coverage_by_type.values()
            ],
            axis=1,
        )
        local_width_all = np.concatenate(
            [
                values.reshape(repetitions, -1)
                for values in local_width_by_type.values()
            ],
            axis=1,
        )
        local_cov_mean, local_cov_mcse = mean_mcse(
            np.mean(local_coverage_all, axis=1)
        )
        local_width_mean, local_width_mcse = mean_mcse(
            np.mean(local_width_all, axis=1)
        )

        summary: dict[str, Any] = {
            "interval_method": method,
            "confidence_level": confidence,
            "repetitions": repetitions,
            "sample_size_per_row": sample_size,
            "local_coverage_overall": local_cov_mean,
            "local_coverage_overall_mcse": local_cov_mcse,
            "local_mean_width_overall": local_width_mean,
            "local_mean_width_overall_mcse": local_width_mcse,
            "mission_cell_coverage_mean": cell_mean,
            "mission_cell_coverage_mcse": cell_mcse,
            "simultaneous_all_cells_successes": joint_successes,
            "simultaneous_all_cells_coverage": joint_successes / repetitions,
            "simultaneous_all_cells_mc95_low": joint_low,
            "simultaneous_all_cells_mc95_high": joint_high,
            "mean_mission_interval_width": width_mean,
            "mean_mission_interval_width_mcse": width_mcse,
            **summarize_policy_rows(model_rows, "credal"),
        }
        for row_type in local_intervals:
            label = {
                "navigation_normal": "navigation",
                "mission_success_given_normal": "mission_normal",
                "mission_success_given_degraded": "mission_degraded",
            }[row_type]
            cov_mean, cov_mcse = mean_mcse(
                np.mean(local_coverage_by_type[row_type], axis=(1, 2))
            )
            row_width_mean, row_width_mcse = mean_mcse(
                np.mean(local_width_by_type[row_type], axis=(1, 2))
            )
            summary[f"local_coverage_{label}"] = cov_mean
            summary[f"local_coverage_{label}_mcse"] = cov_mcse
            summary[f"local_mean_width_{label}"] = row_width_mean
            summary[f"local_mean_width_{label}_mcse"] = row_width_mcse
        interval_summaries.append(summary)

    return DatasetAnalysis(
        sample_size=sample_size,
        point_rows=point_rows,
        interval_rows=interval_rows,
        interval_summaries=interval_summaries,
    )


def build_n_sweep_summary(
    analysis: DatasetAnalysis,
    paired_summary: dict[str, Any],
) -> dict[str, Any]:
    cp = find_summary(
        analysis.interval_summaries, "clopper_pearson", 0.95
    )
    point = summarize_policy_rows(analysis.point_rows, "point")
    return {
        "sample_size": analysis.sample_size,
        "repetitions": len(analysis.point_rows),
        "cp_cell_coverage_mean": cp["mission_cell_coverage_mean"],
        "cp_joint_coverage_successes": cp[
            "simultaneous_all_cells_successes"
        ],
        "cp_joint_coverage": cp["simultaneous_all_cells_coverage"],
        "cp_joint_mc95_low": cp["simultaneous_all_cells_mc95_low"],
        "cp_joint_mc95_high": cp["simultaneous_all_cells_mc95_high"],
        "cp_mean_interval_width": cp["mean_mission_interval_width"],
        "cp_local_mean_interval_width": cp["local_mean_width_overall"],
        "cp_local_mean_interval_width_mcse": cp["local_mean_width_overall_mcse"],
        "cp_regret_mcse": cp["credal_policy_regret_mcse"],
        "point_regret_mcse": point["point_policy_regret_mcse"],
        "cp_mean_interval_width_mcse": cp[
            "mean_mission_interval_width_mcse"
        ],
        "cp_oracle_l1_mean": cp["credal_policy_l1_mean"],
        "cp_regret_mean": cp["credal_policy_regret_mean"],
        "point_oracle_l1_mean": point["point_policy_l1_mean"],
        "point_regret_mean": point["point_policy_regret_mean"],
        "cp_point_policy_l1_mean": paired_summary[
            "cp_point_policy_l1_mean"
        ],
        "cp_point_policy_l1_mcse": paired_summary[
            "cp_point_policy_l1_mcse"
        ],
        "cp_point_policy_l1_mc95_low": paired_summary[
            "cp_point_policy_l1_mc95_low"
        ],
        "cp_point_policy_l1_mc95_high": paired_summary[
            "cp_point_policy_l1_mc95_high"
        ],
        "paired_regret_difference_mean": paired_summary[
            "paired_regret_difference_mean"
        ],
        "paired_regret_difference_mcse": paired_summary[
            "paired_regret_difference_mcse"
        ],
        "paired_regret_difference_mc95_low": paired_summary[
            "paired_regret_difference_mc95_low"
        ],
        "paired_regret_difference_mc95_high": paired_summary[
            "paired_regret_difference_mc95_high"
        ],
    }


def run_repeated_sampling_evaluation(
    output: Path,
    repetitions: int,
    seed: int,
    sample_sizes: Sequence[int],
    make_plots: bool,
    oracle_result: BoundedPolicyResult,
    oracle_regret_reference: float,
) -> None:

    directory = output / "repeated_sampling"
    directory.mkdir(parents=True, exist_ok=True)

    print("  Fixed N=15 interval comparison", flush=True)
    primary = analyze_dataset(
        PRIMARY_SAMPLE_SIZE,
        repetitions,
        seed,
        PRIMARY_INTERVAL_SETTINGS,
        oracle_result,
        oracle_regret_reference,
    )

    save_analysis(directory, primary, seed)
    interval_sensitivity_rows = []
    for summary in primary.interval_summaries:
        interval_sensitivity_rows.append(
            {
                "interval_method": summary["interval_method"],
                "confidence_level": summary["confidence_level"],
                "sample_size": PRIMARY_SAMPLE_SIZE,
                "repetitions": repetitions,
                "cell_coverage": summary["mission_cell_coverage_mean"],
                "joint_coverage_successes": summary[
                    "simultaneous_all_cells_successes"
                ],
                "joint_coverage": summary[
                    "simultaneous_all_cells_coverage"
                ],
                "joint_coverage_mc95_low": summary[
                    "simultaneous_all_cells_mc95_low"
                ],
                "joint_coverage_mc95_high": summary[
                    "simultaneous_all_cells_mc95_high"
                ],
                "mean_mission_interval_width": summary[
                    "mean_mission_interval_width"
                ],
                "oracle_policy_l1": summary["credal_policy_l1_mean"],
                "policy_regret": summary["credal_policy_regret_mean"],
            }
        )
    write_csv(
        directory / "interval_sensitivity_n15.csv",
        interval_sensitivity_rows,
    )

    cp95_rows = primary.interval_rows[SWEEP_PRIMARY_SETTING]
    cp95_point = paired_policy_summary(
        primary.point_rows, cp95_rows, PRIMARY_SAMPLE_SIZE
    )
    write_csv(
        directory / "cp95_point_comparison_n15.csv",
        [
            {
                "sample_size": PRIMARY_SAMPLE_SIZE,
                "repetitions": repetitions,
                "cp_point_policy_l1_mean": cp95_point[
                    "cp_point_policy_l1_mean"
                ],
                "cp_point_policy_l1_mc95_low": cp95_point[
                    "cp_point_policy_l1_mc95_low"
                ],
                "cp_point_policy_l1_mc95_high": cp95_point[
                    "cp_point_policy_l1_mc95_high"
                ],
                "point_minus_cp_regret_mean": cp95_point[
                    "paired_regret_difference_mean"
                ],
                "point_minus_cp_regret_mc95_low": cp95_point[
                    "paired_regret_difference_mc95_low"
                ],
                "point_minus_cp_regret_mc95_high": cp95_point[
                    "paired_regret_difference_mc95_high"
                ],
            }
        ],
    )

    interval_policy_rows = []
    for method, confidence in (("wilson", 0.95), ("jeffreys", 0.95)):
        summary = paired_interval_policy_comparison(
            cp95_rows,
            primary.interval_rows[(method, confidence)],
            PRIMARY_SAMPLE_SIZE,
            method,
            confidence,
        )
        interval_policy_rows.append(
            {
                "reference_method": summary["reference_method"],
                "reference_confidence_level": summary[
                    "reference_confidence_level"
                ],
                "candidate_method": method,
                "candidate_confidence_level": confidence,
                "sample_size": PRIMARY_SAMPLE_SIZE,
                "repetitions": repetitions,
                "policy_l1_mean": summary["policy_l1_mean"],
                "attacker_response_agreement_rate": summary[
                    "attacker_response_agreement_rate"
                ],
            }
        )
    write_csv(
        directory / "interval_policy_comparison_n15.csv",
        interval_policy_rows,
    )

    print("  Sample-size sweep", flush=True)
    sample_size_rows = []
    for index, sample_size in enumerate(sample_sizes, start=1):
        print(f"    N={sample_size} ({index}/{len(sample_sizes)})", flush=True)
        analysis = primary if sample_size == PRIMARY_SAMPLE_SIZE else analyze_dataset(
            sample_size,
            repetitions,
            seed,
            [SWEEP_PRIMARY_SETTING],
            oracle_result,
            oracle_regret_reference,
        )
        paired = paired_policy_summary(
            analysis.point_rows,
            analysis.interval_rows[SWEEP_PRIMARY_SETTING],
            sample_size,
        )
        save_analysis(directory, analysis, seed)
        sample_size_rows.append(build_n_sweep_summary(analysis, paired))

    sample_size_rows.sort(key=lambda row: int(row["sample_size"]))
    sample_size_csv = directory / "sample_size_sensitivity.csv"
    write_csv(sample_size_csv, sample_size_rows)
    if make_plots:
        create_n_sweep_figure_from_csv(sample_size_csv, output / "figures")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


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


def save_analysis(directory, analysis, seed):

    n = analysis.sample_size
    rows = list(analysis.point_rows)
    for values in analysis.interval_rows.values():
        rows.extend(values)
    write_csv(directory / f"policies_N{n}.csv", rows)
    write_csv(directory / f"interval_statistics_N{n}.csv", analysis.interval_summaries)
    nav, normal, degraded = generate_counts(len(analysis.point_rows), n, seed)
    np.savez_compressed(directory / f"counts_N{n}.npz", navigation=nav,
                        mission_normal=normal, mission_degraded=degraded,
                        sample_size=n, seed=seed)


def summarize_records(records, group_keys, metrics):
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[k] for k in group_keys)].append(record)
    output = []
    for key, group in grouped.items():
        row = dict(zip(group_keys, key))
        row["repetitions"] = len(group)
        for metric in metrics:
            data = np.asarray([r[metric] for r in group], dtype=float)
            stats = mean_t_interval(data)
            for suffix in ("mean", "mcse", "mc95_low", "mc95_high"):
                row[f"{metric}_{suffix}"] = stats[suffix]
            row[f"{metric}_q95"] = float(np.quantile(data, .95))
            row[f"{metric}_maximum"] = float(np.max(data))
        output.append(row)
    return output


def paired_method_summary(records, method_a, method_b):
    grouped = defaultdict(dict)
    for row in records:
        grouped[(row["scenario"], row["sample_size"], row["repetition"])][row["method"]] = row
    differences = []
    for (scenario, n, repetition), methods in grouped.items():
        if method_a not in methods or method_b not in methods:
            continue
        a, b = methods[method_a], methods[method_b]
        coordinates = sorted(k for k in a if k.startswith("policy_d") and k[8:].isdigit())
        distance = sum(abs(a[k] - b[k]) for k in coordinates)
        difference = a["policy_regret"] - b["policy_regret"]
        differences.append({
            "scenario": scenario, "sample_size": n, "repetition": repetition,
            "method_a": method_a, "method_b": method_b,
            "regret_a_minus_b": difference, "policy_l1": distance,
            "truth_response_disagreement": float(a["truth_response"] != b["truth_response"]),
            "b_lower_regret": float(difference > 1e-12),
            "a_lower_regret": float(difference < -1e-12),
            "equal_regret": float(abs(difference) <= 1e-12),
        })
    return summarize_records(differences, ["scenario", "sample_size", "method_a", "method_b"],
                             ["regret_a_minus_b", "policy_l1", "truth_response_disagreement",
                              "b_lower_regret", "a_lower_regret", "equal_regret"])


def comparison_record(result, truth, truth_supremum, lower, upper, method,
                      scenario, n, repetition):
    truth_evaluation = evaluate_policy(result.policy, truth)
    interval_evaluation = evaluate_policy(result.policy, lower)
    potential_evaluation = evaluate_potential_policy(result.policy, lower, upper)
    regret = truth_supremum - truth_evaluation.defender_value
    if regret < -1e-8:
        raise RuntimeError("A selected policy exceeded the truth-model supremum.")
    return {
        "scenario": scenario, "sample_size": n, "repetition": repetition,
        "method": method,
        **{f"policy_d{i + 1}": float(v) for i, v in enumerate(result.policy)},
        "model_response": f"a{result.attacker_index + 1}",
        "truth_response": f"a{truth_evaluation.attacker_index + 1}",
        "truth_value": truth_evaluation.defender_value,
        "truth_supremum": truth_supremum,
        "policy_regret": max(0., regret),
        "common_upper_value": interval_evaluation.defender_value,
        "common_potential_value": potential_evaluation.defender_value,
        "potential_response_count": len(potential_evaluation.best_response_indices),
        "own_model_value": result.defender_value,
        "own_model_supremum": result.value_supremum,
        "solver_value_gap": result.optimality_gap,
        "strict_response_margin": result.response_margin_used,
    }


def run_scenario_comparison(output, repetitions, seed):

    directory = output / "scenario_comparison"
    directory.mkdir(parents=True, exist_ok=True)
    definitions, all_summaries, all_pairs = [], [], []
    for case_index, (name, navigation_truth) in enumerate(SCENARIO_NAVIGATION.items()):
        truth = navigation_truth * MISSION_NORMAL_TRUTH + (1. - navigation_truth) * MISSION_DEGRADED_TRUTH
        oracle = optimize_policy_with_bounds(truth)
        uniform = np.full(3, 1. / 3.)
        uniform_payoffs = 1. - uniform @ truth - ATTACK_COSTS
        ordered = np.sort(uniform_payoffs)
        definitions.append({
            "scenario": name, "navigation_truth": navigation_truth.tolist(),
            "mission_normal_truth": MISSION_NORMAL_TRUTH,
            "mission_degraded_truth": MISSION_DEGRADED_TRUTH,
            "mission_truth": truth.tolist(),
            "reference_policy": uniform.tolist(),
            "reference_attacker_payoffs": uniform_payoffs.tolist(),
            "reference_top_two_gap": float(ordered[-1] - ordered[-2]),
            "truth_supremum": oracle.value_supremum,
            "oracle_reference_policy": oracle.policy.tolist(),
        })
        for n in (PRIMARY_SAMPLE_SIZE,):
            print(f"  Scenario {name}, N={n}, R={repetitions}", flush=True)
            stream_seed = seed if name == "table_ii" else derived_seed(seed, 200, case_index, n)
            nav, normal, degraded = generate_scenario_counts(repetitions, n, stream_seed, navigation_truth)
            lower, upper = propagate_mission_intervals(*(conditional_binomial_interval(v, n, .95, "clopper_pearson")
                                                        for v in (nav, normal, degraded)))
            point = point_mission_estimate(nav, normal, degraded, n)
            records = []
            for r in range(repetitions):
                results = {
                    "point": optimize_policy_with_bounds(point[r]),
                    "cp_upper": optimize_policy_with_bounds(lower[r]),
                    "cp_potential": select_potential_policy(lower[r], upper[r]),
                }
                for method, result in results.items():
                    row = comparison_record(result, truth, oracle.value_supremum,
                                            lower[r], upper[r], method, name, n, r + 1)
                    row.update({"data_seed": stream_seed,
                                "mission_interval_width": float(np.mean(upper[r] - lower[r])),
                                "mission_cell_coverage": float(np.mean((lower[r] <= truth) & (truth <= upper[r])))})
                    records.append(row)
                if (r + 1) % 100 == 0:
                    print(f"    {r + 1}/{repetitions}", flush=True)
            stem = f"{name}_N{n}"
            write_csv(directory / f"policies_{stem}.csv", records)
            np.savez_compressed(directory / f"counts_{stem}.npz", navigation=nav,
                                mission_normal=normal, mission_degraded=degraded,
                                sample_size=n, seed=stream_seed, mission_truth=truth)
            all_summaries.extend(summarize_records(records, ["scenario", "sample_size", "method"],
                ["policy_regret", "truth_value", "common_upper_value", "common_potential_value",
                 "solver_value_gap", "potential_response_count", "mission_interval_width",
                 "mission_cell_coverage"]))
            for a, b in [("point", "cp_upper"), ("cp_upper", "cp_potential"), ("point", "cp_potential")]:
                all_pairs.extend(paired_method_summary(records, a, b))
            write_csv(directory / "summary.csv", all_summaries)
            write_csv(directory / "paired_comparisons.csv", all_pairs)
    write_json(directory / "scenario_definitions.json", definitions)


def run_real_world_response_comparison(output):
    directory = output / 'real_world_instantiation'
    directory.mkdir(parents=True, exist_ok=True)
    lower, upper = conditional_binomial_interval(
        REAL_WORLD_SUCCESS_COUNTS, REAL_WORLD_SAMPLE_SIZE, 0.95, 'clopper_pearson')
    ca, cd, budget = (REAL_WORLD_PRIMARY_ATTACK_COST,
                      REAL_WORLD_PRIMARY_DEFENSE_COST, REAL_WORLD_PRIMARY_BUDGET)
    ac, dc = np.array([0., ca]), np.array([0., cd])
    cp = solve_real_world_policy(lower, ca, cd, budget,
                                best_response_tolerance=REFERENCE_BEST_RESPONSE_TOLERANCE)
    point = solve_real_world_policy(REAL_WORLD_SUCCESS_COUNTS / REAL_WORLD_SAMPLE_SIZE,
                                   ca, cd, budget,
                                   best_response_tolerance=REFERENCE_BEST_RESPONSE_TOLERANCE)
    potential = select_potential_policy(lower, upper, ac, dc, budget)
    records = []
    for name, policy in [('cp_upper_representative', cp.policy),
                         ('point_representative', point.policy),
                         ('cp_potential', potential.policy)]:
        upper_evaluation = evaluate_reference_policy(policy, lower, ac, dc)
        potential_evaluation = evaluate_potential_policy(policy, lower, upper, ac, dc)
        records.append({
            'policy_source': name, 'policy_d1': float(policy[0]), 'policy_d2': float(policy[1]),
            'common_cp_upper_response': f'a{upper_evaluation.attacker_index + 1}',
            'common_cp_upper_value': upper_evaluation.defender_value,
            'common_cp_potential_value': potential_evaluation.defender_value,
            'potential_responses': ';'.join(f'a{j + 1}' for j in potential_evaluation.best_response_indices),
            'expected_defense_cost': upper_evaluation.expected_defense_cost,
        })
    write_csv(directory / 'response_sensitivity.csv', records)
    realization = lower.copy()
    realization[:, 0] = upper[:, 0]
    example = evaluate_reference_policy(cp.policy, realization, ac, dc)
    payoffs = 1. - cp.policy @ realization - ac
    write_csv(directory / 'admissible_payoff_example.csv', [{
        'policy_d2': float(cp.policy[1]),
        'upper_rule_response': f'a{cp.attacker_index + 1}',
        'realization_attacker_payoff_a1': float(payoffs[0]),
        'realization_attacker_payoff_a2': float(payoffs[1]),
        'realization_response': f'a{example.attacker_index + 1}',
        'realization_defender_value': example.defender_value,
        'potential_value_intercept': float(lower[0, 1]),
        'potential_value_slope': float(lower[1, 1] - lower[0, 1] - cd),
    }])
    write_json(directory / 'response_comparison_settings.json', {
        'success_counts': REAL_WORLD_SUCCESS_COUNTS.tolist(),
        'trials_per_cell': REAL_WORLD_SAMPLE_SIZE,
        'lower': lower.tolist(), 'upper': upper.tolist(),
        'attack_costs': ac.tolist(), 'defense_costs': dc.tolist(), 'budget': budget,
        'representative_threshold_increment': REAL_WORLD_THRESHOLD_INCREMENT,
        'potential_value_supremum': potential.value_supremum,
        'potential_policy_value_gap': potential.optimality_gap,
        'example_mission_utilities': realization.tolist(),
    })


def _envelope_evaluation(policy, vertices, attack_costs, defense_costs):
    mission = np.min(np.einsum("i,sij->sj", policy, vertices), axis=0)
    payoffs = 1. - mission - attack_costs
    responses = tuple(int(j) for j in np.flatnonzero(payoffs >= np.max(payoffs) - REFERENCE_BEST_RESPONSE_TOLERANCE))
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


def potential_supremum_two_defenses(lower, upper, attack_costs, defense_costs):

    lower, upper = np.asarray(lower), np.asarray(upper)
    ac, dc = np.asarray(attack_costs), np.asarray(defense_costs)
    la, lb = lower[0], lower[1] - lower[0]
    ua, ub = upper[0], upper[1] - upper[0]
    points = {0., 1.}
    for j in range(len(ac)):
        for k in range(len(ac)):
            _add_crossing(points, la[j] - ua[k] + ac[j] - ac[k], lb[j] - ub[k])
            _add_crossing(points, la[j] - la[k], lb[j] - lb[k])
    points = sorted(points)
    best = max(evaluate_potential_policy(np.array([1. - t, t]), lower, upper, ac, dc).defender_value
               for t in points)
    for left, right in zip(points[:-1], points[1:]):
        if right - left < 1e-13:
            continue
        middle = (left + right) / 2.
        response = evaluate_potential_policy(np.array([1. - middle, middle]), lower, upper, ac, dc).attacker_index
        for endpoint in (left, right):
            best = max(best, float(la[response] + lb[response] * endpoint - dc[0] -
                                   (dc[1] - dc[0]) * endpoint))
    return best


def coupling_record(case_name, theta_lower, theta_upper):
    case = COUPLING_CASES[case_name]
    intercept, slope = np.asarray(case["intercept"]), np.asarray(case["slope"])
    ac, dc = np.asarray(case["attack_costs"]), np.asarray(case["defense_costs"])
    vertices = np.stack([intercept + slope * theta_lower, intercept + slope * theta_upper])
    lower, upper = np.min(vertices, axis=0), np.max(vertices, axis=0)
    rectangular = reference_optimize_policy(lower, ac, dc, case["budget"])
    coupled_policy, coupled_evaluation, coupled_supremum = optimize_two_defense_envelope(
        vertices, ac, dc, case["budget"])
    rectangular_in_coupled = _envelope_evaluation(rectangular.policy, vertices, ac, dc)
    coupled_in_rectangular = evaluate_reference_policy(coupled_policy, lower, ac, dc)
    x = np.array([.5, .5])
    coupled_lower = np.min(np.einsum("i,sij->sj", x, vertices), axis=0)
    coupled_upper = np.max(np.einsum("i,sij->sj", x, vertices), axis=0)
    rectangular_lower, rectangular_upper = x @ lower, x @ upper
    truth = intercept + .5 * slope
    truth_supremum = reference_optimize_policy(truth, ac, dc, case["budget"]).value_supremum
    truth_c = evaluate_reference_policy(coupled_policy, truth, ac, dc)
    truth_r = evaluate_reference_policy(rectangular.policy, truth, ac, dc)
    return {
        "scenario": case_name, "theta_lower": theta_lower, "theta_upper": theta_upper,
        "theta_interval_width": theta_upper - theta_lower,
        "uniform_attack_rectangular_lower": float(rectangular_lower[1]),
        "uniform_attack_rectangular_upper": float(rectangular_upper[1]),
        "uniform_attack_coupled_lower": float(coupled_lower[1]),
        "uniform_attack_coupled_upper": float(coupled_upper[1]),
        "uniform_attack_lower_gap": float(coupled_lower[1] - rectangular_lower[1]),
        "uniform_attack_width_reduction": float((rectangular_upper[1] - rectangular_lower[1]) -
                                                  (coupled_upper[1] - coupled_lower[1])),
        "rectangular_policy_d2": float(rectangular.policy[1]),
        "coupled_policy_d2": float(coupled_policy[1]),
        "policy_l1": float(np.sum(np.abs(rectangular.policy - coupled_policy))),
        "rectangular_own_value": rectangular.defender_value,
        "coupled_own_value": coupled_evaluation.defender_value,
        "rectangular_policy_on_coupled_value": rectangular_in_coupled.defender_value,
        "coupled_policy_on_rectangular_value": coupled_in_rectangular.defender_value,
        "coupled_evaluation_gain": coupled_evaluation.defender_value - rectangular_in_coupled.defender_value,
        "coupled_value_supremum": coupled_supremum,
        "coupled_solver_value_gap": max(0., coupled_supremum - coupled_evaluation.defender_value),
        "rectangular_solver_value_gap": rectangular.optimality_gap,
        "rectangular_response": f"a{rectangular.attacker_index + 1}",
        "coupled_response": f"a{coupled_evaluation.attacker_index + 1}",
        "response_disagreement": float(rectangular.attacker_index != coupled_evaluation.attacker_index),
        "rectangular_truth_value": truth_r.defender_value,
        "coupled_truth_value": truth_c.defender_value,
        "rectangular_truth_regret": max(0., truth_supremum - truth_r.defender_value),
        "coupled_truth_regret": max(0., truth_supremum - truth_c.defender_value),
    }


def run_coupling_experiment(output):
    directory = output / 'coupling_sensitivity'
    directory.mkdir(parents=True, exist_ok=True)
    theta_lower, theta_upper = 0.3, 0.7
    record = coupling_record('opposed_defenses', theta_lower, theta_upper)
    write_csv(directory / 'shared_parameter_comparison.csv', [record])
    write_json(directory / 'settings.json', {
        'theta_bounds': [theta_lower, theta_upper], 'generating_theta': 0.5,
        'case': COUPLING_CASES['opposed_defenses'],
        'uniform_policy': [0.5, 0.5], 'evaluated_attack': 'a2',
        'attacker_rule': 'Upper-payoff response with pessimistic ties in both models.',
        'coupled_set': 'One common theta determines all cells.',
        'rectangular_set': 'Each cell varies independently over its marginal interval.',
    })


def run_propagation_experiment(output, repetitions, seed, trajectory_budgets):

    directory = output / "trajectory_propagation"
    directory.mkdir(parents=True, exist_ok=True)
    navigation_given_localization = np.array([.25, .95])
    mission_given_navigation = np.array([.05, .95])
    navigation_truth = .25 + (.95 - .25) * NAVIGATION_TRUTH
    truth = .05 + (.95 - .05) * navigation_truth
    oracle = optimize_policy_with_bounds(truth)
    summaries, pairs = [], []
    for budget in trajectory_budgets:
        print(f"  Complete trajectories per cell B={budget}, R={repetitions}", flush=True)
        stream_seed = derived_seed(seed, 400, 0, budget)
        rng = np.random.default_rng(stream_seed)
        successes = np.zeros((repetitions, 3, 3, 5), dtype=int)
        trials = np.zeros_like(successes)
        endpoint_successes = np.zeros((repetitions, 3, 3), dtype=int)
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
            endpoint_successes[r] = mission.sum(axis=-1)
        local_lower, local_upper = conditional_binomial_interval(successes, trials, .95, "clopper_pearson")
        graph_lower, graph_upper = np.ones((repetitions, 3, 3)), np.zeros((repetitions, 3, 3))
        for bits in itertools.product([False, True], repeat=5):
            p, q1, q0, r1, r0 = [local_upper[..., j] if bit else local_lower[..., j]
                                  for j, bit in enumerate(bits)]
            nav = p * q1 + (1. - p) * q0
            value = nav * r1 + (1. - nav) * r0
            graph_lower = np.minimum(graph_lower, value)
            graph_upper = np.maximum(graph_upper, value)
        endpoint_lower, endpoint_upper = conditional_binomial_interval(endpoint_successes, budget, .95, "clopper_pearson")
        records = []
        for r in range(repetitions):
            for method, lower, upper in [("graph_cp", graph_lower[r], graph_upper[r]),
                                         ("endpoint_cp", endpoint_lower[r], endpoint_upper[r])]:
                result = optimize_policy_with_bounds(lower)
                row = comparison_record(result, truth, oracle.value_supremum,
                                        graph_lower[r], graph_upper[r], method,
                                        "three_node_chain", budget, r + 1)
                covered = (lower <= truth) & (truth <= upper)
                row.update({"data_seed": stream_seed, "trajectories_per_cell": budget,
                            "mission_interval_width": float(np.mean(upper - lower)),
                            "mission_cell_coverage": float(np.mean(covered)),
                            "all_cells_covered": float(np.all(covered)),
                            "fraction_empty_local_rows": float(np.mean(trials[r] == 0)),
                            "minimum_local_row_trials": int(np.min(trials[r])),
                            "mean_local_interval_width": float(np.mean(local_upper[r] - local_lower[r]))})
                records.append(row)
        write_csv(directory / f"policies_B{budget}.csv", records)
        np.savez_compressed(directory / f"counts_B{budget}.npz", local_successes=successes,
                            local_trials=trials, endpoint_successes=endpoint_successes,
                            trajectories_per_cell=budget, seed=stream_seed,
                            row_order=np.array(["localization", "navigation_given_localization",
                                                "navigation_given_no_localization",
                                                "mission_given_navigation", "mission_given_no_navigation"]))
        summaries.extend(summarize_records(records, ["scenario", "sample_size", "method"],
            ["policy_regret", "truth_value", "mission_interval_width", "mission_cell_coverage",
             "all_cells_covered", "fraction_empty_local_rows", "minimum_local_row_trials",
             "mean_local_interval_width", "solver_value_gap"]))
        pairs.extend(paired_method_summary(records, "endpoint_cp", "graph_cp"))
        write_csv(directory / "summary.csv", summaries)
        write_csv(directory / "paired_comparisons.csv", pairs)
    write_json(directory / "settings.json", {
        "graph": "localization -> navigation -> mission",
        "localization_truth": NAVIGATION_TRUTH.tolist(),
        "navigation_normal_given_localization_success": .95,
        "navigation_normal_given_localization_failure": .25,
        "mission_success_given_navigation_normal": .95,
        "mission_success_given_navigation_degraded": .05,
        "mission_truth": truth.tolist(), "empty_row_interval": [0., 1.],
        "observation_design": "B complete trajectories per cell; both methods use the same trajectories.",
        "parameterization": "Conditional rows are estimated separately per attack-defense cell; equal generating constants do not impose shared estimated parameters.",
        "data_type": "Synthetic trajectories; no physical intermediate-state observations.",
    })


def run_validation(output):

    directory = output / "validation"
    directory.mkdir(parents=True, exist_ok=True)
    records = []

    def check(name, actual, expected, tolerance=1e-8):
        error = float(abs(actual - expected))
        passed = bool(error <= tolerance)
        records.append({"check": name, "actual": float(actual), "expected": float(expected),
                        "absolute_error": error, "tolerance": tolerance, "passed": passed})
        if not passed:
            write_csv(directory / "checks.csv", records)
            raise AssertionError(f"{name}: {actual} != {expected} within {tolerance}")

    values = np.array([[.6, .4], [.6, .4]])
    ac, dc = np.array([0., .2]), np.zeros(2)
    nominal = reference_optimize_policy(values, ac, dc, 0.)
    robust = optimize_potential_policy(values, values, ac, dc, 0.)
    check("permanent_tie_oracle", nominal.value_supremum, .4)
    check("permanent_tie_policy", nominal.defender_value, .4)
    check("permanent_tie_singleton", robust.value_supremum, .4)
    single = reference_optimize_policy(np.array([[.2], [.7]]), np.zeros(1), np.array([0., .1]), .1)
    check("single_attack", single.defender_value, .6)
    tied = reference_optimize_policy(np.full((2, 2), .5), np.zeros(2), np.zeros(2), 0.)
    check("equal_cost_ties", tied.defender_value, .5)

    lower, upper = conditional_binomial_interval(REAL_WORLD_SUCCESS_COUNTS, 15, .95, "clopper_pearson")
    ac, dc = np.array([0., .3]), np.array([0., .2])
    tau = (lower[0, 0] - lower[0, 1] - .3) / (lower[1, 1] - lower[0, 1])
    for name, t, expected in [("below_threshold", tau - 1e-5, 1),
                               ("at_threshold", tau, 1), ("above_threshold", tau + 1e-5, 0)]:
        check(name, evaluate_reference_policy(np.array([1-t, t]), lower, ac, dc).attacker_index, expected, 0.)
    result = reference_optimize_policy(lower, ac, dc, .2)
    check("open_region_supremum", result.value_supremum, lower[0, 0] - .2 * tau)
    check("open_region_value_gap", max(0., result.optimality_gap - DEFAULT_VALUE_TOLERANCE), 0.)
    budget_result = reference_optimize_policy(lower, ac, dc, .19)
    check("hard_budget_support", budget_result.policy[1], 0., 1e-12)
    potential = optimize_potential_policy(lower, upper, ac, dc, .2)
    check("real_world_potential_policy", potential.policy[1], 1.)
    check("real_world_potential_value", potential.defender_value, float(np.min(lower[1])) - dc[1])
    cp = solve_real_world_policy(lower, .3, .2, .2, best_response_tolerance=REFERENCE_BEST_RESPONSE_TOLERANCE)
    expected_x2 = min(1., tau + REAL_WORLD_THRESHOLD_INCREMENT)
    expected_policy = np.array([1. - expected_x2, expected_x2])
    expected_cost = float(expected_policy @ dc)
    check("real_world_representative_policy", cp.policy[1], expected_x2)
    check("real_world_representative_upper_value", cp.defender_value, lower[0, 0] - expected_cost)
    check("real_world_representative_potential_value",
          evaluate_potential_policy(cp.policy, lower, upper, ac, dc).defender_value,
          float(np.min(expected_policy @ lower)) - expected_cost)
    ac_high = np.array([0., .7])
    potential_open = optimize_potential_policy(lower, upper, ac_high, dc, .2)
    independent = potential_supremum_two_defenses(lower, upper, ac_high, dc)
    check("potential_open_region_supremum", potential_open.value_supremum, independent)
    check("potential_open_region_value_gap", max(0., potential_open.optimality_gap - DEFAULT_VALUE_TOLERANCE), 0.)

    vertices = [np.where(np.asarray(bits).reshape(2, 2), upper, lower)
                for bits in itertools.product([False, True], repeat=4)]
    worst_error = 0.
    for t in np.linspace(0., 1., 101):
        x = np.array([1. - t, t])
        direct = min(evaluate_reference_policy(x, v, ac, dc).defender_value for v in vertices)
        formula = evaluate_potential_policy(x, lower, upper, ac, dc).defender_value
        worst_error = max(worst_error, abs(direct - formula))
    check("potential_formula_vs_16_vertices", worst_error, 0.)

    rng = np.random.default_rng(73129)
    for case in range(12):
        lo = rng.uniform(.05, .75, size=(2, 3))
        hi = lo + rng.uniform(0., .2, size=(2, 3))
        ac_random = rng.uniform(0., .3, size=3)
        dc_random = np.array([0., .08])
        nominal = reference_optimize_policy(lo, ac_random, dc_random, .08)
        _, _, expected = optimize_two_defense_envelope(lo[None, :, :], ac_random, dc_random, .08)
        check(f"nominal_piecewise_reference_{case}", nominal.value_supremum, expected, 2e-8)
        robust = optimize_potential_policy(lo, hi, ac_random, dc_random, .08)
        expected = potential_supremum_two_defenses(lo, hi, ac_random, dc_random)
        check(f"potential_piecewise_reference_{case}", robust.value_supremum, expected, 2e-8)
        singleton = optimize_potential_policy(lo, lo, ac_random, dc_random, .08)
        check(f"singleton_limit_{case}", singleton.value_supremum, nominal.value_supremum, 2e-8)

    empty_lower, empty_upper = conditional_binomial_interval(np.array([0, 0, 15]), np.array([0, 15, 15]), .95, "clopper_pearson")
    check("empty_row_lower", empty_lower[0], 0.)
    check("empty_row_upper", empty_upper[0], 1.)
    check("all_success_cp_lower", empty_lower[2], (.05 / 2.) ** (1. / 15.))
    check("all_success_cp_upper", empty_upper[2], 1.)
    uniform = np.array([.5, .5])
    opposite_vertices = np.array([[[.2], [.8]], [[.8], [.2]]])
    check("opposed_shared_mixture", float(np.min(np.einsum("i,sij->sj", uniform, opposite_vertices))), .5)
    check("opposed_rectangular_mixture", float(uniform @ np.min(opposite_vertices, axis=0)[:, 0]), .2)
    write_csv(directory / "checks.csv", records)
    print(f"  Numerical checks passed: {len(records)}", flush=True)
    return len(records)


def optimize_policy_with_bounds(mission_lower, attack_costs=ATTACK_COSTS,
                    defense_costs=DEFENSE_COSTS, budget=BUDGET,
                    lambda_attack=LAMBDA_ATTACK, lambda_defense=LAMBDA_DEFENSE,
                    response_margin=DEFAULT_RESPONSE_MARGIN):

    ac, dc = np.asarray(attack_costs), np.asarray(defense_costs)
    selected = optimize_policy(mission_lower, ac, dc, budget,
                                         lambda_attack, lambda_defense, response_margin)
    actual = evaluate_policy(selected.policy, mission_lower, ac, dc,
                                       lambda_attack, lambda_defense)
    if abs(actual.defender_value - selected.defender_value) > 1e-8:
        raise RuntimeError("The selected policy failed re-evaluation.")
    reference = reference_optimize_policy(mission_lower, ac, dc, budget,
                                         lambda_attack, lambda_defense)
    gap = reference.value_supremum - actual.defender_value
    if gap < -1e-8:
        raise RuntimeError("The selected policy exceeded its reference bound.")
    return BoundedPolicyResult(selected.policy.copy(), selected.attacker_index,
                        selected.guaranteed_mission, selected.expected_defense_cost,
                        selected.defender_value, selected.response_margin_used,
                        selected.minimum_payoff_gap, reference.value_supremum,
                        max(0., gap), "upper_payoff")


def select_potential_policy(mission_lower, mission_upper, attack_costs=ATTACK_COSTS,
                            defense_costs=DEFENSE_COSTS, budget=BUDGET,
                            lambda_attack=LAMBDA_ATTACK, lambda_defense=LAMBDA_DEFENSE,
                            response_margin=DEFAULT_RESPONSE_MARGIN):

    lower, ac, dc, feasible = _game_inputs(mission_lower, attack_costs, defense_costs,
                                          budget, lambda_attack, lambda_defense)
    upper = np.asarray(mission_upper, dtype=float)
    if upper.shape != lower.shape or np.any(upper < lower) or np.any(upper > 1):
        raise ValueError("Invalid mission-utility intervals.")
    if not np.all(np.isfinite(upper)) or not math.isfinite(response_margin) or response_margin <= 0:
        raise ValueError("Finite bounds and a positive response margin are required.")
    m, n = lower.shape
    choices = [(-1,) + tuple(k for k in range(n) if k != j) for j in range(n)]
    bounds = [(0., 1.) if f else (0., 0.) for f in feasible] + [(None, None)]
    candidates = []
    for pattern in itertools.product(*choices):
        retained = [j for j, witness in enumerate(pattern) if witness == -1]
        if not retained:
            continue
        constraints, limits = [], []
        for j, witness in enumerate(pattern):
            if witness != -1:
                constraints.append(np.r_[upper[:, witness] - lower[:, j], 0.])
                limits.append(float(ac[j] - ac[witness]) - response_margin)
        constraints.extend(np.r_[dc - lower[:, j], 1.] for j in retained)
        limits.extend([0.] * len(retained))
        solved = _lp(np.r_[np.zeros(m), -1.], constraints, limits, bounds, m)
        if solved is None:
            continue
        x = np.asarray(solved.x[:m])
        ev = evaluate_potential_policy(x, lower, upper, attack_costs, defense_costs,
                                       lambda_attack, lambda_defense)
        candidates.append((ev, x))
    if not candidates:
        raise RuntimeError("No potential-response comparison policy was found.")
    ev, x = max(candidates, key=lambda item: (item[0].defender_value,
                 -item[0].attacker_index, *(-item[1])))
    reference = optimize_potential_policy(lower, upper, attack_costs, defense_costs,
                                          budget, lambda_attack, lambda_defense)
    gap = reference.value_supremum - ev.defender_value
    if gap < -1e-8:
        raise RuntimeError("A comparison policy exceeded the potential-response bound.")
    return BoundedPolicyResult(x, ev.attacker_index, ev.guaranteed_mission,
                        ev.expected_defense_cost, ev.defender_value,
                        response_margin, 0., reference.value_supremum,
                        max(0., gap), "potential_fixed_margin")


def verify_shared_setting_consistency(output):
    summary = output / 'repeated_sampling' / 'sample_size_sensitivity.csv'
    scenarios = output / 'scenario_comparison' / 'summary.csv'
    if not summary.is_file() or not scenarios.is_file():
        return []
    repeated = {int(row['sample_size']): row for row in read_csv(summary)}
    records = []
    for row in read_csv(scenarios):
        n = int(row['sample_size'])
        if row['scenario'] != 'table_ii' or n not in repeated or row['method'] == 'cp_potential':
            continue
        metric = 'point_regret_mean' if row['method'] == 'point' else 'cp_regret_mean'
        difference = abs(float(row['policy_regret_mean']) - float(repeated[n][metric]))
        if difference > 1e-10:
            raise RuntimeError(f'Shared-setting results disagree: N={n}, {metric}.')
        records.append({'comparison': 'scenario_vs_repeated_sampling',
                        'sample_size': n, 'metric': metric,
                        'absolute_difference': difference, 'passed': True})
    return records


def parse_args():
    parser = argparse.ArgumentParser(description='Reproduce the UGV mission-impact experiments.')
    parser.add_argument('--experiment', default='all', choices=[
        'all', 'illustrative', 'repeated', 'scenarios', 'coupling',
        'propagation', 'dependence', 'real-world', 'validate', 'plot'])
    parser.add_argument('--repetitions', type=int, default=DEFAULT_REPETITIONS,
                        help='Independent data sets per synthetic setting (default: 1000).')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Master random seed (default: 1).')
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=list(DEFAULT_SAMPLE_SIZES),
                        help='Conditional-row sample sizes for the repeated-sampling analysis.')
    parser.add_argument('--trajectory-budgets', type=int, nargs='+',
                        default=list(DEFAULT_TRAJECTORY_BUDGETS),
                        help='Complete trajectories per cell for the propagation comparison.')
    parser.add_argument('--output', type=Path, default=Path('results'),
                        help='A new or empty output directory.')
    parser.add_argument('--plot-csv', type=Path,
                        help='sample_size_sensitivity.csv, required for --experiment plot.')
    parser.add_argument('--no-plots', action='store_true', help='Skip figure generation.')
    parser.add_argument('--no-archive', action='store_true', help='Skip the results ZIP archive.')
    return parser.parse_args()


def _prepare_run_directory(path):
    output = path.expanduser().resolve()
    protected = {Path.cwd().resolve(), Path.home().resolve(), Path("/"),
                 Path(__file__).resolve().parent}
    if output in protected:
        raise SystemExit("Choose a dedicated output subdirectory.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _validate_arguments(args):
    if args.repetitions < 2:
        raise SystemExit('--repetitions must be at least 2.')
    if args.seed < 0:
        raise SystemExit('--seed must be nonnegative.')
    for name in ('sample_sizes', 'trajectory_budgets'):
        values = getattr(args, name)
        if not values or any(value < 1 for value in values) or len(values) != len(set(values)):
            raise SystemExit(f"--{name.replace('_', '-')} requires distinct positive integers.")
    if args.experiment == 'plot':
        if args.plot_csv is None or not args.plot_csv.is_file():
            raise SystemExit('--experiment plot requires --plot-csv with an existing CSV file.')
        if args.no_plots:
            raise SystemExit('--no-plots cannot be used with --experiment plot.')
    elif args.plot_csv is not None:
        raise SystemExit('--plot-csv is only used with --experiment plot.')
    if not args.no_archive and Path(str(args.output.expanduser().resolve()) + '.zip').exists():
        raise SystemExit('The results ZIP already exists; choose a different --output path.')


def run_all(args):
    _validate_arguments(args)
    output = _prepare_run_directory(args.output)
    started = time.perf_counter()
    source = Path(__file__).resolve()
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    metadata = {
        'program': 'ugv_mission_impact_evaluation',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
        'python': platform.python_version(), 'platform': platform.platform(),
        'dependencies': {name: version(name) for name in ('numpy', 'scipy', 'matplotlib')},
        'configuration': configuration,
        'selection': {
            'point_and_cp_upper': 'Fixed strict-response-margin selection with pessimistic ties.',
            'response_margin': DEFAULT_RESPONSE_MARGIN,
            'best_response_tolerance': BEST_RESPONSE_TOLERANCE,
            'cp_potential': 'Fixed strict exclusion margin over attainable response sets.',
            'real_world_increment': REAL_WORLD_THRESHOLD_INCREMENT,
            'budget': 'Per-configuration support constraint: C_D(d_i) <= b.',
        },
        'numerics': {
            'solver': 'scipy.optimize.linprog / HiGHS',
            'reference_lp_tolerance': LP_TOLERANCE,
            'reference_response_tolerance': REFERENCE_BEST_RESPONSE_TOLERANCE,
            'strict_margin_floor': STRICT_MARGIN_FLOOR,
            'reference_value_tolerance': DEFAULT_VALUE_TOLERANCE,
        },
        'synthetic_parameters': {
            'navigation_truth': NAVIGATION_TRUTH.tolist(),
            'mission_normal_truth': MISSION_NORMAL_TRUTH,
            'mission_degraded_truth': MISSION_DEGRADED_TRUTH,
            'attack_costs': ATTACK_COSTS.tolist(), 'defense_costs': DEFENSE_COSTS.tolist(),
            'budget': BUDGET, 'lambda_attack': LAMBDA_ATTACK, 'lambda_defense': LAMBDA_DEFENSE,
        },
        'random_streams': {
            'table_ii': 'default_rng(master_seed); navigation, normal, degraded rows per repetition.',
            'other_matrices': 'SeedSequence([master_seed, 200, scenario_index, N]).',
            'trajectories': 'SeedSequence([master_seed, 400, 0, B]).',
            'observation_dependence': 'SeedSequence([master_seed, 500, stream_id, 300]); rho 0.0/0.6 use stream_id 0/2.',
        },
        'scope': 'Synthetic probabilities are controlled settings. Real-world observations are aggregate outcomes.',
    }
    write_json(output / 'run_manifest.json', metadata)
    try:
        snapshots = output / 'source'
        snapshots.mkdir()
        for path in [source, source.parent / 'requirements.txt', source.parent / 'README.md']:
            if path.is_file():
                shutil.copyfile(path, snapshots / path.name)
        if args.experiment == 'plot':
            create_n_sweep_figure_from_csv(args.plot_csv.resolve(), output / 'figures')
        else:
            print('Numerical validation', flush=True)
            metadata['validation_checks_passed'] = run_validation(output)
            if args.experiment in ('all', 'illustrative'):
                print('Illustrative full pipeline', flush=True)
                run_illustrative_full_pipeline(output, args.seed)
            if args.experiment in ('all', 'repeated'):
                print('Repeated sampling', flush=True)
                oracle = optimize_policy_with_bounds(mission_truth_matrix())
                run_repeated_sampling_evaluation(output, args.repetitions, args.seed,
                    args.sample_sizes, not args.no_plots, oracle, oracle.value_supremum)
            if args.experiment in ('all', 'scenarios'):
                print('Generating matrices and response criteria', flush=True)
                run_scenario_comparison(output, args.repetitions, args.seed)
            if args.experiment in ('all', 'propagation'):
                print('Matched-trajectory propagation', flush=True)
                run_propagation_experiment(output, args.repetitions, args.seed,
                                          args.trajectory_budgets)
            if args.experiment in ('all', 'coupling'):
                print('Shared parameter', flush=True)
                run_coupling_experiment(output)
            if args.experiment in ('all', 'dependence'):
                print('Observation dependence', flush=True)
                run_dependence_experiment(output, args.repetitions, args.seed)
            if args.experiment in ('all', 'real-world'):
                print('Reduced real-world case', flush=True)
                run_real_world_instantiation(output)
                run_real_world_response_comparison(output)
            consistency = verify_shared_setting_consistency(output)
            write_csv(output / 'validation' / 'shared_setting_consistency.csv', consistency)
            metadata['shared_setting_checks_passed'] = len(consistency)
        metadata['status'] = 'completed'
    except BaseException as error:
        metadata['status'] = 'failed'
        metadata['failure'] = f'{type(error).__name__}: {error}'
        metadata['elapsed_seconds'] = time.perf_counter() - started
        write_json(output / 'run_manifest.json', metadata)
        raise
    metadata['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['elapsed_seconds'] = time.perf_counter() - started
    write_json(output / 'run_manifest.json', metadata)
    index = [{'file': path.relative_to(output).as_posix(), 'bytes': path.stat().st_size,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(output.rglob('*')) if path.is_file()]
    write_csv(output / 'results_index.csv', index)
    if not args.no_archive:
        archive = shutil.make_archive(str(output), 'zip', root_dir=output.parent, base_dir=output.name)
        print(f'Results archive: {archive}', flush=True)
    print(f"Elapsed seconds: {metadata['elapsed_seconds']:.1f}", flush=True)
    return output


def main():
    output = run_all(parse_args())
    print(f"Evaluation completed successfully. Results: {output}", flush=True)


def create_n_sweep_figure_from_csv(csv_path, figure_directory):
    rows = sorted(read_csv(csv_path), key=lambda row: int(row['sample_size']))
    required = {
        'sample_size', 'repetitions', 'cp_local_mean_interval_width',
        'cp_local_mean_interval_width_mcse', 'cp_mean_interval_width',
        'cp_mean_interval_width_mcse', 'cp_point_policy_l1_mean',
        'cp_point_policy_l1_mc95_low', 'cp_point_policy_l1_mc95_high',
    }
    if not rows or any(required.difference(row) for row in rows):
        raise ValueError('The plotting CSV must contain the sample-size summary columns.')
    n = np.asarray([int(row['sample_size']) for row in rows])
    repetitions = np.asarray([int(row['repetitions']) for row in rows])
    if np.any(n < 1) or len(np.unique(n)) != len(n) or np.any(repetitions < 2):
        raise ValueError('Sample sizes must be positive and distinct; repetitions must be at least two.')
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 7,
        'axes.titlesize': 7.5, 'axes.labelsize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'axes.linewidth': 0.6, 'lines.linewidth': 1,
        'lines.markersize': 3, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.65), layout='constrained')
    for prefix, label, color, marker in [
        ('cp_mean_interval_width', 'Mission', '#3B6F8F', 'o'),
        ('cp_local_mean_interval_width', 'Local', '#B97138', '^'),
    ]:
        means = np.asarray([float(row[prefix]) for row in rows])
        mcse = np.asarray([float(row[prefix + '_mcse']) for row in rows])
        error = student_t.ppf(0.975, repetitions - 1) * mcse
        axes[0].errorbar(n, means, yerr=error, color=color, marker=marker,
                         capsize=1.5, elinewidth=0.6, label=label)
    means = np.asarray([float(row['cp_point_policy_l1_mean']) for row in rows])
    lo = np.asarray([float(row['cp_point_policy_l1_mc95_low']) for row in rows])
    hi = np.asarray([float(row['cp_point_policy_l1_mc95_high']) for row in rows])
    axes[1].errorbar(n, means, yerr=np.maximum(0, np.vstack([means - lo, hi - means])),
                     color='#4F7F52', marker='s', capsize=1.5, elinewidth=0.6)
    axes[0].set_title('(a) Interval width')
    axes[0].set_ylabel('Mean width')
    axes[0].legend(fontsize=6, frameon=False)
    axes[1].set_title('(b) CP-point distance')
    axes[1].set_ylabel(r'Mean policy $\ell_1$')
    ticks = [value for value in (5, 15, 60, 300, 1000) if value in n]
    if len(ticks) < 2:
        ticks = n[np.unique(np.linspace(0, len(n) - 1, min(5, len(n))).astype(int))].tolist()
    for axis in axes:
        axis.set_xscale('log')
        axis.set_xticks(ticks, [str(value) for value in ticks])
        axis.minorticks_off()
        axis.set_ylim(bottom=0)
        axis.set_xlabel(r'Observations per row, $N$')
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
        if 15 in n:
            axis.axvline(15, color='0.5', linestyle='--', linewidth=0.6)
    figure_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ('pdf', 'png'):
        path = figure_directory / f'synthetic_n_sweep.{suffix}'
        fig.savefig(path, dpi=600)
        paths.append(path)
    plt.close(fig)
    return tuple(paths)


def run_dependence_experiment(output, repetitions, seed):
    directory = output / 'observation_dependence'
    directory.mkdir(parents=True, exist_ok=True)
    n = 300
    probabilities = np.stack([
        NAVIGATION_TRUTH,
        np.full((3, 3), MISSION_NORMAL_TRUTH),
        np.full((3, 3), MISSION_DEGRADED_TRUTH),
    ])
    summaries = []
    streams = ((0.0, 0), (0.6, 2))
    for rho, stream_id in streams:
        stream_seed = derived_seed(seed, 500, stream_id, n)
        rng = np.random.default_rng(stream_seed)
        state = rng.random((repetitions, 3, 3, 3)) < probabilities
        counts = state.astype(int)
        for _ in range(n - 1):
            transition = (1. - rho) * probabilities + rho * state
            state = rng.random(state.shape) < transition
            counts += state
        lower, upper = conditional_binomial_interval(counts, n, 0.95, 'clopper_pearson')
        coverage = np.mean((lower <= probabilities) & (probabilities <= upper), axis=(1, 2, 3))
        records = [{'sample_size': n, 'serial_correlation': rho, 'repetition': r + 1,
                    'local_row_coverage': float(coverage[r])} for r in range(repetitions)]
        write_csv(directory / f'row_coverage_N{n}_rho{rho:.1f}.csv', records)
        np.savez_compressed(directory / f'counts_N{n}_rho{rho:.1f}.npz',
                            counts=counts, probabilities=probabilities, sample_size=n,
                            serial_correlation=rho, seed=stream_seed)
        summaries.extend(summarize_records(records, ['sample_size', 'serial_correlation'],
                                            ['local_row_coverage']))
    write_csv(directory / 'summary.csv', summaries)
    write_json(directory / 'settings.json', {
        'transition_probability': 'P(Y_t=1|Y_(t-1))=(1-rho)*p+rho*Y_(t-1)',
        'initial_state': 'Bernoulli(p), stationary',
        'sample_size': n, 'serial_correlations': [rho for rho, _ in streams],
        'stream_ids': {str(rho): stream_id for rho, stream_id in streams},
        'independence': 'Rows and repetitions are independent.',
        'probabilities': probabilities.tolist(),
    })


if __name__ == "__main__":
    main()
