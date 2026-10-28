# UGV Mission-Impact Assessment and Defense Selection

Numerical experiments for **UGV Mission-Impact Assessment and Defense Selection under Small-Sample Uncertainty** (PRDC 2026).

## Run

Use Python 3.10 and install the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python ugv_mission_impact_evaluation.py
```

The default run uses seed `1` and `1000` repetitions. The generating probabilities, costs, budgets, sample sizes, and interval settings are specified in the source. Results are written to `results/`.

The output directory must be new or empty. An alternative destination can be specified with:

```bash
python ugv_mission_impact_evaluation.py --output my_results
```

A short run for checking the installation is:

```bash
python ugv_mission_impact_evaluation.py --repetitions 10 --seed 1 --output smoke_results
```

Use the default `1000` repetitions for the paper experiments. A short run has larger Monte Carlo uncertainty.

## Files

| File | Purpose |
| --- | --- |
| `ugv_mission_impact_evaluation.py` | Experiment settings, sampling, interval propagation, evaluations, tables, and figure generation. |
| `policy_optimization.py` | Upper-payoff and potential-best-response policy selection and evaluation. |
| `test_policy_optimization.py` | Analytical boundary cases and independent numerical checks. |
| `requirements.txt` | Pinned package versions. |

## Results

- `table_03_*.csv` through `table_11_*.csv`: numerical results for Tables III–XI. Table IV has separate files for panels (a) and (b).
- `text_results.csv`: numerical comparisons for the evaluation text, identified by section and quantity.
- `figure_04.pdf` and `figure_04.png`: Fig. 4. The plotted values and 95% Monte Carlo intervals are also in `table_05_sample_size.csv`.
- `optimization_summary.csv`: counts of completed upper-payoff and potential-response solves, maximum observed objective gaps, and numerical tolerances. This file is written after all experiments finish successfully.
- `run_config.json`: experiment settings, dependency versions, and source-file SHA-256 hashes for the run.

CSV files contain unrounded numerical values. The settings in Tables I–II are defined by the input constants. All outputs are computed during execution; the package does not require result files as inputs.

## Policy computation

The upper-payoff rule selects an attacker best response from the mission lower bounds. Attacker-payoff ties are resolved by the lowest defender mission utility. If that utility is also tied, the lowest attack index is used. The hard budget excludes any defense whose individual cost exceeds the budget.

For each admissible response region, the solver computes a zero-margin closure bound on defender value. Strict feasibility is checked separately. When a supremum lies on an excluded boundary, the solver constructs a policy inside the region with defender-value loss at most `1e-6`. The potential-response comparison enumerates nonempty response sets and strict exclusion witnesses, with the same value criterion.

Every returned policy is evaluated under its actual response rule and checked against the largest admissible region bound. Attacker-payoff ties use a numerical tolerance of `1e-9`; the LP primal and dual feasibility tolerances are `1e-10`. Bounds and objective gaps are numerical quantities computed in floating-point arithmetic. If a policy cannot satisfy the requested value criterion, execution raises an error; no repetition is omitted or resampled.

The shared-parameter example uses a piecewise affine two-defense envelope. The reduced real-world illustration constructs its deterrence policy with the specified increment `x2 = tau + 0.001`, capped at one. That explicit policy construction is distinct from the `1e-6` value criterion; the real-world CP-P comparison uses the potential-response solver.

Sampling streams are determined by the seed and experiment identifiers. Methods within a comparison use the same observations, and regret is evaluated under the common generating model with its attacker response recomputed.

## Tests

```bash
python -m unittest -v
```

The tests cover narrow response regions, persistent ties, boundary-only response sets, unattained suprema, hard-budget feasibility, cost coefficients, and potential responses. Analytical examples, independent one-dimensional breakpoint enumeration, and simplex-grid comparisons check policy values without loading experiment outputs.
