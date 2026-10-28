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

## Files

| File | Purpose |
| --- | --- |
| `ugv_mission_impact_evaluation.py` | Experiment settings, sampling, interval propagation, evaluations, tables, and figure generation. |
| `policy_optimization.py` | Upper-payoff and potential-best-response policy selection and evaluation. |
| `test_policy_optimization.py` | Analytical boundary cases and independent numerical checks. |
| `requirements.txt` | Pinned package versions. |

CSV files contain unrounded numerical values. The settings in Tables I–II are defined by the input constants. All outputs are computed during execution; the package does not require result files as inputs.
