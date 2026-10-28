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

## Results

- `table_03_*.csv` through `table_11_*.csv`: numerical results for Tables III–XI. Table IV has separate files for panels (a) and (b).
- `text_results.csv`: numerical comparisons for the evaluation text, identified by section and quantity.
- `figure_04.pdf` and `figure_04.png`: Fig. 4. The plotted values and 95% Monte Carlo intervals are also in `table_05_sample_size.csv`.
- `optimization_summary.csv`: counts of completed upper-payoff and potential-response solves, maximum observed objective gaps, and numerical tolerances. This file is written after all experiments finish successfully.
- `run_config.json`: experiment settings, dependency versions, and source-file SHA-256 hashes for the run.

CSV files contain unrounded numerical values. The settings in Tables I–II are defined by the input constants. All outputs are computed during execution; the package does not require result files as inputs.
