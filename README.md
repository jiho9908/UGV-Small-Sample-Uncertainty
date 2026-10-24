# UGV Mission-Impact Assessment and Defense Selection

Numerical experiments for **UGV Mission-Impact Assessment and Defense Selection under Small-Sample Uncertainty** (PRDC 2026).

## Run

Use Python 3.10 and the packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python ugv_mission_impact_evaluation.py
```

The default run uses seed `1` and `1000` repetitions, with the sample sizes and experiment settings specified in the paper. To use a different output directory:

```bash
python ugv_mission_impact_evaluation.py --output my_results
```

The output directory must be new or empty. An installation check can use `--repetitions 4 --output smoke_results`; this does not reproduce the paper's estimates.

## Results

The `results/` directory contains:

- `table_03_*.csv` through `table_11_*.csv`: numerical results for Tables III–XI, with Table IV split into panels (a) and (b).
- `text_results.csv`: numerical comparisons reported in the evaluation text. Section and quantity columns identify each result.
- `figure_04.pdf` and `figure_04.png`: interval widths and CP–Point policy distances. Their data and 95% Monte Carlo intervals are in `table_05_sample_size.csv`.

CSV values retain computational precision. Tables I–II are specified by the input constants in the script. `reference_results.zip` contains the default-run outputs; extract it separately when inspecting them.

The real-world inputs are the aggregate TurtleBot 2 outcomes reported by Wang et al., *Physical Attacks on Robot Navigation Systems* (reference [36] in the paper). Model definitions and experimental interpretation are given in the paper.
