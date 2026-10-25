# UGV Mission-Impact Assessment and Defense Selection

Numerical experiments for **UGV Mission-Impact Assessment and Defense Selection under Small-Sample Uncertainty** (PRDC 2026).

## Run

Use Python 3.10 and install the required packages:

```bash
python -m pip install -r requirements.txt
python ugv_mission_impact_evaluation.py
```

The default run uses seed `1` and `1000` repetitions with the experiment settings specified in the paper. Results are saved to the `results/` directory.

The output directory must be new or empty. To save results to a different directory:

```bash
python ugv_mission_impact_evaluation.py --output results
```

## Results

- `table_03_*.csv` through `table_11_*.csv`: numerical results for Tables III–XI. Table IV has separate files for panels (a) and (b).
- `text_results.csv`: numerical comparisons reported in the evaluation text, identified by section and quantity.
- `figure_04.pdf` and `figure_04.png`: Fig. 4. The plotted values and 95% Monte Carlo intervals are included in `table_05_sample_size.csv`.

CSV files contain unrounded numerical values. The settings in Tables I–II are defined by the input constants in the script.
