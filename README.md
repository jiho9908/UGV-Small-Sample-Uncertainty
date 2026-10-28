# Run

Use Python 3.10. From this directory, run:

```bash
python -m pip install -r requirements.txt
python ugv_mission_impact_evaluation.py
```

The default run uses seed `1` and `1000` repetitions. It writes the paper's table data (`table_*.csv`), numerical values reported in the text (`text_results.csv`), and Figure 4 (`figure_04.pdf`) to `results/`.

The output directory must be new or empty. To use another directory:

```bash
python ugv_mission_impact_evaluation.py --output my_results
```
