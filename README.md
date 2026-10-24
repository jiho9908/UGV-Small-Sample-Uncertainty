# UGV Mission-Impact Assessment and Defense Selection

Python implementation of the numerical experiments in **UGV Mission-Impact Assessment and Defense Selection under Small-Sample Uncertainty**, by Jiho Kang, Sanghoon Jeon, and Huy Kang Kim, accepted for PRDC 2026.

The implementation constructs local binomial intervals, propagates them to mission-utility bounds, and selects mixed defense policies with strategy costs and a hard budget. It includes synthetic experiments and a reduced case based on reported aggregate robot-navigation outcomes.

## Installation

Use Python 3.10 and the versions pinned in `requirements.txt`:

- NumPy 1.24.4
- SciPy 1.15.3
- Matplotlib 3.10.8

The reference results were generated with Python 3.10.1 on Windows 10. The exact execution environment is recorded in `results/run_manifest.json` inside `reference_results.zip`.

Create an environment and install the dependencies. On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe ugv_mission_impact_evaluation.py --experiment all --output results
```

On Linux or macOS:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python ugv_mission_impact_evaluation.py --experiment all --output results
```

The commands below use `python` to mean the interpreter in this environment. No GUI, LaTeX installation, pandas, or notebook is required.

## Full run

```bash
python ugv_mission_impact_evaluation.py --experiment all --output results
```

The default settings are:

| Experiment | Settings |
| --- | --- |
| Illustrative synthetic pipeline | First realization at `N=15`, master seed `1` |
| Repeated sampling | `R=1000` data sets per `N`; `N=5,10,15,30,60,120,300,1000` |
| Interval constructions | CP 90%, 95%, 99%; Wilson 95%; equal-tailed Jeffreys 95%, all at `N=15` |
| Generating matrices | Six matrices, `N=15`, `R=1000`; Point, CP-U, and CP-P |
| Matched trajectories | `B=15,60,300` complete trajectories per cell; `R=1000` |
| Shared parameter | Two defenses and two attacks; common `theta` in `[0.3,0.7]`, generating `theta=0.5` |
| Observation dependence | `N=300`, `R=1000`, serial correlations `0.0` and `0.6` |
| Reduced real-world case | Four counts `[[15,6],[15,13]]`, 15 trials per cell; interval, cost-budget, and response comparisons |

`--output` must name a new or empty directory. Existing result directories and ZIP archives are not overwritten. A successful run creates `results/` and `results.zip`. The archive includes computed results, sampled counts, settings, numerical checks, a run manifest, and a source snapshot. Use `--no-archive` to omit the ZIP or `--no-plots` to omit figure generation.

Synthetic generating matrices and reported aggregate observations define the experiment inputs. Intervals, policies, and summary statistics are computed from these inputs.

## Reference results

[`reference_results.zip`](reference_results.zip) contains the default `R=1000`, seed `1` results under a top-level `results/` directory. It includes CSV summaries, per-repetition policies, sampled counts, figures, numerical checks, and the executed Python source. `run_manifest.json` records the execution environment and script hash; `results_index.csv` lists file hashes.

Extract the archive into a separate directory when inspecting it, so that `--output results` remains available for a new run. Use the numerical CSV columns for comparisons. LP solvers can choose different policies when an optimum is not unique, so retain the reported dependency versions and inspect the selected policies as well as objective values.

## Individual experiments

```bash
python ugv_mission_impact_evaluation.py --experiment illustrative --output results_illustrative
python ugv_mission_impact_evaluation.py --experiment repeated --output results_repeated
python ugv_mission_impact_evaluation.py --experiment scenarios --output results_scenarios
python ugv_mission_impact_evaluation.py --experiment propagation --output results_propagation
python ugv_mission_impact_evaluation.py --experiment coupling --output results_coupling
python ugv_mission_impact_evaluation.py --experiment dependence --output results_dependence
python ugv_mission_impact_evaluation.py --experiment real-world --output results_real_world
```

A small installation check runs every experiment with fewer repetitions:

```bash
python ugv_mission_impact_evaluation.py --experiment all --repetitions 4 --output smoke_results
```

This checks execution; it does not produce the paper's repeated-sampling estimates. The full run uses the default `R=1000` and seed `1`. `--sample-sizes` controls the repeated-sampling sweep, and `--trajectory-budgets` controls the propagation comparison. The fixed-`N` interval comparison and generating-matrix comparison use `N=15`.

Numerical checks run before simulation and policy experiments. They cover analytically solvable small games, pessimistic ties, budget feasibility, empty rows, interval endpoints, and independent one-dimensional optimization calculations. Expected values in these checks follow from their test inputs. They can be run alone:

```bash
python ugv_mission_impact_evaluation.py --experiment validate --output numerical_checks
```

## Outputs and paper correspondence

Paths in this table are relative to the selected output directory.

| Paper content | Output |
| --- | --- |
| Tables I–II: synthetic strategies, costs, and generating probabilities | Script constants and `run_manifest.json` |
| Table III: worked cell `(d2,a2)` | `illustrative_full_pipeline/worked_cell_d2_a2.csv` |
| Table IV: mission intervals and policies | `illustrative_full_pipeline/mission_utility_intervals.csv`, `defense_policies.csv` |
| Table V: sample-size analysis | `repeated_sampling/sample_size_sensitivity.csv` |
| Table VI: interval constructions | `repeated_sampling/interval_sensitivity_n15.csv` |
| Paired CP–Point and interval-method comparisons | `repeated_sampling/cp95_point_comparison_n15.csv`, `interval_policy_comparison_n15.csv` |
| Fig. 4: widths and policy distances | `figures/synthetic_n_sweep.pdf`, `synthetic_n_sweep.png` |
| Table VII: six generating matrices | `scenario_comparison/summary.csv`; paired comparisons in `paired_comparisons.csv` |
| Table VIII: local propagation and endpoint intervals | `trajectory_propagation/summary.csv`; paired comparisons in `paired_comparisons.csv` |
| Section IV-A5: shared parameter | `coupling_sensitivity/shared_parameter_comparison.csv` |
| Section IV-A5: serial dependence | `observation_dependence/summary.csv` |
| Table IX: physical-trial aggregate counts and CP intervals | `real_world_instantiation/observations_cp95.csv` |
| Table X: real-world interval sensitivity | `real_world_instantiation/interval_sensitivity.csv` |
| Table XI: real-world cost-budget sensitivity | `real_world_instantiation/cost_budget_table.csv`; all 27 settings in `cost_budget_sensitivity.csv` |
| Cost and response boundaries | `real_world_instantiation/cost_budget_regime_boundaries.csv` |
| Section IV-B4: common-CP response comparison | `real_world_instantiation/response_sensitivity.csv` |
| Section IV-B4: admissible payoff realization example | `real_world_instantiation/admissible_payoff_example.csv` |

The workflow and graph diagrams are conceptual figures, not outputs of this numerical script.

Per-repetition CSV files and compressed NumPy `.npz` files retain policies and sampled counts. `settings.json` and `scenario_definitions.json` describe the inputs for each comparison. CSV numbers are unrounded except for the compact presentation strings in `cost_budget_table.csv` and `policy_latex`; use the numerical columns for analysis.

`run_manifest.json` records the configuration, source hash, dependency versions, numerical settings, status, and elapsed time. `results_index.csv` records output file hashes. A failed run records its error in the manifest and leaves completed outputs available for inspection. It does not resume partial runs.

## Rebuild Fig. 4

The figure is generated from the stored summary CSV. To rebuild it without rerunning simulations:

```bash
python ugv_mission_impact_evaluation.py --experiment plot --plot-csv results/repeated_sampling/sample_size_sensitivity.csv --output figure_only
```

The left panel shows local and propagated mission interval widths. The right panel shows paired CP–Point policy distances. Error bars are 95% Monte Carlo intervals; the horizontal axis is logarithmic.

## Statistical design and interpretation

The reduced real-world input is taken from M. Wang, Y. Hayamizu, M. Tang, K. Gopalan, S. Zhang, and P. Yang, [Physical Attacks on Robot Navigation Systems](https://openreview.net/forum?id=A4AWclA4aC), RSS 2025 Workshop on Reliable Robotics: Safety and Security in the Face of Generative AI. It uses the reported TurtleBot 2 navigation outcomes: 15, 6, 15, and 13 successes out of 15 trials for `(default, no attack)`, `(default, table repositioning)`, `(enhanced, no attack)`, and `(enhanced, table repositioning)`, respectively. The input contains aggregate counts rather than individual trial trajectories. Strategy costs and the budget are scenario parameters specified by the accompanying UGV mission-impact paper.

For each conditional-row sample size, `numpy.random.default_rng(seed)` draws nine navigation rows, nine mission-success-given-normal rows, and nine mission-success-given-degraded rows in that order for every repetition. Methods at the same `N` share counts. Different sample sizes are separate experiments and are not analyzed as paired observations. The fixed-`N` and sample-size analyses use the same `N=15` sequence; its first realization supplies the illustrative example.

The baseline generating matrix in the scenario comparison uses the same seed and draw order. Other matrices use `SeedSequence([seed,200,scenario_index,N])`. Matched trajectories use `SeedSequence([seed,400,0,B])`. Serial-dependence settings use `SeedSequence([seed,500,stream_id,300])`, with stream IDs `0` and `2` for correlations `0.0` and `0.6`. These identifiers are part of the reproducibility settings.

Graph CP and Endpoint CP use the same complete trajectories. Graph CP estimates five local rows using their observed parent-state counts and assigns `[0,1]` to empty rows. Equal generating constants across cells do not impose shared estimated parameters. The shared-parameter comparison uses a specified interval for one common parameter; it is a deterministic comparison, not an estimated confidence interval.

Cell coverage averages over the nine mission intervals and repetitions. Joint coverage requires all nine intervals in a repetition to contain their generating probabilities. Mean widths and policy distances are averaged over independent repetitions. Regret is the generating-model value supremum minus the value of the selected policy, with the attacker response and pessimistic ties recomputed under the generating probabilities.

Wilson Monte Carlo intervals describe joint-coverage frequencies. Student-t Monte Carlo intervals describe means and paired differences. These intervals quantify simulation uncertainty; they are distinct from the local probability intervals used in the credal model. CP's nominal level is row-wise under the binomial assumptions. The serial-dependence experiment deliberately applies unadjusted binomial intervals to dependent observations.

## Decision model and numerical settings

Point and CP-U use the fixed upper-payoff attacker response with pessimistic tie-breaking. Their synthetic policy selection uses a strict attacker-payoff margin of `1e-4`, with best-response tolerance `1e-8`. Exact defender-value ties across response-region candidates use attacker index and then lexicographic policy order. There is no secondary defender objective within an individual LP.

CP-P considers attacker responses that can be optimal under admissible rectangular mission-utility realizations. Each cell's attacker and defender payoffs use the same mission utility. CP-P uses a strict exclusion margin of `1e-4` and response tolerance `1e-9`. Auxiliary response-region calculations provide value suprema and numerical gaps, using LP tolerance `1e-10` and value tolerance `1e-6`. A fixed selection margin does not itself bound defender-value error; the computed gaps are retained in the result records.

The hard budget restricts which complete defense configurations may receive positive probability. It is not an expected-cost budget. The reduced real-world case uses the representative strict-deterrence policy `x2=tau+0.001` where applicable. Its upper-payoff and potential-response values are evaluated under the same CP95 intervals for the response comparison.

Mission lower values and defense choices are conditional on the specified graph, costs, budget, and attacker model. The real-world case uses aggregate end-to-end counts; the multistage propagation experiments use synthetic data. Costs are normalized scenario parameters.

## Citation

Bibliographic information for the accompanying paper is provided in `CITATION.bib`.
