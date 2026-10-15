# Reproducing the CSI500 project

This repository preserves the source files and outputs from the May 2026 course submission. Its main README is a new guide; `original_submission_README.md` and `ReportML.pdf` retain the original project documentation.

## Included evidence

- Thirteen Python source files, copied without algorithm changes.
- Two final portfolio CSVs, each containing 30 distinct stock codes and normalized non-negative weights.
- Detailed results for 21 historical windows at each of six horizon/portfolio-size configurations, plus the original summary CSV.
- The original project report and submission README.

Submission-platform metadata is omitted. The duplicate root-level second-phase portfolio file is omitted because it is identical to `submissions/week2_SUBMIT.csv`.

## Data and environment

Install the dependencies in `requirements.txt` in an isolated Python environment. Those are the original version ranges, not a lockfile for the original environment. On macOS, XGBoost may additionally require an OpenMP runtime; see the original README for the environment used during the project.

The code expects:

```text
data/constituents.csv
data/prices.parquet
data/index.parquet
```

The archive did not contain these files. The original documentation states that the final data refresh ended on May 8, 2026, using `--end 20260510`. The downloader refreshes the constituent list at download time. Fetching data again with a historical end date can therefore change the constituent universe and historical adjusted prices. Exact numerical reproduction needs the original snapshot as well as compatible dependencies.

## Generate the four model score files

The commands below use an explicit historical as-of date to keep the prediction cutoff visible. They require the data files above.

```bash
python my_pipeline.py --as-of 20260508 --horizon 3 --top-k 50 \
  --out submissions/week2_tmp_normal_h3.csv \
  --scores-out results/week2_scores_normal_h3.csv
python my_pipeline.py --as-of 20260508 --horizon 5 --top-k 50 \
  --out submissions/week2_tmp_normal_h5.csv \
  --scores-out results/week2_scores_normal_h5.csv
python my_pipeline_conservative.py --as-of 20260508 --horizon 3 --top-k 50 \
  --out submissions/week2_tmp_cons_h3.csv \
  --scores-out results/week2_scores_cons_h3.csv
python my_pipeline_conservative.py --as-of 20260508 --horizon 5 --top-k 50 \
  --out submissions/week2_tmp_cons_h5.csv \
  --scores-out results/week2_scores_cons_h5.csv

python make_week2_4model_ensemble_h5strong.py --top-k 30 \
  --out submissions/week2_reproduced.csv \
  --debug-out results/week2_reproduced_scores.csv
python validate_submission.py submissions/week2_reproduced.csv
```

The second-phase final ensemble weights are 0.10 normal h3, 0.40 normal h5, 0.10 conservative h3, and 0.40 conservative h5 after score standardization. The first-phase weights are 0.35, 0.15, 0.35, and 0.15 respectively. First-phase commands and filenames are preserved in the original README; that phase requires its own historical data cutoff.

## Inspect or rerun the historical evaluation

The preserved results are:

```text
results/selftest_results_21windows.csv
results/selftest_summary_21windows.csv
```

`python my_backtest.py` writes new runs to `results/selftest_results.csv` and `results/selftest_summary.csv`, leaving the submitted evidence filenames intact. It evaluates the normal XGBoost configuration at horizons of 3 and 5 trading days and top-k values of 30, 50, and 80. It does not backtest the final four-model ensemble.

Portfolio return is the weighted close-to-close return over each window. Excess return subtracts the CSI500 index return over that window. Win rate is the fraction of windows with positive excess return. These values are not annualized, compounded, or net of trading costs.

The saved mean validation rank IC is approximately -0.000244 for h3 and -0.012096 for h5. These rolling averages should be distinguished from the early single-run validation figures described in the report. An exact IC of 0.2925 is not supported by the supplied report or retained CSVs and is not used in the new project summary.

## Checks completed during organization

Python source syntax, original-source integrity, submission CSV structure and weights, and consistency between the detailed and summary historical CSVs were checked. The original validator can check constituent membership only when `data/constituents.csv` is present. No market data was downloaded and no training or backtest was rerun during this organization step.

## Original code provenance

The original README identifies `baseline_xgboost.py`, `features.py`, `download_data.py`, `validate_submission.py`, and `score_submission.py` as supplied course starting materials or utilities. Their contents are preserved. The `my_*` files and ensemble scripts contain the project extensions described in the report. The report also retains its original academic-integrity and AI-assistance statement.
