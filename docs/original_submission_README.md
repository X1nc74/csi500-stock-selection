# CSI500 Stock Selection Competition - README

## 1. Project Overview

This project builds a machine-learning-based stock selection pipeline for the CSI500 Stock Selection Competition. The goal is to construct a long-only portfolio of CSI500 constituents that is expected to outperform the CSI500 index over the next evaluation window.

The main pipeline trains XGBoost regression models on historical price-volume data. Instead of predicting raw future stock returns, the improved model predicts future excess returns relative to the CSI500 index:

```text
target_h = stock_forward_return_h - index_forward_return_h
```

This target is better aligned with the competition scoring metric, which evaluates portfolio excess return over the CSI500 benchmark.

## 2. Main Files

- `my_features.py`: Feature engineering and target construction. This file builds technical features, cross-sectional rank features, and the forward excess-return target.
- `my_pipeline.py`: Main model training and portfolio generation pipeline. It trains the normal XGBoost model, predicts stock scores, constructs a constrained long-only portfolio, and writes a valid submission CSV.
- `my_pipeline_conservative.py`: A more conservative XGBoost pipeline with shallower trees, stronger regularization, lower learning rate, and more conservative sampling. This is used as part of the four-model ensemble.
- `my_backtest.py`: Rolling self-test script. It evaluates historical performance across multiple test windows, horizons, and top-k portfolio sizes.
- `make_ensemble_submission.py`: Builds a simple Week 1 ensemble submission by combining h=3 and h=5 model scores.
- `make_4model_ensemble_submission.py`: Builds the final Phase 1 / Week 1 four-model ensemble submission.
- `make_week2_4model_ensemble_submission.py`: Builds the Week 2 h5-lead ensemble candidate.
- `make_week2_4model_ensemble_h5strong.py`: Builds the final Week 2 h5-strong four-model ensemble submission. This is the script used for the final Phase 2 portfolio.
- `baseline_xgboost.py`: Original course-provided XGBoost baseline. I keep this file for baseline comparison.
- `features.py`: Original course-provided feature engineering file.
- `validate_submission.py`: Official submission validator. I run this before every submission.
- `download_data.py`: Course-provided data download script.
- `score_submission.py`: Course-provided realized-return scoring script.
- `requirements.txt`: Python dependencies.
- `data/`: Data folder containing `constituents.csv`, `prices.parquet`, and `index.parquet`. This folder is not included in the code zip because the data can be reproduced with `download_data.py`.
- `submissions/`: Generated portfolio CSV files.
- `results/`: Generated model scores and self-test result files.

## 3. Environment Setup

Python >= 3.10 is recommended.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS, XGBoost may require OpenMP. If `xgboost` fails to load with a `libomp.dylib` error, install `libomp` with Homebrew:

```bash
brew install libomp
```

The experiments in this project were run on macOS with a Python virtual environment.

## 4. Data

The project uses the data files under `data/`:

```text
data/constituents.csv
data/prices.parquet
data/index.parquet
```

The stock price data contains daily OHLCV data for CSI500 constituents. The index data contains daily CSI500 benchmark prices. The effective universe is the set of stock codes in `data/constituents.csv`.

For Phase 2, I refreshed the data with:

```bash
python download_data.py --update --end 20260510
```

After updating, the latest available trading day in both stock prices and index data was 2026-05-08.

Every generated submission is checked with:

```bash
python validate_submission.py <submission_file>
```

## 5. Method Summary

### 5.1 Features

The model uses technical and cross-sectional features, including:

```text
ret_1d
ret_5d
ret_10d
ret_20d
ret_60d
vol_20d
volume_z_20d
turnover_ma_20d
close_over_ma20
close_over_ma60
rsi_14
ret_5d_rank
ret_20d_rank
vol_20d_rank
```

These features capture momentum/reversal, volatility, liquidity and trading activity, moving-average trend, and cross-sectional relative strength.

### 5.2 Target

The baseline predicts raw future stock returns. My improved pipeline predicts forward excess return relative to the CSI500 index:

```text
target_h = stock_return_h - index_return_h
```

This directly matches the competition’s live evaluation metric:

```text
excess_return = portfolio_return - benchmark_return
```

### 5.3 Models

The main model is an XGBoost regressor trained from scratch.

The normal model uses:

```text
n_estimators = 400
max_depth = 5
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
min_child_weight = 10
reg_lambda = 1.0
tree_method = hist
early_stopping_rounds = 30
random_state = 42
```

The conservative model uses shallower trees and stronger regularization:

```text
n_estimators = 800
max_depth = 3
learning_rate = 0.03
subsample = 0.7
colsample_bytree = 0.7
min_child_weight = 20
reg_lambda = 5.0
tree_method = hist
early_stopping_rounds = 50
random_state = 42
```

### 5.4 Portfolio Construction

The model outputs one predicted score for each stock. The portfolio is constructed as follows:

1. Rank stocks by predicted score.
2. Select the top-k stocks.
3. Assign rank-based weights.
4. Normalize weights to sum to 1.
5. Enforce the 10% maximum weight constraint.
6. Output a CSV with exactly two columns: `stock_code` and `weight`.

Rank-based weights are used instead of raw score weights to reduce sensitivity to model score scale and calibration.

## 6. Phase 1 / Week 1 Submission

For Phase 1, the live evaluation window had three trading days. Therefore, the final Phase 1 ensemble was h=3 dominant.

The final Phase 1 four-model ensemble used:

```text
ensemble_score =
0.35 * normal_h3_z
+ 0.15 * normal_h5_z
+ 0.35 * conservative_h3_z
+ 0.15 * conservative_h5_z
```

This gives 70% total weight to h=3 signals and 30% total weight to h=5 signals.

The final Phase 1 submission was:

```text
submissions/week1_SUBMIT.csv
```

It passed validation with:

```bash
python validate_submission.py submissions/week1_SUBMIT.csv
```

Phase 1 received the full live evaluation score. The realized portfolio return reported by Gradescope was:

```text
0.087653593
```

which is approximately 8.77%.

## 7. Reproduce Phase 1 / Week 1 Submission

Generate the normal h=3 and h=5 score files:

```bash
python my_pipeline.py \
  --horizon 3 \
  --top-k 50 \
  --out submissions/tmp_h3.csv \
  --scores-out results/week1_scores_h3.csv

python my_pipeline.py \
  --horizon 5 \
  --top-k 50 \
  --out submissions/tmp_h5.csv \
  --scores-out results/week1_scores_h5.csv
```

Generate the conservative h=3 and h=5 score files:

```bash
python my_pipeline_conservative.py \
  --horizon 3 \
  --top-k 50 \
  --out submissions/tmp_cons_h3.csv \
  --scores-out results/week1_scores_cons_h3.csv

python my_pipeline_conservative.py \
  --horizon 5 \
  --top-k 50 \
  --out submissions/tmp_cons_h5.csv \
  --scores-out results/week1_scores_cons_h5.csv
```

Build and validate the final Phase 1 submission:

```bash
python make_4model_ensemble_submission.py \
  --top-k 30 \
  --out submissions/week1_SUBMIT.csv \
  --debug-out results/week1_4model_ensemble_scores_top30.csv

python validate_submission.py submissions/week1_SUBMIT.csv
```

## 8. Reproduce Self-Test

Run:

```bash
python my_backtest.py
```

This script performs rolling historical self-tests over multiple dates, horizons, and top-k portfolio sizes.

It outputs:

```text
results/selftest_results.csv
results/selftest_summary.csv
```

The saved 21-window self-test files included in the code zip are:

```text
results/selftest_results_21windows.csv
results/selftest_summary_21windows.csv
```

The self-test compares:

```text
horizon = 3 and 5
top_k = 30, 50, and 80
```

The main reported metrics include validation rank IC, portfolio return, benchmark return, excess return, and win rate.

## 9. Self-Test Results Summary

The 21-window self-test summary is:

```text
horizon  top_k  avg_excess_return  median_excess_return  win_rate  n_windows
3        30     0.010772           0.007389              0.857143  21
5        30     0.009476           0.007384              0.619048  21
3        50     0.007915           0.005222              0.809524  21
5        50     0.006158           0.007591              0.666667  21
3        80     0.005592           0.004848              0.714286  21
5        80     0.003871           0.002368              0.619048  21
```

The h=3, top-30 portfolio had the highest average excess return and a high win rate in historical self-tests. This supported using a concentrated top-30 portfolio for Phase 1.

## 10. Phase 2 / Week 2 Adjustment

Since the Phase 1 portfolio received the full live evaluation score, I kept the same overall pipeline for Phase 2. However, Phase 2 has a five-trading-day live evaluation window, so I shifted the ensemble toward h=5 models.

I tested both a 70% h5-lead ensemble and an 80% h5-strong ensemble. The h5-strong version changed only 1 out of 30 names relative to the 70% h5-lead portfolio, so it was a small adjustment rather than a major strategy change.

The final Phase 2 h5-strong ensemble uses:

```text
ensemble_score =
0.10 * normal_h3_z
+ 0.40 * normal_h5_z
+ 0.10 * conservative_h3_z
+ 0.40 * conservative_h5_z
```

This gives 20% total weight to h=3 signals and 80% total weight to h=5 signals.

The final Phase 2 submission is:

```text
submissions/week2_SUBMIT.csv
```

## 11. Reproduce Phase 2 / Week 2 Submission

Before Phase 2, refresh the data:

```bash
python download_data.py --update --end 20260510
```

Generate the four model score files:

```bash
python my_pipeline.py \
  --horizon 3 \
  --top-k 50 \
  --out submissions/week2_tmp_normal_h3.csv \
  --scores-out results/week2_scores_normal_h3.csv

python my_pipeline.py \
  --horizon 5 \
  --top-k 50 \
  --out submissions/week2_tmp_normal_h5.csv \
  --scores-out results/week2_scores_normal_h5.csv

python my_pipeline_conservative.py \
  --horizon 3 \
  --top-k 50 \
  --out submissions/week2_tmp_cons_h3.csv \
  --scores-out results/week2_scores_cons_h3.csv

python my_pipeline_conservative.py \
  --horizon 5 \
  --top-k 50 \
  --out submissions/week2_tmp_cons_h5.csv \
  --scores-out results/week2_scores_cons_h5.csv
```

Build and validate the final Phase 2 submission:

```bash
python make_week2_4model_ensemble_h5strong.py \
  --top-k 30 \
  --out submissions/week2_SUBMIT.csv \
  --debug-out results/week2_EXPERIMENT_h5strong_scores_top30.csv

python validate_submission.py submissions/week2_SUBMIT.csv
```

Expected result:

```text
OK: submissions/week2_SUBMIT.csv passes all checks
```

## 12. Submission Format

All submitted portfolio CSV files must have exactly two columns:

```text
stock_code,weight
```

No extra columns should be included in the submitted CSV.

The submitted file must satisfy:

```text
stock_code is a zero-padded 6-digit string
all stocks are in data/constituents.csv
at least 30 stocks have positive weight
each weight is between 0 and 0.10
weights sum to 1.0 within tolerance
```

I validate each submission with:

```bash
python validate_submission.py <submission_file>
```

## 13. Important Generated Files

Phase 1 final submission:

```text
submissions/week1_SUBMIT.csv
```

Phase 2 final submission:

```text
submissions/week2_SUBMIT.csv
```

Phase 2 final model script:

```text
make_week2_4model_ensemble_h5strong.py
```

Self-test outputs included in the code zip:

```text
results/selftest_results_21windows.csv
results/selftest_summary_21windows.csv
```

## 14. Random Seeds and Reproducibility

The XGBoost models use:

```text
random_state = 42
```

The portfolio construction is deterministic given the model scores.

The final portfolios can be reproduced by running the commands in this README using the same data snapshot and package versions.

## 15. Academic Integrity and LLM Assistance

I completed this project individually. I used the course-provided baseline code as a starting point.

I used ChatGPT as a coding and brainstorming assistant for debugging, pipeline organization, and report planning. The submitted portfolios were generated by my own trained XGBoost models using the provided market data. I did not use an LLM to directly choose stocks or generate portfolio weights.

No code, features, or portfolios were shared with other students.
