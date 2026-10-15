# CSI500 Stock Selection with XGBoost

A machine learning course project by Xinpeng (Stefan) Cheng. The project ranks CSI500 stocks using price-volume features and predicted forward excess returns, then constructs a constrained long-only portfolio. It combines normal and conservative XGBoost models across three- and five-trading-day horizons.

The submitted report describes an effective universe of 499 stocks. This repository contains the submitted source code, historical result CSVs, portfolio submissions, and [project report](docs/ReportML.pdf).

## Approach

- Build momentum, reversal, volatility, liquidity, trend, and daily cross-sectional rank features.
- Predict stock returns relative to the CSI500 benchmark rather than raw returns.
- Use chronological training and validation splits, horizon-aware label cutoffs, and an embargo between training and validation.
- Combine four signals: normal and conservative XGBoost models at horizons of 3 and 5 trading days.
- Select the top 30 stocks, assign rank-based weights, and enforce a 10% single-stock weight cap.

## Saved historical results

The included self-test files cover 21 historical as-of dates from August 2025 to April 2026. Each configuration was evaluated on those dates. These are per-window historical results used during project development, rather than annualized returns or an independent evaluation of the final four-model ensemble.

| Horizon in trading days | Stocks | Mean excess return per window | Median excess return | Windows beating benchmark | Mean validation rank IC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 30 | 1.0772% | 0.7389% | 85.71% | -0.000244 |
| 5 | 30 | 0.9476% | 0.7384% | 61.90% | -0.012096 |
| 3 | 50 | 0.7915% | 0.5222% | 80.95% | -0.000244 |
| 5 | 50 | 0.6158% | 0.7591% | 66.67% | -0.012096 |
| 3 | 80 | 0.5592% | 0.4848% | 71.43% | -0.000244 |
| 5 | 80 | 0.3871% | 0.2368% | 61.90% | -0.012096 |

Source: [saved summary](results/selftest_summary_21windows.csv) and [window-level results](results/selftest_results_21windows.csv). Excess return is portfolio return minus benchmark return over the same window. The saved backtest uses the normal XGBoost configuration; the final submissions use four-model ensembles. The weak average validation rank IC is included alongside the portfolio metrics to show the variability of the signal.

The report records an 8.77% portfolio return for the first three-trading-day competition evaluation window. That figure is a report-stated competition result, separate from the rolling self-test table; the original grading record is not included here.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `my_features.py` | Project feature engineering and forward excess-return labels |
| `my_pipeline.py` | Normal XGBoost training and portfolio generation |
| `my_pipeline_conservative.py` | More regularized XGBoost configuration |
| `my_backtest.py` | Rolling historical evaluation across horizons and portfolio sizes |
| `make_4model_ensemble_submission.py` | Final first-phase ensemble |
| `make_week2_4model_ensemble_h5strong.py` | Final second-phase ensemble |
| `make_ensemble_submission.py` | Earlier two-signal ensemble |
| `make_week2_4model_ensemble_submission.py` | Alternative second-phase weighting |
| `results/` | Original detailed and summary historical result CSVs |
| `submissions/` | Original first- and second-phase portfolio submissions |
| `docs/ReportML.pdf` | Original eight-page project report |
| `docs/original_submission_README.md` | Unedited submission documentation |
| `docs/REPRODUCIBILITY.md` | Data requirements, commands, and evaluation limits |

The course-provided starting files are `baseline_xgboost.py`, `features.py`, `download_data.py`, `validate_submission.py`, and `score_submission.py`. They are retained with their original code and attribution.

## Setup and usage

Run commands from the repository root. The original project recommends Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Training requires local `data/constituents.csv`, `data/prices.parquet`, and `data/index.parquet`. These data files were not included in the submitted archive. The downloader fetches constituents as of the download date, so a fresh download does not recreate the original constituent snapshot exactly.

```bash
# Download historical data. This makes external data requests.
python download_data.py --start 20250101 --end 20260510

# Train a five-day model and write a portfolio and stock-level scores.
python my_pipeline.py --as-of 20260508 --horizon 5 --top-k 30 \
  --out submissions/example_h5.csv --scores-out results/example_h5_scores.csv

# Validate the generated portfolio against the local constituent list.
python validate_submission.py submissions/example_h5.csv

# Run the rolling historical self-test.
python my_backtest.py
```

For all four-model ensemble commands and the two competition phases, see the [reproduction guide](docs/REPRODUCIBILITY.md) and [original submission README](docs/original_submission_README.md).

## Evaluation and reproducibility limits

- Historical portfolio returns are gross of transaction costs and slippage; the supplied backtest does not simulate trade execution.
- The same historical configurations informed model and portfolio choices. The saved self-tests are not a new untouched test set.
- Constituents and forward-adjusted prices can change when re-downloaded. The original market-data snapshot and an exact dependency lockfile are unavailable.
- The retained scripts and CSVs have been checked without re-downloading data or retraining the models. Reproduction requires the missing market data and a compatible environment.

## Attribution

This was an individual course project built from a supplied baseline. The project work extends the prediction target, features, model configurations, ensemble construction, rolling self-tests, and portfolio generation. The original report documents the use of ChatGPT for debugging, pipeline organization, and report planning.
