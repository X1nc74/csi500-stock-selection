"""
Improved XGBoost pipeline for the CSI500 stock-selection competition.

Main improvement over the provided baseline:
- The model predicts h-day forward excess return relative to CSI500 instead of
  raw stock return.
- The horizon can be changed with --horizon, so we can use h=3 for the first
  live evaluation window and h=5 for the second live evaluation window.
- The script can save stock-level scores for analysis and report writing.

Usage
-----
python my_pipeline.py --horizon 5 --out submissions/week1_excess_h5.csv
python my_pipeline.py --horizon 3 --out submissions/week1_excess_h3.csv --scores-out results/week1_scores_h3.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from my_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FORWARD_HORIZON,
    build_features,
    training_frame,
    prediction_frame,
)


DATA_DIR = Path(__file__).parent / "data"

VAL_DAYS = 10
EMBARGO_DAYS = 5
MIN_STOCKS = 30
MAX_WEIGHT = 0.10
DEFAULT_TOP_K = 50


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> xgb.XGBRegressor:
    """Train a more conservative XGBoost regression model."""
    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=20,
        reg_lambda=5.0,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """Daily cross-sectional Spearman correlation, averaged over dates."""
    ics = []

    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue

        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)

    return float(np.mean(ics)) if ics else float("nan")


def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    """Top-K names with rank-based weights and 10% cap."""
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")

    chosen = scores.sort_values(ascending=False).head(top_k).copy()

    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break

        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT

        free = ~over
        if not free.any():
            break

        w[free] += excess * w[free] / w[free].sum()

    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w <= MAX_WEIGHT + 1e-9).all(), "max weight cap violated"
    assert (w >= 0).all(), "negative weight found"
    assert (w > 0).sum() >= MIN_STOCKS, "too few names"

    return w


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument(
        "--as-of",
        default=None,
        help="YYYYMMDD; defaults to latest date in data",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=FORWARD_HORIZON,
        help="forward return horizon in trading days",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument(
        "--scores-out",
        default=None,
        help="optional path to save stock-level scores for analysis",
    )

    args = parser.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])

    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(f">> Loading {args.index}")
    index = pd.read_parquet(args.index)
    index["date"] = pd.to_datetime(index["date"])

    print(
        f"   index rows: {len(index):,}, "
        f"dates {index['date'].min().date()} to {index['date'].max().date()}"
    )

    print(f">> Building features with horizon={args.horizon}")
    panel = build_features(prices, index=index, horizon=args.horizon)

    # Bound training data so --as-of does not leak future rows.
    # The target at date t uses future prices up to t+horizon, so training dates
    # must be capped at as_of - horizon trading days.
    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()

    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - args.horizon)

    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)

    all_dates = np.sort(train_pool["date"].unique())
    embargo_days = max(EMBARGO_DAYS, args.horizon)

    if len(all_dates) < VAL_DAYS + embargo_days + 20:
        raise RuntimeError("Not enough dates to train; download more history.")

    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + embargo_days + 1)])

    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]

    print(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    print(f"   embargo: {embargo_days} trading days")
    print(f"   val:   {len(val_df):,} rows from {val_start.date()}")

    print(">> Training XGBoost")
    model = train_model(train_df, val_df)

    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    ic = rank_ic(
        val_df[TARGET_COLUMN].to_numpy(),
        val_pred,
        val_df["date"].to_numpy(),
    )
    print(f"   validation rank IC: {ic:.4f}")

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of)

    if pred_df.empty:
        raise RuntimeError(f"No rows available for as_of={args.as_of}. Check data.")

    pred_date = pred_df["date"].iloc[0]
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")

    pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))

    if args.scores_out is not None:
        scores_path = Path(args.scores_out)
        scores_path.parent.mkdir(parents=True, exist_ok=True)

        score_cols = ["date", "stock_code", "score"] + FEATURE_COLUMNS
        pred_df[score_cols].sort_values("score", ascending=False).to_csv(
            scores_path,
            index=False,
        )
        print(f">> Wrote scores to {scores_path}")

    scores = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame(
        {
            "stock_code": weights.index.astype(str).str.zfill(6),
            "weight": weights.values,
        }
    )

    out.to_csv(out_path, index=False)

    print(f">> Wrote {len(out)} names to {out_path}")
    print(
        f"   weight summary: min={out['weight'].min():.4f} "
        f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}"
    )


if __name__ == "__main__":
    main()