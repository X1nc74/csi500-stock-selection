"""
Rolling self-test / backtest for the CSI500 stock-selection project.

This script evaluates the improved excess-return model on historical windows.
It is designed for the report's Self-test section.

It reports:
- validation rank IC
- realized portfolio return
- benchmark return
- excess return
- win rate across windows
- comparison across horizon and top-k choices

Usage:
python my_backtest.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from my_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    training_frame,
    prediction_frame,
)


DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

VAL_DAYS = 10
EMBARGO_DAYS = 5
MIN_STOCKS = 30
MAX_WEIGHT = 0.10


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
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
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def build_portfolio(scores: pd.Series, top_k: int) -> pd.Series:
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

    return w


def realized_stock_returns(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int,
) -> pd.Series:
    """Return close(t+h)/close(t)-1 for each stock."""
    dates = np.sort(prices["date"].unique())
    as_of_idx = int(np.searchsorted(dates, np.datetime64(as_of)))

    if as_of_idx + horizon >= len(dates):
        raise ValueError("Not enough future dates for this test window.")

    exit_date = pd.Timestamp(dates[as_of_idx + horizon])

    entry = prices[prices["date"] == as_of][["stock_code", "close"]].rename(
        columns={"close": "entry_close"}
    )
    exit_ = prices[prices["date"] == exit_date][["stock_code", "close"]].rename(
        columns={"close": "exit_close"}
    )

    merged = entry.merge(exit_, on="stock_code", how="inner")
    merged["ret"] = merged["exit_close"] / merged["entry_close"] - 1.0

    return merged.set_index("stock_code")["ret"]


def realized_index_return(
    index: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int,
) -> float:
    dates = np.sort(index["date"].unique())
    as_of_idx = int(np.searchsorted(dates, np.datetime64(as_of)))

    if as_of_idx + horizon >= len(dates):
        raise ValueError("Not enough future index dates for this test window.")

    exit_date = pd.Timestamp(dates[as_of_idx + horizon])

    entry_close = float(index.loc[index["date"] == as_of, "close"].iloc[0])
    exit_close = float(index.loc[index["date"] == exit_date, "close"].iloc[0])

    return exit_close / entry_close - 1.0


def run_one_window(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    index: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int,
    top_k: int,
) -> dict:
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))

    train_cutoff_idx = max(0, as_of_idx - horizon)
    train_cutoff = pd.Timestamp(trading_dates[train_cutoff_idx])

    train_pool = training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(train_pool["date"].unique())

    embargo_days = max(EMBARGO_DAYS, horizon)

    if len(all_dates) < VAL_DAYS + embargo_days + 20:
        raise RuntimeError(f"Not enough training data for as_of={as_of.date()}")

    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + embargo_days + 1)])

    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]

    model = train_model(train_df, val_df)

    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    ic = rank_ic(
        val_df[TARGET_COLUMN].to_numpy(),
        val_pred,
        val_df["date"].to_numpy(),
    )

    pred_df = prediction_frame(panel, as_of=as_of)
    pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))

    scores = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=top_k)

    stock_rets = realized_stock_returns(prices, as_of=as_of, horizon=horizon)
    stock_rets = stock_rets.reindex(weights.index).fillna(0.0)

    portfolio_ret = float((weights * stock_rets).sum())
    benchmark_ret = realized_index_return(index, as_of=as_of, horizon=horizon)
    excess_ret = portfolio_ret - benchmark_ret

    return {
        "as_of": as_of.date().isoformat(),
        "horizon": horizon,
        "top_k": top_k,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "val_rank_ic": ic,
        "portfolio_return": portfolio_ret,
        "benchmark_return": benchmark_ret,
        "excess_return": excess_ret,
        "n_names": int((weights > 0).sum()),
        "max_weight": float(weights.max()),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print(">> Loading data")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    index = pd.read_parquet(DATA_DIR / "index.parquet")

    prices["date"] = pd.to_datetime(prices["date"])
    index["date"] = pd.to_datetime(index["date"])

    print(
        f"   prices: {len(prices):,} rows, "
        f"{prices['stock_code'].nunique()} stocks, "
        f"{prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    # Test windows before the final available date.
    # These are historical self-test dates, not live evaluation dates.
    candidate_as_of_dates = [
    "2025-08-01",
    "2025-08-15",
    "2025-08-29",
    "2025-09-12",
    "2025-09-26",
    "2025-10-10",
    "2025-10-24",
    "2025-11-07",
    "2025-11-21",
    "2025-12-05",
    "2025-12-19",
    "2026-01-02",
    "2026-01-16",
    "2026-01-30",
    "2026-02-13",
    "2026-02-27",
    "2026-03-06",
    "2026-03-13",
    "2026-03-20",
    "2026-03-27",
    "2026-04-03",
    "2026-04-10",
]

    horizons = [3, 5]
    top_ks = [30, 50, 80]

    all_results = []

    for horizon in horizons:
        print(f"\n>> Building feature panel for horizon={horizon}")
        panel = build_features(prices, index=index, horizon=horizon)

        available_dates = set(pd.to_datetime(panel["date"]).dt.date)

        for as_of_str in candidate_as_of_dates:
            as_of = pd.Timestamp(as_of_str)

            if as_of.date() not in available_dates:
                print(f"   skip {as_of_str}: not a trading date")
                continue

            # Make sure future dates exist.
            price_dates = np.sort(prices["date"].unique())
            as_of_idx = int(np.searchsorted(price_dates, np.datetime64(as_of)))
            if as_of_idx + horizon >= len(price_dates):
                print(f"   skip {as_of_str}: not enough future days")
                continue

            for top_k in top_ks:
                print(f"   testing as_of={as_of_str}, horizon={horizon}, top_k={top_k}")
                result = run_one_window(
                    panel=panel,
                    prices=prices,
                    index=index,
                    as_of=as_of,
                    horizon=horizon,
                    top_k=top_k,
                )
                all_results.append(result)

    results = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "selftest_results.csv"
    results.to_csv(out_path, index=False)

    print(f"\n>> Wrote detailed results to {out_path}")

    summary = (
        results.groupby(["horizon", "top_k"])
        .agg(
            avg_val_rank_ic=("val_rank_ic", "mean"),
            avg_excess_return=("excess_return", "mean"),
            median_excess_return=("excess_return", "median"),
            win_rate=("excess_return", lambda x: float((x > 0).mean())),
            avg_portfolio_return=("portfolio_return", "mean"),
            avg_benchmark_return=("benchmark_return", "mean"),
            n_windows=("excess_return", "count"),
        )
        .reset_index()
        .sort_values(["avg_excess_return", "avg_val_rank_ic"], ascending=False)
    )

    summary_path = RESULTS_DIR / "selftest_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f">> Wrote summary to {summary_path}")

    print("\n=== Self-test summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()