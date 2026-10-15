"""
Feature engineering for the CSI500 stock-selection model.

This version extends the baseline by using a competition-aligned target:
h-trading-day forward excess return relative to the CSI500 index,

    target = stock_forward_return_h - index_forward_return_h

This matches the live scoring metric more closely than predicting raw stock return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "vol_20d", "volume_z_20d", "turnover_ma_20d",
    "close_over_ma20", "close_over_ma60", "rsi_14",
    "ret_5d_rank", "ret_20d_rank", "vol_20d_rank",
]

TARGET_COLUMN = "target_excess"
FORWARD_HORIZON = 5


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that only depend on a single stock's time series."""
    df = df.sort_values("date").copy()
    close = df["close"]

    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)

    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vol_mean) / vol_std

    if "turnover" in df.columns:
        df["turnover_ma_20d"] = df["turnover"].astype(float).rolling(20).mean()
    else:
        df["turnover_ma_20d"] = np.nan

    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = up / down
    df["rsi_14"] = 100 - 100 / (1 + rs)

    return df


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional rank of selected features, values in [0, 1]."""
    for base in ["ret_5d", "ret_20d", "vol_20d"]:
        panel[f"{base}_rank"] = (
            panel.groupby("date")[base].rank(method="average", pct=True)
        )
    return panel


def _add_excess_return_target(
    panel: pd.DataFrame,
    index: pd.DataFrame,
    horizon: int = FORWARD_HORIZON,
) -> pd.DataFrame:
    """Add h-day forward excess return target.

    target = stock_forward_return_h - index_forward_return_h
    """
    panel = panel.sort_values(["stock_code", "date"]).copy()

    panel["stock_fwd_ret"] = (
        panel.groupby("stock_code")["close"].shift(-horizon) / panel["close"] - 1.0
    )

    idx = index.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")

    if "close" not in idx.columns:
        raise ValueError("index data must contain a close column")

    idx["index_fwd_ret"] = idx["close"].shift(-horizon) / idx["close"] - 1.0
    idx = idx[["date", "index_fwd_ret"]]

    panel = panel.merge(idx, on="date", how="left")
    panel[TARGET_COLUMN] = panel["stock_fwd_ret"] - panel["index_fwd_ret"]

    return panel


def build_features(
    prices: pd.DataFrame,
    index: pd.DataFrame | None = None,
    horizon: int = FORWARD_HORIZON,
) -> pd.DataFrame:
    """Build a (date, stock_code) panel of features + target."""
    required = {"date", "stock_code", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    panel = (
        prices.groupby("stock_code", group_keys=False)
        .apply(_per_stock_features)
        .reset_index(drop=True)
    )

    panel = _cross_sectional_ranks(panel)

    if index is not None:
        panel = _add_excess_return_target(panel, index=index, horizon=horizon)

    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    """Rows usable for supervised training: all features present and target present."""
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]

    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Rows for a single prediction date. Defaults to the latest date."""
    if as_of is None:
        as_of = panel["date"].max()

    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()

    return df