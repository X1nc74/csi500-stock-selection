"""
Create a four-model ensemble portfolio.

Models:
- normal h=3
- normal h=5
- conservative h=3
- conservative h=5

For Week 1, h=3 should dominate because the live evaluation window has 3 trading days.
Conservative models are included because they achieved higher validation IC.

Usage:
python make_4model_ensemble_submission.py \
  --top-k 30 \
  --out submissions/week1_4model_ensemble_top30.csv \
  --debug-out results/week1_4model_ensemble_scores_top30.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MIN_STOCKS = 30
MAX_WEIGHT = 0.10


def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0 or pd.isna(std):
        return s * 0
    return (s - s.mean()) / std


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

    assert abs(w.sum() - 1.0) < 1e-6
    assert (w <= MAX_WEIGHT + 1e-9).all()
    assert (w >= 0).all()
    assert (w > 0).sum() >= MIN_STOCKS

    return w


def load_score(path: str, score_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_code": str})
    return df[["stock_code", "score"]].rename(columns={"score": score_name})


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--normal-h3", default="results/week1_scores_h3.csv")
    parser.add_argument("--normal-h5", default="results/week1_scores_h5.csv")
    parser.add_argument("--cons-h3", default="results/week1_scores_cons_h3.csv")
    parser.add_argument("--cons-h5", default="results/week1_scores_cons_h5.csv")

    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--out", default="submissions/week1_4model_ensemble_top30.csv")
    parser.add_argument("--debug-out", default=None)

    args = parser.parse_args()

    normal_h3 = load_score(args.normal_h3, "normal_h3")
    normal_h5 = load_score(args.normal_h5, "normal_h5")
    cons_h3 = load_score(args.cons_h3, "cons_h3")
    cons_h5 = load_score(args.cons_h5, "cons_h5")

    df = normal_h3.merge(normal_h5, on="stock_code", how="inner")
    df = df.merge(cons_h3, on="stock_code", how="inner")
    df = df.merge(cons_h5, on="stock_code", how="inner")

    for col in ["normal_h3", "normal_h5", "cons_h3", "cons_h5"]:
        df[col + "_z"] = zscore(df[col])

    df["ensemble_score"] = (
        0.35 * df["normal_h3_z"]
        + 0.15 * df["normal_h5_z"]
        + 0.35 * df["cons_h3_z"]
        + 0.15 * df["cons_h5_z"]
    )

    scores = df.set_index("stock_code")["ensemble_score"]
    weights = build_portfolio(scores, top_k=args.top_k)

    out = pd.DataFrame(
        {
            "stock_code": weights.index.astype(str).str.zfill(6),
            "weight": weights.values,
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f">> Wrote {len(out)} names to {out_path}")
    print(
        f"   weight summary: min={out['weight'].min():.4f} "
        f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}"
    )

    if args.debug_out is not None:
        debug_path = Path(args.debug_out)
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        df.sort_values("ensemble_score", ascending=False).to_csv(debug_path, index=False)
        print(f">> Wrote debug scores to {debug_path}")


if __name__ == "__main__":
    main()