"""
Create ensemble portfolio from h=3 and h=5 model scores.

The idea:
- h=3 matches the first live evaluation window.
- h=5 had slightly higher validation IC.
- A weighted ensemble may be more stable than either single horizon.

Usage:
python make_ensemble_submission.py --top-k 30 --w3 0.7 --out submissions/week1_ensemble_h3h5_top30.csv
python validate_submission.py submissions/week1_ensemble_h3h5_top30.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MIN_STOCKS = 30
MAX_WEIGHT = 0.10


def build_portfolio(scores: pd.Series, top_k: int) -> pd.Series:
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")

    chosen = scores.sort_values(ascending=False).head(top_k).copy()

    # rank-based weights
    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    # cap and redistribute
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


def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0 or pd.isna(std):
        return s * 0
    return (s - s.mean()) / std


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores-h3", default="results/week1_scores_h3.csv")
    p.add_argument("--scores-h5", default="results/week1_scores_h5.csv")
    p.add_argument("--w3", type=float, default=0.7, help="weight on h=3 score")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--out", default="submissions/week1_ensemble_h3h5_top30.csv")
    p.add_argument("--debug-out", default=None)
    args = p.parse_args()

    h3 = pd.read_csv(args.scores_h3, dtype={"stock_code": str})
    h5 = pd.read_csv(args.scores_h5, dtype={"stock_code": str})

    h3 = h3[["stock_code", "score"]].rename(columns={"score": "score_h3"})
    h5 = h5[["stock_code", "score"]].rename(columns={"score": "score_h5"})

    df = h3.merge(h5, on="stock_code", how="inner")

    # Normalize each model's score scale before combining.
    df["score_h3_z"] = zscore(df["score_h3"])
    df["score_h5_z"] = zscore(df["score_h5"])

    w3 = args.w3
    w5 = 1.0 - w3
    df["ensemble_score"] = w3 * df["score_h3_z"] + w5 * df["score_h5_z"]

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
        debug = df.sort_values("ensemble_score", ascending=False)
        debug_path = Path(args.debug_out)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug.to_csv(debug_path, index=False)
        print(f">> Wrote debug scores to {debug_path}")


if __name__ == "__main__":
    main()