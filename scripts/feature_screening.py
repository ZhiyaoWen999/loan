#!/usr/bin/env python3
"""Automatic feature screening (cheap filters before modeling).

This script computes per-feature signals to help you prune the feature space:
- IV (information value) after coarse binning
- univariate AUC (numeric features directly; categorical via category bad-rate)
- low variance / nearly-constant checks
- high correlation pairs (numeric-only) with a suggested drop candidate

Outputs (CSV):
- iv_auc_summary.csv
- corr_pairs.csv
- feature_screening_flags.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    add_log_level_arg,
    configure_logging,
    ensure_dir,
    log_saved_paths,
    record_manifest,
    resolve_outdir,
    resolve_run_dir,
)


def auc_from_scores(y: pd.Series, score: pd.Series) -> float:
    """Rank-based AUC (Mann-Whitney) that tolerates missing values."""
    y = pd.to_numeric(y, errors="coerce")
    score = pd.to_numeric(score, errors="coerce")
    mask = y.notna() & score.notna()
    y = y[mask].astype(int)
    score = score[mask]
    if y.nunique() < 2:
        return float("nan")
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = score.rank(method="average")
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def bin_series(series: pd.Series, bins: int, top_n: int) -> tuple[pd.Series, str]:
    """Coarse binning used for IV computation.

    - Numeric: quantile bins (qcut), fallback to cut() or categorical if unique small
    - Categorical: keep top-N and collapse the rest into OTHER
    """
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce")
        if s.nunique(dropna=True) <= bins:
            binned = s.astype("string")
            binned = binned.fillna("MISSING")
            return binned, "numeric_as_categorical"
        try:
            binned = pd.qcut(s, q=bins, duplicates="drop")
        except ValueError:
            binned = pd.cut(s, bins=bins)
        binned = binned.cat.add_categories(["MISSING"]).fillna("MISSING")
        return binned.astype(str), "numeric_binned"

    # Categorical: keep top-N, collapse others.
    s = s.astype("string").fillna("MISSING")
    top = s.value_counts().head(top_n).index
    binned = s.where(s.isin(top), "OTHER")
    return binned, "categorical"


def calc_iv(binned: pd.Series, y: pd.Series, eps: float = 0.5) -> float:
    """Classic IV with Laplace smoothing to avoid div-by-zero."""
    df = pd.DataFrame({"bin": binned, "y": y})
    df = df[df["y"].notna()].copy()
    df["bin"] = df["bin"].astype("string").fillna("MISSING")

    agg = df.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    agg = agg.rename(columns={"sum": "bad"})
    agg["good"] = agg["count"] - agg["bad"]

    bad_dist = (agg["bad"] + eps) / (agg["bad"].sum() + eps * len(agg))
    good_dist = (agg["good"] + eps) / (agg["good"].sum() + eps * len(agg))
    iv = ((bad_dist - good_dist) * np.log(bad_dist / good_dist)).sum()
    return float(iv)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Feature screening: IV, univariate AUC, low variance, correlation.")
    parser.add_argument("--train", type=str, default="data/features_train_90d.csv", help="Train CSV")
    parser.add_argument("--label", type=str, default="y_dpd30_ever", help="Label column")
    parser.add_argument("--outdir", type=str, default="data/eda", help="Output directory")
    parser.add_argument("--bins", type=int, default=10, help="Bins for numeric IV")
    parser.add_argument("--top-n", type=int, default=10, help="Top categories to keep for IV")
    parser.add_argument("--iv-threshold", type=float, default=0.02, help="Low IV threshold")
    parser.add_argument("--auc-threshold", type=float, default=0.52, help="Low AUC threshold (abs direction)")
    parser.add_argument("--top-ratio-threshold", type=float, default=0.95, help="High mode ratio threshold")
    parser.add_argument("--low-unique-threshold", type=int, default=3, help="Low unique count threshold")
    parser.add_argument("--skew-threshold", type=float, default=5.0, help="Absolute skewness threshold")
    parser.add_argument("--corr-threshold", type=float, default=0.9, help="Correlation threshold")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional run directory (e.g., data/run_YYYYMMDD or 'auto'). If set, outputs go under it.",
    )
    add_log_level_arg(parser)
    args = parser.parse_args()

    logger = configure_logging(args.log_level)
    run_dir = resolve_run_dir(args.run_dir)
    df = pd.read_csv(args.train)
    if args.label not in df.columns:
        raise SystemExit(f"Missing label column: {args.label}")

    outdir = resolve_outdir(args.outdir, run_dir, name="eda")

    y = df[args.label]
    features = [c for c in df.columns if c != args.label]

    summary_rows = []
    for col in features:
        series = df[col]
        missing_rate = float(series.isna().mean())
        nunique = int(series.nunique(dropna=True))
        top_ratio = float(series.value_counts(dropna=False).iloc[0] / len(series)) if len(series) else float("nan")

        if pd.api.types.is_numeric_dtype(series):
            skew = float(pd.to_numeric(series, errors="coerce").skew())
        else:
            skew = float("nan")

        binned, bin_type = bin_series(series, bins=args.bins, top_n=args.top_n)
        iv = calc_iv(binned, y)

        if pd.api.types.is_numeric_dtype(series):
            auc = auc_from_scores(y, series)
        else:
            # Use category-level bad rate as score.
            tmp = pd.DataFrame({"x": series.astype("string").fillna("MISSING"), "y": y})
            br = tmp.groupby("x", observed=True)["y"].mean()
            auc = auc_from_scores(y, tmp["x"].map(br))

        auc_adj = float(max(auc, 1 - auc)) if not np.isnan(auc) else float("nan")

        summary_rows.append(
            {
                "feature": col,
                "dtype": str(series.dtype),
                "nunique": nunique,
                "missing_rate": missing_rate,
                "top_ratio": top_ratio,
                "skew": skew,
                "bin_type": bin_type,
                "iv": iv,
                "auc": auc,
                "auc_adj": auc_adj,
                "low_iv": iv < args.iv_threshold,
                "low_auc": auc_adj < args.auc_threshold if not np.isnan(auc_adj) else False,
                "low_unique": nunique <= args.low_unique_threshold,
                "high_top_ratio": top_ratio >= args.top_ratio_threshold if not np.isnan(top_ratio) else False,
                "high_skew": abs(skew) >= args.skew_threshold if not np.isnan(skew) else False,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["low_iv", "low_auc", "iv"], ascending=[False, False, True])
    iv_path = outdir / "iv_auc_summary.csv"
    summary.to_csv(iv_path, index=False)

    # Correlation screening for numeric columns.
    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    numeric_df = df[numeric_cols].copy()
    # Drop constant columns to avoid NaN correlations.
    constant_cols = [c for c in numeric_cols if numeric_df[c].nunique(dropna=True) <= 1]
    numeric_df = numeric_df.drop(columns=constant_cols, errors="ignore")

    corr_pairs = []
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                val = float(corr.loc[c1, c2])
                if np.isnan(val) or abs(val) < args.corr_threshold:
                    continue
                s1 = summary.loc[summary["feature"] == c1, "auc_adj"]
                s2 = summary.loc[summary["feature"] == c2, "auc_adj"]
                score1 = float(s1.iloc[0]) if len(s1) else float("nan")
                score2 = float(s2.iloc[0]) if len(s2) else float("nan")
                # Prefer keeping the feature with higher univariate AUC.
                if np.isnan(score1) or np.isnan(score2):
                    drop = c2 if abs(val) >= abs(val) else c1
                else:
                    drop = c1 if score1 < score2 else c2
                corr_pairs.append(
                    {
                        "feature_1": c1,
                        "feature_2": c2,
                        "corr": val,
                        "suggest_drop": drop,
                    }
                )

    corr_df = pd.DataFrame(corr_pairs).sort_values("corr", ascending=False)
    corr_path = outdir / "corr_pairs.csv"
    corr_df.to_csv(corr_path, index=False)

    # Merge flags into a single suggestion table.
    summary_flags = summary.copy()
    summary_flags["high_corr_drop"] = summary_flags["feature"].isin(corr_df["suggest_drop"])
    summary_flags["drop_candidate"] = (
        summary_flags["low_iv"]
        | summary_flags["low_auc"]
        | summary_flags["low_unique"]
        | summary_flags["high_top_ratio"]
        | summary_flags["high_corr_drop"]
    )
    flags_path = outdir / "feature_screening_flags.csv"
    summary_flags.to_csv(flags_path, index=False)

    log_saved_paths(logger, iv_path, corr_path, flags_path)
    record_manifest(
        run_dir,
        step="feature_screening",
        outputs=[iv_path, corr_path, flags_path],
        note=f"iv_thr={args.iv_threshold}, auc_thr={args.auc_threshold}, corr_thr={args.corr_threshold}",
    )


if __name__ == "__main__":
    main()
