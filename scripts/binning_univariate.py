#!/usr/bin/env python3
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


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def bin_numeric(series: pd.Series, bins: int) -> tuple[pd.Series, str, bool]:
    s = pd.to_numeric(series, errors="coerce")
    if s.nunique(dropna=True) <= bins:
        binned = s.astype("string").fillna("MISSING")
        return binned, "numeric_as_categorical", False
    try:
        binned = pd.qcut(s, q=bins, duplicates="drop")
    except ValueError:
        binned = pd.cut(s, bins=bins)
    binned = binned.cat.add_categories(["MISSING"]).fillna("MISSING")
    return binned, "numeric_binned", True


def bin_categorical(series: pd.Series, top_n: int) -> tuple[pd.Series, str, bool]:
    s = series.astype("string").fillna("MISSING")
    top = s.value_counts().head(top_n).index
    return s.where(s.isin(top), "OTHER"), "categorical", False


def bad_rate_table(df: pd.DataFrame, feature: str, label: str, ordered: bool) -> pd.DataFrame:
    work = df[[feature, label]].copy()
    work["bad"] = pd.to_numeric(work[label], errors="coerce")
    work = work.dropna(subset=["bad"])
    agg = work.groupby(feature, as_index=False, sort=False, observed=True).agg(
        total=("bad", "size"),
        bad_cnt=("bad", "sum"),
        bad_rate=("bad", "mean"),
    )
    agg["bad_rate"] = agg["bad_rate"].astype(float)
    if ordered:
        if isinstance(work[feature].dtype, pd.CategoricalDtype):
            agg = agg.sort_values(feature)
        return agg
    return agg.sort_values("bad_rate", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Univariate binning and bad-rate tables.")
    parser.add_argument("--train", type=str, default="data/features_train_90d.csv", help="Train CSV")
    parser.add_argument("--label", type=str, default="y_dpd30_ever", help="Label column")
    parser.add_argument("--bins", type=int, default=10, help="Quantile bins for numeric")
    parser.add_argument("--top-n", type=int, default=10, help="Top N categories to keep")
    parser.add_argument("--outdir", type=str, default="data/binning", help="Output directory")
    parser.add_argument("--max-features", type=int, default=None, help="Optional limit of features to process")
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

    outdir = resolve_outdir(args.outdir, run_dir, name="binning")

    features = [c for c in df.columns if c != args.label]
    if args.max_features:
        features = features[: args.max_features]

    summary_rows = []

    for feature in features:
        series = df[feature]
        if is_numeric(series):
            binned, bin_type, ordered = bin_numeric(series, bins=args.bins)
        else:
            binned, bin_type, ordered = bin_categorical(series, top_n=args.top_n)

        tmp = df[[args.label]].copy()
        tmp[feature] = binned
        table = bad_rate_table(tmp, feature, args.label, ordered=ordered)
        table.to_csv(outdir / f"binning_{feature}.csv", index=False)

        monotonic = False
        if ordered and isinstance(tmp[feature].dtype, pd.CategoricalDtype):
            idx = table[feature]
            if idx.dtype.name == "category":
                ordered_rates = (
                    table.set_index(feature)["bad_rate"]
                    .reindex(idx.cat.categories)
                    .dropna()
                )
                ordered_rates = ordered_rates[ordered_rates.index != "MISSING"]
                if len(ordered_rates) >= 2:
                    diffs = np.diff(ordered_rates.values)
                    monotonic = (np.all(diffs <= 0)) or (np.all(diffs >= 0))
                else:
                    monotonic = True

        summary_rows.append(
            {
                "feature": feature,
                "bins": len(table),
                "min_bad_rate": float(table["bad_rate"].min()),
                "max_bad_rate": float(table["bad_rate"].max()),
                "monotonic": monotonic,
                "bin_type": bin_type,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["monotonic", "max_bad_rate"], ascending=[True, False]
    )
    summary_path = outdir / "binning_summary.csv"
    summary.to_csv(summary_path, index=False)

    logger.info("Saved per-feature binning tables to: %s", outdir)
    log_saved_paths(logger, summary_path)
    record_manifest(run_dir, step="binning_univariate", outputs=[summary_path], note=f"bins={args.bins}")


if __name__ == "__main__":
    main()
