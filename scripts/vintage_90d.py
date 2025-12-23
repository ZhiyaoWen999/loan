#!/usr/bin/env python3
"""Point-in-time (PIT) vintage summary using a fixed observation window.

This script answers: "For each disbursement cohort (e.g., month), what is the
90D bad rate?" where bad is defined by a label column (default `y_dpd30_ever`).

It is useful to:
- validate label maturity / stability across cohorts
- compare segments (channel/product/score decile) at a fixed window

Output is a cohort table (optionally segmented) with loan_cnt/bad_cnt/bad_rate.
"""

import argparse
from pathlib import Path

import pandas as pd

from common import (
    add_log_level_arg,
    add_risk_score_decile,
    configure_logging,
    ensure_dir,
    log_saved_paths,
    parse_mixed_datetime,
    record_manifest,
    resolve_outdir,
    resolve_run_dir,
    safe_filename,
)


def cohort_table(
    df: pd.DataFrame,
    cohort_col: str,
    y_col: str,
    group_cols: list[str],
    amount_col: str | None = "principal",
    score_col: str | None = "risk_score",
) -> pd.DataFrame:
    """Aggregate label outcomes by cohort (and optional segment columns)."""
    work = df.copy()
    work = work[work[y_col].notna()].copy()
    work["bad"] = work[y_col].astype("int64")

    keys = group_cols + [cohort_col]
    agg = {
        "loan_cnt": ("bad", "size"),
        "bad_cnt": ("bad", "sum"),
        "bad_rate": ("bad", "mean"),
    }
    if amount_col and amount_col in work.columns:
        agg["amount_mean"] = (amount_col, "mean")
        agg["amount_sum"] = (amount_col, "sum")
    if score_col and score_col in work.columns:
        agg["score_mean"] = (score_col, "mean")

    out = work.groupby(keys, as_index=False).agg(**agg)
    out["bad_rate"] = out["bad_rate"].astype("float64")
    return out.sort_values(keys)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Vintage (90D point-in-time) analysis on labeled loans.")
    parser.add_argument(
        "--infile",
        type=str,
        default="data/model_train_90d.csv",
        help="Input labeled modeling table (default: data/model_train_90d.csv)",
    )
    parser.add_argument("--outdir", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--cohort-col",
        type=str,
        default="disburse_dt",
        help="Datetime column used to form cohorts (default: disburse_dt)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="M",
        choices=["M", "W", "D"],
        help="Cohort frequency: M=month, W=week, D=day (default: M)",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default="y_dpd30_ever",
        help="Label column (default: y_dpd30_ever)",
    )
    parser.add_argument(
        "--by",
        type=str,
        default="",
        help="Comma-separated segment columns (e.g. channel,product_type,risk_score_decile)",
    )
    parser.add_argument("--score-col", type=str, default="risk_score", help="Score column (default: risk_score)")
    parser.add_argument("--score-bins", type=int, default=10, help="Number of quantile bins for *_decile (default: 10)")
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
    infile = Path(args.infile)
    outdir = resolve_outdir(args.outdir, run_dir, name="vintage")

    df = pd.read_csv(infile)
    if args.cohort_col not in df.columns:
        raise KeyError(f"Missing cohort column: {args.cohort_col}")
    df[args.cohort_col] = parse_mixed_datetime(df[args.cohort_col])
    if df[args.cohort_col].isna().any():
        raise ValueError(f"{args.cohort_col} contains NaT; cannot build cohorts reliably.")

    # Convert timestamps to cohort buckets (month/week/day).
    cohort_period = df[args.cohort_col].dt.to_period(args.freq)
    df["cohort"] = cohort_period.astype(str)

    # Optional segmentation (e.g., by channel/product/score decile).
    by_cols = [c.strip() for c in args.by.split(",") if c.strip()]
    if "risk_score_decile" in by_cols:
        df = add_risk_score_decile(df, args.score_col, args.score_bins)
        by_cols = [f"{args.score_col}_decile" if c == "risk_score_decile" else c for c in by_cols]

    base = cohort_table(df, cohort_col="cohort", y_col=args.y_col, group_cols=by_cols)
    suffix = f"_{safe_filename('_'.join(by_cols))}" if by_cols else ""
    outpath = outdir / f"vintage_90d_{args.freq.lower()}{suffix}.csv"
    base.to_csv(outpath, index=False)

    cohorts = base["cohort"].nunique()
    loans = int(base["loan_cnt"].sum())
    bad_rate = (base["bad_cnt"].sum() / loans) if loans else float("nan")

    logger.info("Input: %s", infile)
    logger.info("Cohorts: %s", cohorts)
    logger.info("Loans (labeled): %s", f"{loans:,}")
    logger.info("Overall bad rate: %.2f%%", bad_rate * 100)
    log_saved_paths(logger, outpath)
    record_manifest(run_dir, step="vintage_90d", outputs=[outpath], note=f"by={by_cols or 'none'}")


if __name__ == "__main__":
    main()
