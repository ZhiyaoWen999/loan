#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from common import (
    add_log_level_arg,
    add_risk_score_decile,
    configure_logging,
    ensure_dir,
    log_saved_paths,
    parse_datetime_columns,
    record_manifest,
    resolve_outdir,
    resolve_run_dir,
)


LABEL_COLS = ["dpd30_ever_30d", "dpd30_ever_60d", "dpd30_ever", "dpd30_ever_180d"]
LABEL_RENAME = {
    "dpd30_ever_30d": "dpd30_30d",
    "dpd30_ever_60d": "dpd30_60d",
    "dpd30_ever": "dpd30_90d",
    "dpd30_ever_180d": "dpd30_180d",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare bad-rate growth across label windows (30/60/90/180D).")
    parser.add_argument(
        "--infile",
        type=str,
        default="data/model_train_90d.csv",
        help="Input labeled model table (default: data/model_train_90d.csv)",
    )
    parser.add_argument("--outdir", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--cohort-col",
        type=str,
        default="disburse_dt",
        help="Cohort datetime column (default: disburse_dt)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="M",
        choices=["M", "W"],
        help="Cohort frequency (default: M)",
    )
    parser.add_argument("--score-col", type=str, default="risk_score", help="Score column for deciles")
    parser.add_argument("--score-bins", type=int, default=10, help="Number of score quantiles (default: 10)")
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
    df = pd.read_csv(args.infile)
    df = parse_datetime_columns(df, [args.cohort_col])
    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing label columns: {missing}. Rebuild model table after regenerating data.")

    df = df[df[args.cohort_col].notna()].copy()
    df["cohort"] = df[args.cohort_col].dt.to_period(args.freq).astype(str)

    df = add_risk_score_decile(df, args.score_col, args.score_bins)

    outdir = resolve_outdir(args.outdir, run_dir, name="label_windows")

    overall = {}
    for col in LABEL_COLS:
        overall[LABEL_RENAME[col]] = df[col].mean()
    overall_df = pd.DataFrame([overall])
    overall_path = outdir / "label_window_bad_rates_overall.csv"
    overall_df.to_csv(overall_path, index=False)

    cohort_rows = []
    for col in LABEL_COLS:
        tmp = df.groupby("cohort", as_index=False)[col].mean()
        tmp = tmp.rename(columns={col: LABEL_RENAME[col]})
        cohort_rows.append(tmp.set_index("cohort"))
    cohort_df = pd.concat(cohort_rows, axis=1).reset_index()
    cohort_path = outdir / "label_window_bad_rates_by_cohort.csv"
    cohort_df.to_csv(cohort_path, index=False)

    decile_rows = []
    for col in LABEL_COLS:
        tmp = df.groupby("risk_score_decile", as_index=False)[col].mean()
        tmp = tmp.rename(columns={col: LABEL_RENAME[col]})
        decile_rows.append(tmp.set_index("risk_score_decile"))
    decile_df = pd.concat(decile_rows, axis=1).reset_index()
    decile_path = outdir / "label_window_bad_rates_by_score_decile.csv"
    decile_df.to_csv(decile_path, index=False)

    log_saved_paths(logger, overall_path, cohort_path, decile_path)
    record_manifest(
        run_dir,
        step="compare_label_windows",
        outputs=[overall_path, cohort_path, decile_path],
        note=f"freq={args.freq}",
    )


if __name__ == "__main__":
    main()
