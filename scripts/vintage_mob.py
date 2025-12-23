#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from common import (
    add_log_level_arg,
    add_risk_score_decile,
    configure_logging,
    ensure_parent,
    log_saved_paths,
    read_csv_table,
    record_manifest,
    resolve_path,
    resolve_run_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build classic cohort×MOB vintage tables from DPD snapshots.")
    parser.add_argument(
        "--snapshots",
        type=str,
        default="data/loan_dpd_snapshots.csv",
        help="Loan-level DPD snapshot table (default: data/loan_dpd_snapshots.csv)",
    )
    parser.add_argument(
        "--applications",
        type=str,
        default="data/applications.csv",
        help="Applications table with risk_score/channel/product (default: data/applications.csv)",
    )
    parser.add_argument("--out", type=str, default="data/vintage_mob_dpd30.csv", help="Output CSV path")
    parser.add_argument(
        "--cohort-freq",
        type=str,
        default="M",
        choices=["M", "W"],
        help="Cohort frequency based on disburse_dt: M=month, W=week (default: M)",
    )
    parser.add_argument(
        "--mob-max",
        type=int,
        default=6,
        help="Max MOB to keep (default: 6, meaning 0..6)",
    )
    parser.add_argument(
        "--by",
        type=str,
        default="",
        help="Comma-separated segment columns (e.g. channel,product_type,risk_score_decile)",
    )
    parser.add_argument("--score-col", type=str, default="risk_score", help="Score column (default: risk_score)")
    parser.add_argument("--score-bins", type=int, default=10, help="Quantile bins for *_decile (default: 10)")
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
    snap = read_csv_table(
        Path(args.snapshots),
        datetime_cols=["disburse_dt", "snapshot_dt", "paid_off_dt", "chargeoff_dt"],
    )
    apps = read_csv_table(Path(args.applications), datetime_cols=["apply_dt", "approve_dt"])

    if "app_id" not in snap.columns:
        raise KeyError("snapshots missing app_id; regenerate data with updated generator.")

    apps_keep = ["app_id", "risk_score", "channel", "product_type", "requested_amount", "requested_term"]
    snap = snap.merge(apps[apps_keep], on="app_id", how="left")

    cohort = snap["disburse_dt"].dt.to_period(args.cohort_freq).astype(str)
    snap["cohort"] = cohort

    snap["mob"] = pd.to_numeric(snap["mob"], errors="coerce").astype("Int64")
    snap = snap[snap["mob"].notna()].copy()
    snap = snap[snap["mob"] <= args.mob_max].copy()

    snap["dpd"] = pd.to_numeric(snap["dpd"], errors="coerce").fillna(0).astype(int)
    snap["dpd30"] = (snap["dpd"] >= 30).astype(int)
    snap = snap.sort_values(["loan_id", "mob"])
    snap["ever_dpd30"] = snap.groupby("loan_id")["dpd30"].cummax()

    by_cols = [c.strip() for c in args.by.split(",") if c.strip()]
    if "risk_score_decile" in by_cols:
        snap = add_risk_score_decile(snap, args.score_col, args.score_bins)
        by_cols = [f"{args.score_col}_decile" if c == "risk_score_decile" else c for c in by_cols]

    keys = by_cols + ["cohort", "mob"]
    out = snap.groupby(keys, as_index=False).agg(
        loan_cnt=("loan_id", "nunique"),
        dpd30_rate=("dpd30", "mean"),
        ever_dpd30_rate=("ever_dpd30", "mean"),
        active_loan_cnt=("is_active", "sum"),
    )
    out["dpd30_rate"] = out["dpd30_rate"].astype("float64")
    out["ever_dpd30_rate"] = out["ever_dpd30_rate"].astype("float64")

    outpath = resolve_path(args.out, run_dir)
    ensure_parent(outpath)
    out.sort_values(keys).to_csv(outpath, index=False)

    log_saved_paths(logger, outpath)
    record_manifest(run_dir, step="vintage_mob", outputs=[outpath], note=f"by={by_cols or 'none'}")


if __name__ == "__main__":
    main()
