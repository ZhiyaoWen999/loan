#!/usr/bin/env python3
"""Build model-ready feature tables (train/valid) for the 90D label setup.

Inputs:
- `model_train_90d.csv` (mature, labeled loans) produced by `build_model_table.py`

Key steps:
- parse datetimes and derive time features (hour/day-of-week/month, lags)
- drop label/leakage columns and optionally auto-screened features
- split train/valid by time (default: split quantile on disburse_dt)

Outputs:
- `features_train_90d.csv`, `features_valid_90d.csv`
- `feature_columns_90d.txt` (feature list used for modeling)
"""

import argparse
from pathlib import Path

import pandas as pd

from common import (
    add_log_level_arg,
    configure_logging,
    ensure_dir,
    log_saved_paths,
    parse_datetime_columns,
    record_manifest,
    resolve_outdir,
    resolve_run_dir,
)


DATE_COLS = [
    "apply_dt",
    "approve_dt",
    "register_dt",
    "disburse_dt",
    "first_due_dt",
    "chargeoff_dt",
    "paid_off_dt",
    "first_seen_min",
    "first_seen_max",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive simple calendar/recency features from datetime columns."""
    df = df.copy()
    if "apply_dt" in df.columns:
        df["apply_hour"] = df["apply_dt"].dt.hour
        df["apply_dayofweek"] = df["apply_dt"].dt.dayofweek
        df["apply_month"] = df["apply_dt"].dt.month

    if "register_dt" in df.columns and "apply_dt" in df.columns:
        df["register_days_to_apply"] = (df["apply_dt"] - df["register_dt"]).dt.days

    if "first_seen_min" in df.columns and "apply_dt" in df.columns:
        df["device_first_seen_days"] = (df["apply_dt"] - df["first_seen_min"]).dt.days
    if "first_seen_max" in df.columns and "apply_dt" in df.columns:
        df["device_last_seen_days"] = (df["apply_dt"] - df["first_seen_max"]).dt.days

    return df


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Build 90D feature tables with time split.")
    parser.add_argument(
        "--infile",
        type=str,
        default="data/model_train_90d.csv",
        help="Input labeled dataset (default: data/model_train_90d.csv)",
    )
    parser.add_argument("--outdir", type=str, default="data", help="Output directory")
    parser.add_argument("--label", type=str, default="y_dpd30_ever", help="Label column")
    parser.add_argument(
        "--split-col",
        type=str,
        default="disburse_dt",
        help="Datetime column used for time split (default: disburse_dt)",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default=None,
        help="Split date YYYY-MM-DD (train <= date, valid > date).",
    )
    parser.add_argument(
        "--split-quantile",
        type=float,
        default=0.8,
        help="Quantile for time split if split-date not provided (default: 0.8).",
    )
    parser.add_argument(
        "--keep-risk-score",
        action="store_true",
        help="Keep risk_score as a baseline feature (default: drop).",
    )
    parser.add_argument(
        "--keep-ids",
        action="store_true",
        help="Keep app_id/user_id in outputs (default: drop).",
    )
    parser.add_argument(
        "--drop-screened",
        action="store_true",
        help="Drop auto-screened features (low IV/AUC, low variance, high corr).",
    )
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
    df = parse_datetime_columns(df, DATE_COLS)
    if args.label not in df.columns:
        raise SystemExit(f"Missing label column: {args.label}")
    if args.split_col not in df.columns:
        raise SystemExit(f"Missing split column: {args.split_col}")

    df = df[df[args.label].notna()].copy()

    df[args.split_col] = pd.to_datetime(df[args.split_col], errors="coerce")
    df = df[df[args.split_col].notna()].copy()

    df = add_time_features(df)

    # Drop columns that are labels, IDs, or leakage (post-loan) fields.
    drop_cols = {
        args.label,
        "loan_id",
        "disburse_dt",
        "principal",
        "term_days",
        "apr",
        "first_due_dt",
        "dpd_max_30d",
        "dpd_max_60d",
        "dpd_max_90d",
        "dpd_max_180d",
        "dpd30_ever_30d",
        "dpd30_ever_60d",
        "dpd30_ever",
        "dpd30_ever_180d",
        "chargeoff_dt",
        "paid_off_dt",
        "loan_age_days",
        "label_available",
        "decision",
        "approved_amount",
        "approved_term",
        "approve_dt",
    }
    # High-PSI time-related features from EDA; drop to reduce temporal leakage/instability.
    drop_cols.update(
        {
            "apply_month",
            "register_days_to_apply",
            "device_first_seen_days",
            "device_last_seen_days",
        }
    )

    # Auto-screened drop candidates (low IV/AUC, low variance, or high correlation).
    screened_drop_cols = {
        "channel",
        "device_change_90d",
        "has_credit_card",
        "gender",
        "emulator_any",
        "marital_status",
        "device_count",
        "device_cnt",
        "geo_consistent_rate",
        "login_30d",
        "emulator_rate",
        "ip_risk_level_max",
        "device_risk_score_max",
        "sms_cnt_7d",
        "phone_age_months",
        "credit_utilization",
        "apply_dayofweek",
        "ip_risk_level_mean",
        "apply_hour",
        "device_risk_score_mean",
        "brand_nunique",
        "product_type",
        "rooted_any",
    }
    if args.drop_screened:
        drop_cols.update(screened_drop_cols)

    if not args.keep_risk_score:
        drop_cols.add("risk_score")

    if not args.keep_ids:
        drop_cols.update({"app_id", "user_id"})

    # Remove raw datetime columns after deriving features.
    drop_cols.update({c for c in DATE_COLS if c in df.columns})

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Time split: train uses earlier disburse dates; valid uses later dates.
    if args.split_date:
        split_date = pd.to_datetime(args.split_date).normalize()
    else:
        split_date = df[args.split_col].quantile(args.split_quantile)

    train_mask = df[args.split_col] <= split_date
    train = df.loc[train_mask].copy()
    valid = df.loc[~train_mask].copy()

    outdir = resolve_outdir(args.outdir, run_dir, name="features")

    train_out = outdir / "features_train_90d.csv"
    valid_out = outdir / "features_valid_90d.csv"
    feature_list_out = outdir / "feature_columns_90d.txt"

    train[feature_cols + [args.label]].to_csv(train_out, index=False)
    valid[feature_cols + [args.label]].to_csv(valid_out, index=False)

    feature_list_out.write_text("\n".join(feature_cols) + "\n", encoding="utf-8")

    def summary(name: str, d: pd.DataFrame) -> str:
        rate = d[args.label].mean()
        return f"{name}: n={len(d):,}, bad_rate={rate:.2%}"

    logger.info("Split date: %s", split_date.date())
    logger.info("%s", summary("Train", train))
    logger.info("%s", summary("Valid", valid))
    log_saved_paths(logger, train_out, valid_out, feature_list_out)
    record_manifest(
        run_dir,
        step="build_features_90d",
        outputs=[train_out, valid_out, feature_list_out],
        note=f"split_col={args.split_col}, split_quantile={args.split_quantile}",
    )


if __name__ == "__main__":
    main()
