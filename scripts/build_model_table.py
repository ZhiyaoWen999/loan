#!/usr/bin/env python3
"""Build the application-level modeling table (pre-loan features + post-loan label).

This script joins the mock source tables into one wide table at application grain:
- application info + behavioral aggregates (app_id)
- user profile + bureau + device aggregates (user_id)
- loan info + outcomes (approved apps only)

Label (default):
- `y_dpd30_ever`: whether DPD30+ ever happened within an observation window (90D),
  and only defined for "mature" loans (loan_age_days >= maturity window).

Outputs:
- `model_dataset.csv`: all applications (including rejected/unlabeled)
- `model_train_90d.csv`: mature, labeled loans used for training/validation
"""

import argparse
from pathlib import Path

import pandas as pd

from common import (
    add_log_level_arg,
    configure_logging,
    ensure_parent,
    read_csv_table,
    record_manifest,
    resolve_path,
    resolve_run_dir,
)


def assert_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    """Fail fast if the given key(s) are not unique in a table."""
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, keys].head(10)
        raise ValueError(f"{name} has duplicated keys={keys}. Sample:\n{sample}")


def read_table(datadir: Path, filename: str, datetime_cols: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV from datadir and parse selected datetime columns if present."""
    return read_csv_table(datadir / filename, datetime_cols=datetime_cols)


def build_device_agg(devices: pd.DataFrame) -> pd.DataFrame:
    """Aggregate device-level records to user-level features."""
    devices = devices.copy()
    devices["is_android"] = (devices["os"] == "android").astype("int64")

    device_agg = devices.groupby("user_id", as_index=False).agg(
        device_cnt=("device_id", "nunique"),
        brand_nunique=("brand", "nunique"),
        first_seen_min=("first_seen_dt", "min"),
        first_seen_max=("first_seen_dt", "max"),
        emulator_any=("emulator_flag", "max"),
        emulator_rate=("emulator_flag", "mean"),
        rooted_any=("rooted_flag", "max"),
        rooted_rate=("rooted_flag", "mean"),
        device_risk_score_mean=("device_risk_score", "mean"),
        device_risk_score_max=("device_risk_score", "max"),
        ip_risk_level_mean=("ip_risk_level", "mean"),
        ip_risk_level_max=("ip_risk_level", "max"),
        geo_consistent_rate=("geo_consistent_flag", "mean"),
        android_share=("is_android", "mean"),
    )

    numeric_cols = [
        "emulator_rate",
        "rooted_rate",
        "device_risk_score_mean",
        "device_risk_score_max",
        "ip_risk_level_mean",
        "ip_risk_level_max",
        "geo_consistent_rate",
        "android_share",
    ]
    device_agg[numeric_cols] = device_agg[numeric_cols].astype("float64")
    return device_agg


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Build an application-level modeling table for pre-loan risk.")
    parser.add_argument("--datadir", type=str, default="data", help="Directory containing source CSVs")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path for all applications")
    parser.add_argument("--out-train", type=str, default=None, help="Output CSV path for labeled mature samples")
    parser.add_argument("--maturity-days", type=int, default=90, help="Label observation window in days")
    parser.add_argument("--asof", type=str, default=None, help="As-of date YYYY-MM-DD (default: today)")
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
    datadir = Path(args.datadir)
    if args.out:
        out_path = resolve_path(args.out, run_dir)
    else:
        out_path = (run_dir / "model_dataset.csv") if run_dir else datadir / "model_dataset.csv"
    if args.out_train:
        out_train_path = resolve_path(args.out_train, run_dir)
    else:
        out_train_path = (run_dir / "model_train_90d.csv") if run_dir else datadir / "model_train_90d.csv"
    ensure_parent(out_path)
    ensure_parent(out_train_path)

    applications = read_table(datadir, "applications.csv", datetime_cols=["apply_dt", "approve_dt"])
    behavior = read_table(datadir, "behavior_agg.csv", datetime_cols=["apply_dt"])
    bureau = read_table(datadir, "bureau.csv")
    users = read_table(datadir, "users.csv", datetime_cols=["register_dt"])
    devices = read_table(datadir, "devices.csv", datetime_cols=["first_seen_dt"])
    loans = read_table(datadir, "loans.csv", datetime_cols=["disburse_dt", "first_due_dt"])
    outcomes = read_table(datadir, "loan_outcomes.csv", datetime_cols=["chargeoff_dt", "paid_off_dt"])

    assert_unique(applications, ["app_id"], "applications")
    assert_unique(behavior, ["app_id"], "behavior_agg")
    assert_unique(users, ["user_id"], "users")
    assert_unique(bureau, ["user_id"], "bureau")
    assert_unique(devices, ["device_id"], "devices")
    assert_unique(loans, ["loan_id"], "loans")
    assert_unique(loans, ["app_id"], "loans(app_id)")
    assert_unique(outcomes, ["loan_id"], "loan_outcomes")

    device_agg = build_device_agg(devices)

    # Join app-level sources first (app_id grain), then user-level enrichments.
    df = applications.merge(
        behavior.drop(columns=["apply_dt"]),
        on=["app_id", "user_id"],
        how="left",
    )
    df = df.merge(users, on="user_id", how="left")
    df = df.merge(bureau, on="user_id", how="left")
    df = df.merge(device_agg, on="user_id", how="left")

    # Bring in loan and outcome only for approved applications that disbursed.
    loan_cols = ["loan_id", "app_id", "disburse_dt", "principal", "term_days", "apr", "first_due_dt"]
    df = df.merge(loans[loan_cols], on="app_id", how="left")

    desired_outcome_cols = [
        "loan_id",
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
    ]
    outcome_cols = [c for c in desired_outcome_cols if c in outcomes.columns]
    df = df.merge(outcomes[outcome_cols], on="loan_id", how="left")

    # Maturity filter: we can only compute "ever in X days" once the window is fully observed.
    as_of_date = pd.to_datetime(args.asof).normalize() if args.asof else pd.Timestamp.today().normalize()
    df["loan_age_days"] = (as_of_date - df["disburse_dt"]).dt.days
    df["label_available"] = df["loan_id"].notna() & (df["loan_age_days"] >= args.maturity_days)
    df["y_dpd30_ever"] = df["dpd30_ever"].where(df["label_available"]).astype("Int64")

    df.to_csv(out_path, index=False)

    train_df = df[df["label_available"]].copy()
    train_df.to_csv(out_train_path, index=False)

    total_apps = len(df)
    approved_apps = (df["decision"] == "approved").sum()
    matured = len(train_df)
    bad_rate = train_df["y_dpd30_ever"].mean() if matured else float("nan")

    logger.info("All applications: %s", f"{total_apps:,}")
    logger.info("Approved applications: %s (%.2f%%)", f"{approved_apps:,}", approved_apps / total_apps * 100)
    logger.info("Mature labeled (>= %sd): %s", args.maturity_days, f"{matured:,}")
    if matured:
        logger.info("Bad rate (y_dpd30_ever): %.2f%%", bad_rate * 100)
    logger.info("Saved: %s", out_path)
    logger.info("Saved: %s", out_train_path)
    record_manifest(
        run_dir,
        step="build_model_table",
        outputs=[out_path, out_train_path],
        note=f"maturity_days={args.maturity_days}",
    )


if __name__ == "__main__":
    main()
