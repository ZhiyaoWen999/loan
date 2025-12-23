#!/usr/bin/env python3
import argparse
from datetime import datetime
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clamp(series, low, high):
    return np.minimum(np.maximum(series, low), high)


def random_dates(start, end, n, rng):
    start_u = start.value // 10**9
    end_u = end.value // 10**9
    return pd.to_datetime(rng.integers(start_u, end_u, n), unit="s")


def main():
    parser = argparse.ArgumentParser(description="Generate mock pre-loan risk data tables.")
    parser.add_argument("--users", type=int, default=5000, help="Number of users")
    parser.add_argument("--apps", type=int, default=8000, help="Number of applications")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory")
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
    rng = np.random.default_rng(args.seed)
    outdir = resolve_outdir(args.outdir, run_dir, name="")

    today = pd.Timestamp(datetime.now().date())
    start_date = today - pd.Timedelta(days=730)

    user_id = np.arange(1, args.users + 1)
    register_dt = random_dates(start_date, today, args.users, rng)
    age = rng.integers(18, 56, args.users)
    gender = rng.choice(["M", "F"], args.users, p=[0.55, 0.45])
    city_tier = rng.choice([1, 2, 3, 4, 5], args.users, p=[0.12, 0.25, 0.3, 0.23, 0.1])
    education = rng.choice(
        ["middle", "high", "college", "graduate"], args.users, p=[0.15, 0.35, 0.4, 0.1]
    )
    marital_status = rng.choice(["single", "married", "divorced"], args.users, p=[0.45, 0.5, 0.05])
    residence_years = rng.integers(0, 11, args.users)
    company_type = rng.choice(
        ["private", "state", "self_employed", "unemployed"], args.users, p=[0.5, 0.15, 0.25, 0.1]
    )
    monthly_income = rng.lognormal(mean=8.2, sigma=0.35, size=args.users)
    monthly_income = clamp(monthly_income, 1500, 30000).round(0)
    phone_age_months = rng.integers(1, 121, args.users)
    contact_count = rng.integers(50, 501, args.users)
    emergency_contact_count = rng.integers(1, 6, args.users)
    device_count = rng.integers(1, 4, args.users)
    has_credit_card = rng.choice([0, 1], args.users, p=[0.6, 0.4])

    users = pd.DataFrame(
        {
            "user_id": user_id,
            "register_dt": register_dt,
            "age": age,
            "gender": gender,
            "city_tier": city_tier,
            "education": education,
            "marital_status": marital_status,
            "residence_years": residence_years,
            "company_type": company_type,
            "monthly_income": monthly_income,
            "phone_age_months": phone_age_months,
            "contact_count": contact_count,
            "emergency_contact_count": emergency_contact_count,
            "device_count": device_count,
            "has_credit_card": has_credit_card,
        }
    )

    bureau_score = clamp(rng.normal(600, 80, args.users), 300, 850).round(0)
    delinq_12m = rng.choice([0, 1, 2, 3, 4, 5], args.users, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01])
    credit_utilization = clamp(rng.beta(2, 4, args.users), 0, 1).round(2)
    open_accounts = rng.integers(0, 11, args.users)
    inquiry_3m = rng.choice([0, 1, 2, 3, 4, 5, 6], args.users, p=[0.4, 0.25, 0.15, 0.1, 0.06, 0.03, 0.01])

    bureau = pd.DataFrame(
        {
            "user_id": user_id,
            "bureau_score": bureau_score,
            "delinq_12m": delinq_12m,
            "credit_utilization": credit_utilization,
            "open_accounts": open_accounts,
            "inquiry_3m": inquiry_3m,
        }
    )

    device_rows = []
    device_id = 1
    for uid, reg_dt, dcnt in zip(user_id, register_dt, device_count):
        for _ in range(dcnt):
            first_seen = reg_dt + pd.Timedelta(days=int(rng.integers(0, 120)))
            device_os = rng.choice(["android", "ios"], p=[0.8, 0.2])
            brand = rng.choice(
                ["xiaomi", "oppo", "vivo", "huawei", "apple", "samsung", "other"],
                p=[0.22, 0.18, 0.16, 0.14, 0.12, 0.08, 0.1],
            )
            emulator_flag = int(rng.random() < 0.02)
            rooted_flag = int(device_os == "android" and rng.random() < 0.06)
            device_risk_score = int(clamp(rng.normal(40, 20), 0, 100))
            ip_risk_level = rng.choice([1, 2, 3, 4, 5], p=[0.5, 0.25, 0.15, 0.07, 0.03])
            geo_consistent_flag = int(rng.random() < 0.9)
            device_rows.append(
                {
                    "device_id": device_id,
                    "user_id": uid,
                    "first_seen_dt": first_seen,
                    "os": device_os,
                    "brand": brand,
                    "emulator_flag": emulator_flag,
                    "rooted_flag": rooted_flag,
                    "device_risk_score": device_risk_score,
                    "ip_risk_level": ip_risk_level,
                    "geo_consistent_flag": geo_consistent_flag,
                }
            )
            device_id += 1

    devices = pd.DataFrame(device_rows)

    app_user_id = rng.choice(user_id, size=args.apps, replace=True)
    apply_dt = random_dates(start_date + pd.Timedelta(days=30), today, args.apps, rng)
    channel = rng.choice(["app", "web", "partner"], size=args.apps, p=[0.7, 0.2, 0.1])
    product_type = rng.choice(["short_term", "installment"], size=args.apps, p=[0.55, 0.45])
    requested_amount = rng.integers(500, 5001, size=args.apps)
    requested_term = np.where(product_type == "short_term", rng.choice([7, 14, 21, 30], args.apps),
                              rng.choice([60, 90, 120, 180], args.apps))

    applications = pd.DataFrame(
        {
            "app_id": np.arange(1, args.apps + 1),
            "user_id": app_user_id,
            "apply_dt": apply_dt,
            "channel": channel,
            "product_type": product_type,
            "requested_amount": requested_amount,
            "requested_term": requested_term,
        }
    )

    behavior_rows = []
    for app_id, uid, adt in applications[["app_id", "user_id", "apply_dt"]].itertuples(index=False):
        login_7d = int(clamp(rng.normal(5, 3), 0, 20))
        login_30d = int(clamp(rng.normal(20, 10), 0, 80))
        night_ratio = float(clamp(rng.beta(2, 5), 0, 1).round(2))
        device_change_90d = int(rng.random() < 0.1)
        app_install_cnt = int(clamp(rng.normal(60, 20), 10, 200))
        sms_cnt_7d = int(clamp(rng.normal(30, 15), 0, 150))
        behavior_rows.append(
            {
                "app_id": app_id,
                "user_id": uid,
                "apply_dt": adt,
                "login_7d": login_7d,
                "login_30d": login_30d,
                "night_ratio": night_ratio,
                "device_change_90d": device_change_90d,
                "app_install_cnt": app_install_cnt,
                "sms_cnt_7d": sms_cnt_7d,
            }
        )

    behavior = pd.DataFrame(behavior_rows)

    device_main = (
        devices.sort_values(["user_id", "first_seen_dt"])
        .groupby("user_id")
        .tail(1)[
            [
                "user_id",
                "emulator_flag",
                "rooted_flag",
                "device_risk_score",
                "ip_risk_level",
                "geo_consistent_flag",
            ]
        ]
    )

    app_features = (
        applications.merge(users, on="user_id", how="left")
        .merge(bureau, on="user_id", how="left")
        .merge(device_main, on="user_id", how="left")
        .merge(behavior.drop(columns=["apply_dt"]), on=["app_id", "user_id"], how="left")
    )

    risk_score = (
        650
        + (app_features["monthly_income"] - 5000) / 100
        + (app_features["bureau_score"] - 600) * 0.2
        - app_features["delinq_12m"] * 25
        - app_features["inquiry_3m"] * 10
        - app_features["emulator_flag"] * 60
        - app_features["rooted_flag"] * 25
        - app_features["device_risk_score"] * 0.3
        - (1 - app_features["geo_consistent_flag"]) * 20
        - app_features["night_ratio"] * 40
        + rng.normal(0, 15, size=args.apps)
    )
    risk_score = clamp(risk_score, 300, 850).round(0)

    approve_prob = sigmoid((risk_score - 600) / 40)
    decision = rng.random(args.apps) < approve_prob
    approve_delay_days = rng.integers(0, 3, args.apps)
    approve_dt = (applications["apply_dt"] + pd.to_timedelta(approve_delay_days, unit="D")).where(decision)

    credit_line = clamp(app_features["monthly_income"] * 1.5, 500, 8000)
    approved_amount = np.where(decision, np.minimum(requested_amount, credit_line), 0).round(0)
    approved_term = np.where(decision, requested_term, 0)

    applications = applications.assign(
        risk_score=risk_score.astype(int),
        decision=np.where(decision, "approved", "rejected"),
        approve_dt=approve_dt,
        approved_amount=approved_amount.astype(int),
        approved_term=approved_term.astype(int),
    )

    approved_apps = applications[applications["decision"] == "approved"].copy()
    loan_id = np.arange(1, len(approved_apps) + 1)
    disburse_dt = approved_apps["approve_dt"] + pd.to_timedelta(rng.integers(0, 2, len(approved_apps)), unit="D")
    apr = clamp(rng.normal(0.24, 0.05, len(approved_apps)), 0.1, 0.36).round(3)
    first_due_dt = disburse_dt + pd.to_timedelta(approved_apps["approved_term"] // 4 + 7, unit="D")

    loans = pd.DataFrame(
        {
            "loan_id": loan_id,
            "app_id": approved_apps["app_id"].values,
            "user_id": approved_apps["user_id"].values,
            "disburse_dt": disburse_dt.values,
            "principal": approved_apps["approved_amount"].values,
            "term_days": approved_apps["approved_term"].values,
            "apr": apr,
            "first_due_dt": first_due_dt.values,
        }
    )

    loan_features = approved_apps.merge(
        app_features, on=["app_id", "user_id"], how="left", suffixes=("", "_feat")
    )
    bad_prob = sigmoid(
        -6
        + 0.0006 * loan_features["requested_amount"]
        - 0.0004 * loan_features["monthly_income"]
        + 0.9 * loan_features["delinq_12m"]
        + 0.5 * loan_features["inquiry_3m"]
        + 1.3 * loan_features["emulator_flag"]
        + 0.7 * loan_features["rooted_flag"]
        + 0.9 * (loan_features["device_risk_score"] > 70).astype(int)
        + 0.8 * loan_features["night_ratio"]
        + 0.8 * (loan_features["credit_utilization"] > 0.8).astype(int)
        - 0.004 * (loan_features["bureau_score"] - 600)
    )

    bad_flag = rng.random(len(loan_features)) < bad_prob
    dpd_max_90d = np.where(
        bad_flag,
        rng.choice([30, 60, 90], len(loan_features), p=[0.5, 0.3, 0.2]),
        rng.choice([0, 5, 15, 25], len(loan_features), p=[0.6, 0.2, 0.15, 0.05]),
    )

    dpd_max_90d = dpd_max_90d.astype(int)
    dpd30_ever = (dpd_max_90d >= 30).astype(int)

    # Extend delinquency horizon for classic cohort×MOB vintage curves.
    n_loans = len(loans)
    dpd_max_180d = dpd_max_90d.copy()

    early_bad = dpd_max_90d >= 30
    dpd_max_180d = np.where(
        early_bad,
        dpd_max_90d + rng.choice([0, 30, 60], n_loans, p=[0.75, 0.2, 0.05]),
        dpd_max_180d,
    )

    late_bad_prob = 0.01
    late_bad = (~early_bad) & (rng.random(n_loans) < late_bad_prob)
    dpd_max_180d = np.where(late_bad, rng.choice([30, 60], n_loans, p=[0.7, 0.3]), dpd_max_180d)

    dpd_max_180d = np.minimum(dpd_max_180d, 180)
    dpd_max_180d = np.minimum(dpd_max_180d, 90 + dpd_max_90d).astype(int)

    # DPD(t) is monotonic after dpd_start_day: dpd(t)=min(dpd_max_180d, max(0, t-dpd_start_day)).
    dpd_start_day = np.full(n_loans, 999, dtype=int)

    same_horizon = (~late_bad) & (dpd_max_180d == dpd_max_90d) & (dpd_max_90d > 0)
    dpd_start_day[same_horizon] = rng.integers(0, (90 - dpd_max_90d[same_horizon]) + 1)

    longer_horizon = (dpd_max_180d > dpd_max_90d) & (dpd_max_90d > 0)
    dpd_start_day[longer_horizon] = 90 - dpd_max_90d[longer_horizon]

    if late_bad.any():
        latest_start = np.maximum(91, 180 - dpd_max_180d + 1)
        dpd_start_day[late_bad] = rng.integers(91, latest_start[late_bad])

    def dpd_at(day: int) -> np.ndarray:
        return np.clip(day - dpd_start_day, 0, dpd_max_180d).astype(int)

    dpd_max_30d = dpd_at(30)
    dpd_max_60d = dpd_at(60)
    dpd_max_90d_check = dpd_at(90)
    dpd_max_180d_check = dpd_at(180)

    # Keep original dpd_max_90d as the intended 90D max (dpd_at(90) should match by construction).
    dpd_max_90d = dpd_max_90d_check
    dpd_max_180d = dpd_max_180d_check

    dpd30_ever_30d = (dpd_max_30d >= 30).astype(int)
    dpd30_ever_60d = (dpd_max_60d >= 30).astype(int)
    dpd30_ever_90d = (dpd_max_90d >= 30).astype(int)
    dpd30_ever_180d = (dpd_max_180d >= 30).astype(int)
    chargeoff_delay_days = rng.integers(45, 120, len(loans))
    chargeoff_dt = (loans["disburse_dt"] + pd.to_timedelta(chargeoff_delay_days, unit="D")).where(dpd30_ever == 1)

    paid_off_delay_days = loans["term_days"] + rng.integers(0, 15, len(loans))
    paid_off_dt = (loans["disburse_dt"] + pd.to_timedelta(paid_off_delay_days, unit="D")).where(dpd30_ever == 0)

    outcomes = pd.DataFrame(
        {
            "loan_id": loans["loan_id"].values,
            "user_id": loans["user_id"].values,
            "dpd_max_30d": dpd_max_30d,
            "dpd_max_60d": dpd_max_60d,
            "dpd_max_90d": dpd_max_90d,
            "dpd_max_180d": dpd_max_180d,
            "dpd30_ever_30d": dpd30_ever_30d,
            "dpd30_ever_60d": dpd30_ever_60d,
            "dpd30_ever": dpd30_ever_90d,
            "dpd30_ever_180d": dpd30_ever_180d,
            "chargeoff_dt": chargeoff_dt,
            "paid_off_dt": paid_off_dt,
        }
    )

    offsets = pd.DataFrame({"days_since_disburse": [0, 30, 60, 90, 120, 150, 180]})
    dpd_params = pd.DataFrame(
        {
            "loan_id": loans["loan_id"].values,
            "dpd_start_day": dpd_start_day,
            "dpd_max_180d": dpd_max_180d,
        }
    )
    as_of_dt = pd.Timestamp(datetime.now())
    snapshots = (
        loans[["loan_id", "app_id", "user_id", "disburse_dt", "principal", "term_days"]]
        .merge(offsets, how="cross")
        .merge(dpd_params, on="loan_id", how="left")
    )
    snapshots = snapshots.merge(
        pd.DataFrame(
            {
                "loan_id": loans["loan_id"].values,
                "paid_off_dt": paid_off_dt.values,
                "chargeoff_dt": chargeoff_dt.values,
            }
        ),
        on="loan_id",
        how="left",
    )
    snapshots["snapshot_dt"] = snapshots["disburse_dt"] + pd.to_timedelta(
        snapshots["days_since_disburse"], unit="D"
    )
    snapshots = snapshots[snapshots["snapshot_dt"] <= as_of_dt].copy()
    snapshots["mob"] = (snapshots["days_since_disburse"] // 30).astype(int)
    snapshots["dpd"] = np.clip(
        snapshots["days_since_disburse"] - snapshots["dpd_start_day"], 0, snapshots["dpd_max_180d"]
    ).astype(int)
    snapshots["dpd_bucket"] = pd.cut(
        snapshots["dpd"],
        bins=[-1, 0, 30, 60, 90, 10_000],
        labels=["M0", "M1", "M2", "M3", "M4P"],
    ).astype(str)
    snapshots["is_active"] = (
        (snapshots["paid_off_dt"].fillna(pd.Timestamp.max) >= snapshots["snapshot_dt"])
        & (snapshots["chargeoff_dt"].fillna(pd.Timestamp.max) >= snapshots["snapshot_dt"])
    ).astype(int)

    users_path = outdir / "users.csv"
    bureau_path = outdir / "bureau.csv"
    devices_path = outdir / "devices.csv"
    applications_path = outdir / "applications.csv"
    behavior_path = outdir / "behavior_agg.csv"
    loans_path = outdir / "loans.csv"
    outcomes_path = outdir / "loan_outcomes.csv"
    snapshots_path = outdir / "loan_dpd_snapshots.csv"

    users.to_csv(users_path, index=False)
    bureau.to_csv(bureau_path, index=False)
    devices.to_csv(devices_path, index=False)
    applications.to_csv(applications_path, index=False)
    behavior.to_csv(behavior_path, index=False)
    loans.to_csv(loans_path, index=False)
    outcomes.to_csv(outcomes_path, index=False)
    snapshots.to_csv(snapshots_path, index=False)

    logger.info(
        "Saved %s users, %s apps, %s devices, %s loans.",
        f"{len(users):,}",
        f"{len(applications):,}",
        f"{len(devices):,}",
        f"{len(loans):,}",
    )
    outputs = [
        users_path,
        bureau_path,
        devices_path,
        applications_path,
        behavior_path,
        loans_path,
        outcomes_path,
        snapshots_path,
    ]
    log_saved_paths(
        logger,
        *outputs,
    )
    record_manifest(
        run_dir,
        step="generate_mock_data",
        outputs=outputs,
        note=f"seed={args.seed}, users={args.users}, apps={args.apps}",
    )


if __name__ == "__main__":
    main()
