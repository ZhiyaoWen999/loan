#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    add_log_level_arg,
    configure_logging,
    ensure_dir,
    record_manifest,
    resolve_outdir,
    resolve_run_dir,
)


def infer_categorical(nunique_train: int, nunique_valid: int, max_unique: int) -> bool:
    return (nunique_train <= max_unique) and (nunique_valid <= max_unique)


def psi_from_counts(p: pd.Series, q: pd.Series, eps: float = 1e-6) -> float:
    p = p.astype(float).replace(0, eps)
    q = q.astype(float).replace(0, eps)
    return float(((p - q) * np.log(p / q)).sum())


def psi_numeric(train: pd.Series, valid: pd.Series, bins: int = 10) -> float:
    t = pd.to_numeric(train, errors="coerce").dropna()
    v = pd.to_numeric(valid, errors="coerce").dropna()
    if t.empty or v.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = t.quantile(quantiles).to_numpy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    t_bins = pd.cut(t, bins=edges, include_lowest=True)
    v_bins = pd.cut(v, bins=edges, include_lowest=True)
    p = t_bins.value_counts(normalize=True).sort_index()
    q = v_bins.value_counts(normalize=True).reindex(p.index, fill_value=0)
    return psi_from_counts(p, q)


def psi_categorical(train: pd.Series, valid: pd.Series) -> float:
    t = train.astype("string").fillna("MISSING")
    v = valid.astype("string").fillna("MISSING")
    p = t.value_counts(normalize=True)
    q = v.value_counts(normalize=True)
    all_idx = p.index.union(q.index)
    p = p.reindex(all_idx, fill_value=0)
    q = q.reindex(all_idx, fill_value=0)
    return psi_from_counts(p, q)


def summary_numeric(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic EDA report for train/valid feature tables.")
    parser.add_argument("--train", type=str, default="data/features_train_90d.csv", help="Train CSV")
    parser.add_argument("--valid", type=str, default="data/features_valid_90d.csv", help="Valid CSV")
    parser.add_argument("--label", type=str, default="y_dpd30_ever", help="Label column")
    parser.add_argument("--outdir", type=str, default="data/eda", help="Output directory")
    parser.add_argument("--top-n", type=int, default=10, help="Top N categories to list")
    parser.add_argument("--psi-bins", type=int, default=10, help="Bins for numeric PSI")
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.2,
        help="Flag columns with missing rate >= threshold (default: 0.2)",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.1,
        help="Flag columns with PSI >= threshold (default: 0.1)",
    )
    parser.add_argument(
        "--cat-unique-max",
        type=int,
        default=20,
        help="Treat numeric columns with <=N unique values as categorical",
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
    train = pd.read_csv(args.train)
    valid = pd.read_csv(args.valid)

    if args.label not in train.columns:
        raise SystemExit(f"Missing label in train: {args.label}")
    if args.label not in valid.columns:
        raise SystemExit(f"Missing label in valid: {args.label}")

    outdir = resolve_outdir(args.outdir, run_dir, name="eda")

    features = [c for c in train.columns if c != args.label]

    missing_rows = []
    for col in features:
        t = train[col]
        v = valid[col]
        missing_rows.append(
            {
                "column": col,
                "dtype": str(t.dtype),
                "train_missing_rate": float(t.isna().mean()),
                "valid_missing_rate": float(v.isna().mean()),
                "train_nunique": int(t.nunique(dropna=True)),
                "valid_nunique": int(v.nunique(dropna=True)),
            }
        )
    missing_df = pd.DataFrame(missing_rows).sort_values("train_missing_rate", ascending=False)
    missing_df.to_csv(outdir / "missing_summary.csv", index=False)

    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    cat_cols = [c for c in features if train[c].dtype == "object"]

    # Add low-cardinality numeric columns as categorical.
    for col in numeric_cols:
        t_nu = int(train[col].nunique(dropna=True))
        v_nu = int(valid[col].nunique(dropna=True))
        if infer_categorical(t_nu, v_nu, args.cat_unique_max):
            if col not in cat_cols:
                cat_cols.append(col)

    num_rows = []
    for col in numeric_cols:
        t_stats = summary_numeric(train[col])
        v_stats = summary_numeric(valid[col])
        num_rows.append(
            {
                "column": col,
                "train_mean": t_stats["mean"],
                "valid_mean": v_stats["mean"],
                "train_std": t_stats["std"],
                "valid_std": v_stats["std"],
                "train_p50": t_stats["p50"],
                "valid_p50": v_stats["p50"],
                "train_p95": t_stats["p95"],
                "valid_p95": v_stats["p95"],
                "psi": psi_numeric(train[col], valid[col], bins=args.psi_bins),
            }
        )
    numeric_df = pd.DataFrame(num_rows).sort_values("psi", ascending=False)
    numeric_df.to_csv(outdir / "numeric_summary.csv", index=False)

    cat_rows = []
    cat_psi_rows = []
    for col in cat_cols:
        t = train[col].astype("string").fillna("MISSING")
        v = valid[col].astype("string").fillna("MISSING")
        t_counts = t.value_counts(dropna=False).head(args.top_n)
        v_counts = v.value_counts(dropna=False).head(args.top_n)

        for value, count in t_counts.items():
            cat_rows.append(
                {
                    "column": col,
                    "dataset": "train",
                    "value": value,
                    "count": int(count),
                    "rate": float(count) / len(train),
                }
            )
        for value, count in v_counts.items():
            cat_rows.append(
                {
                    "column": col,
                    "dataset": "valid",
                    "value": value,
                    "count": int(count),
                    "rate": float(count) / len(valid),
                }
            )

        cat_psi_rows.append({"column": col, "psi": psi_categorical(t, v)})

    pd.DataFrame(cat_rows).to_csv(outdir / "categorical_top_values.csv", index=False)
    cat_psi_df = pd.DataFrame(cat_psi_rows).sort_values("psi", ascending=False)
    cat_psi_df.to_csv(outdir / "categorical_psi.csv", index=False)

    label_summary = pd.DataFrame(
        [
            {
                "dataset": "train",
                "n": len(train),
                "bad_rate": float(train[args.label].mean()),
            },
            {
                "dataset": "valid",
                "n": len(valid),
                "bad_rate": float(valid[args.label].mean()),
            },
        ]
    )
    label_summary.to_csv(outdir / "label_summary.csv", index=False)

    missing_map = missing_df.set_index("column")
    psi_num_map = numeric_df.set_index("column")["psi"]
    psi_cat_map = cat_psi_df.set_index("column")["psi"]

    flags_rows = []
    for col in features:
        row = {
            "column": col,
            "dtype": str(train[col].dtype),
            "train_missing_rate": float(missing_map.loc[col, "train_missing_rate"]),
            "valid_missing_rate": float(missing_map.loc[col, "valid_missing_rate"]),
            "psi_numeric": float(psi_num_map[col]) if col in psi_num_map.index else float("nan"),
            "psi_categorical": float(psi_cat_map[col]) if col in psi_cat_map.index else float("nan"),
        }
        if col in cat_cols:
            row["psi_used"] = row["psi_categorical"]
            row["psi_source"] = "categorical"
        else:
            row["psi_used"] = row["psi_numeric"]
            row["psi_source"] = "numeric"
        row["missing_flag"] = max(row["train_missing_rate"], row["valid_missing_rate"]) >= args.missing_threshold
        row["psi_flag"] = row["psi_used"] >= args.psi_threshold if not np.isnan(row["psi_used"]) else False
        flags_rows.append(row)

    flags_df = pd.DataFrame(flags_rows).sort_values(
        ["psi_flag", "missing_flag", "psi_used"], ascending=[False, False, False]
    )
    flags_df.to_csv(outdir / "feature_flags.csv", index=False)

    flagged = flags_df[(flags_df["missing_flag"]) | (flags_df["psi_flag"])]
    flagged.to_csv(outdir / "feature_flags_high.csv", index=False)

    logger.info("Saved EDA outputs to: %s", outdir)
    record_manifest(
        run_dir,
        step="eda_report",
        outputs=[
            outdir / "missing_summary.csv",
            outdir / "numeric_summary.csv",
            outdir / "categorical_top_values.csv",
            outdir / "categorical_psi.csv",
            outdir / "label_summary.csv",
            outdir / "feature_flags.csv",
            outdir / "feature_flags_high.csv",
        ],
        note=f"psi_bins={args.psi_bins}",
    )


if __name__ == "__main__":
    main()
