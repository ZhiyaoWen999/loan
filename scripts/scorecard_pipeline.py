#!/usr/bin/env python3
"""WOE scorecard pipeline (binning → WOE → logistic regression → points).

This script implements a simplified, scorecard-style workflow:
1) For each feature, build a coarse binning on the train set
2) Compute WOE and IV; keep features above `--iv-threshold`
3) Transform both train/valid into WOE values
4) Fit logistic regression on WOE features (optionally stepwise selection)
5) Convert coefficients into scorecard points using `--pdo`, `--base-score`, `--base-odds`

Outputs (CSV) under `data/scorecard/` (or a run dir):
- scorecard_woe_bins.csv, scorecard_iv.csv, scorecard_points.csv
- scored train/valid with `score`, decile tables, summary, selected features
"""

import argparse
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import (
    add_log_level_arg,
    configure_logging,
    ensure_dir,
    record_manifest,
    resolve_outdir,
    resolve_run_dir,
)


def auc_from_scores(y: pd.Series, score: pd.Series) -> float:
    """Rank-based AUC (Mann-Whitney). Accepts either raw scores or predicted probabilities."""
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


def ks_stat(y: pd.Series, score: pd.Series) -> float:
    """Compute KS statistic from predicted scores (max CDF separation)."""
    y = pd.to_numeric(y, errors="coerce")
    score = pd.to_numeric(score, errors="coerce")
    mask = y.notna() & score.notna()
    y = y[mask].astype(int)
    score = score[mask]
    if y.nunique() < 2:
        return float("nan")
    df = pd.DataFrame({"y": y, "score": score}).sort_values("score", ascending=False)
    df["cum_bad"] = df["y"].cumsum() / df["y"].sum()
    df["cum_good"] = (1 - df["y"]).cumsum() / (1 - df["y"]).sum()
    return float((df["cum_bad"] - df["cum_good"]).abs().max())


def bin_series(series: pd.Series, bins: int, top_n: int) -> tuple[pd.Series, dict]:
    """Create train-time binning for a single feature and return (binned, params)."""
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce")
        if s.nunique(dropna=True) <= bins:
            binned = s.astype("string").fillna("MISSING")
            return binned, {"type": "numeric_as_categorical", "categories": sorted(binned.dropna().unique())}
        quantiles = np.linspace(0, 1, bins + 1)
        edges = s.quantile(quantiles).to_numpy()
        edges[0] = -np.inf
        edges[-1] = np.inf
        edges = np.unique(edges)
        if len(edges) < 3:
            binned = s.astype("string").fillna("MISSING")
            return binned, {"type": "numeric_as_categorical", "categories": sorted(binned.dropna().unique())}
        binned = pd.cut(s, bins=edges, include_lowest=True)
        binned = binned.cat.add_categories(["MISSING"]).fillna("MISSING")
        return binned.astype(str), {"type": "numeric_binned", "edges": edges.tolist()}

    s = s.astype("string").fillna("MISSING")
    top = s.value_counts().head(top_n).index.tolist()
    binned = s.where(s.isin(top), "OTHER")
    return binned, {"type": "categorical", "categories": top}


def apply_binning(series: pd.Series, params: dict) -> pd.Series:
    """Apply a stored binning spec (from train) to a new Series."""
    if params["type"] == "numeric_binned":
        s = pd.to_numeric(series, errors="coerce")
        edges = np.array(params["edges"], dtype=float)
        binned = pd.cut(s, bins=edges, include_lowest=True)
        binned = binned.cat.add_categories(["MISSING"]).fillna("MISSING")
        return binned.astype(str)
    if params["type"] == "numeric_as_categorical":
        s = pd.to_numeric(series, errors="coerce").astype("string").fillna("MISSING")
        cats = set(params.get("categories", []))
        return s.where(s.isin(cats), "OTHER")
    s = series.astype("string").fillna("MISSING")
    cats = set(params.get("categories", []))
    return s.where(s.isin(cats), "OTHER")


def woe_table(binned: pd.Series, y: pd.Series, eps: float = 0.5) -> pd.DataFrame:
    """Compute WOE table for a binned feature and binary label."""
    df = pd.DataFrame({"bin": binned.astype("string"), "y": y})
    df = df[df["y"].notna()].copy()
    agg = df.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    agg = agg.rename(columns={"sum": "bad"})
    agg["good"] = agg["count"] - agg["bad"]

    bad_dist = (agg["bad"] + eps) / (agg["bad"].sum() + eps * len(agg))
    good_dist = (agg["good"] + eps) / (agg["good"].sum() + eps * len(agg))
    agg["woe"] = np.log(good_dist / bad_dist)
    agg["bad_rate"] = agg["bad"] / agg["count"]
    agg["iv_component"] = (good_dist - bad_dist) * agg["woe"]
    agg = agg.reset_index()
    return agg


def make_scorecard(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    label: str,
    bins: int,
    top_n: int,
    iv_threshold: float,
    pdo: float,
    base_score: float,
    base_odds: float,
    stepwise: bool,
    p_enter: float,
    p_remove: float,
    max_iter: int,
    outdir: Path,
) -> None:
    """Train a scorecard and write scorecard + scoring artifacts to `outdir`."""
    y_train = train[label]
    y_valid = valid[label]
    features = [c for c in train.columns if c != label]

    binning_params = {}
    woe_maps = {}
    woe_rows = []
    iv_rows = []

    for feature in features:
        binned_train, params = bin_series(train[feature], bins=bins, top_n=top_n)
        table = woe_table(binned_train, y_train)
        iv = float(table["iv_component"].sum())

        if iv < iv_threshold:
            continue

        binning_params[feature] = params
        woe_map = table.set_index("bin")["woe"].to_dict()
        woe_maps[feature] = woe_map

        table["feature"] = feature
        table["iv"] = iv
        table["bin_type"] = params["type"]
        woe_rows.append(table)
        iv_rows.append({"feature": feature, "iv": iv, "bin_type": params["type"]})

    if not binning_params:
        raise SystemExit("No features left after IV filtering. Lower --iv-threshold or check inputs.")

    woe_table_df = pd.concat(woe_rows, ignore_index=True)
    iv_df = pd.DataFrame(iv_rows).sort_values("iv", ascending=False)

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        """Map raw features into WOE values using train-derived binning + WOE maps."""
        out = {}
        for feature, params in binning_params.items():
            binned = apply_binning(df[feature], params)
            woe_map = woe_maps[feature]
            out[feature] = binned.map(woe_map).fillna(0.0).astype(float)
        return pd.DataFrame(out, index=df.index)

    X_train = transform(train)
    X_valid = transform(valid)

    selected_features = list(X_train.columns)
    if stepwise:
        import statsmodels.api as sm

        remaining = list(X_train.columns)
        selected: list[str] = []
        iteration = 0

        while iteration < max_iter:
            iteration += 1
            changed = False

            # Forward step: add one feature with the lowest p-value under threshold.
            pvals = {}
            for col in remaining:
                try:
                    X = sm.add_constant(X_train[selected + [col]], has_constant="add")
                    model = sm.Logit(y_train, X).fit(disp=0)
                    pvals[col] = model.pvalues.get(col, np.nan)
                except Exception:
                    continue

            if pvals:
                best_col = min(pvals, key=pvals.get)
                best_p = pvals[best_col]
                if pd.notna(best_p) and best_p < p_enter:
                    selected.append(best_col)
                    remaining.remove(best_col)
                    changed = True

            # Backward step: remove the worst p-value feature if it exceeds threshold.
            if selected:
                try:
                    X = sm.add_constant(X_train[selected], has_constant="add")
                    model = sm.Logit(y_train, X).fit(disp=0)
                    pvals = model.pvalues.drop("const", errors="ignore")
                    worst_p = pvals.max()
                    if pd.notna(worst_p) and worst_p > p_remove:
                        worst_col = pvals.idxmax()
                        selected.remove(worst_col)
                        remaining.append(worst_col)
                        changed = True
                except Exception:
                    pass

            if not changed:
                break

        if not selected:
            raise SystemExit("Stepwise selection removed all features. Loosen p thresholds.")

        selected_features = selected
        X_train = X_train[selected_features]
        X_valid = X_valid[selected_features]

    model = LogisticRegression(solver="liblinear", max_iter=1000)
    model.fit(X_train, y_train)

    train_pred = pd.Series(model.predict_proba(X_train)[:, 1], index=train.index, name="pred")
    valid_pred = pd.Series(model.predict_proba(X_valid)[:, 1], index=valid.index, name="pred")

    train_auc = auc_from_scores(y_train, train_pred)
    valid_auc = auc_from_scores(y_valid, valid_pred)
    train_ks = ks_stat(y_train, train_pred)
    valid_ks = ks_stat(y_valid, valid_pred)

    factor = pdo / log(2)
    offset = base_score - factor * log(base_odds)
    intercept = float(model.intercept_[0])

    base_points = offset - factor * intercept
    coef = pd.Series(model.coef_[0], index=X_train.columns)

    score_train = base_points - factor * (X_train @ coef)
    score_valid = base_points - factor * (X_valid @ coef)

    scorecard_rows = []
    for feature in coef.index:
        woe_map = woe_maps[feature]
        for bin_label, woe in woe_map.items():
            points = -factor * coef[feature] * woe
            scorecard_rows.append(
                {
                    "feature": feature,
                    "bin": bin_label,
                    "woe": woe,
                    "coef": coef[feature],
                    "points": points,
                }
            )

    scorecard_df = pd.DataFrame(scorecard_rows)

    def decile_table(y: pd.Series, score: pd.Series) -> pd.DataFrame:
        """Aggregate score distribution into deciles with bad rates."""
        df = pd.DataFrame({"y": y, "score": score}).dropna()
        df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop") + 1
        agg = df.groupby("decile", as_index=False).agg(
            n=("y", "size"),
            bad_cnt=("y", "sum"),
            bad_rate=("y", "mean"),
            score_mean=("score", "mean"),
        )
        return agg.sort_values("decile")

    ensure_dir(outdir)
    woe_table_df.to_csv(outdir / "scorecard_woe_bins.csv", index=False)
    iv_df.to_csv(outdir / "scorecard_iv.csv", index=False)
    scorecard_df.to_csv(outdir / "scorecard_points.csv", index=False)

    pd.DataFrame(
        [
            {
                "base_score": base_score,
                "pdo": pdo,
                "base_odds": base_odds,
                "base_points": base_points,
                "intercept": intercept,
                "train_auc": train_auc,
                "valid_auc": valid_auc,
                "train_ks": train_ks,
                "valid_ks": valid_ks,
                "iv_threshold": iv_threshold,
                "feature_count": len(coef),
                "stepwise": stepwise,
                "p_enter": p_enter,
                "p_remove": p_remove,
            }
        ]
    ).to_csv(outdir / "scorecard_summary.csv", index=False)

    train_out = train.copy()
    valid_out = valid.copy()
    train_out["score"] = score_train
    valid_out["score"] = score_valid
    train_out.to_csv(outdir / "scorecard_scored_train.csv", index=False)
    valid_out.to_csv(outdir / "scorecard_scored_valid.csv", index=False)

    decile_table(y_train, score_train).to_csv(outdir / "scorecard_deciles_train.csv", index=False)
    decile_table(y_valid, score_valid).to_csv(outdir / "scorecard_deciles_valid.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(outdir / "scorecard_selected_features.csv", index=False)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Scorecard pipeline (WOE + logistic regression).")
    parser.add_argument("--train", type=str, default="data/features_train_90d.csv", help="Train CSV")
    parser.add_argument("--valid", type=str, default="data/features_valid_90d.csv", help="Valid CSV")
    parser.add_argument("--label", type=str, default="y_dpd30_ever", help="Label column")
    parser.add_argument("--outdir", type=str, default="data/scorecard", help="Output directory")
    parser.add_argument("--bins", type=int, default=5, help="Bins for numeric WOE")
    parser.add_argument("--top-n", type=int, default=10, help="Top categories for WOE")
    parser.add_argument("--iv-threshold", type=float, default=0.02, help="IV threshold")
    parser.add_argument("--pdo", type=float, default=50, help="Points to double the odds")
    parser.add_argument("--base-score", type=float, default=600, help="Base score")
    parser.add_argument("--base-odds", type=float, default=20, help="Base odds (good:bad)")
    parser.add_argument("--stepwise", action="store_true", help="Enable bidirectional stepwise selection")
    parser.add_argument("--p-enter", type=float, default=0.05, help="Stepwise enter p-value")
    parser.add_argument("--p-remove", type=float, default=0.1, help="Stepwise remove p-value")
    parser.add_argument("--max-iter", type=int, default=50, help="Max stepwise iterations")
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
    if args.label not in train.columns or args.label not in valid.columns:
        raise SystemExit(f"Missing label: {args.label}")

    outdir = resolve_outdir(args.outdir, run_dir, name="scorecard")
    make_scorecard(
        train=train,
        valid=valid,
        label=args.label,
        bins=args.bins,
        top_n=args.top_n,
        iv_threshold=args.iv_threshold,
        pdo=args.pdo,
        base_score=args.base_score,
        base_odds=args.base_odds,
        stepwise=args.stepwise,
        p_enter=args.p_enter,
        p_remove=args.p_remove,
        max_iter=args.max_iter,
        outdir=outdir,
    )

    logger.info("Saved scorecard outputs to: %s", outdir)
    record_manifest(
        run_dir,
        step="scorecard_pipeline",
        outputs=[
            outdir / "scorecard_points.csv",
            outdir / "scorecard_summary.csv",
            outdir / "scorecard_deciles_train.csv",
            outdir / "scorecard_deciles_valid.csv",
            outdir / "scorecard_selected_features.csv",
        ],
        note=f"bins={args.bins}, iv_thr={args.iv_threshold}, stepwise={args.stepwise}",
    )


if __name__ == "__main__":
    main()
