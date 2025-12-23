#!/usr/bin/env python3
import argparse
import importlib.util
import os
from pathlib import Path

import pandas as pd

from common import (
    add_log_level_arg,
    configure_logging,
    ensure_dir,
    log_saved_paths,
    record_manifest,
    resolve_outdir,
    resolve_run_dir,
    safe_filename,
)


def ensure_seaborn(mpl_config_dir: Path) -> None:
    missing = [pkg for pkg in ["matplotlib", "seaborn"] if importlib.util.find_spec(pkg) is None]
    if missing:
        raise SystemExit(
            "Missing dependency: "
            + ", ".join(missing)
            + ". Install with:\n"
            + "  .venv/bin/pip install matplotlib seaborn\n"
        )
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def sorted_cohorts(values: pd.Series) -> list[str]:
    # Works for YYYY-MM (monthly) and YYYY-MM-DD/weekly strings alike by parsing to Period when possible.
    v = values.dropna().astype(str).unique().tolist()
    try:
        p = pd.PeriodIndex(v, freq="M")
        return [str(x) for x in p.sort_values()]
    except Exception:
        return sorted(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot classic vintage charts using seaborn.")
    parser.add_argument("--infile", type=str, default="data/vintage_mob_dpd30.csv", help="Input vintage table")
    parser.add_argument("--outdir", type=str, default="data/figures", help="Output directory")
    parser.add_argument(
        "--metric",
        type=str,
        default="ever_dpd30_rate",
        choices=["dpd30_rate", "ever_dpd30_rate"],
        help="Metric to plot (default: ever_dpd30_rate)",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="both",
        choices=["line", "heatmap", "both"],
        help="Chart type (default: both)",
    )
    parser.add_argument("--max-cohorts", type=int, default=12, help="Max cohorts in line chart (default: 12)")
    parser.add_argument(
        "--hue-col",
        type=str,
        default="cohort",
        help="Hue column for line plot (default: cohort). Use e.g. risk_score_decile for score-segment vintage.",
    )
    parser.add_argument(
        "--segment-col",
        type=str,
        default="",
        help="Optional segment column (e.g. risk_score_decile); draws one chart per segment value.",
    )
    parser.add_argument(
        "--segment-values",
        type=str,
        default="",
        help="Optional comma-separated segment values to plot (default: all values in data).",
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
    outdir = resolve_outdir(args.outdir, run_dir, name="figures")
    ensure_seaborn(outdir / ".mplconfig")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter

    df = pd.read_csv(args.infile)
    if "cohort" not in df.columns or "mob" not in df.columns or args.metric not in df.columns:
        raise SystemExit(f"Expected columns cohort,mob,{args.metric} in {args.infile}")

    df["mob"] = pd.to_numeric(df["mob"], errors="coerce").astype("Int64")
    df = df[df["mob"].notna()].copy()
    df["mob"] = df["mob"].astype(int)
    stem = safe_filename(Path(args.infile).stem)

    segment_col = args.segment_col.strip() or None
    if segment_col:
        if segment_col not in df.columns:
            raise SystemExit(f"segment-col not found in data: {segment_col}")
        seg_values = (
            [v.strip() for v in args.segment_values.split(",") if v.strip()]
            if args.segment_values.strip()
            else sorted(df[segment_col].dropna().astype(str).unique().tolist())
        )
        segments = seg_values
    else:
        segments = [None]

    sns.set_theme(style="whitegrid")

    saved_paths: list[Path] = []
    for seg in segments:
        work = df.copy()
        seg_suffix = ""
        title_suffix = ""
        if segment_col and seg is not None:
            work = work[work[segment_col].astype(str) == str(seg)].copy()
            seg_suffix = f"_{safe_filename(segment_col)}_{safe_filename(str(seg))}"
            title_suffix = f" | {segment_col}={seg}"

        hue_col = args.hue_col.strip() or "cohort"
        if hue_col not in work.columns:
            raise SystemExit(f"hue-col not found in data: {hue_col}")

        cohorts_sorted = sorted_cohorts(work["cohort"]) if "cohort" in work.columns else []
        cohorts_line = cohorts_sorted[-args.max_cohorts :] if cohorts_sorted else []

        if args.kind in ("line", "both"):
            if hue_col == "cohort":
                line = work[work["cohort"].isin(cohorts_line)].copy()
                title = f"Vintage (cohort lines) - {args.metric}{title_suffix}"
            else:
                if "loan_cnt" in work.columns:
                    tmp = work[[hue_col, "mob", args.metric, "loan_cnt"]].copy()
                    tmp["_wx"] = tmp[args.metric] * tmp["loan_cnt"]
                    grouped = tmp.groupby([hue_col, "mob"], as_index=False).agg(
                        _w=("loan_cnt", "sum"),
                        _wx=("_wx", "sum"),
                    )
                    grouped[args.metric] = grouped["_wx"] / grouped["_w"]
                    grouped = grouped[[hue_col, "mob", args.metric]]
                else:
                    grouped = work.groupby([hue_col, "mob"], as_index=False)[args.metric].mean()
                line = grouped.copy()
                title = f"Vintage (by {hue_col}) - {args.metric}{title_suffix}"

            plt.figure(figsize=(12, 5))
            ax = sns.lineplot(
                data=line,
                x="mob",
                y=args.metric,
                hue=hue_col,
                marker="o",
                linewidth=2,
            )
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_title(title)
            ax.set_xlabel("MOB")
            ax.set_ylabel(args.metric)
            ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
            plt.tight_layout()
            outpath = outdir / f"{stem}_{args.metric}_lines_{safe_filename(hue_col)}{seg_suffix}.png"
            plt.savefig(outpath, dpi=160)
            plt.close()
            saved_paths.append(outpath)

        if args.kind in ("heatmap", "both"):
            if hue_col == "cohort":
                pivot = (
                    work.pivot_table(index="cohort", columns="mob", values=args.metric, aggfunc="mean")
                    .reindex(index=cohorts_sorted)
                    .sort_index()
                )
                y_label = "cohort"
                title = f"Vintage (cohort×MOB heatmap) - {args.metric}{title_suffix}"
            else:
                if "loan_cnt" in work.columns:
                    tmp = work[[hue_col, "mob", args.metric, "loan_cnt"]].copy()
                    tmp["_wx"] = tmp[args.metric] * tmp["loan_cnt"]
                    grouped = tmp.groupby([hue_col, "mob"], as_index=False).agg(
                        _w=("loan_cnt", "sum"),
                        _wx=("_wx", "sum"),
                    )
                    grouped[args.metric] = grouped["_wx"] / grouped["_w"]
                    pivot = grouped.pivot(index=hue_col, columns="mob", values=args.metric)
                else:
                    pivot = work.pivot_table(index=hue_col, columns="mob", values=args.metric, aggfunc="mean")
                y_label = hue_col
                title = f"Vintage ({hue_col}×MOB heatmap) - {args.metric}{title_suffix}"
            plt.figure(figsize=(12, 7))
            ax = sns.heatmap(
                pivot,
                cmap="Reds",
                vmin=0,
                vmax=float(pivot.max().max()) if pivot.size else 0,
                linewidths=0.3,
                linecolor="#e5e7eb",
                cbar_kws={"format": PercentFormatter(1.0)},
            )
            ax.set_title(title)
            ax.set_xlabel("MOB")
            ax.set_ylabel(y_label)
            plt.tight_layout()
            outpath = outdir / f"{stem}_{args.metric}_heatmap_{safe_filename(hue_col)}{seg_suffix}.png"
            plt.savefig(outpath, dpi=160)
            plt.close()
            saved_paths.append(outpath)

    if saved_paths:
        log_saved_paths(logger, *saved_paths)
        record_manifest(run_dir, step="plot_vintage", outputs=saved_paths, note=f"metric={args.metric}, kind={args.kind}")


if __name__ == "__main__":
    main()
