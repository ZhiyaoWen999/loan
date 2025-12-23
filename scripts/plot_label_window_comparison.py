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
)


def ensure_matplotlib(mpl_config_dir: Path) -> None:
    missing = [pkg for pkg in ["matplotlib"] if importlib.util.find_spec(pkg) is None]
    if missing:
        raise SystemExit(
            "Missing dependency: matplotlib. Install with:\n"
            "  .venv/bin/pip install matplotlib seaborn\n"
        )
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def plot_overall(df: pd.DataFrame, outpath: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    cols = ["dpd30_30d", "dpd30_60d", "dpd30_90d", "dpd30_180d"]
    values = df[cols].iloc[0].tolist()
    x = [30, 60, 90, 180]

    plt.figure(figsize=(8, 4))
    plt.plot(x, values, marker="o", linewidth=2)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.title("Bad rate vs observation window (overall)")
    plt.xlabel("Observation window (days)")
    plt.ylabel("Bad rate (DPD30+)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_by_decile(df: pd.DataFrame, outpath: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    cols = ["dpd30_30d", "dpd30_60d", "dpd30_90d", "dpd30_180d"]
    x = [30, 60, 90, 180]

    plt.figure(figsize=(10, 5))
    for _, row in df.sort_values("risk_score_decile").iterrows():
        y = [row[c] for c in cols]
        label = f"D{int(row['risk_score_decile'])}"
        plt.plot(x, y, marker="o", linewidth=1.8, alpha=0.8, label=label)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.title("Bad rate vs observation window by risk_score decile")
    plt.xlabel("Observation window (days)")
    plt.ylabel("Bad rate (DPD30+)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bad-rate comparison across observation windows.")
    parser.add_argument(
        "--overall",
        type=str,
        default="data/label_window_bad_rates_overall.csv",
        help="Overall bad rate table",
    )
    parser.add_argument(
        "--by-decile",
        type=str,
        default="data/label_window_bad_rates_by_score_decile.csv",
        help="Bad rate by score decile table",
    )
    parser.add_argument("--outdir", type=str, default="data/figures", help="Output directory")
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
    ensure_matplotlib(outdir / ".mplconfig")

    overall = pd.read_csv(args.overall)
    decile = pd.read_csv(args.by_decile)

    overall_path = outdir / "label_window_bad_rate_overall.png"
    decile_path = outdir / "label_window_bad_rate_by_decile.png"
    plot_overall(overall, overall_path)
    plot_by_decile(decile, decile_path)

    log_saved_paths(logger, overall_path, decile_path)
    record_manifest(
        run_dir,
        step="plot_label_windows",
        outputs=[overall_path, decile_path],
    )


if __name__ == "__main__":
    main()
