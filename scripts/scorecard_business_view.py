#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from common import add_log_level_arg, configure_logging, ensure_parent, log_saved_paths, record_manifest, resolve_path, resolve_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Business-friendly scorecard view.")
    parser.add_argument("--woe", type=str, default="data/scorecard/scorecard_woe_bins.csv", help="WOE bins CSV")
    parser.add_argument("--points", type=str, default="data/scorecard/scorecard_points.csv", help="Points CSV")
    parser.add_argument("--iv", type=str, default="data/scorecard/scorecard_iv.csv", help="IV CSV")
    parser.add_argument("--out", type=str, default="data/scorecard/scorecard_business_view.csv", help="Output CSV")
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
    woe = pd.read_csv(args.woe)
    pts = pd.read_csv(args.points)
    iv = pd.read_csv(args.iv)

    pts = pts[["feature", "bin", "coef", "points"]]
    woe = woe[["feature", "bin", "bad_rate", "woe", "iv", "bin_type"]]

    out = woe.merge(pts, on=["feature", "bin"], how="left").merge(iv, on="feature", how="left", suffixes=("", "_iv"))
    out = out.rename(columns={"iv": "iv_feature"})
    out = out[
        [
            "feature",
            "bin",
            "bin_type",
            "bad_rate",
            "woe",
            "coef",
            "points",
            "iv_feature",
        ]
    ].sort_values(["feature", "bin"])

    out_path = resolve_path(args.out, run_dir)
    ensure_parent(out_path)
    out.to_csv(out_path, index=False)

    log_saved_paths(logger, out_path)
    record_manifest(run_dir, step="scorecard_business_view", outputs=[out_path])


if __name__ == "__main__":
    main()
