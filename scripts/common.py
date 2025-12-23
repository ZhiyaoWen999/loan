#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def add_log_level_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )


def configure_logging(level: str) -> logging.Logger:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(levelname)s %(message)s")
    return logging.getLogger("loan-risk")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def log_saved_paths(logger: logging.Logger, *paths: Path) -> None:
    for path in paths:
        logger.info("Saved: %s", path)


def record_manifest(run_dir: Path | None, step: str, outputs: list[Path], note: str = "") -> None:
    if run_dir is None:
        return
    manifest = run_dir / "manifest.csv"
    ensure_parent(manifest)
    timestamp = datetime.utcnow().isoformat()
    rows = [{"timestamp": timestamp, "step": step, "path": str(p), "note": note} for p in outputs]
    write_header = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "step", "path", "note"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.to_datetime(series)
    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.to_datetime(series, format=DATETIME_FORMAT, errors="coerce")
    if numeric.notna().any():
        parsed_numeric = pd.to_datetime(numeric, unit="ns", errors="coerce")
        parsed = parsed.fillna(parsed_numeric)
    return parsed


def parse_datetime_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = parse_mixed_datetime(df[col])
    return df


def read_csv_table(path: Path, datetime_cols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if datetime_cols:
        df = parse_datetime_columns(df, datetime_cols)
    return df


def add_risk_score_decile(df: pd.DataFrame, score_col: str, bins: int) -> pd.DataFrame:
    if score_col not in df.columns:
        raise KeyError(f"Missing score column: {score_col}")
    df = df.copy()
    score = pd.to_numeric(df[score_col], errors="coerce")
    try:
        decile = pd.qcut(score, q=bins, labels=False, duplicates="drop") + 1
    except ValueError:
        decile = pd.Series(pd.NA, index=df.index)
    df[f"{score_col}_decile"] = decile.astype("Int64")
    return df


def safe_filename(value: str, extra_allowed: str = "") -> str:
    allowed = set("-_." + extra_allowed)
    return "".join(ch if ch.isalnum() or ch in allowed else "_" for ch in value)


def resolve_run_dir(run_dir: str | None, base: Path = Path("data")) -> Path | None:
    if run_dir is None:
        return None
    if run_dir == "auto":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return ensure_dir(base / f"run_{stamp}")
    return ensure_dir(Path(run_dir))


def resolve_outdir(outdir: str, run_dir: Path | None, name: str | None = None) -> Path:
    if run_dir:
        target_name = Path(outdir).name if name is None else name
        target = run_dir / target_name if target_name else run_dir
        return ensure_dir(target)
    return ensure_dir(Path(outdir))


def resolve_path(path_str: str, run_dir: Path | None) -> Path:
    p = Path(path_str)
    if run_dir and not p.is_absolute():
        return run_dir / p
    return p
