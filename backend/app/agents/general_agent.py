"""
general_agent.py

General statistical profiling agent for ContextCap.

This module is intentionally decoupled from the FastAPI backend so that it can
be reused both:
  - offline (command–line profiling of a CSV), and
  - online (imported by the backend during the ingestion step).

It computes a numeric summary for each metric in a multivariate time–series
dataset, restricted to an optional time range, and emits a JSON–serialisable
profile that can be used as a lightweight RAG source for Q/A and captioning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataclasses describing the profile
# ---------------------------------------------------------------------------


@dataclass
class MetricExtremes:
    name: str
    min_value: float
    min_timestamp: Optional[str]
    max_value: float
    max_timestamp: Optional[str]
    mean: float
    median: float
    std: float
    p05: float
    p25: float
    p75: float
    p95: float


@dataclass
class ExtremePoint:
    metric: str
    timestamp: str
    value: float
    z_score: float
    rank: int


@dataclass
class GeneralProfile:
    n_rows: int
    n_cols: int
    time_start: Optional[str]
    time_end: Optional[str]
    metrics: List[MetricExtremes]
    extreme_points: List[ExtremePoint]
    correlations: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "metrics": [asdict(m) for m in self.metrics],
            "extreme_points": [asdict(e) for e in self.extreme_points],
            "correlations": self.correlations,
        }


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def slice_time_range(
    df: pd.DataFrame,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Return a view of *df* restricted to [date_from, date_to] on the index."""
    out = df
    if date_from is not None:
        start = pd.to_datetime(date_from)
        out = out[out.index >= start]
    if date_to is not None:
        end = pd.to_datetime(date_to)
        out = out[out.index <= end]
    return out


def detect_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """Heuristic detection of numeric metric columns."""
    exclude = set(exclude or [])
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols


def compute_metric_extremes(df: pd.DataFrame, metric_cols: List[str]) -> List[MetricExtremes]:
    metrics: List[MetricExtremes] = []
    for col in metric_cols:
        series = df[col]

        if not pd.api.types.is_numeric_dtype(series):
            continue

        series = series.dropna()
        if series.empty:
            continue

        idx_min = series.idxmin()
        idx_max = series.idxmax()
        metrics.append(
            MetricExtremes(
                name=col,
                min_value=float(series.min()),
                min_timestamp=idx_min.isoformat() if hasattr(idx_min, "isoformat") else str(idx_min),
                max_value=float(series.max()),
                max_timestamp=idx_max.isoformat() if hasattr(idx_max, "isoformat") else str(idx_max),
                mean=float(series.mean()),
                median=float(series.median()),
                std=float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                p05=float(series.quantile(0.05)),
                p25=float(series.quantile(0.25)),
                p75=float(series.quantile(0.75)),
                p95=float(series.quantile(0.95)),
            )
        )
    return metrics


def compute_extreme_points(
    df: pd.DataFrame,
    metric_cols: List[str],
    top_k: int = 10,
) -> List[ExtremePoint]:
    """
    Identify extreme observations across all metrics using simple z–scores.

    For each metric, compute (value - mean) / std and keep the top_k absolute
    deviations. The result is flattened and re–sorted globally by |z|.
    """
    points: List[ExtremePoint] = []

    for col in metric_cols:
        series = df[col]

        if not pd.api.types.is_numeric_dtype(series):
            continue

        series = series.dropna()
        if len(series) < 3:
            continue

        mean = series.mean()
        std = series.std(ddof=1)
        if std <= 0:
            continue

        z = (series - mean) / std
        # Take the largest absolute deviations for this metric
        top_idx = z.abs().nlargest(top_k).index
        for ts in top_idx:
            points.append(
                ExtremePoint(
                    metric=col,
                    timestamp=ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    value=float(series.loc[ts]),
                    z_score=float(z.loc[ts]),
                    rank=0,  # will be filled after global sort
                )
            )

    # Global ranking by absolute z–score
    points.sort(key=lambda p: abs(p.z_score), reverse=True)
    for i, p in enumerate(points, start=1):
        p.rank = i
    return points


def compute_correlations(df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """Return a nested dict with Pearson correlations between metric columns."""
    if not metric_cols:
        return {}

    numeric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return {}

    corr = df[numeric_cols].corr()
    result: Dict[str, Dict[str, float]] = {}
    for i in numeric_cols:
        result[i] = {}
        for j in numeric_cols:
            val = float(corr.loc[i, j]) if not pd.isna(corr.loc[i, j]) else 0.0
            result[i][j] = val
    return result



def build_general_profile(
    df: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    top_k_extremes: int = 10,
) -> GeneralProfile:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    df_range = slice_time_range(df, date_from, date_to)
    if df_range.empty:
        raise ValueError("No data in specified date range")

    # Work on a copy and normalise metric columns
    df_range = df_range.copy()
    if metric_cols is None:
        metric_cols = detect_numeric_columns(df_range)
    else:
        metric_cols = [c for c in metric_cols if c in df_range.columns]

    # Force metrics to numeric, coercing bad values to NaN
    df_range[metric_cols] = df_range[metric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    metrics = compute_metric_extremes(df_range, metric_cols)
    extreme_points = compute_extreme_points(df_range, metric_cols, top_k=top_k_extremes)
    correlations = compute_correlations(df_range, metric_cols)

    time_start = df_range.index.min().isoformat()
    time_end = df_range.index.max().isoformat()

    profile = GeneralProfile(
        n_rows=int(df_range.shape[0]),
        n_cols=int(df_range.shape[1]),
        time_start=time_start,
        time_end=time_end,
        metrics=metrics,
        extreme_points=extreme_points,
        correlations=correlations,
    )
    return profile


def profile_to_json(profile: GeneralProfile) -> str:
    """Return the profile as a JSON string (pretty–printed)."""
    return json.dumps(profile.to_dict(), indent=2)


def save_profile_json(profile: GeneralProfile, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(profile_to_json(profile))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="General statistical profiler for ContextCap.")
    p.add_argument("csv_path", help="Path to the CSV file.")
    p.add_argument(
        "--datetime-col",
        default=None,
        help="Name of the datetime column. If omitted, the first column is parsed as datetime.",
    )
    p.add_argument(
        "--date-from",
        dest="date_from",
        default=None,
        help="Optional start date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--date-to",
        dest="date_to",
        default=None,
        help="Optional end date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--out",
        default="general_profile.json",
        help="Output JSON path (default: general_profile.json).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.csv_path)

    # Basic datetime handling: assume a column, then set index.
    dt_col = args.datetime_col or df.columns[0]
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()

    profile = build_general_profile(
        df,
        metric_cols=None,
        date_from=args.date_from,
        date_to=args.date_to,
        top_k_extremes=10,
    )
    save_profile_json(profile, args.out)
    print(f"Saved general profile to {args.out}")


if __name__ == "__main__":
    main()
