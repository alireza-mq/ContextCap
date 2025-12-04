"""
traffic_agent.py

Domain–specific profiling agent for a traffic + weather dataset.

This module focuses on higher–level aggregates that are meaningful for urban
traffic analysis, such as:
  - traffic by hour of day,
  - traffic by weekday,
  - traffic by holiday flag,
  - traffic by coarse weather condition.

It can be imported by the backend to provide domain–aware JSON snippets that
can be fed into Q/A prompts or used to support hand–crafted captions.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def slice_time_range(
    df: pd.DataFrame,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    out = df
    if date_from is not None:
        start = pd.to_datetime(date_from)
        out = out[out.index >= start]
    if date_to is not None:
        end = pd.to_datetime(date_to)
        out = out[out.index <= end]
    return out


def build_traffic_profile(
    df: pd.DataFrame,
    target_col: str = "traffic_volume",
    weather_col: str = "weather_main",
    holiday_col: str = "holiday",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict:
    """
    Build a JSON–serialisable traffic profile for the given DataFrame.

    The DataFrame is assumed to be indexed by timestamp and to contain the
    columns given by *target_col*, *weather_col* and *holiday_col*. Any
    missing columns are silently skipped.
    """
    df_range = slice_time_range(df, date_from, date_to)
    if df_range.empty:
        raise ValueError("No data in specified date range for traffic profile.")

    if target_col not in df_range.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    s = df_range[target_col].dropna()
    global_stats = {
        "target": target_col,
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
        "p10": float(s.quantile(0.10)),
        "p90": float(s.quantile(0.90)),
    }

    # Top congested timestamps
    top_congested = []
    top_idx = s.nlargest(10).index
    for ts in top_idx:
        top_congested.append(
            {
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "value": float(s.loc[ts]),
            }
        )

    # Traffic by hour of day
    by_hour: List[Dict] = []
    hours = df_range.index.hour
    grp_hour = df_range.groupby(hours)[target_col].mean().dropna()
    for h, v in grp_hour.items():
        by_hour.append({"hour": int(h), "mean": float(v)})

    # Traffic by weekday
    by_weekday: List[Dict] = []
    weekday_index = df_range.index.dayofweek  # 0=Mon
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grp_weekday = df_range.groupby(weekday_index)[target_col].mean().dropna()
    for idx, v in grp_weekday.items():
        name = weekday_names[int(idx)] if 0 <= int(idx) < 7 else str(idx)
        by_weekday.append({"weekday": name, "mean": float(v)})

    # Traffic by holiday indicator
    by_holiday: List[Dict] = []
    if holiday_col in df_range.columns:
        grp_holiday = df_range.groupby(holiday_col)[target_col].mean().dropna()
        for k, v in grp_holiday.items():
            by_holiday.append({"holiday": str(k), "mean": float(v)})

    # Traffic by weather
    by_weather: List[Dict] = []
    if weather_col in df_range.columns:
        grp_weather = df_range.groupby(weather_col)[target_col].mean().dropna()
        for k, v in grp_weather.items():
            by_weather.append({"weather_main": str(k), "mean": float(v)})

    profile = {
        "time_start": df_range.index.min().isoformat(),
        "time_end": df_range.index.max().isoformat(),
        "global_stats": global_stats,
        "top_congested_periods": top_congested,
        "by_hour": by_hour,
        "by_weekday": by_weekday,
        "by_holiday": by_holiday,
        "by_weather": by_weather,
    }
    return profile


def save_profile_json(profile: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Traffic + weather domain profiler for ContextCap.")
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
        "--target-col",
        dest="target_col",
        default="traffic_volume",
        help="Name of the traffic target column (default: traffic_volume).",
    )
    p.add_argument(
        "--out",
        default="traffic_profile.json",
        help="Output JSON path (default: traffic_profile.json).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.csv_path)

    dt_col = args.datetime_col or df.columns[0]
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()

    profile = build_traffic_profile(
        df,
        target_col=args.target_col,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    save_profile_json(profile, args.out)
    print(f"Saved traffic profile to {args.out}")


if __name__ == "__main__":
    main()
