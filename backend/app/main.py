import os
import re
import io
import uuid
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .config.paths import STATIC_DIR, QA_LOG_DIR
from .config import env as _env  # load .env on import
from .agents.general_agent import build_general_profile
from .agents.traffic_agent import build_traffic_profile

# Load pre-context (for demo paper) once at import time
PRECONTEXT_PATH = os.path.join(os.path.dirname(__file__), "demo_paper_context.txt")
try:
    with open(PRECONTEXT_PATH, "r", encoding="utf-8") as _f:
        DEMO_PAPER_CONTEXT = _f.read().strip()
except FileNotFoundError:
    DEMO_PAPER_CONTEXT = ""


# Optional OpenAI client
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    OpenAI = None
    _HAS_OPENAI = False


def prettify_metric_name(name: Optional[str]) -> str:
    """
    Turn an internal metric name like 'traffic_volume' into a user-facing label
    like 'Traffic volume'. If name is None, return 'dataset'.
    """
    if not name:
        return "dataset"
    s = str(name).replace("_", " ")
    return s[:1].upper() + s[1:]



# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    is_metric: bool
    semantic_type: str
    unit: Optional[str]
    stats: Dict[str, float]


@dataclass
class DomainContext:
    domain: str
    probabilities: Dict[str, float]


@dataclass
class Insight:
    id: str
    insight_type: str
    metric: Optional[str]
    time_window: Tuple[pd.Timestamp, pd.Timestamp]
    segment: Dict[str, Any]
    severity: str
    confidence: float
    details: Dict[str, Any]
    caption: Optional[str] = None
    score: Optional[float] = None  # ranking score


# Global in-memory store for this prototype (not production-safe)
DATASETS: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# OpenAI client helper
# ---------------------------------------------------------------------------

def get_openai_client() -> Optional[Any]:
    """
    Create an OpenAI client if the package and API key are available.

    Reads from:
      - environment variable OPENAI_API_KEY
      - or local file openai_key.txt next to this file
    """
    if not _HAS_OPENAI:
        return None

    if os.getenv("OPENAI_API_KEY"):
        return OpenAI()

    key_path = os.path.join(os.path.dirname(__file__), "openai_key.txt")
    if os.path.exists(key_path):
        with open(key_path, "r", encoding="utf-8") as f:
            key = f.read().strip()
        os.environ["OPENAI_API_KEY"] = key
        return OpenAI()

    return None


# ---------------------------------------------------------------------------
# 1. Data Ingestion and Storage
# ---------------------------------------------------------------------------

def load_dataset_from_upload(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Ingest the Metro Interstate Traffic Volume dataset from an uploaded CSV.

    Expected columns (at minimum):
      - date_time
      - traffic_volume
      - temp
      - rain_1h
      - snow_1h
      - clouds_all
      - holiday
      - weather_main
    """
    if "date_time" not in df.columns:
        raise ValueError("Expected a 'date_time' column in the CSV.")

    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").set_index("date_time")

    # Convert temperature from Kelvin to Celsius if present
    if "temp" in df.columns:
        df["temp_c"] = df["temp"] - 273.15
    elif "temp_c" not in df.columns:
        df["temp_c"] = np.nan

    metric_cols = [c for c in ["traffic_volume", "temp_c", "rain_1h", "snow_1h", "clouds_all"]
                   if c in df.columns]

    cat_cols = [c for c in ["holiday", "weather_main"]
                if c in df.columns]

    return df, metric_cols, cat_cols


def compute_quality_summary(df: pd.DataFrame,
                            metric_cols: List[str]) -> Dict[str, Any]:
    """
    Very simple data quality summary:
      - number of rows / columns
      - time range
      - per-metric missing ratio, outlier ratio, min, max
    """
    n_rows = int(len(df))
    n_cols = int(len(df.columns))
    if len(df.index) > 0:
        time_start = df.index.min().isoformat()
        time_end = df.index.max().isoformat()
    else:
        time_start = None
        time_end = None

    per_metric: Dict[str, Any] = {}
    for col in metric_cols:
        series = df[col]
        missing_ratio = float(series.isna().mean())
        vals = series.dropna()
        if vals.empty:
            outlier_ratio = 0.0
            vmin = None
            vmax = None
        else:
            mean = float(vals.mean())
            std = float(vals.std())
            if std == 0 or np.isnan(std):
                outlier_ratio = 0.0
            else:
                z = (vals - mean) / std
                outlier_ratio = float((np.abs(z) > 3).mean())
            vmin = float(vals.min())
            vmax = float(vals.max())
        per_metric[col] = {
            "missing_ratio": missing_ratio,
            "outlier_ratio": outlier_ratio,
            "min": vmin,
            "max": vmax,
        }

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "time_start": time_start,
        "time_end": time_end,
        "per_metric": per_metric,
    }

def build_preview(df: pd.DataFrame,
                  date_from: Optional[str] = None,
                  date_to: Optional[str] = None,
                  n_rows: int = 10) -> Dict[str, Any]:
    """
    Build a small preview table for the given time-indexed DataFrame,
    restricted to [date_from, date_to] if provided, and returning at most
    n_rows rows.
    """
    df_range = df
    if date_from:
        try:
            start_ts = pd.to_datetime(date_from)
            df_range = df_range[df_range.index >= start_ts]
        except Exception:
            pass

    if date_to:
        try:
            end_ts = pd.to_datetime(date_to)
            df_range = df_range[df_range.index <= end_ts]
        except Exception:
            pass

    if df_range.empty:
        preview_df = df.head(0)  # just to keep columns
    else:
        preview_df = df_range.reset_index().head(n_rows)

    return {
        "columns": list(preview_df.columns),
        "rows": preview_df.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 2. Semantic and Domain Analysis
# ---------------------------------------------------------------------------

def infer_semantic_type_and_unit(col_name: str) -> Tuple[str, Optional[str]]:
    """Very simple heuristic mapping from column names to semantic types."""
    name = col_name.lower()
    if "traffic" in name:
        return "traffic_volume", "vehicles/hour"
    if "temp" in name:
        return "temperature", "C"
    if "rain" in name:
        return "rainfall", "mm/hour"
    if "snow" in name:
        return "snowfall", "mm/hour"
    if "cloud" in name:
        return "cloud_cover", "%"
    return "generic_metric", None


def profile_columns(df: pd.DataFrame,
                    metric_cols: List[str],
                    cat_cols: List[str]) -> Dict[str, ColumnProfile]:
    profiles: Dict[str, ColumnProfile] = {}

    for col in metric_cols:
        series = df[col]
        stats = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "missing_ratio": float(series.isna().mean()),
        }
        sem_type, unit = infer_semantic_type_and_unit(col)
        profiles[col] = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            is_metric=True,
            semantic_type=sem_type,
            unit=unit,
            stats=stats,
        )

    for col in cat_cols:
        series = df[col]
        stats = {
            "num_unique": float(series.nunique()),
            "missing_ratio": float(series.isna().mean()),
        }
        profiles[col] = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            is_metric=False,
            semantic_type="categorical",
            unit=None,
            stats=stats,
        )

    return profiles


def classify_domain_heuristic(profiles: Dict[str, ColumnProfile]) -> DomainContext:
    """
    Fallback domain classifier:
      - If 'traffic_volume' metric exists -> 'traffic'
      - Else -> 'generic'
    """
    metric_names = [p.name for p in profiles.values() if p.is_metric]
    if "traffic_volume" in metric_names:
        domain = "traffic"
        probs = {"traffic": 0.9, "generic": 0.1}
    else:
        domain = "generic"
        probs = {"generic": 0.8, "traffic": 0.2}
    return DomainContext(domain=domain, probabilities=probs)


def llm_profile_domain(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Use domain_prompt.txt with OpenAI to infer domains + variable roles.
    Returns the parsed JSON (or None on failure).
    """
    client = get_openai_client()
    if client is None:
        return None

    prompt_path = os.path.join(os.path.dirname(__file__), "domain_prompt.txt")
    if not os.path.exists(prompt_path):
        return None

    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    snippet_df = df.reset_index().head(5)
    csv_snippet = snippet_df.to_csv(index=False)
    user_prompt = template.replace("{{CSV_CONTENT}}", csv_snippet)

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0,
        )
        raw = completion.choices[0].message.content.strip()

        # 1) Strip optional markdown code fences
        if raw.startswith("```"):
            # remove ```json ... ``` or ``` ... ```
            lines = raw.splitlines()
            # drop first and last fence lines if present
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        # 2) Try direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 3) Fallback: extract substring between first '{' and last '}'
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw[start : end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

        # If we reach here, we failed to parse
        print("LLM domain profiling failed to parse JSON. Raw content was:")
        print(raw)
        return None

    except Exception as e:
        print("LLM domain profiling failed:", repr(e))
        return None



def build_domain_context_from_llm(llm_json: Dict[str, Any],
                                  fallback: DomainContext) -> DomainContext:
    domains = llm_json.get("domains") or []
    if not isinstance(domains, list) or len(domains) == 0:
        return fallback

    probs: Dict[str, float] = {}
    for d in domains:
        try:
            name = str(d["name"])
            conf = float(d.get("confidence", 0.0))
            probs[name] = conf
        except Exception:
            continue

    if not probs:
        return fallback

    top_name = max(probs, key=probs.get)
    return DomainContext(domain=top_name, probabilities=probs)


# ---------------------------------------------------------------------------
# 3. Insight Generation Engines
# ---------------------------------------------------------------------------

def _time_to_numeric(dt_index: pd.DatetimeIndex) -> np.ndarray:
    t = dt_index.view("int64") // 10**9  # seconds
    return (t - t.min()).astype(float)


def trend_insights(df: pd.DataFrame,
                   metric_cols: List[str]) -> List[Insight]:
    insights: List[Insight] = []
    if df.empty:
        return insights

    t_num = _time_to_numeric(df.index)

    for col in metric_cols:
        series = df[col].dropna()
        if len(series) < 100:
            continue
        y = series.values
        t = t_num[-len(series):]
        slope, intercept = np.polyfit(t, y, 1)
        total_change = float(slope * (t.max() - t.min()))
        mean_val = float(y.mean())
        if mean_val == 0:
            continue
        rel = abs(total_change) / (abs(mean_val) + 1e-8)
        if rel < 0.1:
            continue
        direction = "increase" if total_change > 0 else "decrease"
        severity = "medium" if rel < 0.5 else "high"
        insights.append(
            Insight(
                id=str(uuid.uuid4()),
                insight_type="trend",
                metric=col,
                time_window=(df.index.min(), df.index.max()),
                segment={},
                severity=severity,
                confidence=0.8,
                details={
                    "direction": direction,
                    "relative_change": rel,
                    "total_change": total_change,
                    "mean": mean_val,
                },
            )
        )
    return insights


def seasonality_insights(df: pd.DataFrame) -> List[Insight]:
    insights: List[Insight] = []
    if "traffic_volume" not in df.columns or df.empty:
        return insights

    hours = df.index.hour
    dow = df.index.dayofweek

    hourly_means = df.groupby(hours)["traffic_volume"].mean()
    dow_means = df.groupby(dow)["traffic_volume"].mean()

    top_hours = hourly_means.sort_values(ascending=False).head(2).index.tolist()
    top_dows = dow_means.sort_values(ascending=False).head(2).index.tolist()

    insights.append(
        Insight(
            id=str(uuid.uuid4()),
            insight_type="seasonality",
            metric="traffic_volume",
            time_window=(df.index.min(), df.index.max()),
            segment={},
            severity="info",
            confidence=0.9,
            details={
                "hourly_means": {int(h): float(v) for h, v in hourly_means.items()},
                "dow_means": {int(d): float(v) for d, v in dow_means.items()},
                "peak_hours": [int(h) for h in top_hours],
                "peak_days_of_week": [int(d) for d in top_dows],
            },
        )
    )
    return insights


def anomaly_insights(df: pd.DataFrame,
                     metric: str = "traffic_volume",
                     z_thresh: float = 2.5,
                     max_points: int = 5) -> List[Insight]:
    insights: List[Insight] = []
    if metric not in df.columns or df.empty:
        return insights

    series = df[metric].dropna()
    mean = float(series.mean())
    std = float(series.std())
    if std == 0:
        return insights

    z = (series - mean) / std
    anoms = z[abs(z) > z_thresh].sort_values(
        key=lambda s: abs(s), ascending=False
    ).head(max_points)

    for ts, z_val in anoms.items():
        window_start = ts - pd.Timedelta(hours=1)
        window_end = ts + pd.Timedelta(hours=1)
        insights.append(
            Insight(
                id=str(uuid.uuid4()),
                insight_type="anomaly",
                metric=metric,
                time_window=(window_start, window_end),
                segment={},
                severity="high",
                confidence=0.85,
                details={
                    "timestamp": str(ts),
                    "z_score": float(z_val),
                    "value": float(series.loc[ts]),
                    "mean": mean,
                    "std": std,
                },
            )
        )

    return insights


def segment_comparison_insights(df: pd.DataFrame) -> List[Insight]:
    insights: List[Insight] = []
    if "traffic_volume" not in df.columns or df.empty:
        return insights

    # Holiday vs non-holiday
    if "holiday" in df.columns:
        normal = df[df["holiday"] == "None"]["traffic_volume"]
        holiday = df[df["holiday"] != "None"]["traffic_volume"]
        if len(normal) > 50 and len(holiday) > 10:
            diff = float(holiday.mean() - normal.mean())
            rel = diff / (float(normal.mean()) + 1e-8)
            severity = "medium" if abs(rel) > 0.1 else "low"
            insights.append(
                Insight(
                    id=str(uuid.uuid4()),
                    insight_type="segment_comparison",
                    metric="traffic_volume",
                    time_window=(df.index.min(), df.index.max()),
                    segment={
                        "dimension": "holiday",
                        "segment_a": "holiday",
                        "segment_b": "None",
                    },
                    severity=severity,
                    confidence=0.8,
                    details={
                        "mean_holiday": float(holiday.mean()),
                        "mean_non_holiday": float(normal.mean()),
                        "absolute_diff": diff,
                        "relative_diff": rel,
                    },
                )
            )

    # Weather: Clear vs Rain
    if "weather_main" in df.columns:
        base = df[df["weather_main"] == "Clear"]["traffic_volume"]
        rain = df[df["weather_main"] == "Rain"]["traffic_volume"]
        if len(base) > 50 and len(rain) > 10:
            diff = float(rain.mean() - base.mean())
            rel = diff / (float(base.mean()) + 1e-8)
            severity = "medium" if abs(rel) > 0.1 else "low"
            insights.append(
                Insight(
                    id=str(uuid.uuid4()),
                    insight_type="segment_comparison",
                    metric="traffic_volume",
                    time_window=(df.index.min(), df.index.max()),
                    segment={
                        "dimension": "weather_main",
                        "segment_a": "Rain",
                        "segment_b": "Clear",
                    },
                    severity=severity,
                    confidence=0.8,
                    details={
                        "mean_rain": float(rain.mean()),
                        "mean_clear": float(base.mean()),
                        "absolute_diff": diff,
                        "relative_diff": rel,
                    },
                )
            )

    return insights


def correlation_insights(df: pd.DataFrame,
                         metric_cols: List[str],
                         threshold: float = 0.25) -> List[Insight]:
    insights: List[Insight] = []
    if len(metric_cols) < 2 or df.empty:
        return insights

    corr = df[metric_cols].corr()
    for i, col_i in enumerate(metric_cols):
        for j, col_j in enumerate(metric_cols):
            if j <= i:
                continue
            value = float(corr.loc[col_i, col_j])
            if np.isnan(value) or abs(value) < threshold:
                continue
            severity = "info" if abs(value) < 0.6 else "medium"
            insights.append(
                Insight(
                    id=str(uuid.uuid4()),
                    insight_type="correlation",
                    metric=None,
                    time_window=(df.index.min(), df.index.max()),
                    segment={},
                    severity=severity,
                    confidence=0.8,
                    details={
                        "metric_a": col_i,
                        "metric_b": col_j,
                        "correlation": value,
                    },
                )
            )
    return insights


def run_insight_pipeline(df: pd.DataFrame,
                         metric_cols: List[str],
                         domain_ctx: DomainContext) -> List[Insight]:
    """
    Simple planner:
      - Always run trend, anomaly, correlation.
      - If `traffic_volume` exists, also run seasonality & segment comparisons.
    """
    insights: List[Insight] = []
    insights.extend(trend_insights(df, metric_cols))
    insights.extend(anomaly_insights(df))
    insights.extend(correlation_insights(df, metric_cols))

    if "traffic_volume" in metric_cols:
        insights.extend(seasonality_insights(df))
        insights.extend(segment_comparison_insights(df))

    return insights


# ---------------------------------------------------------------------------
# 4. LLM Narration and Ranking
# ---------------------------------------------------------------------------

def generate_caption_for_insight(client: Any,
                                 insight: Insight,
                                 domain_ctx: DomainContext) -> str:
    """
    Generate a short, metric-aware caption for a single insight.

    IMPORTANT: The caption MUST describe the metric given in the 'Metric' field,
    not some generic variable like traffic volume, unless the metric name itself
    clearly indicates traffic volume.
    """
    if client is None:
        # Fallback template if no client / key
        return f"[FALLBACK] {insight.insight_type} on {insight.metric or 'metrics'} with details {insight.details}"

    metric_raw = insight.metric or "dataset"
    metric_human = prettify_metric_name(metric_raw)

    system_prompt = (
        "You are ContextCap, an assistant that generates concise, "
        "domain-aware natural-language captions for multivariate time-series insights. "
        "You will be given a specific metric and an insight over that metric. "
        "Your job is to describe the behaviour of THAT METRIC, not some other variable. "
        "If the metric is temperature, talk about temperature; if it is rainfall, talk "
        "about rainfall; if it is traffic volume, talk about traffic volume. "
        "Be factual, cautious about causality, and write 2–3 sentences."
    )

    user_prompt = (
        "Domain context: {domain}\n"
        "Metric: {metric_human} (variable name: {metric_raw})\n"
        "Insight type: {itype}\n"
        "Time window: {tstart} to {tend}\n"
        "Segment: {segment}\n"
        "Severity: {severity}, confidence: {conf}\n"
        "Details (computed from this metric only): {details}\n\n"
        "Write a short caption explaining this insight to a non-expert. "
        "The caption MUST clearly describe the behaviour of the metric given in the "
        "'Metric' field and MUST NOT claim it is about generic 'urban road traffic' "
        "unless the metric name explicitly indicates traffic volume."
    ).format(
        domain=domain_ctx.domain,
        metric_human=metric_human,
        metric_raw=metric_raw,
        itype=insight.insight_type,
        tstart=str(insight.time_window[0]),
        tend=str(insight.time_window[1]),
        segment=insight.segment,
        severity=insight.severity,
        conf=insight.confidence,
        details=insight.details,
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-5.1",  # or your working chat model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("LLM caption generation failed:", repr(e))
        return f"[FALLBACK] {insight.insight_type} on {insight.metric or 'metrics'} with details {insight.details}"


def caption_and_rank_insights(insights: List[Insight],
                              domain_ctx: DomainContext,
                              max_insights: int = 15) -> List[Insight]:
    client = get_openai_client()
    severity_weight = {"high": 2.0, "medium": 1.0, "info": 0.0}

    for ins in insights:
        ins.caption = generate_caption_for_insight(client, ins, domain_ctx)
        base = severity_weight.get(ins.severity, 0.0)
        ins.score = base + float(ins.confidence)

    # sort descending by score
    insights_sorted = sorted(
        insights,
        key=lambda i: (i.score if i.score is not None else 0.0),
        reverse=True,
    )
    return insights_sorted[:max_insights]


def insight_to_dict(ins: Insight) -> Dict[str, Any]:
    d = asdict(ins)
    d["time_window"] = [
        ins.time_window[0].isoformat(),
        ins.time_window[1].isoformat(),
    ]
    return d


# ---------------------------------------------------------------------------
# 5. Visualization summary for Interactive Visualization
# ---------------------------------------------------------------------------

def build_viz_summary(df: pd.DataFrame,
                      metric_cols: List[str],
                      cat_cols: List[str],
                      targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a small summary used by the front-end plots and Q/A:
      - daily series for target
      - holiday vs non-holiday average
      - weather_main average
      - hourly average
      - max and min day
      - weekday averages
    """
    # 1) Choose a target variable
    target_name: Optional[str] = None
    if targets:
        target_name = str(targets[0]["name"])
    elif "traffic_volume" in df.columns:
        target_name = "traffic_volume"
    elif metric_cols:
        target_name = metric_cols[0]

    # If no valid target, return empty summary
    if target_name is None or target_name not in df.columns:
        return {
            "target_name": None,
            "daily_target": [],
            "holiday_summary": [],
            "weather_summary": [],
            "hourly_summary": [],
            "weekday_summary": [],
            "max_day": None,
            "min_day": None,
        }

    # 2) Daily mean of target
    daily = df[target_name].resample("D").mean().dropna()
    daily_target = [
        {"date": ts.date().isoformat(), "value": float(val)}
        for ts, val in daily.items()
    ]

    # 3) Max/min traffic day over the whole dataset
    if not daily.empty:
        max_ts = daily.idxmax()
        max_val = daily.loc[max_ts]
        max_day = {
            "date": max_ts.date().isoformat(),
            "value": float(max_val),
        }

        min_ts = daily.idxmin()
        min_val = daily.loc[min_ts]
        min_day = {
            "date": min_ts.date().isoformat(),
            "value": float(min_val),
        }
    else:
        max_day = None
        min_day = None

    # 4) Holiday summary (mean by holiday, plus non-holiday)
    holiday_summary: List[Dict[str, Any]] = []
    if "holiday" in df.columns:
        tmp = df.copy()
        # Normalise holiday labels and collapse all non-holiday codes
        h = tmp["holiday"].astype(str).str.strip().str.lower()

        non_holiday_mask = h.isna() | h.isin(
            ["none", "non-holiday", "non holiday", "no_holiday", "no holiday", "0", ""]
        )
        tmp["holiday_group"] = h.where(~non_holiday_mask, "non_holiday")

        grp = tmp.groupby("holiday_group")[target_name].mean().dropna()
        for k, v in grp.items():
            holiday_summary.append({"holiday": str(k), "mean": float(v)})



    # 5) Weather summary (mean by weather_main)
    weather_summary: List[Dict[str, Any]] = []
    if "weather_main" in df.columns:
        grp = df.groupby("weather_main")[target_name].mean().dropna()
        for k, v in grp.items():
            weather_summary.append({"weather_main": str(k), "mean": float(v)})

    # 6) Hour-of-day summary (mean by hour)
    hourly_summary: List[Dict[str, Any]] = []
    hours = df.index.hour
    grp_hour = df.groupby(hours)[target_name].mean().dropna()
    for h, v in grp_hour.items():
        hourly_summary.append({"hour": int(h), "mean": float(v)})

    # 7) Weekday summary (mean by Monday, Tuesday, ...)
    weekday_summary: List[Dict[str, Any]] = []
    weekday_index = df.index.dayofweek  # 0=Mon, 6=Sun
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
    grp_weekday = df.groupby(weekday_index)[target_name].mean().dropna()
    for idx, v in grp_weekday.items():
        name = weekday_names[int(idx)] if 0 <= int(idx) < 7 else str(idx)
        weekday_summary.append({"weekday": name, "mean": float(v)})

    return {
        "target_name": target_name,
        "daily_target": daily_target,
        "holiday_summary": holiday_summary,
        "weather_summary": weather_summary,
        "hourly_summary": hourly_summary,
        "weekday_summary": weekday_summary,
        "max_day": max_day,
        "min_day": min_day,
    }

# ---------------------------------------------------------------------------
# 6. Q&A: dataset-aware question answering
# ---------------------------------------------------------------------------

def extract_date_from_question(question: str) -> Optional[str]:
    """
    Look for a YYYY-MM-DD date in the question.
    Returns the ISO date string (YYYY-MM-DD) or None.
    """
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", question)
    if not m:
        return None
    try:
        dt = pd.to_datetime(m.group(1)).date()
        return dt.isoformat()
    except Exception:
        return None

def answer_question_with_dataset(dataset: Dict[str, Any],
                                 question: str) -> str:
    client = get_openai_client()
    domain_ctx: DomainContext = dataset["domain_ctx"]
    quality = dataset.get("range_quality") or dataset["quality"]
    viz_summary = dataset.get("viz_summary") or {}
    insights: List[Insight] = dataset.get("insights") or []
    general_profile = dataset.get("general_profile")
    traffic_profile = dataset.get("traffic_profile")

    lines: List[str] = []

    # --- demo paper pre-context (highest-priority) ---
    if DEMO_PAPER_CONTEXT:
        lines.append("Domain pre-context (trusted background notes):")
        lines.append(DEMO_PAPER_CONTEXT)
        lines.append("")  # blank line

    # --- existing range summary ---
    lines.append(f"Domain: {domain_ctx.domain}")
    lines.append(
        f"Rows: {quality.get('n_rows')} from {quality.get('time_start')} to {quality.get('time_end')}"
    )
    target_name = viz_summary.get("target_name")
    if target_name:
        lines.append(f"Target variable: {target_name}")

    # --- numeric stats from general_profile for the target ---
    if general_profile and target_name:
        m_stats = next(
            (m for m in general_profile.get("metrics", []) if m["name"] == target_name),
            None,
        )
        if m_stats:
            lines.append(
                f"{target_name} numeric range in this window: "
                f"min={m_stats['min_value']:.1f}, "
                f"q05={m_stats['p05']:.1f}, "
                f"median={m_stats['median']:.1f}, "
                f"q95={m_stats['p95']:.1f}, "
                f"max={m_stats['max_value']:.1f}."
            )
    if traffic_profile:
        weekday_means = traffic_profile.get("by_weekday") or []
        if weekday_means:
            parts = [f"{w['weekday']} -> {w['mean']:.1f}" for w in weekday_means]
            lines.append("Average traffic by weekday: " + ", ".join(parts))

        weekend_vs_weekday = traffic_profile.get("weekend_vs_weekday") or {}
        if weekend_vs_weekday:
            lines.append(
                "Weekend vs weekday traffic (vehicles/hour): "
                f"weekday_mean={weekend_vs_weekday.get('weekday_mean', 0):.1f}, "
                f"weekend_mean={weekend_vs_weekday.get('weekend_mean', 0):.1f}."
            )

        weather_impact = traffic_profile.get("weather_impact") or []
        if weather_impact:
            parts = [f"{w['condition']} -> {w['mean']:.1f}" for w in weather_impact]
            lines.append("Average traffic by weather condition: " + ", ".join(parts))

    # Holiday stats
    holiday_summary = viz_summary.get("holiday_summary") or []
    if holiday_summary:
        parts = [f"{h['holiday']} -> {h['mean']:.1f}" for h in holiday_summary]
        lines.append("Average target by holiday: " + ", ".join(parts))

    # Weather stats
    weather_summary = viz_summary.get("weather_summary") or []
    if weather_summary:
        parts = [f"{w['weather_main']} -> {w['mean']:.1f}" for w in weather_summary]
        lines.append("Average target by weather_main: " + ", ".join(parts))

    # Weekday stats
    weekday_summary = viz_summary.get("weekday_summary") or []
    if weekday_summary:
        parts = [f"{w['weekday']} -> {w['mean']:.1f}" for w in weekday_summary]
        lines.append("Average target by weekday: " + ", ".join(parts))

    # Max / min daily target
    max_day = viz_summary.get("max_day") or None
    if max_day and max_day.get("date") is not None:
        lines.append(
            f"Max_daily_target: {max_day['date']} -> {max_day['value']:.2f}"
        )

    min_day = viz_summary.get("min_day") or None
    if min_day and min_day.get("date") is not None:
        lines.append(
            f"Min_daily_target: {min_day['date']} -> {min_day['value']:.2f}"
        )

    # A few insight captions
    if insights:
        caps = [f"- {i.insight_type}: {i.caption}" for i in insights[:5] if i.caption]
        if caps:
            lines.append("Key insights:")
            lines.extend(caps)

    context_text = "\n".join(lines)

    if client is None:
        return (
            "I cannot reach the language model, but here is a basic summary:\n\n"
            + context_text
            + "\n\nBased on this summary, you can reason manually about your question."
        )

    system_prompt = (
        "You are a data analysis assistant for a specific multivariate time-series dataset. "
        "You receive a question and a compact context built from several sources:\n"
        "  (1) curated domain pre-context (trusted background notes),\n"
        "  (2) numeric profiles for each metric (general_profile),\n"
        "  (3) traffic-domain summaries (traffic_profile), and\n"
        "  (4) time-window quality and generated insights.\n"
        "Use the numeric statistics whenever possible: quote concrete means, ranges, "
        "and differences instead of generic phrases like 'increased' or 'decreased'. "
        "Ground every claim in the provided context and be explicit about uncertainty."
    )


    user_prompt = (
        "User question:\n"
        f"{question}\n\n"
        "Dataset summary:\n"
        f"{context_text}\n\n"
        "Now answer the question in 2–4 sentences, grounding your answer in the dataset summary."
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("LLM Q&A failed:", repr(e))
        return (
            "An error occurred while contacting the language model. "
            "However, here is the dataset summary I have:\n\n"
            + context_text
        )



# ---------------------------------------------------------------------------
# 7. FastAPI app + static UI
# ---------------------------------------------------------------------------

app = FastAPI(title="ContextCap EDBT Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the standalone frontend directory under /static.
# All assets (index.html, style.css, app.js) live in the top-level `frontend/`.
static_dir = str(STATIC_DIR)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.post("/upload_profile")
async def upload_profile(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Step 1: upload CSV and return domain candidates, variables, and quality summary.
    """
    contents = await file.read()
    raw_df = pd.read_csv(io.BytesIO(contents))

    ts_df, metric_cols, cat_cols = load_dataset_from_upload(raw_df)
    profiles = profile_columns(ts_df, metric_cols, cat_cols)
    quality = compute_quality_summary(ts_df, metric_cols)

        
    preview = build_preview(ts_df, date_from=None, date_to=None, n_rows=10)



    heuristic_ctx = classify_domain_heuristic(profiles)
    llm_json = llm_profile_domain(ts_df)

    if llm_json is not None:
        domain_ctx = build_domain_context_from_llm(llm_json, heuristic_ctx)
        domains = llm_json.get("domains") or []
        variables = llm_json.get("variables") or []
        targets = llm_json.get("targets") or []
    else:
        domain_ctx = heuristic_ctx
        domains = [{"name": domain_ctx.domain, "confidence": 1.0}]
        # build variables from profiles
        variables = []
        for name, p in profiles.items():
            variables.append({
                "name": name,
                "role": "feature" if p.is_metric else "meta",
                "description": name.replace("_", " "),
                "unit": p.unit or "unknown",
                "unit_confidence": 0.5,
            })
        targets = []
        if "traffic_volume" in profiles:
            targets.append({"name": "traffic_volume", "confidence": 0.8})

    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = {
        "df": ts_df,
        "metric_cols": metric_cols,
        "cat_cols": cat_cols,
        "profiles": profiles,
        "llm_profile": llm_json,
        "domain_ctx": domain_ctx,
        "quality": quality,
        "targets": targets,
        "chosen_domain": domain_ctx.domain,
        "insights": [],
        "viz_summary": None,
    }

    return {
        "dataset_id": dataset_id,
        "domains": domains,
        "domain_context": {
            "domain": domain_ctx.domain,
            "probabilities": domain_ctx.probabilities,
        },
        "variables": variables,
        "targets": targets,
        "quality": quality,
        "preview": preview,  
    }

@app.post("/preview_range")
async def preview_range(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Return a preview table (first N rows) for the selected analysis range
    for a previously uploaded dataset.
    """
    dataset_id = payload.get("dataset_id")
    date_from = payload.get("date_from")
    date_to = payload.get("date_to")

    if not dataset_id or dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Unknown dataset_id")

    ds = DATASETS[dataset_id]
    df = ds["df"]

    preview = build_preview(df, date_from=date_from, date_to=date_to, n_rows=10)
    return {"preview": preview}



@app.post("/analyze")
async def analyze_compat(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Backward-compatible alias for /upload_profile (used in an earlier UI version).
    Allows older frontends that POST to /analyze to still function.
    """
    return await upload_profile(file)



@app.post("/generate_insights")
async def generate_insights(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Step 2: generate insights and visualization summary for a dataset,
    restricted to an optional date range.
    """
    dataset_id = payload.get("dataset_id")
    chosen_domain = payload.get("chosen_domain")
    date_from = payload.get("date_from")
    date_to = payload.get("date_to")

    if not dataset_id or dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Unknown dataset_id")

    ds = DATASETS[dataset_id]
    df_full = ds["df"]
    metric_cols = ds["metric_cols"]
    domain_ctx: DomainContext = ds["domain_ctx"]

    # --- apply optional date filter on the index ---
    df = df_full
    if date_from:
        try:
            start_ts = pd.to_datetime(date_from)
            df = df[df.index >= start_ts]
        except Exception:
            pass
    if date_to:
        try:
            end_ts = pd.to_datetime(date_to)
            df = df[df.index <= end_ts]
        except Exception:
            pass

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="No data available in the selected date range.",
        )

    # Recompute quality for analysis range and store it
    range_quality = compute_quality_summary(df, metric_cols)
    ds["range_quality"] = range_quality

    # Keep track of the effective analysis range
    ds["analysis_range"] = {
        "start": df.index.min().isoformat(),
        "end": df.index.max().isoformat(),
    }

    # Update domain if user explicitly chose one
    if chosen_domain:
        domain_ctx.domain = str(chosen_domain)
        ds["chosen_domain"] = domain_ctx.domain

        # ---- NEW: general + traffic profiles for this analysis range ----
    try:
        general_profile = build_general_profile(
            df,
            metric_cols=metric_cols,  # can be None, but this keeps it aligned
        )
        # store as dict so it’s JSON-serialisable
        ds["general_profile"] = general_profile.to_dict()
    except Exception as e:
        print("general_profile failed:", repr(e))
        ds["general_profile"] = None

    try:
        traffic_profile = build_traffic_profile(
            df,
            target_col="traffic_volume",
            weather_col="weather_main",
            holiday_col="holiday",
        )
        ds["traffic_profile"] = traffic_profile  # already a dict
    except Exception as e:
        print("traffic_profile failed:", repr(e))
        ds["traffic_profile"] = None


    # --- run insight engines on the filtered df ---
    raw_insights = run_insight_pipeline(df, metric_cols, domain_ctx)
    ranked_insights = caption_and_rank_insights(
        raw_insights, domain_ctx, max_insights=15
    )
    ds["insights"] = ranked_insights

    # --- build viz summary on the filtered df ---
    viz_summary = build_viz_summary(df, metric_cols, ds["cat_cols"], ds.get("targets") or [])
    ds["viz_summary"] = viz_summary

    return {
        "insights": [insight_to_dict(i) for i in ranked_insights],
        "summary": viz_summary,
    }


@app.post("/qa")
async def qa(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Step 3: question answering on top of the dataset + insights.
    """
    dataset_id = payload.get("dataset_id")
    question = payload.get("question")

    if not dataset_id or dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Unknown dataset_id")
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    ds = DATASETS[dataset_id]
    answer = answer_question_with_dataset(ds, question)
    return {
        "answer": answer,
    }


@app.post("/qa_feedback")
async def qa_feedback(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Store Q/A feedback for future use.
    """
    dataset_id = payload.get("dataset_id", "unknown")
    question = payload.get("question", "")
    answer = payload.get("answer", "")
    feedback = payload.get("feedback", "")
    corrected = payload.get("corrected_answer", "")

    logs_dir = os.path.join(os.path.dirname(__file__), "qa_logs")
    os.makedirs(logs_dir, exist_ok=True)

    if feedback == "up":
        path = os.path.join(logs_dir, "accepted.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"DATASET: {dataset_id}\n")
            f.write(f"Q: {question}\n")
            f.write(f"A: {answer}\n\n")
    elif feedback == "down":
        path = os.path.join(logs_dir, "corrections.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"DATASET: {dataset_id}\n")
            f.write(f"Q: {question}\n")
            f.write(f"ORIGINAL: {answer}\n")
            f.write(f"CORRECTED: {corrected}\n\n")

    return {"status": "ok"}
