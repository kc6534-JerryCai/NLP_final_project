#!/usr/bin/env python
"""Aggregate classified macro-event metrics to one standardized event per date."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "outputs" / "macro_event_identification"

EQUITY_DIRECTION_SIGN = {
    "headline_yoy": -1.0,
    "headline_mom": -1.0,
    "core_yoy": -1.0,
    "core_mom": -1.0,
    "unemployment_rate": -1.0,
    "nonfarm_payroll_change": 1.0,
    "target_lower": -1.0,
    "target_upper": -1.0,
    "rate_change_bps": -1.0,
}


def join_unique(values: pd.Series) -> str:
    return " | ".join(sorted(set(values.dropna().astype(str))))


def equity_label(score: float, threshold: float) -> str:
    if pd.isna(score):
        return "无法判断"
    if score > threshold:
        return "利好"
    if score < -threshold:
        return "利空"
    return "中性"


def refresh_intraday_metadata(output_dir: Path, daily: pd.DataFrame) -> None:
    """Keep downloaded prices while synchronizing corrected event metadata."""
    metadata_columns = [
        "release_date", "event_types", "official_release_datetime_et",
        "earliest_news_datetime_et", "earliest_news_lag_minutes",
        "equity_label", "equity_direction_score", "equity_label_pre",
        "equity_direction_pre_score", "event_surprise_mean",
        "event_surprise_pre_mean", "event_surprise_unit", "equity_label_basis",
    ]
    metadata = daily.set_index("event_ids")
    for filename in (
        "QQQ_event_intraday_1min_extended.csv",
        "QQQ_event_intraday_summary_extended.csv",
    ):
        path = output_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "event_ids" not in frame.columns:
            continue
        matched = frame["event_ids"].isin(metadata.index)
        for column in metadata_columns:
            if column in frame.columns and column in metadata.columns:
                frame[column] = frame[column].astype("object")
                frame.loc[matched, column] = frame.loc[matched, "event_ids"].map(metadata[column]).to_numpy()
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        temporary_path.replace(path)
        print(f"Refreshed corrected event metadata in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    events = pd.read_csv(output_dir / "macro_events.csv")
    metrics = pd.read_csv(output_dir / "event_metrics_actual_vs_expected.csv")
    for column in ("actual_value", "expected_value", "surprise"):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    for column in ("official_release_datetime_et", "earliest_news_datetime_et"):
        events[column] = pd.to_datetime(events[column], errors="coerce")

    components = metrics.merge(
        events[["event_id", "official_release_datetime_et"]].drop_duplicates("event_id"),
        on="event_id", how="left", validate="many_to_one",
    )
    components = components.loc[components["surprise"].notna()].copy()
    components["release_date"] = components["official_release_datetime_et"].dt.normalize()
    scales = components.groupby("metric")["surprise"].std(ddof=1)
    components["metric_surprise_scale"] = components["metric"].map(scales)
    components["standardized_surprise"] = np.where(
        components["metric_surprise_scale"].gt(0),
        components["surprise"] / components["metric_surprise_scale"],
        0.0,
    )
    components["equity_direction_sign"] = components["metric"].map(EQUITY_DIRECTION_SIGN)
    components["equity_direction_component"] = (
        components["standardized_surprise"] * components["equity_direction_sign"]
    )
    components["is_pre"] = components["expectation_observation_phase"].eq("pre")
    components["surprise_pp"] = components["surprise"].where(components["unit"].eq("percent"))
    components["surprise_jobs"] = components["surprise"].where(components["unit"].eq("jobs"))
    components["surprise_thousand_jobs"] = components["surprise"].where(components["unit"].eq("thousand_jobs"))
    components["surprise_bps"] = components["surprise"].where(components["unit"].eq("basis_points"))

    if components.empty:
        raise RuntimeError("No metrics contain both actual and expected values")

    daily = (
        components.groupby("release_date", as_index=False)
        .agg(
            event_surprise_mean=("standardized_surprise", "mean"),
            equity_direction_score=("equity_direction_component", "mean"),
            surprise_count=("surprise", "size"),
            pre_surprise_count=("is_pre", "sum"),
            official_release_count=("event_id", "nunique"),
            event_ids=("event_id", join_unique),
            event_types=("event_type", join_unique),
            raw_mean_percentage_points=("surprise_pp", "mean"),
            raw_mean_jobs=("surprise_jobs", "mean"),
            raw_mean_thousand_jobs=("surprise_thousand_jobs", "mean"),
            raw_mean_basis_points=("surprise_bps", "mean"),
        )
    )
    daily["post_surprise_count"] = daily["surprise_count"] - daily["pre_surprise_count"]
    daily_pre = (
        components.loc[components["is_pre"]]
        .groupby("release_date", as_index=False)
        .agg(
            event_surprise_pre_mean=("standardized_surprise", "mean"),
            equity_direction_pre_score=("equity_direction_component", "mean"),
        )
    )
    daily = daily.merge(daily_pre, on="release_date", how="left", validate="one_to_one")

    scaffold_source = events.assign(
        release_date=events["official_release_datetime_et"].dt.normalize()
    )
    scaffold = scaffold_source.groupby("release_date", as_index=False).agg(
        official_release_datetime_et=("official_release_datetime_et", "min"),
        earliest_news_datetime_et=("earliest_news_datetime_et", "min"),
        earliest_news_lag_minutes=("publication_lag_minutes", "min"),
        official_release_count=("event_id", "nunique"),
        event_ids=("event_id", join_unique),
        event_types=("event_type", join_unique),
    )
    result = (
        scaffold.merge(
            daily.drop(columns=["official_release_count", "event_ids", "event_types"]),
            on="release_date", how="left", validate="one_to_one",
        )
        .sort_values("release_date")
        .reset_index(drop=True)
    )
    for column in ("surprise_count", "pre_surprise_count", "post_surprise_count"):
        result[column] = result[column].fillna(0).astype(int)
    result["event_surprise_unit"] = "standard deviations"
    result["equity_label"] = result["equity_direction_score"].map(
        lambda value: equity_label(value, args.threshold)
    )
    result["equity_label_pre"] = result["equity_direction_pre_score"].map(
        lambda value: equity_label(value, args.threshold)
    )
    result["equity_label_basis"] = "conventional broad U.S. equity interpretation"
    columns = [
        "release_date", "official_release_datetime_et", "earliest_news_datetime_et",
        "earliest_news_lag_minutes", "official_release_count", "event_ids", "event_types",
        "event_surprise_mean", "event_surprise_pre_mean", "event_surprise_unit",
        "equity_direction_score", "equity_label", "equity_direction_pre_score",
        "equity_label_pre", "equity_label_basis", "surprise_count",
        "pre_surprise_count", "post_surprise_count", "raw_mean_percentage_points",
        "raw_mean_jobs", "raw_mean_thousand_jobs", "raw_mean_basis_points",
    ]
    result = result[columns]

    daily_path = output_dir / "macro_event_surprises_by_release_date.csv"
    component_path = output_dir / "macro_event_surprise_components.csv"
    result.to_csv(daily_path, index=False, encoding="utf-8-sig")
    components.to_csv(component_path, index=False, encoding="utf-8-sig")
    refresh_intraday_metadata(output_dir, result)
    print(f"Saved {len(result)} release-date rows to {daily_path}")
    print(f"Rows with earliest news: {int(result['earliest_news_datetime_et'].notna().sum())}")
    print(f"Rows with comparable surprise: {int(result['event_surprise_mean'].notna().sum())}")
    print(result["equity_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
