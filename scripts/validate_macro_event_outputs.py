#!/usr/bin/env python3
"""Validate the event-identification CSV outputs."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "macro_event_identification"


def read(name: str) -> list[dict[str, str]]:
    with (BASE / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    events = read("macro_events.csv")
    metrics = read("event_metrics_actual_vs_expected.csv")
    expectations = read("expectation_news.csv")
    audit = read("candidate_audit.csv")
    issues: list[str] = []

    event_ids = [row["event_id"] for row in events]
    if len(event_ids) != len(set(event_ids)):
        issues.append("duplicate event IDs")
    releases = {
        row["event_id"]: datetime.fromisoformat(row["official_release_datetime_et"])
        for row in events
    }
    for row in events:
        if row["earliest_news_datetime_et"]:
            published = datetime.fromisoformat(row["earliest_news_datetime_et"])
            if published < releases[row["event_id"]]:
                issues.append(f"result before release: {row['event_id']}")
    for row in expectations:
        published = datetime.fromisoformat(row["news_datetime_et"])
        phase = row.get("observation_phase", "")
        if phase == "pre" and published >= releases[row["event_id"]]:
            issues.append(f"pre expectation not before release: {row['event_id']} {row['candidate_id']}")
        elif phase == "post" and published < releases[row["event_id"]]:
            issues.append(f"post expectation before release: {row['event_id']} {row['candidate_id']}")
        elif phase not in {"pre", "post"}:
            issues.append(f"unknown expectation phase: {row['event_id']} {row['candidate_id']}")

    selected_counts = Counter(
        row["event_id"] for row in audit if row["is_earliest_result"].lower() == "true"
    )
    for row in events:
        expected_count = 1 if row["earliest_news_url"] else 0
        if selected_counts[row["event_id"]] != expected_count:
            issues.append(
                f"earliest marker count: {row['event_id']} expected={expected_count} "
                f"actual={selected_counts[row['event_id']]}"
            )
    for row in audit:
        if row["is_earliest_result"].lower() == "true" and row["model_classification"] != "OFFICIAL_RESULT_REPORT":
            issues.append(f"selected candidate has wrong class: {row['event_id']} {row['candidate_id']}")

    metric_keys = [
        (row["event_id"], row["reference_period"], row["metric"], row["unit"])
        for row in metrics
    ]
    if len(metric_keys) != len(set(metric_keys)):
        duplicates = [key for key, count in Counter(metric_keys).items() if count > 1]
        issues.append(f"duplicate metric keys: {duplicates[:10]}")
    for row in metrics:
        phase = row.get("expectation_observation_phase", "")
        if row["expected_value"] and phase not in {"pre", "post"}:
            issues.append(f"expected value without valid phase: {row['event_id']} {row['metric']}")
        if row["actual_value"] and row["expected_value"]:
            expected_surprise = float(row["actual_value"]) - float(row["expected_value"])
            if abs(float(row["surprise"]) - expected_surprise) > 1e-9:
                issues.append(f"surprise arithmetic: {row['event_id']} {row['metric']}")

    print(
        f"events={len(events)} metrics={len(metrics)} expectations={len(expectations)} "
        f"audit={len(audit)} issues={len(issues)}"
    )
    for issue in issues:
        print(issue)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
