#!/usr/bin/env python3
"""Classify macro-event news candidates with DeepSeek and build analysis tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "macro_event_identification"
MODEL = "deepseek-flash"

METRICS = [
    "headline_mom", "headline_yoy", "core_mom", "core_yoy",
    "unemployment_rate", "nonfarm_payroll_change",
    "target_lower", "target_upper", "rate_change_bps",
    "hold_probability", "cut_probability", "hike_probability",
]
UNITS = ["percent", "percentage_points", "jobs", "thousand_jobs", "basis_points", "probability_percent"]


def normalize_rate_change_sign(
    metric: str | None,
    unit: str | None,
    value: Any,
    evidence: str | None,
    title: str | None = None,
) -> Any:
    """Make FOMC basis-point changes negative for cuts and positive for hikes."""
    if metric != "rate_change_bps" or unit != "basis_points" or not isinstance(value, (int, float)):
        return value
    context = f"{evidence or ''} {title or ''}".casefold()
    cut_terms = ("cut", "lower", "reduc", "easing")
    hike_terms = ("hike", "raise", "increas", "tightening")
    has_cut = any(term in context for term in cut_terms)
    has_hike = any(term in context for term in hike_terms)
    if has_cut and not has_hike:
        return -abs(value)
    if has_hike and not has_cut:
        return abs(value)
    return value

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "event_status": {"type": "string", "enum": ["CONFIRMED", "MANUAL_REVIEW", "NO_FMP_MATCH"]},
        "earliest_result_candidate_id": {"type": "string"},
        "earliest_result_confidence": {"type": "number"},
        "earliest_result_evidence": {"type": "string"},
        "official_metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": METRICS},
                    "reference_period": {"type": "string"},
                    "actual_value": {"type": ["number", "null"]},
                    "unit": {"type": "string", "enum": UNITS},
                    "evidence": {"type": "string"},
                    "source_basis": {"type": "string", "enum": ["official_release", "result_news", "unavailable"]},
                },
                "required": ["metric", "reference_period", "actual_value", "unit", "evidence", "source_basis"],
                "additionalProperties": False,
            },
        },
        "expectation_reports": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "forecast_type": {"type": "string", "enum": ["CONSENSUS", "INDIVIDUAL"]},
                    "metric": {"type": "string", "enum": METRICS},
                    "reference_period": {"type": "string"},
                    "expected_value": {"type": "number"},
                    "unit": {"type": "string", "enum": UNITS},
                    "expectation_source": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "candidate_id", "forecast_type", "metric", "reference_period", "expected_value",
                    "unit", "expectation_source", "evidence", "confidence",
                ],
                "additionalProperties": False,
            },
        },
        "post_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": ["OFFICIAL_RESULT_REPORT", "POST_RELEASE_REACTION", "COMMENTARY", "UNRELATED"],
                    },
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["candidate_id", "classification", "evidence", "confidence"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "event_status", "earliest_result_candidate_id", "earliest_result_confidence",
        "earliest_result_evidence", "official_metrics", "expectation_reports", "post_assessments", "notes",
    ],
    "additionalProperties": False,
}

INSTRUCTIONS = """You are creating a reproducible U.S. macroeconomic event-study dataset.

For each supplied official event, perform three tasks:
1. Extract explicitly supported official ACTUAL metrics. Prefer official_release_text. If it says the page fetch failed, use a clearly matching result-news report and set source_basis=result_news. Never invent a value.
2. From both PRE and POST candidates, extract every explicit numeric expectation for the same event. A POST article may restate a consensus that existed before release (for example, "matching the forecast of economists"); retain it because it is useful for surprise measurement, but its candidate phase will remain POST in the output. CONSENSUS requires a survey, median, consensus, or plural economists/analysts. A named bank, person, or single forecaster is INDIVIDUAL. Do not treat a previous value as an expectation. Keep MoM separate from YoY and headline separate from core.
3. Classify every POST candidate. OFFICIAL_RESULT_REPORT must report at least one actual outcome from this exact release, not merely discuss the indicator, policy outlook, or market reaction. Select the earliest valid OFFICIAL_RESULT_REPORT by publishedDate.

The FMP timestamps are treated as America/New_York wall-clock times. The candidates are already split at the official release time. Do not use PRE candidates as result reports. A POST story that clearly describes the newly released indicator outcome qualifies as OFFICIAL_RESULT_REPORT even when its short FMP snippet omits a number or uses analytical framing. Evidence must be a short exact phrase from the supplied text. Candidate IDs must exactly match supplied IDs. Use an empty string when no earliest result exists and NO_FMP_MATCH when appropriate.

Metric mapping:
- CPI/PCE: headline_mom, headline_yoy, core_mom, core_yoy.
- Employment: unemployment_rate and nonfarm_payroll_change.
- FOMC: target_lower, target_upper, rate_change_bps, and explicit pre-release probabilities.
For jobs, preserve whether the article states jobs or thousand jobs. For rates and inflation, numeric values are percentage values (for example, 3.2 means 3.2%, not 0.032)."""


def output_text(body: dict[str, Any]) -> str:
    if body.get("output_text"):
        return str(body["output_text"])
    for item in body.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                return str(content["text"])
    raise ValueError("response contained no output text")


def slim_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "publishedDate": row["publishedDate"],
        "publisher": row.get("publisher"),
        "title": str(row.get("title") or "")[:500],
        "text": str(row.get("text") or "")[:1200],
    }


def request_event(bundle: dict[str, Any], key: str, endpoint: str) -> dict[str, Any]:
    input_data = {
        "event_id": bundle["event_id"],
        "event_type": bundle["event_type"],
        "reference_period": bundle["reference_period"],
        "release_datetime_et": bundle["release_datetime_et"],
        "official_url": bundle["official_url"],
        "official_release_text": bundle.get("official_text", "")[:12000],
        "pre_candidates": [slim_candidate(row) for row in bundle["pre_candidates"]],
        "post_candidates": [slim_candidate(row) for row in bundle["post_candidates"]],
    }
    payload = {
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "input": json.dumps(input_data, ensure_ascii=False, separators=(",", ":")),
        "reasoning": {"effort": "none"},
        "text": {"format": {"type": "json_schema", "name": "macro_event_detection", "schema": SCHEMA}},
        "max_output_tokens": 6000,
    }
    last_error = ""
    for attempt in range(6):
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=(20, 240),
            )
            if response.status_code == 200:
                body = response.json()
                return {
                    "event_id": bundle["event_id"],
                    "ok": True,
                    "annotation": json.loads(output_text(body)),
                    "usage": body.get("usage") or {},
                    "model": body.get("model", MODEL),
                    "response_id": body.get("id"),
                }
            last_error = f"HTTP {response.status_code}: {response.text[:800]}"
            if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(30, 2**attempt + random.random()))
    return {"event_id": bundle["event_id"], "ok": False, "error": last_error}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(
    bundles: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    event_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    expectation_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for bundle in bundles:
        event_id = bundle["event_id"]
        result = results.get(event_id) or {"ok": False, "error": "missing model result"}
        annotation = result.get("annotation") or {}
        candidates = {
            row["candidate_id"]: row
            for row in bundle["pre_candidates"] + bundle["post_candidates"]
        }
        assessments = {
            row.get("candidate_id"): row
            for row in annotation.get("post_assessments") or []
            if row.get("candidate_id") in candidates
        }
        selected_id = str(annotation.get("earliest_result_candidate_id") or "")
        selected = candidates.get(selected_id)
        if not selected or selected.get("phase") != "post" or assessments.get(selected_id, {}).get("classification") != "OFFICIAL_RESULT_REPORT":
            confirmed = [
                candidates[cid]
                for cid, assessment in assessments.items()
                if assessment.get("classification") == "OFFICIAL_RESULT_REPORT"
                and float(assessment.get("confidence") or 0) >= 0.50
            ]
            selected = min(confirmed, key=lambda row: row["publishedDate"]) if confirmed else None
            selected_id = selected["candidate_id"] if selected else ""

        release_dt = datetime.fromisoformat(bundle["release_datetime_et"])
        selected_dt = datetime.fromisoformat(selected["publishedDate"]) if selected else None
        lag_minutes = round((selected_dt - release_dt).total_seconds() / 60, 2) if selected_dt else None
        status = annotation.get("event_status") if result.get("ok") else "MANUAL_REVIEW"
        selected_assessment_confidence = (
            float(assessments.get(selected_id, {}).get("confidence") or 0)
            if selected_id else 0
        )
        if selected is not None and selected_assessment_confidence >= 0.80:
            status = "CONFIRMED"
        if selected is None and status == "CONFIRMED":
            status = "MANUAL_REVIEW"
        event_rows.append({
            "event_id": event_id,
            "event_type": bundle["event_type"],
            "reference_period": bundle["reference_period"],
            "official_release_datetime_et": bundle["release_datetime_et"],
            "official_source_url": bundle["official_url"],
            "status": status,
            "earliest_news_datetime_et": selected.get("publishedDate") if selected else "",
            "publication_lag_minutes": lag_minutes,
            "earliest_publisher": selected.get("publisher") if selected else "",
            "earliest_title": selected.get("title") if selected else "",
            "earliest_text": selected.get("text") if selected else "",
            "earliest_news_url": selected.get("url") if selected else "",
            "earliest_confidence": selected_assessment_confidence if selected else None,
            "earliest_evidence": annotation.get("earliest_result_evidence") if selected else "",
            "model_notes": " | ".join(annotation.get("notes") or []),
            "model_error": "" if result.get("ok") else result.get("error", ""),
        })

        valid_expectations: list[dict[str, Any]] = []
        for report in annotation.get("expectation_reports") or []:
            candidate = candidates.get(report.get("candidate_id"))
            if not candidate:
                continue
            expected_value = normalize_rate_change_sign(
                report.get("metric"),
                report.get("unit"),
                report.get("expected_value"),
                report.get("evidence"),
                candidate.get("title"),
            )
            row = {
                "event_id": event_id,
                "event_type": bundle["event_type"],
                "reference_period": report.get("reference_period") or bundle["reference_period"],
                "candidate_id": report["candidate_id"],
                "news_datetime_et": candidate.get("publishedDate"),
                "observation_phase": candidate.get("phase"),
                "publisher": candidate.get("publisher"),
                "title": candidate.get("title"),
                "url": candidate.get("url"),
                "forecast_type": report.get("forecast_type"),
                "metric": report.get("metric"),
                "expected_value": expected_value,
                "unit": report.get("unit"),
                "expectation_source": report.get("expectation_source"),
                "evidence": report.get("evidence"),
                "confidence": report.get("confidence"),
            }
            valid_expectations.append(row)
            expectation_rows.append(row)

        official_metrics = annotation.get("official_metrics") or []
        for official_metric in official_metrics:
            actual = normalize_rate_change_sign(
                official_metric.get("metric"),
                official_metric.get("unit"),
                official_metric.get("actual_value"),
                official_metric.get("evidence"),
            )
            if not isinstance(actual, (int, float)):
                continue
            metric = official_metric.get("metric")
            unit = official_metric.get("unit")
            metric_reference_period = official_metric.get("reference_period") or bundle["reference_period"]
            matches = [
                row for row in valid_expectations
                if row["forecast_type"] == "CONSENSUS"
                and row["metric"] == metric
                and row["unit"] == unit
                and row["reference_period"] == metric_reference_period
            ]
            pre_matches = sorted(
                (row for row in matches if row["observation_phase"] == "pre"),
                key=lambda row: row["news_datetime_et"], reverse=True,
            )
            post_matches = sorted(
                (row for row in matches if row["observation_phase"] == "post"),
                key=lambda row: row["news_datetime_et"],
            )
            chosen = pre_matches[0] if pre_matches else (post_matches[0] if post_matches else None)
            expected = chosen.get("expected_value") if chosen else None
            surprise = round(actual - expected, 6) if isinstance(expected, (int, float)) else None
            metric_rows.append({
                "event_id": event_id,
                "event_type": bundle["event_type"],
                "reference_period": metric_reference_period,
                "metric": metric,
                "unit": unit,
                "actual_value": actual,
                "expected_value": expected,
                "surprise": surprise,
                "expectation_source": chosen.get("expectation_source") if chosen else "",
                "expectation_news_datetime_et": chosen.get("news_datetime_et") if chosen else "",
                "expectation_observation_phase": chosen.get("observation_phase") if chosen else "",
                "expectation_news_url": chosen.get("url") if chosen else "",
                "actual_evidence": official_metric.get("evidence"),
                "actual_source_basis": official_metric.get("source_basis"),
            })

        for candidate in bundle["pre_candidates"] + bundle["post_candidates"]:
            assessment = assessments.get(candidate["candidate_id"], {})
            pre_reports = [row for row in valid_expectations if row["candidate_id"] == candidate["candidate_id"]]
            if candidate["phase"] == "pre":
                classification = ";".join(sorted({row["forecast_type"] for row in pre_reports})) or "PRE_OTHER"
                evidence = " | ".join(row["evidence"] for row in pre_reports)
                confidence = max((float(row["confidence"] or 0) for row in pre_reports), default=None)
            else:
                classification = assessment.get("classification", "UNASSESSED")
                evidence = assessment.get("evidence", "")
                confidence = assessment.get("confidence")
            audit_rows.append({
                "event_id": event_id,
                "event_type": bundle["event_type"],
                "reference_period": bundle["reference_period"],
                "official_release_datetime_et": bundle["release_datetime_et"],
                **candidate,
                "rule_reasons": "|".join(candidate.get("rule_reasons") or []),
                "model_classification": classification,
                "model_confidence": confidence,
                "model_evidence": evidence,
                "is_earliest_result": candidate["candidate_id"] == selected_id,
            })

    write_csv(output_dir / "macro_events.csv", event_rows, list(event_rows[0]))
    write_csv(output_dir / "event_metrics_actual_vs_expected.csv", metric_rows, list(metric_rows[0]))
    write_csv(output_dir / "expectation_news.csv", expectation_rows, list(expectation_rows[0]) if expectation_rows else [
                "event_id", "event_type", "reference_period", "candidate_id", "news_datetime_et", "observation_phase", "publisher", "title", "url",
        "forecast_type", "metric", "expected_value", "unit", "expectation_source", "evidence", "confidence",
    ])
    audit_fields = [
        "event_id", "event_type", "reference_period", "official_release_datetime_et", "candidate_id", "phase",
        "rule_score", "rule_reasons", "publishedDate", "publisher", "site", "title", "text", "url",
        "model_classification", "model_confidence", "model_evidence", "is_earliest_result",
    ]
    write_csv(output_dir / "candidate_audit.csv", audit_rows, audit_fields)

    statuses: dict[str, int] = {}
    for row in event_rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    summary = {
        "events": len(event_rows),
        "events_with_earliest_news": sum(bool(row["earliest_news_url"]) for row in event_rows),
        "status_counts": statuses,
        "official_metric_rows": len(metric_rows),
        "metrics_with_consensus_expectation": sum(row["expected_value"] not in (None, "") for row in metric_rows),
        "expectation_report_rows": len(expectation_rows),
        "candidate_audit_rows": len(audit_rows),
        "mean_publication_lag_minutes": statistics.mean(
            row["publication_lag_minutes"] for row in event_rows if isinstance(row["publication_lag_minutes"], (int, float))
        ) if any(isinstance(row["publication_lag_minutes"], (int, float)) for row in event_rows) else None,
    }
    (output_dir / "classification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "workbook_data.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "events": event_rows,
                "metrics": metric_rows,
                "expectations": expectation_rows,
                "audit": audit_rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    bundles = read_jsonl(output_dir / "event_candidate_bundles.jsonl")
    if not bundles:
        raise SystemExit("event_candidate_bundles.jsonl is missing or empty")

    load_dotenv(ROOT / ".env", override=False)
    key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    base = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip().rstrip("/")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is missing from .env")
    endpoint = f"{base}/responses"
    raw_path = output_dir / "deepseek_event_results_v2.jsonl"
    existing = {row["event_id"]: row for row in read_jsonl(raw_path)}
    pending = [bundle for bundle in bundles if not existing.get(bundle["event_id"], {}).get("ok")]
    lock = threading.Lock()
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(request_event, bundle, key, endpoint): bundle["event_id"] for bundle in pending}
            completed = 0
            for future in as_completed(future_map):
                result = future.result()
                existing[result["event_id"]] = result
                with lock, raw_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed += 1
                print(f"classified {completed}/{len(pending)}: {result['event_id']} ok={result.get('ok')}", flush=True)
    summary = build_outputs(bundles, existing, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
