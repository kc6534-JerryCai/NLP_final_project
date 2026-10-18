from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import openpyxl
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek-flash"
TOPICS = [
    "cost_of_living_income",
    "labor_market",
    "household_economy",
    "macro_growth",
    "economic_policy",
]
FRAMES = [
    "household_hardship",
    "economic_resilience",
    "unequal_outcomes",
    "macro_household_disconnect",
    "future_risk",
]
FACT_STATUSES = ["consistent", "mixed", "inconsistent", "not_assessable"]
RELEVANCE = ["relevant", "irrelevant", "uncertain"]
REVIEW_FLAGS = [
    "missing_abstract",
    "truncated_or_corrupted",
    "ambiguous_topic",
    "ambiguous_tone",
    "ambiguous_framing",
    "unclear_relevance",
    "reference_period_uncertain",
    "reference_metric_unavailable",
    "possible_vintage_mismatch",
    "headline_abstract_conflict",
]


def obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


SCHEMA = obj(
    {
        "article_id": {"type": "string"},
        "relevance": {"type": "string", "enum": RELEVANCE},
        "primary_topic": {"anyOf": [{"type": "string", "enum": TOPICS}, {"type": "null"}]},
        "tone": {"anyOf": [{"type": "string", "enum": ["-1", "0", "1"]}, {"type": "null"}]},
        "topic_evidence": {"type": "string"},
        "tone_evidence": {"type": "string"},
        "frames": {
            "type": "array",
            "items": obj(
                {"frame": {"type": "string", "enum": FRAMES}, "evidence": {"type": "string"}},
                ["frame", "evidence"],
            ),
        },
        "factual_calibration": obj(
            {
                "status": {"type": "string", "enum": FACT_STATUSES},
                "claim_evidence": {"type": "string"},
                "reference_matches": {
                    "type": "array",
                    "items": obj(
                        {
                            "metric": {"type": "string"},
                            "period": {"type": "string"},
                            "value": {"type": "number"},
                            "assessment": {"type": "string", "enum": ["consistent", "inconsistent"]},
                        },
                        ["metric", "period", "value", "assessment"],
                    ),
                },
                "note": {"type": "string"},
            },
            ["status", "claim_evidence", "reference_matches", "note"],
        ),
        "needs_review": {"type": "boolean"},
        "review_flags": {"type": "array", "items": {"type": "string", "enum": REVIEW_FLAGS}},
    },
    [
        "article_id",
        "relevance",
        "primary_topic",
        "tone",
        "topic_evidence",
        "tone_evidence",
        "frames",
        "factual_calibration",
        "needs_review",
        "review_flags",
    ],
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            rows.append(
                {
                    "article_id": str(source["article_id"]),
                    "headline": str(source.get("headline") or source.get("title") or ""),
                    "abstract": str(source.get("abstract") or ""),
                    "published_at": source.get("published_at"),
                    "month": source.get("month") or str(source.get("published_at") or "")[:7],
                    "duplicate_group_id": source.get("duplicate_group_id"),
                }
            )
    return rows


def excel_records(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[sheet_name]
    values = sheet.iter_rows(values_only=True)
    headers = [str(value) for value in next(values)]
    records = [dict(zip(headers, row)) for row in values if any(value is not None for value in row)]
    workbook.close()
    return records


def period_key(value: Any, quarterly: bool = False) -> str:
    if isinstance(value, datetime):
        if quarterly:
            return f"{value.year}Q{((value.month - 1) // 3) + 1}"
        return value.strftime("%Y-%m")
    return str(value)[:7]


class EconomicReference:
    def __init__(self, path: Path):
        monthly = excel_records(path, "Monthly")
        quarterly = excel_records(path, "Quarterly GDP")
        self.monthly = {period_key(row["month"]): row for row in monthly}
        self.quarterly = {str(row["quarter"]): row for row in quarterly}

    @staticmethod
    def _shift_month(month: str, delta: int) -> str:
        year, number = map(int, month.split("-"))
        serial = year * 12 + number - 1 + delta
        return f"{serial // 12:04d}-{serial % 12 + 1:02d}"

    @staticmethod
    def _quarter(month: str) -> tuple[int, int]:
        year, number = map(int, month.split("-"))
        return year, (number - 1) // 3 + 1

    @staticmethod
    def _shift_quarter(year: int, quarter: int, delta: int) -> str:
        serial = year * 4 + quarter - 1 + delta
        return f"{serial // 4}Q{serial % 4 + 1}"

    @staticmethod
    def _compact(row: dict[str, Any] | None, fields: list[str]) -> dict[str, Any] | None:
        if row is None:
            return None
        return {key: round(value, 4) if isinstance(value, float) else value for key in fields if (value := row.get(key)) is not None}

    def context(self, month: str) -> dict[str, Any]:
        monthly_fields = [
            "month",
            "michigan_sentiment",
            "michigan_current_conditions",
            "michigan_expectations",
            "survey_status",
            "u3_unemployment_pct_sa",
            "pce_inflation_yoy_pct",
            "real_disposable_income_bn2017usd_saar",
            "u6_underutilization_pct_sa",
            "pce_cumulative_since_jan2010_pct",
        ]
        gdp_fields = [
            "quarter",
            "real_gdp_bn2017usd_saar",
            "real_gdp_qoq_annualized_official_pct",
            "real_gdp_yoy_pct",
        ]
        year, quarter = self._quarter(month)
        monthly_keys = [self._shift_month(month, delta) for delta in (-12, -2, -1, 0)]
        quarterly_keys = [self._shift_quarter(year, quarter, delta) for delta in (-4, -2, -1, 0)]
        return {
            "snapshot_as_of": "2026-10-14",
            "vintage_warning": "current revised series; observation period is not release date",
            "monthly": [self._compact(self.monthly.get(key), monthly_fields) for key in monthly_keys if key in self.monthly],
            "quarterly_gdp": [self._compact(self.quarterly.get(key), gdp_fields) for key in quarterly_keys if key in self.quarterly],
        }


def response_text(body: dict[str, Any]) -> str:
    parts = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "".join(parts)


def validate(row: dict[str, Any], annotation: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    source = f"{row['headline']}\n{row['abstract']}"
    if annotation.get("article_id") != row["article_id"]:
        issues.append("article_id_mismatch")
    relevance = annotation.get("relevance")
    if relevance not in RELEVANCE:
        issues.append("invalid_relevance")
    if relevance == "relevant":
        if annotation.get("primary_topic") not in TOPICS:
            issues.append("invalid_topic")
        if str(annotation.get("tone")) not in {"-1", "0", "1"}:
            issues.append("invalid_tone")
        for field in ("topic_evidence", "tone_evidence"):
            evidence = annotation.get(field)
            if not isinstance(evidence, str) or not evidence or evidence not in source:
                issues.append(f"invalid_{field}")
    else:
        if annotation.get("primary_topic") is not None or annotation.get("tone") is not None:
            issues.append("nonrelevant_nonnull_label")
        if annotation.get("topic_evidence") != "" or annotation.get("tone_evidence") != "":
            issues.append("nonrelevant_nonempty_evidence")
    frames = annotation.get("frames")
    if not isinstance(frames, list):
        issues.append("invalid_frames")
    else:
        seen = set()
        for frame in frames:
            label, evidence = frame.get("frame"), frame.get("evidence")
            if label not in FRAMES or label in seen:
                issues.append("invalid_frame_label")
            seen.add(label)
            if not isinstance(evidence, str) or not evidence or evidence not in source:
                issues.append("invalid_frame_evidence")
    if relevance != "relevant" and frames:
        issues.append("nonrelevant_with_frames")
    fact = annotation.get("factual_calibration") or {}
    status = fact.get("status")
    matches = fact.get("reference_matches")
    claim = fact.get("claim_evidence")
    if status not in FACT_STATUSES or not isinstance(matches, list):
        issues.append("invalid_factual_calibration")
    elif status == "not_assessable":
        if claim != "" or matches:
            issues.append("not_assessable_with_evidence")
    elif not isinstance(claim, str) or not claim or claim not in source or not matches:
        issues.append("assessed_without_valid_support")
    if relevance != "relevant" and status != "not_assessable":
        issues.append("nonrelevant_assessed_fact")
    flags = annotation.get("review_flags")
    if not isinstance(flags, list) or any(flag not in REVIEW_FLAGS for flag in flags):
        issues.append("invalid_review_flags")
        flags = []
    if annotation.get("needs_review") != bool(flags):
        issues.append("review_boolean_mismatch")
    if relevance == "uncertain" and "unclear_relevance" not in flags:
        issues.append("uncertain_without_flag")
    return sorted(set(issues))


def request_one(row: dict[str, Any], prompt: str, context: dict[str, Any], key: str, endpoint: str) -> dict[str, Any]:
    input_data = {
        "article_id": row["article_id"],
        "published_at": row["published_at"],
        "headline": row["headline"],
        "abstract": row["abstract"],
        "economic_reference": context,
    }
    payload = {
        "model": MODEL,
        "instructions": prompt,
        "input": json.dumps(input_data, ensure_ascii=False, separators=(",", ":"), default=str),
        "reasoning": {"effort": "none"},
        "text": {"format": {"type": "json_schema", "name": "full_economic_annotation", "schema": SCHEMA}},
        "max_output_tokens": 1000,
    }
    last_error = ""
    last_result: dict[str, Any] | None = None
    validation_attempt = 0
    usage_total = Counter()
    for attempt in range(1, 7):
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=(15, 180),
            )
            if response.status_code == 200:
                body = response.json()
                response_usage = body.get("usage") or {}
                usage_total["input_tokens"] += int(response_usage.get("input_tokens") or 0)
                usage_total["output_tokens"] += int(response_usage.get("output_tokens") or 0)
                usage_details = response_usage.get("input_tokens_details") or {}
                usage_total["cached_input_tokens"] += int(
                    usage_details.get("cached_tokens") or response_usage.get("cached_input_tokens") or 0
                )
                annotation = json.loads(response_text(body))
                raw_tone = annotation.get("tone")
                annotation["tone"] = int(raw_tone) if str(raw_tone) in {"-1", "0", "1"} else raw_tone
                issues = validate(row, annotation)
                result = {
                    **row,
                    "ok": True,
                    "model": body.get("model", MODEL),
                    "response_id": body.get("id"),
                    "annotation": annotation,
                    "validation_issues": issues,
                    "normalization_actions": ["tone_string_to_integer"] if isinstance(raw_tone, str) else [],
                    "usage": dict(usage_total),
                }
                last_result = result
                if issues and validation_attempt < 2:
                    validation_attempt += 1
                    input_data["previous_output_validation_errors"] = issues
                    input_data["instruction"] = "Regenerate the full annotation and correct every listed validation error."
                    payload["input"] = json.dumps(input_data, ensure_ascii=False, separators=(",", ":"), default=str)
                    continue
                return result
            last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
            if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(30, 2 ** (attempt - 1) + random.random()))
    if last_result is not None:
        last_result["usage"] = dict(usage_total)
        return last_result
    return {**row, "ok": False, "error": last_error}


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def aggregate(path: Path, total_source_rows: int) -> dict[str, Any]:
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    latest_by_id = {row["article_id"]: row for row in rows}
    latest = list(latest_by_id.values())
    successful = [row for row in latest if row.get("ok")]
    usage = Counter()
    # The file is an append-only request log. Sum all successful calls so retry
    # costs remain counted, while label distributions use only the latest row.
    for row in rows:
        if not row.get("ok"):
            continue
        item = row.get("usage") or {}
        usage["input_tokens"] += int(item.get("input_tokens") or 0)
        usage["output_tokens"] += int(item.get("output_tokens") or 0)
        details = item.get("input_tokens_details") or {}
        usage["cached_input_tokens"] += int(details.get("cached_tokens") or item.get("cached_input_tokens") or 0)
    uncached = max(0, usage["input_tokens"] - usage["cached_input_tokens"])
    peak = usage["cached_input_tokens"] / 1_000_000 * 0.006 + uncached / 1_000_000 * 0.30 + usage["output_tokens"] / 1_000_000 * 1.20
    validation_counts = Counter(issue for row in successful for issue in row.get("validation_issues") or [])
    topic_counts = Counter(row["annotation"].get("primary_topic") for row in successful)
    tone_counts = Counter(str(row["annotation"].get("tone")) for row in successful)
    relevance_counts = Counter(row["annotation"].get("relevance") for row in successful)
    frame_counts = Counter(frame["frame"] for row in successful for frame in row["annotation"].get("frames") or [])
    fact_counts = Counter(row["annotation"].get("factual_calibration", {}).get("status") for row in successful)
    return {
        "source_total": total_source_rows,
        "request_log_rows": len(rows),
        "attempted": len(latest),
        "successful": len(successful),
        "failed": len(latest) - len(successful),
        "validation_issue_rows": sum(bool(row.get("validation_issues")) for row in successful),
        "validation_issue_counts": dict(validation_counts),
        "relevance_counts": dict(relevance_counts),
        "topic_counts": dict(topic_counts),
        "tone_counts": dict(tone_counts),
        "frame_counts": dict(frame_counts),
        "factual_calibration_counts": dict(fact_counts),
        "usage": dict(usage),
        "estimated_peak_cost_usd": peak,
        "estimated_off_peak_cost_usd": peak / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/clean/relevant_only.jsonl")
    parser.add_argument("--reference", type=Path, default=ROOT / "Consumer_Sentiment_Economic_Data_2010_2026.xlsx")
    parser.add_argument(
        "--base-prompt",
        type=Path,
        default=ROOT / "data/annotations/gpt6_luna_compact_10000/prompt_used.txt",
    )
    parser.add_argument(
        "--extra-prompt",
        type=Path,
        default=ROOT / "prompts/deepseek_full_relevant_v1_2026-10-15.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/all_relevant/deepseek")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is missing")
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    endpoint = f"{base}/responses"
    all_rows = read_rows(args.input)
    rows = all_rows
    if args.limit is not None and args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "annotations.jsonl"
    progress_path = args.output_dir / "progress.json"
    report_path = args.output_dir / "report.json"
    completed = set()
    if output_path.exists():
        existing_by_id = {}
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    existing_by_id[item["article_id"]] = item
        completed = {
            article_id
            for article_id, item in existing_by_id.items()
            if item.get("ok") and not item.get("validation_issues")
        }
    pending = [row for row in rows if row["article_id"] not in completed]
    base_prompt = args.base_prompt.read_text(encoding="utf-8-sig").rstrip()
    extra_prompt = args.extra_prompt.read_text(encoding="utf-8-sig").rstrip()
    prompt = f"{base_prompt}\n\n{extra_prompt}\n"
    (args.output_dir / "prompt_used.txt").write_text(prompt, encoding="utf-8")
    reference = EconomicReference(args.reference)
    lock = threading.Lock()
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(request_one, row, prompt, reference.context(row["month"]), key, endpoint): row
            for row in pending
        }
        for completed_now, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            append_jsonl(output_path, result, lock)
            if completed_now % 25 == 0 or completed_now == len(pending):
                report = aggregate(output_path, len(all_rows))
                report.update(
                    {
                        "status": "running" if completed_now < len(pending) else "complete_for_requested_scope",
                        "requested_scope": len(rows),
                        "requested_scope_completed": len(completed) + completed_now,
                        "elapsed_seconds": time.time() - started,
                    }
                )
                progress_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = aggregate(output_path, len(all_rows))
    fully_valid = report["successful"] == len(all_rows) and report["validation_issue_rows"] == 0
    report.update(
        {
            "status": "complete" if fully_valid else "incomplete",
            "requested_scope": len(rows),
            "base_prompt": str(args.base_prompt.relative_to(ROOT)),
            "extra_prompt": str(args.extra_prompt.relative_to(ROOT)),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "reference": str(args.reference.relative_to(ROOT)),
            "model": MODEL,
            "elapsed_seconds": time.time() - started,
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
