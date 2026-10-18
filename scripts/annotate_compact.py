from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "clean" / "relevant_only.jsonl"
DEFAULT_OLD_MANIFEST = PROJECT_ROOT / "data" / "annotations" / "gpt6_luna_5000" / "batch_manifest.jsonl"
DEFAULT_PROMPT = PROJECT_ROOT / "config" / "annotation_compact.v2.zh.txt"
DEFAULT_RUN_DIR = PROJECT_ROOT / "data" / "annotations" / "gpt6_luna_compact_10000"
API_BASE = "https://api.openai.com/v1"
MODEL = "gpt-6-luna"
SEED = "media-indicator-v2"

TOPICS = [
    "cost_of_living_income",
    "labor_market",
    "household_economy",
    "macro_growth",
    "economic_policy",
]
RELEVANCE = ["relevant", "irrelevant", "uncertain"]
REVIEW_FLAGS = [
    "missing_abstract",
    "truncated_or_corrupted",
    "unclear_relevance",
    "ambiguous_topic",
    "ambiguous_tone",
    "headline_abstract_conflict",
]


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


ANNOTATION_SCHEMA = _object(
    {
        "article_id": {"type": "string"},
        "relevance": {"type": "string", "enum": RELEVANCE},
        "primary_topic": {
            "anyOf": [
                {"type": "string", "enum": TOPICS},
                {"type": "null"},
            ]
        },
        "tone": {
            "anyOf": [
                {"type": "integer", "enum": [-1, 0, 1]},
                {"type": "null"},
            ]
        },
        "topic_evidence": {"type": "string"},
        "tone_evidence": {"type": "string"},
        "needs_review": {"type": "boolean"},
        "review_flags": {
            "type": "array",
            "items": {"type": "string", "enum": REVIEW_FLAGS},
        },
    },
    [
        "article_id",
        "relevance",
        "primary_topic",
        "tone",
        "topic_evidence",
        "tone_evidence",
        "needs_review",
        "review_flags",
    ],
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_rank(value: str, salt: str) -> str:
    return sha256_text(f"{salt}|{value}")


def redact(value: str, secret: str = "") -> str:
    if secret:
        value = value.replace(secret, "<redacted>")
    return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<redacted>", value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env or the environment")
    return key


def article_type(row: dict[str, Any]) -> str:
    return str(row.get("article_type") or "unknown")


def month_of(row: dict[str, Any]) -> str:
    published = str(row.get("published_at") or "")
    if not re.match(r"^\d{4}-\d{2}", published):
        raise ValueError(f"Invalid published_at for {row.get('article_id')}: {published!r}")
    return published[:7]


def manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "article_id": row["article_id"],
        "headline": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "published_at": row.get("published_at", ""),
        "month": month_of(row),
        "article_type": article_type(row),
        "section": row.get("section", ""),
        "duplicate_group_id": row.get("duplicate_group_id", ""),
    }


def proportional_select(rows: list[dict[str, Any]], quota: int, salt: str) -> list[dict[str, Any]]:
    if quota > len(rows):
        raise ValueError(f"Quota {quota} exceeds stratum size {len(rows)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[article_type(row)].append(row)
    exact = {key: quota * len(value) / len(rows) for key, value in groups.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remaining = quota - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda key: (
            -(exact[key] - allocation[key]),
            stable_rank(key, f"{salt}|type-tie"),
        ),
    )
    for key in order[:remaining]:
        allocation[key] += 1
    chosen: list[dict[str, Any]] = []
    for key in sorted(groups):
        ranked = sorted(
            groups[key],
            key=lambda row: stable_rank(str(row["article_id"]), f"{salt}|{key}"),
        )
        chosen.extend(ranked[: allocation[key]])
    if len(chosen) != quota:
        raise AssertionError(f"Selected {len(chosen)} rows, expected {quota}")
    return sorted(chosen, key=lambda row: stable_rank(str(row["article_id"]), f"{salt}|order"))


def counter_dict(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def prepare(args: argparse.Namespace) -> int:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text(encoding="utf-8-sig").strip() + "\n"
    source = read_jsonl(args.dataset)
    old_manifest = read_jsonl(args.old_manifest)

    by_id: dict[str, dict[str, Any]] = {}
    duplicate_source_ids: list[str] = []
    for row in source:
        article_id = str(row.get("article_id") or "")
        if not article_id:
            raise ValueError("Source contains a row without article_id")
        if article_id in by_id:
            duplicate_source_ids.append(article_id)
        else:
            by_id[article_id] = row
    old_ids = {str(row["article_id"]) for row in old_manifest}
    missing_old_ids = sorted(old_ids - set(by_id))
    old_groups = {
        str(by_id[article_id].get("duplicate_group_id") or "")
        for article_id in old_ids & set(by_id)
        if by_id[article_id].get("duplicate_group_id")
    }

    excluded_old_id = 0
    excluded_old_group = 0
    eligible: list[dict[str, Any]] = []
    for article_id, row in by_id.items():
        group = str(row.get("duplicate_group_id") or "")
        if article_id in old_ids:
            excluded_old_id += 1
            continue
        if group and group in old_groups:
            excluded_old_group += 1
            continue
        eligible.append(row)

    # Keep at most one article per duplicate group in the new pool.
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        group = str(row.get("duplicate_group_id") or f"id:{row['article_id']}")
        by_group[group].append(row)
    deduped = [
        min(rows, key=lambda row: stable_rank(str(row["article_id"]), f"{SEED}|dedupe"))
        for rows in by_group.values()
    ]
    removed_within_pool = len(eligible) - len(deduped)

    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deduped:
        by_month[month_of(row)].append(row)
    months = sorted(by_month)
    if args.count < len(months):
        raise ValueError(f"Count {args.count} is smaller than {len(months)} month strata")
    base, remainder = divmod(args.count, len(months))
    extra_months = set(
        sorted(months, key=lambda month: stable_rank(month, f"{SEED}|month-extra"))[:remainder]
    )
    month_quotas = {month: base + int(month in extra_months) for month in months}
    undersized = {
        month: {"available": len(by_month[month]), "quota": month_quotas[month]}
        for month in months
        if len(by_month[month]) < month_quotas[month]
    }
    # Preserve naturally short partial months and spread their deficit as evenly as
    # possible over months with remaining capacity. Never duplicate a record.
    deficit = 0
    for month, sizes in undersized.items():
        deficit += sizes["quota"] - sizes["available"]
        month_quotas[month] = sizes["available"]
    while deficit:
        candidates = [month for month in months if month_quotas[month] < len(by_month[month])]
        if not candidates:
            raise ValueError(f"Not enough eligible records to fill {args.count} rows")
        month = min(
            candidates,
            key=lambda value: (
                month_quotas[value],
                stable_rank(value, f"{SEED}|month-deficit"),
            ),
        )
        month_quotas[month] += 1
        deficit -= 1

    selected: list[dict[str, Any]] = []
    for month in months:
        selected.extend(
            proportional_select(by_month[month], month_quotas[month], f"{SEED}|month|{month}")
        )
    selected = sorted(selected, key=lambda row: (month_of(row), stable_rank(str(row["article_id"]), f"{SEED}|final")))

    # Pilot: year-balanced sample from the frozen 10,000, proportional by article type within year.
    by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_year[month_of(row)[:4]].append(row)
    years = sorted(by_year)
    pilot_base, pilot_remainder = divmod(args.pilot_count, len(years))
    extra_years = set(
        sorted(years, key=lambda year: stable_rank(year, f"{SEED}|pilot-year-extra"))[:pilot_remainder]
    )
    pilot: list[dict[str, Any]] = []
    for year in years:
        quota = pilot_base + int(year in extra_years)
        pilot.extend(proportional_select(by_year[year], quota, f"{SEED}|pilot|{year}"))
    pilot = sorted(pilot, key=lambda row: stable_rank(str(row["article_id"]), f"{SEED}|pilot-order"))
    repeats = sorted(
        pilot, key=lambda row: stable_rank(str(row["article_id"]), f"{SEED}|repeat")
    )[: args.repeat_count]

    selected_manifest = [manifest_row(row) for row in selected]
    pilot_manifest = [
        {"request_id": f"pilot-{index:04d}", "is_repeat": False, **manifest_row(row)}
        for index, row in enumerate(pilot, 1)
    ]
    pilot_id_to_request = {row["article_id"]: row["request_id"] for row in pilot_manifest}
    repeat_manifest = [
        {
            "request_id": f"repeat-{index:04d}",
            "is_repeat": True,
            "original_request_id": pilot_id_to_request[str(row["article_id"])],
            **manifest_row(row),
        }
        for index, row in enumerate(repeats, 1)
    ]

    selected_ids = {row["article_id"] for row in selected_manifest}
    selected_groups = [row["duplicate_group_id"] for row in selected_manifest if row["duplicate_group_id"]]
    audit = {
        "created_at": utc_now(),
        "selection_seed": SEED,
        "source_path": str(args.dataset.resolve()),
        "old_manifest_path": str(args.old_manifest.resolve()),
        "source_rows": len(source),
        "source_unique_ids": len(by_id),
        "source_duplicate_id_count": len(duplicate_source_ids),
        "old_unique_ids": len(old_ids),
        "old_ids_missing_from_source": missing_old_ids,
        "excluded_old_ids": excluded_old_id,
        "excluded_old_duplicate_groups": excluded_old_group,
        "eligible_before_new_dedupe": len(eligible),
        "removed_new_duplicate_group_members": removed_within_pool,
        "eligible_unique_groups": len(deduped),
        "selected_count": len(selected_manifest),
        "selected_unique_ids": len(selected_ids),
        "selected_unique_duplicate_groups": len(set(selected_groups)),
        "old_id_overlap": len(selected_ids & old_ids),
        "old_duplicate_group_overlap": len(set(selected_groups) & old_groups),
        "month_count": len(months),
        "capacity_constrained_months": undersized,
        "month_quota_distribution": counter_dict(month_quotas.values()),
        "selected_by_month": counter_dict(row["month"] for row in selected_manifest),
        "selected_by_year": counter_dict(row["month"][:4] for row in selected_manifest),
        "selected_by_article_type": counter_dict(row["article_type"] for row in selected_manifest),
        "pilot_unique_count": len(pilot_manifest),
        "pilot_repeat_count": len(repeat_manifest),
        "pilot_by_year": counter_dict(row["month"][:4] for row in pilot_manifest),
        "pilot_by_article_type": counter_dict(row["article_type"] for row in pilot_manifest),
    }
    if not (
        len(selected_manifest) == args.count
        and len(selected_ids) == args.count
        and len(set(selected_groups)) == len(selected_groups)
        and audit["old_id_overlap"] == 0
        and audit["old_duplicate_group_overlap"] == 0
        and len(pilot_manifest) == args.pilot_count
        and len(repeat_manifest) == args.repeat_count
    ):
        raise AssertionError(f"Selection invariant failed: {audit}")

    write_jsonl(args.run_dir / "selection_manifest.jsonl", selected_manifest)
    write_jsonl(args.run_dir / "pilot_manifest.jsonl", pilot_manifest)
    write_jsonl(args.run_dir / "pilot_repeat_manifest.jsonl", repeat_manifest)
    (args.run_dir / "selection_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.run_dir / "prompt_used.txt").write_text(prompt, encoding="utf-8")
    (args.run_dir / "schema.json").write_text(
        json.dumps(ANNOTATION_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "created_at": utc_now(),
        "model": MODEL,
        "reasoning_effort": "none",
        "service_mode": "standard_responses_api",
        "batch_api_used": False,
        "prompt_sha256": sha256_text(prompt),
        "schema_sha256": sha256_text(json.dumps(ANNOTATION_SCHEMA, sort_keys=True)),
        "selection_manifest_sha256": sha256_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected_manifest)
        ),
        "selection_seed": SEED,
    }
    (args.run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return 0


def article_payload(row: dict[str, Any]) -> dict[str, str]:
    return {
        "article_id": str(row["article_id"]),
        "headline": str(row.get("headline") or ""),
        "abstract": str(row.get("abstract") or ""),
    }


def request_body(prompt: str, row: dict[str, Any], service_tier: str | None) -> dict[str, Any]:
    prompt_hash = sha256_text(prompt)
    body: dict[str, Any] = {
        "model": MODEL,
        "instructions": prompt,
        "input": json.dumps(article_payload(row), ensure_ascii=False, separators=(",", ":")),
        "reasoning": {"effort": "none"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "compact_economic_annotation",
                "strict": True,
                "schema": ANNOTATION_SCHEMA,
            }
        },
        "max_output_tokens": 512,
        "prompt_cache_key": f"compact-economic-labels-{prompt_hash[:24]}",
        "prompt_cache_options": {"ttl": "30m"},
        "store": False,
    }
    if service_tier:
        body["service_tier"] = service_tier
    return body


def output_text(body: dict[str, Any]) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    chunks: list[str] = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    if not chunks:
        raise ValueError("Response contained no output_text")
    return "".join(chunks)


def parse_json_output(raw_text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse one leading JSON object and explicitly flag any trailing model text."""
    stripped = raw_text.lstrip()
    parsed, end = json.JSONDecoder().raw_decode(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Structured output was not a JSON object")
    issues = ["response_trailing_text"] if stripped[end:].strip() else []
    return parsed, issues


def post_response(session: requests.Session, key: str, body: dict[str, Any]) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_error = ""
    for attempt in range(1, 7):
        retryable = False
        try:
            response = session.post(f"{API_BASE}/responses", headers=headers, json=body, timeout=180)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            retryable = True
        else:
            if response.status_code < 400:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:2000]}"
            retryable = response.status_code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable:
                break
        if attempt < 6 and retryable:
            time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
    raise RuntimeError(redact(last_error, key))


def validate_annotation(annotation: dict[str, Any], row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if annotation.get("article_id") != row["article_id"]:
        issues.append("article_id_mismatch")
    relevance = annotation.get("relevance")
    primary_topic = annotation.get("primary_topic")
    tone = annotation.get("tone")
    flags = annotation.get("review_flags") or []
    if len(flags) != len(set(flags)):
        issues.append("duplicate_review_flags")
    if bool(flags) != bool(annotation.get("needs_review")):
        issues.append("needs_review_flags_mismatch")
    if relevance == "relevant":
        if primary_topic not in TOPICS:
            issues.append("relevant_invalid_topic")
        if tone not in {-1, 0, 1} or isinstance(tone, bool):
            issues.append("relevant_invalid_tone")
        if not annotation.get("topic_evidence"):
            issues.append("relevant_missing_topic_evidence")
        if not annotation.get("tone_evidence"):
            issues.append("relevant_missing_tone_evidence")
    elif relevance in {"irrelevant", "uncertain"}:
        for field, expected in {
            "primary_topic": None,
            "tone": None,
            "topic_evidence": "",
            "tone_evidence": "",
        }.items():
            if annotation.get(field) != expected:
                issues.append(f"nonrelevant_{field}_invalid")
        if relevance == "uncertain" and "unclear_relevance" not in flags:
            issues.append("uncertain_without_unclear_relevance")
    else:
        issues.append("invalid_relevance")
    source = f"{row.get('headline', '')}\n{row.get('abstract', '')}"
    for field in ("topic_evidence", "tone_evidence"):
        evidence = annotation.get(field, "")
        if evidence and evidence not in source:
            issues.append(f"{field}_not_exact_substring")
    return issues


def usage_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for record in records:
        usage = record.get("usage") or {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            totals[key] += int(usage.get(key) or 0)
        details = usage.get("input_tokens_details") or {}
        totals["cached_tokens"] += int(details.get("cached_tokens") or 0)
        totals["cache_write_tokens"] += int(details.get("cache_write_tokens") or 0)
    return dict(totals)


def attempt_weighted_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    """Approximate billed usage when earlier retry responses were not separately retained."""
    totals = Counter()
    for record in records:
        multiplier = max(1, int(record.get("attempts") or 1))
        usage = record.get("usage") or {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            totals[key] += int(usage.get(key) or 0) * multiplier
        details = usage.get("input_tokens_details") or {}
        totals["cached_tokens"] += int(details.get("cached_tokens") or 0) * multiplier
        totals["cache_write_tokens"] += int(details.get("cache_write_tokens") or 0) * multiplier
    return dict(totals)


def fast_cost_usd(usage: dict[str, int]) -> float:
    """GPT-6 Luna Fast short-context rates as of 2026-10-15, USD per 1M tokens."""
    cached = int(usage.get("cached_tokens") or 0)
    cache_write = int(usage.get("cache_write_tokens") or 0)
    input_tokens = int(usage.get("input_tokens") or 0)
    uncached = max(0, input_tokens - cached - cache_write)
    output = int(usage.get("output_tokens") or 0)
    return round(
        (uncached * 0.20 + cached * 0.02 + cache_write * 0.25 + output * 1.00)
        / 1_000_000,
        6,
    )


def summarize_pilot(run_dir: Path) -> dict[str, Any]:
    records = read_jsonl(run_dir / "pilot_results.jsonl")
    latest = {str(record["request_id"]): record for record in records}
    originals = [record for record in latest.values() if not record.get("is_repeat")]
    repeats = [record for record in latest.values() if record.get("is_repeat")]
    successful_originals = [record for record in originals if record.get("ok")]
    successful_repeats = [record for record in repeats if record.get("ok")]
    annotations = [record["annotation"] for record in successful_originals]
    issue_counter = Counter(
        issue for record in latest.values() for issue in (record.get("validation_issues") or [])
    )

    original_by_request = {record["request_id"]: record for record in successful_originals}
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for repeat in successful_repeats:
        original = original_by_request.get(str(repeat.get("original_request_id")))
        if original:
            paired.append((original["annotation"], repeat["annotation"]))

    def agreement(field: str, eligible_relevance: bool = False) -> dict[str, Any]:
        compared = 0
        agreed = 0
        for first, second in paired:
            if eligible_relevance and not (
                first.get("relevance") == "relevant" and second.get("relevance") == "relevant"
            ):
                continue
            compared += 1
            agreed += int(first.get(field) == second.get(field))
        return {
            "compared": compared,
            "agreed": agreed,
            "rate": round(agreed / compared, 4) if compared else None,
        }

    full_agreed = sum(
        int(
            (first.get("relevance"), first.get("primary_topic"), first.get("tone"))
            == (second.get("relevance"), second.get("primary_topic"), second.get("tone"))
        )
        for first, second in paired
    )
    nonrepeat_usage = usage_totals(successful_originals)
    projected_remaining_usage = {
        key: round(value / len(successful_originals) * 9800)
        for key, value in nonrepeat_usage.items()
    } if successful_originals else {}
    summary = {
        "generated_at": utc_now(),
        "model": MODEL,
        "requested_unique": 200,
        "requested_blind_repeats": 50,
        "successful_unique": len(successful_originals),
        "failed_unique": len(originals) - len(successful_originals),
        "successful_repeats": len(successful_repeats),
        "failed_repeats": len(repeats) - len(successful_repeats),
        "relevance": counter_dict(annotation.get("relevance") for annotation in annotations),
        "primary_topic_relevant_only": counter_dict(
            annotation.get("primary_topic")
            for annotation in annotations
            if annotation.get("relevance") == "relevant"
        ),
        "tone_relevant_only": counter_dict(
            annotation.get("tone")
            for annotation in annotations
            if annotation.get("relevance") == "relevant"
        ),
        "needs_review": counter_dict(annotation.get("needs_review") for annotation in annotations),
        "validation_issues": dict(sorted(issue_counter.items())),
        "blind_repeat_agreement": {
            "paired": len(paired),
            "relevance": agreement("relevance"),
            "primary_topic_when_both_relevant": agreement("primary_topic", True),
            "tone_when_both_relevant": agreement("tone", True),
            "full_label_tuple": {
                "compared": len(paired),
                "agreed": full_agreed,
                "rate": round(full_agreed / len(paired), 4) if paired else None,
            },
        },
        "usage_all_successes": usage_totals(successful_originals + successful_repeats),
        "usage_unique_only": nonrepeat_usage,
        "projected_usage_remaining_9800": projected_remaining_usage,
    }
    (run_dir / "pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def run_pilot(args: argparse.Namespace) -> int:
    key = load_key()
    prompt = (args.run_dir / "prompt_used.txt").read_text(encoding="utf-8")
    originals = read_jsonl(args.run_dir / "pilot_manifest.jsonl")
    repeats = read_jsonl(args.run_dir / "pilot_repeat_manifest.jsonl")
    requests_to_run = originals + repeats
    results_path = args.run_dir / "pilot_results.jsonl"
    existing = {
        str(record["request_id"])
        for record in (read_jsonl(results_path) if results_path.exists() else [])
        if record.get("ok")
    }
    pending = [row for row in requests_to_run if str(row["request_id"]) not in existing]
    print(
        f"pilot requested={len(requests_to_run)} existing={len(existing)} pending={len(pending)} "
        f"workers={args.workers} tier={args.service_tier}",
        flush=True,
    )
    completed = len(existing)
    lock = threading.Lock()

    def worker(row: dict[str, Any]) -> dict[str, Any]:
        session = requests.Session()
        started = time.perf_counter()
        last_error = ""
        raw_text = ""
        for attempt in range(1, 4):
            try:
                raw = post_response(session, key, request_body(prompt, row, args.service_tier))
                raw_text = output_text(raw)
                parsed, parse_issues = parse_json_output(raw_text)
                issues = parse_issues + validate_annotation(parsed, row)
                return {
                    "request_id": row["request_id"],
                    "article_id": row["article_id"],
                    "is_repeat": row.get("is_repeat", False),
                    "original_request_id": row.get("original_request_id"),
                    "ok": True,
                    "annotation": parsed,
                    "validation_issues": issues,
                    "usage": raw.get("usage") or {},
                    "response_id": raw.get("id"),
                    "model": raw.get("model"),
                    "status": raw.get("status"),
                    "service_tier": raw.get("service_tier"),
                    "attempts": attempt,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "completed_at": utc_now(),
                }
            except Exception as exc:
                last_error = redact(f"{type(exc).__name__}: {exc}", key)
                if attempt < 3:
                    time.sleep(attempt + random.random())
        return {
            "request_id": row["request_id"],
            "article_id": row["article_id"],
            "is_repeat": row.get("is_repeat", False),
            "original_request_id": row.get("original_request_id"),
            "ok": False,
            "error": last_error,
            "raw_output": raw_text,
            "attempts": 3,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "completed_at": utc_now(),
        }

    if pending:
        first, pending = pending[0], pending[1:]
        first_result = worker(first)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(first_result, ensure_ascii=False) + "\n")
        completed += 1
        print(f"pilot progress={completed}/{len(requests_to_run)} ok={first_result['ok']}", flush=True)
        if not first_result["ok"]:
            print(first_result["error"], file=sys.stderr, flush=True)
            return 2

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, row): row for row in pending}
        with results_path.open("a", encoding="utf-8") as handle:
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed += 1
                if completed % 25 == 0 or not result["ok"] or completed == len(requests_to_run):
                    print(
                        f"pilot progress={completed}/{len(requests_to_run)} ok={result['ok']}",
                        flush=True,
                    )
    summary = summarize_pilot(args.run_dir)
    return 0 if summary["successful_unique"] == len(originals) and summary["successful_repeats"] == len(repeats) else 2


def summarize_production(run_dir: Path) -> dict[str, Any]:
    manifest = read_jsonl(run_dir / "selection_manifest.jsonl")
    manifest_by_id = {str(row["article_id"]): row for row in manifest}
    pilot_records = read_jsonl(run_dir / "pilot_results.jsonl")
    pilot_latest = {
        str(record["article_id"]): record
        for record in pilot_records
        if not record.get("is_repeat") and record.get("ok")
    }
    production_path = run_dir / "production_results.jsonl"
    production_records = read_jsonl(production_path) if production_path.exists() else []
    production_latest: dict[str, dict[str, Any]] = {}
    for record in production_records:
        production_latest[str(record["article_id"])] = record
    combined = {**pilot_latest}
    combined.update(
        {article_id: record for article_id, record in production_latest.items() if record.get("ok")}
    )
    issue_counter = Counter(
        issue for record in combined.values() for issue in (record.get("validation_issues") or [])
    )
    annotations_path = run_dir / "annotations_10000.jsonl"
    raw_annotations_path = run_dir / "annotations_10000_raw.jsonl"
    review_path = run_dir / "review_queue.jsonl"
    corrected_ids = 0
    review_count = 0
    with (
        annotations_path.open("w", encoding="utf-8") as handle,
        raw_annotations_path.open("w", encoding="utf-8") as raw_handle,
        review_path.open("w", encoding="utf-8") as review_handle,
    ):
        for article_id in [str(row["article_id"]) for row in manifest]:
            record = combined.get(article_id)
            if not record:
                continue
            raw_annotation = record["annotation"]
            clean_annotation = dict(raw_annotation)
            normalization_actions: list[str] = []
            if clean_annotation.get("article_id") != article_id:
                clean_annotation["article_id"] = article_id
                corrected_ids += 1
                normalization_actions.append("article_id_corrected_from_request_manifest")
            raw_output = {
                **manifest_by_id[article_id],
                "annotation": raw_annotation,
                "validation_issues": record.get("validation_issues") or [],
                "response_id": record.get("response_id"),
                "model": record.get("model"),
            }
            output = {
                **manifest_by_id[article_id],
                "annotation": clean_annotation,
                "validation_issues": record.get("validation_issues") or [],
                "normalization_actions": normalization_actions,
                "response_id": record.get("response_id"),
                "model": record.get("model"),
            }
            raw_handle.write(json.dumps(raw_output, ensure_ascii=False) + "\n")
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            if clean_annotation.get("needs_review") or output["validation_issues"]:
                review_handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                review_count += 1
    annotations = [record["annotation"] for record in combined.values()]
    successful_production = [record for record in production_latest.values() if record.get("ok")]
    failed_production = [record for record in production_latest.values() if not record.get("ok")]
    usage_pilot = usage_totals(list(pilot_latest.values()))
    usage_production = usage_totals(successful_production)
    usage_all = usage_totals(list(combined.values()))
    weighted_production = attempt_weighted_usage(successful_production)
    pilot_all_records = read_jsonl(run_dir / "pilot_results.jsonl")
    pilot_api_calls = sum(max(1, int(record.get("attempts") or 1)) for record in pilot_all_records)
    pilot_latest_all = {
        str(record["request_id"]): record
        for record in pilot_all_records
        if record.get("ok")
    }
    pilot_success_usage = usage_totals(list(pilot_latest_all.values()))
    pilot_success_cost = fast_cost_usd(pilot_success_usage)
    pilot_success_count = len(pilot_latest_all)
    estimated_pilot_cost = round(
        pilot_success_cost * pilot_api_calls / pilot_success_count,
        6,
    ) if pilot_success_count else 0.0
    summary = {
        "generated_at": utc_now(),
        "model": MODEL,
        "selected_total": len(manifest),
        "pilot_unique_included": len(pilot_latest),
        "production_successful": len(successful_production),
        "production_failed": len(failed_production),
        "combined_successful": len(combined),
        "combined_missing": len(manifest) - len(combined),
        "relevance": counter_dict(annotation.get("relevance") for annotation in annotations),
        "primary_topic_relevant_only": counter_dict(
            annotation.get("primary_topic")
            for annotation in annotations
            if annotation.get("relevance") == "relevant"
        ),
        "tone_relevant_only": counter_dict(
            annotation.get("tone")
            for annotation in annotations
            if annotation.get("relevance") == "relevant"
        ),
        "needs_review": counter_dict(annotation.get("needs_review") for annotation in annotations),
        "validation_issues": dict(sorted(issue_counter.items())),
        "normalization": {
            "article_id_corrected_from_request_manifest": corrected_ids,
            "evidence_text_not_auto_modified": True,
        },
        "review_queue_count": review_count,
        "usage_pilot_unique": usage_pilot,
        "usage_production": usage_production,
        "usage_all_10000": usage_all,
        "estimated_api_calls": {
            "pilot_including_repeats_and_parse_retries": pilot_api_calls,
            "production_including_validation_retries": sum(
                max(1, int(record.get("attempts") or 1)) for record in successful_production
            ),
        },
        "cost_estimate_usd": {
            "rate_snapshot_date": "2026-10-15",
            "service_tier": "fast",
            "official_rate_url": "https://developers.openai.com/api/docs/models/gpt-6-luna",
            "final_success_responses_only": fast_cost_usd(usage_all),
            "pilot_including_blind_repeats_and_parse_retries_approx": estimated_pilot_cost,
            "production_including_validation_retries_approx": fast_cost_usd(weighted_production),
            "total_run_including_repeats_and_retries_approx": round(
                estimated_pilot_cost + fast_cost_usd(weighted_production), 6
            ),
            "note": "Retry response usage was not stored separately; retry cost uses each final response's token mix as an approximation.",
        },
    }
    (run_dir / "production_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def run_production(args: argparse.Namespace) -> int:
    key = load_key()
    prompt = (args.run_dir / "prompt_used.txt").read_text(encoding="utf-8")
    selected = read_jsonl(args.run_dir / "selection_manifest.jsonl")
    pilot_ids = {str(row["article_id"]) for row in read_jsonl(args.run_dir / "pilot_manifest.jsonl")}
    production = [row for row in selected if str(row["article_id"]) not in pilot_ids]
    results_path = args.run_dir / "production_results.jsonl"
    existing_records = read_jsonl(results_path) if results_path.exists() else []
    existing = {
        str(record["article_id"])
        for record in existing_records
        if record.get("ok")
    }
    pending = [row for row in production if str(row["article_id"]) not in existing]
    print(
        f"production requested={len(production)} existing={len(existing)} pending={len(pending)} "
        f"workers={args.workers} tier={args.service_tier}",
        flush=True,
    )
    completed = len(existing)
    started_run = time.perf_counter()
    lock = threading.Lock()

    def worker(row: dict[str, Any]) -> dict[str, Any]:
        session = requests.Session()
        started = time.perf_counter()
        last_error = ""
        raw_text = ""
        last_parsed: dict[str, Any] | None = None
        last_issues: list[str] = []
        last_raw: dict[str, Any] = {}
        for attempt in range(1, 4):
            try:
                raw = post_response(session, key, request_body(prompt, row, args.service_tier))
                last_raw = raw
                raw_text = output_text(raw)
                parsed, parse_issues = parse_json_output(raw_text)
                issues = parse_issues + validate_annotation(parsed, row)
                last_parsed, last_issues = parsed, issues
                if issues and attempt < 3:
                    time.sleep(0.2 + random.random() * 0.3)
                    continue
                return {
                    "article_id": row["article_id"],
                    "ok": True,
                    "annotation": parsed,
                    "validation_issues": issues,
                    "usage": raw.get("usage") or {},
                    "response_id": raw.get("id"),
                    "model": raw.get("model"),
                    "status": raw.get("status"),
                    "service_tier": raw.get("service_tier"),
                    "attempts": attempt,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "completed_at": utc_now(),
                }
            except Exception as exc:
                last_error = redact(f"{type(exc).__name__}: {exc}", key)
                if attempt < 3:
                    time.sleep(attempt + random.random())
        if last_parsed is not None:
            return {
                "article_id": row["article_id"],
                "ok": True,
                "annotation": last_parsed,
                "validation_issues": last_issues,
                "usage": last_raw.get("usage") or {},
                "response_id": last_raw.get("id"),
                "model": last_raw.get("model"),
                "status": last_raw.get("status"),
                "service_tier": last_raw.get("service_tier"),
                "attempts": 3,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "completed_at": utc_now(),
            }
        return {
            "article_id": row["article_id"],
            "ok": False,
            "error": last_error,
            "raw_output": raw_text,
            "attempts": 3,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "completed_at": utc_now(),
        }

    if pending:
        first, pending = pending[0], pending[1:]
        first_result = worker(first)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(first_result, ensure_ascii=False) + "\n")
        completed += 1
        print(f"production progress={completed}/{len(production)} ok={first_result['ok']}", flush=True)
        if not first_result["ok"]:
            print(first_result["error"], file=sys.stderr, flush=True)
            return 2

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, row): row for row in pending}
        with results_path.open("a", encoding="utf-8") as handle:
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed += 1
                if completed % 100 == 0 or not result["ok"] or completed == len(production):
                    elapsed = max(time.perf_counter() - started_run, 0.001)
                    newly_done = completed - len(existing)
                    rate = newly_done / elapsed
                    remaining = len(production) - completed
                    eta_minutes = remaining / rate / 60 if rate else None
                    print(
                        f"production progress={completed}/{len(production)} ok={result['ok']} "
                        f"rate={rate:.2f}/s eta_minutes={eta_minutes:.1f}",
                        flush=True,
                    )
    summary = summarize_production(args.run_dir)
    return 0 if summary["combined_successful"] == len(selected) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and run compact GPT economic-news labels")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--old-manifest", type=Path, default=DEFAULT_OLD_MANIFEST)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--count", type=int, default=10000)
    prepare_parser.add_argument("--pilot-count", type=int, default=200)
    prepare_parser.add_argument("--repeat-count", type=int, default=50)
    prepare_parser.set_defaults(func=prepare)

    pilot_parser = subparsers.add_parser("pilot")
    pilot_parser.add_argument("--workers", type=int, default=20)
    pilot_parser.add_argument("--service-tier", choices=["default", "fast"], default="fast")
    pilot_parser.set_defaults(func=run_pilot)

    production_parser = subparsers.add_parser("run-production")
    production_parser.add_argument("--workers", type=int, default=30)
    production_parser.add_argument("--service-tier", choices=["default", "fast"], default="fast")
    production_parser.set_defaults(func=run_production)

    production_summary_parser = subparsers.add_parser("summarize-production")
    production_summary_parser.set_defaults(func=lambda args: (summarize_production(args.run_dir) and 0))

    summary_parser = subparsers.add_parser("summarize-pilot")
    summary_parser.set_defaults(func=lambda args: (summarize_pilot(args.run_dir) and 0))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for field in ("dataset", "old_manifest", "prompt", "run_dir"):
        setattr(args, field, getattr(args, field).resolve())
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
