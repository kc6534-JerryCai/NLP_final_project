from __future__ import annotations

import argparse
import csv
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
DEFAULT_DATASET = PROJECT_ROOT / "data" / "clean" / "news_articles.csv"
DEFAULT_RUN_DIR = PROJECT_ROOT / "data" / "annotations" / "gpt6_luna_5000"
API_BASE = "https://api.openai.com/v1"
MODEL = "gpt-6-luna"
SEED = "media-indicator-v1"

TOPICS = [
    "prices_inflation",
    "employment",
    "income_purchasing_power",
    "housing",
    "consumption",
    "growth_outlook",
    "economic_policy",
    "household_finance",
]
TONES = ["positive", "negative", "neutral", "mixed", "insufficient_information"]
TIME_FOCI = ["current_or_past", "future", "both", "unclear"]
FRAMES = [
    "household_hardship",
    "economic_resilience",
    "unequal_outcomes",
    "macro_household_disconnect",
    "future_risk",
]
REVIEW_FLAGS = [
    "missing_abstract",
    "truncated_or_corrupted",
    "unclear_us_scope",
    "headline_abstract_conflict",
    "ambiguous_tone",
    "ambiguous_attribution",
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
        "relevance": {"type": "string", "enum": ["relevant", "irrelevant", "uncertain"]},
        "us_scope": {"type": "string", "enum": ["explicit", "implicit", "not_us", "unclear"]},
        "relevance_evidence": {"type": "string"},
        "primary_topic": {"type": "string", "enum": [*TOPICS, "multiple", "none"]},
        "overall_tone": {
            "anyOf": [
                {"type": "string", "enum": TONES},
                {"type": "null"},
            ]
        },
        "overall_evidence": {"type": "string"},
        "topics": {
            "type": "array",
            "items": _object(
                {
                    "topic": {"type": "string", "enum": TOPICS},
                    "condition_tone": {"type": "string", "enum": TONES},
                    "time_focus": {"type": "string", "enum": TIME_FOCI},
                    "evidence": {"type": "string"},
                },
                ["topic", "condition_tone", "time_focus", "evidence"],
            ),
        },
        "frames": {
            "type": "array",
            "items": _object(
                {
                    "frame": {"type": "string", "enum": FRAMES},
                    "evidence": {"type": "string"},
                },
                ["frame", "evidence"],
            ),
        },
        "political_attribution": _object(
            {
                "status": {
                    "type": "string",
                    "enum": ["present", "not_observed", "uncertain", "not_assessed"],
                },
                "items": {
                    "type": "array",
                    "items": _object(
                        {
                            "target": {"type": "string"},
                            "attribution_type": {
                                "type": "string",
                                "enum": ["credit", "blame", "mixed", "causal_without_evaluation"],
                            },
                            "claim_source": {
                                "type": "string",
                                "enum": ["article_narration", "quoted_or_attributed_source", "unclear"],
                            },
                            "evidence": {"type": "string"},
                        },
                        ["target", "attribution_type", "claim_source", "evidence"],
                    ),
                },
            },
            ["status", "items"],
        ),
        "factual_calibration": _object(
            {"status": {"type": "string", "enum": ["not_assessed"]}},
            ["status"],
        ),
        "needs_review": {"type": "boolean"},
        "review_flags": {
            "type": "array",
            "items": {"type": "string", "enum": REVIEW_FLAGS},
        },
    },
    [
        "article_id",
        "relevance",
        "us_scope",
        "relevance_evidence",
        "primary_topic",
        "overall_tone",
        "overall_evidence",
        "topics",
        "frames",
        "political_attribution",
        "factual_calibration",
        "needs_review",
        "review_flags",
    ],
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(value: str, secret: str = "") -> str:
    if secret:
        value = value.replace(secret, "<redacted>")
    return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<redacted>", value)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_prompt(prompt_path: Path, suffix_path: Path | None = None) -> str:
    prompt = prompt_path.read_text(encoding="utf-8-sig").rstrip()
    if suffix_path is not None:
        suffix = suffix_path.read_text(encoding="utf-8-sig").strip()
        if suffix:
            prompt = f"{prompt}\n\n{suffix}"
    return prompt + "\n"


def load_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env or the environment")
    return key


def load_relevant(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("us_relevance") == "relevant"]
    if not rows:
        raise RuntimeError(f"No relevant rows found in {path}")
    return rows


def stable_rank(article_id: str, salt: str) -> str:
    return sha256_text(f"{salt}|{article_id}")


def select_production(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count > len(rows):
        raise ValueError(f"Requested {count} rows, but only {len(rows)} relevant rows exist")
    return sorted(rows, key=lambda row: stable_rank(row["article_id"], SEED))[:count]


def select_year_stratified(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    by_year: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        year = (row.get("published_at") or "unknown")[:4]
        by_year[year].append(row)
    years = sorted(by_year)
    if count < len(years):
        raise ValueError(f"Pilot count {count} is smaller than the {len(years)} publication-year strata")
    base, remainder = divmod(count, len(years))
    chosen: list[dict[str, str]] = []
    leftovers: list[dict[str, str]] = []
    for index, year in enumerate(years):
        ranked = sorted(by_year[year], key=lambda row: stable_rank(row["article_id"], f"{SEED}|pilot"))
        quota = base + (1 if index < remainder else 0)
        chosen.extend(ranked[:quota])
        leftovers.extend(ranked[quota:])
    if len(chosen) < count:
        leftovers.sort(key=lambda row: stable_rank(row["article_id"], f"{SEED}|pilot-fill"))
        chosen.extend(leftovers[: count - len(chosen)])
    return sorted(chosen, key=lambda row: stable_rank(row["article_id"], f"{SEED}|pilot-order"))


def article_payload(row: dict[str, str]) -> dict[str, str]:
    return {
        "article_id": row["article_id"],
        "headline": row.get("title", ""),
        "abstract": row.get("abstract", ""),
    }


def request_body(
    prompt: str, row: dict[str, str], service_tier: str | None = None
) -> dict[str, Any]:
    prompt_hash = sha256_text(prompt)
    body = {
        "model": MODEL,
        "instructions": prompt,
        "input": json.dumps(article_payload(row), ensure_ascii=False, separators=(",", ":")),
        "reasoning": {"effort": "none"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "monthly_media_annotation",
                "strict": True,
                "schema": ANNOTATION_SCHEMA,
            }
        },
        "max_output_tokens": 2048,
        "prompt_cache_key": f"monthly-media-labels-{prompt_hash[:24]}",
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


def post_response(session: requests.Session, key: str, body: dict[str, Any]) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_error = ""
    for attempt in range(1, 7):
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


def validate_annotation(annotation: dict[str, Any], row: dict[str, str]) -> list[str]:
    issues: list[str] = []
    if annotation.get("article_id") != row["article_id"]:
        issues.append("article_id_mismatch")
    relevance = annotation.get("relevance")
    topics = annotation.get("topics") or []
    frames = annotation.get("frames") or []
    topic_names = [item.get("topic") for item in topics]
    frame_names = [item.get("frame") for item in frames]
    if len(topic_names) != len(set(topic_names)):
        issues.append("duplicate_topics")
    if len(frame_names) != len(set(frame_names)):
        issues.append("duplicate_frames")
    primary = annotation.get("primary_topic")
    if primary in TOPICS and primary not in topic_names:
        issues.append("primary_topic_missing_from_topics")
    if relevance == "relevant":
        if not topics:
            issues.append("relevant_without_topics")
        if primary == "none":
            issues.append("relevant_with_primary_none")
        if annotation.get("overall_tone") is None:
            issues.append("relevant_with_null_overall_tone")
        if (annotation.get("political_attribution") or {}).get("status") == "not_assessed":
            issues.append("relevant_with_unassessed_political_attribution")
    if relevance in {"irrelevant", "uncertain"}:
        expected = {
            "primary_topic": "none",
            "overall_tone": None,
            "overall_evidence": "",
            "topics": [],
            "frames": [],
        }
        for field, value in expected.items():
            if annotation.get(field) != value:
                issues.append(f"nonrelevant_{field}_invalid")
        political = annotation.get("political_attribution") or {}
        if political.get("status") != "not_assessed" or political.get("items") != []:
            issues.append("nonrelevant_political_attribution_invalid")
    if relevance == "uncertain" and not annotation.get("needs_review"):
        issues.append("uncertain_without_review")
    political = annotation.get("political_attribution") or {}
    if political.get("status") != "present" and political.get("items"):
        issues.append("political_items_without_present_status")
    if political.get("status") == "present" and not political.get("items"):
        issues.append("present_political_attribution_without_items")
    source = f"{row.get('title', '')}\n{row.get('abstract', '')}"
    evidence_values = [annotation.get("relevance_evidence", ""), annotation.get("overall_evidence", "")]
    evidence_values.extend(item.get("evidence", "") for item in topics)
    evidence_values.extend(item.get("evidence", "") for item in frames)
    evidence_values.extend(item.get("evidence", "") for item in political.get("items") or [])
    for evidence in evidence_values:
        if evidence and evidence not in source:
            issues.append("evidence_not_exact_substring")
            break
    return issues


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_run_metadata(run_dir: Path, prompt: str, dataset: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt_used.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "schema.json").write_text(
        json.dumps(ANNOTATION_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "created_at": utc_now(),
        "model": MODEL,
        "reasoning_effort": "none",
        "dataset": str(dataset.resolve()),
        "prompt_sha256": sha256_text(prompt),
        "schema_sha256": sha256_text(json.dumps(ANNOTATION_SCHEMA, sort_keys=True)),
        "selection_seed": SEED,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_pilot(args: argparse.Namespace) -> int:
    key = load_key()
    prompt = load_prompt(args.prompt, args.prompt_suffix)
    rows = load_relevant(args.dataset)
    production = select_production(rows, args.production_count)
    selected = select_year_stratified(production, args.count)
    write_run_metadata(args.run_dir, prompt, args.dataset)
    pilot_input = args.run_dir / "pilot_input.jsonl"
    with pilot_input.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(article_payload(row), ensure_ascii=False) + "\n")

    results_path = args.run_dir / "pilot_results.jsonl"
    existing = {record["article_id"] for record in jsonl_rows(results_path) if record.get("ok")}
    pending = [row for row in selected if row["article_id"] not in existing]
    print(f"pilot selected={len(selected)} existing={len(existing)} pending={len(pending)}", flush=True)
    lock = threading.Lock()
    completed = len(existing)

    def worker(row: dict[str, str]) -> dict[str, Any]:
        session = requests.Session()
        started = time.perf_counter()
        raw_text = ""
        try:
            raw = post_response(session, key, request_body(prompt, row))
            raw_text = output_text(raw)
            parsed = json.loads(raw_text)
            issues = validate_annotation(parsed, row)
            return {
                "article_id": row["article_id"],
                "ok": True,
                "annotation": parsed,
                "validation_issues": issues,
                "usage": raw.get("usage") or {},
                "response_id": raw.get("id"),
                "model": raw.get("model"),
                "status": raw.get("status"),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "completed_at": utc_now(),
            }
        except Exception as exc:
            return {
                "article_id": row["article_id"],
                "ok": False,
                "error": redact(f"{type(exc).__name__}: {exc}", key),
                "raw_output": raw_text,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "completed_at": utc_now(),
            }

    if pending:
        # Warm the shared prompt cache with one request before concurrent calls.
        first, pending = pending[0], pending[1:]
        first_result = worker(first)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(first_result, ensure_ascii=False) + "\n")
        completed += 1
        print(f"pilot progress={completed}/{len(selected)} ok={first_result['ok']}", flush=True)
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
                if completed % 5 == 0 or not result["ok"] or completed == len(selected):
                    print(f"pilot progress={completed}/{len(selected)} ok={result['ok']}", flush=True)
    summarize_pilot(args.run_dir, selected)
    return 0


def summarize_pilot(run_dir: Path, selected: list[dict[str, str]] | None = None) -> dict[str, Any]:
    records = list(jsonl_rows(run_dir / "pilot_results.jsonl"))
    successful = [record for record in records if record.get("ok")]
    latest_by_id = {record["article_id"]: record for record in successful}
    successful = list(latest_by_id.values())
    annotations = [record["annotation"] for record in successful]
    usage_keys = ["input_tokens", "output_tokens", "total_tokens"]
    usage = {key: sum((record.get("usage") or {}).get(key, 0) for record in successful) for key in usage_keys}
    usage["cached_tokens"] = sum(
        ((record.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens", 0)
        for record in successful
    )
    usage["cache_write_tokens"] = sum(
        ((record.get("usage") or {}).get("input_tokens_details") or {}).get("cache_write_tokens", 0)
        for record in successful
    )
    issue_counts = Counter(issue for record in successful for issue in record.get("validation_issues") or [])
    summary = {
        "generated_at": utc_now(),
        "requested": len(selected) if selected is not None else None,
        "successful_unique": len(successful),
        "failed_records": sum(not record.get("ok") for record in records),
        "model": MODEL,
        "relevance": Counter(item.get("relevance") for item in annotations),
        "us_scope": Counter(item.get("us_scope") for item in annotations),
        "overall_tone": Counter(str(item.get("overall_tone")) for item in annotations),
        "primary_topic": Counter(item.get("primary_topic") for item in annotations),
        "needs_review": Counter(str(item.get("needs_review")) for item in annotations),
        "frames": Counter(frame.get("frame") for item in annotations for frame in item.get("frames") or []),
        "validation_issues": issue_counts,
        "usage": usage,
    }
    serializable = json.loads(json.dumps(summary))
    (run_dir / "pilot_summary.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(serializable, ensure_ascii=False, indent=2), flush=True)
    return serializable


def prepare_batch(args: argparse.Namespace) -> int:
    prompt = load_prompt(args.prompt, args.prompt_suffix)
    rows = select_production(load_relevant(args.dataset), args.count)
    write_run_metadata(args.run_dir, prompt, args.dataset)
    batch_path = args.run_dir / "batch_input_5000.jsonl"
    manifest_path = args.run_dir / "batch_manifest.jsonl"
    with batch_path.open("w", encoding="utf-8") as batch_handle, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_handle:
        for index, row in enumerate(rows, 1):
            custom_id = f"annotation-{index:05d}"
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": request_body(prompt, row),
            }
            batch_handle.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            manifest_handle.write(
                json.dumps(
                    {"custom_id": custom_id, **article_payload(row)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    metadata = {
        "prepared_at": utc_now(),
        "requests": len(rows),
        "bytes": batch_path.stat().st_size,
        "input_file_sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest(),
        "prompt_sha256": sha256_text(prompt),
        "model": MODEL,
    }
    (args.run_dir / "batch_prepared.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def api_headers(key: str, json_content: bool = True) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {key}"}
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


def submit_batch(args: argparse.Namespace) -> int:
    key = load_key()
    state_path = args.run_dir / "batch_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("batch_id"):
            print(f"Existing batch_id={state['batch_id']}; refusing duplicate submission.")
            return 0
    batch_path = args.run_dir / "batch_input_5000.jsonl"
    if not batch_path.exists():
        raise RuntimeError(f"Missing prepared batch file: {batch_path}")
    with batch_path.open("rb") as handle:
        response = requests.post(
            f"{API_BASE}/files",
            headers=api_headers(key, json_content=False),
            data={"purpose": "batch"},
            files={"file": (batch_path.name, handle, "application/jsonl")},
            timeout=300,
        )
    if response.status_code >= 400:
        raise RuntimeError(redact(f"File upload HTTP {response.status_code}: {response.text}", key))
    uploaded = response.json()
    create = requests.post(
        f"{API_BASE}/batches",
        headers=api_headers(key),
        json={
            "input_file_id": uploaded["id"],
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "metadata": {
                "project": "NLP_Final_Assignment",
                "task": "monthly_media_annotation_5000",
                "model": MODEL,
            },
        },
        timeout=120,
    )
    if create.status_code >= 400:
        raise RuntimeError(redact(f"Batch create HTTP {create.status_code}: {create.text}", key))
    batch = create.json()
    state = {
        "submitted_at": utc_now(),
        "input_file_id": uploaded["id"],
        "batch_id": batch["id"],
        "status": batch.get("status"),
        "request_counts": batch.get("request_counts"),
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def get_batch(key: str, batch_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE}/batches/{batch_id}", headers=api_headers(key), timeout=120
    )
    if response.status_code >= 400:
        raise RuntimeError(redact(f"Batch status HTTP {response.status_code}: {response.text}", key))
    return response.json()


def batch_status(args: argparse.Namespace) -> int:
    key = load_key()
    state_path = args.run_dir / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch = get_batch(key, state["batch_id"])
    state.update(
        {
            "checked_at": utc_now(),
            "status": batch.get("status"),
            "request_counts": batch.get("request_counts"),
            "output_file_id": batch.get("output_file_id"),
            "error_file_id": batch.get("error_file_id"),
            "completed_at": batch.get("completed_at"),
            "failed_at": batch.get("failed_at"),
            "expired_at": batch.get("expired_at"),
            "errors": batch.get("errors"),
        }
    )
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def cancel_batch(args: argparse.Namespace) -> int:
    key = load_key()
    state_path = args.run_dir / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch_id = state["batch_id"]
    current = get_batch(key, batch_id)
    if current.get("status") not in {"completed", "failed", "expired", "cancelled"}:
        response = requests.post(
            f"{API_BASE}/batches/{batch_id}/cancel",
            headers=api_headers(key),
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                redact(f"Batch cancel HTTP {response.status_code}: {response.text}", key)
            )
        current = response.json()
    state.update(
        {
            "cancel_requested_at": utc_now(),
            "status": current.get("status"),
            "request_counts": current.get("request_counts"),
            "output_file_id": current.get("output_file_id"),
            "error_file_id": current.get("error_file_id"),
        }
    )
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return 0


def wait_batch_cancelled(args: argparse.Namespace) -> int:
    key = load_key()
    state_path = args.run_dir / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + args.max_minutes * 60
    while time.monotonic() < deadline:
        batch = get_batch(key, state["batch_id"])
        state.update(
            {
                "checked_at": utc_now(),
                "status": batch.get("status"),
                "request_counts": batch.get("request_counts"),
                "output_file_id": batch.get("output_file_id"),
                "error_file_id": batch.get("error_file_id"),
            }
        )
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {"status": state["status"], "request_counts": state["request_counts"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if state["status"] == "cancelled":
            return 0
        if state["status"] in {"completed", "failed", "expired"}:
            return 2
        time.sleep(args.interval_seconds)
    return 4


def summarize_standard(run_dir: Path, requested: int) -> dict[str, Any]:
    records = list(jsonl_rows(run_dir / "standard_results.jsonl"))
    latest_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        latest_by_id[record["article_id"]] = record
    successful = [record for record in latest_by_id.values() if record.get("ok")]
    failed = [record for record in latest_by_id.values() if not record.get("ok")]
    counters: dict[str, Counter] = {
        "relevance": Counter(),
        "us_scope": Counter(),
        "overall_tone": Counter(),
        "primary_topic": Counter(),
        "needs_review": Counter(),
        "frames": Counter(),
        "validation_issues": Counter(),
    }
    usage = Counter()
    annotations_path = run_dir / "annotations_5000.jsonl"
    with annotations_path.open("w", encoding="utf-8") as handle:
        for record in successful:
            annotation = record["annotation"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counters["relevance"][annotation.get("relevance")] += 1
            counters["us_scope"][annotation.get("us_scope")] += 1
            counters["overall_tone"][str(annotation.get("overall_tone"))] += 1
            counters["primary_topic"][annotation.get("primary_topic")] += 1
            counters["needs_review"][str(annotation.get("needs_review"))] += 1
            counters["frames"].update(
                frame.get("frame") for frame in annotation.get("frames") or []
            )
            counters["validation_issues"].update(record.get("validation_issues") or [])
            body_usage = record.get("usage") or {}
            usage.update(
                {key: body_usage.get(key, 0) for key in ("input_tokens", "output_tokens", "total_tokens")}
            )
            usage["cached_tokens"] += (body_usage.get("input_tokens_details") or {}).get(
                "cached_tokens", 0
            )
            usage["cache_write_tokens"] += (body_usage.get("input_tokens_details") or {}).get(
                "cache_write_tokens", 0
            )
    summary = {
        "generated_at": utc_now(),
        "requested": requested,
        "successful_unique": len(successful),
        "failed_unique": len(failed),
        "model": MODEL,
        "service_tier_requested": "fast",
        **{name: dict(counter) for name, counter in counters.items()},
        "usage": dict(usage),
    }
    (run_dir / "standard_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def run_standard(args: argparse.Namespace) -> int:
    key = load_key()
    prompt = (args.run_dir / "prompt_used.txt").read_text(encoding="utf-8")
    selected = select_production(load_relevant(args.dataset), args.count)
    results_path = args.run_dir / "standard_results.jsonl"
    existing = {
        record["article_id"]
        for record in jsonl_rows(results_path)
        if record.get("ok")
    }
    pending = [row for row in selected if row["article_id"] not in existing]
    print(
        f"standard selected={len(selected)} existing={len(existing)} pending={len(pending)} "
        f"workers={args.workers} tier={args.service_tier}",
        flush=True,
    )
    completed = len(existing)
    lock = threading.Lock()

    def worker(row: dict[str, str]) -> dict[str, Any]:
        session = requests.Session()
        started = time.perf_counter()
        last_error = ""
        raw_text = ""
        for attempt in range(1, 4):
            try:
                raw = post_response(
                    session,
                    key,
                    request_body(prompt, row, service_tier=args.service_tier),
                )
                raw_text = output_text(raw)
                parsed = json.loads(raw_text)
                issues = validate_annotation(parsed, row)
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
        print(f"standard progress={completed}/{len(selected)} ok={first_result['ok']}", flush=True)
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
                if completed % 25 == 0 or not result["ok"] or completed == len(selected):
                    print(
                        f"standard progress={completed}/{len(selected)} ok={result['ok']}",
                        flush=True,
                    )
    summary = summarize_standard(args.run_dir, len(selected))
    return 0 if summary["successful_unique"] == len(selected) else 2


def download_file(key: str, file_id: str, destination: Path) -> None:
    response = requests.get(
        f"{API_BASE}/files/{file_id}/content",
        headers=api_headers(key, json_content=False),
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(redact(f"File download HTTP {response.status_code}: {response.text}", key))
    destination.write_bytes(response.content)


def download_batch(args: argparse.Namespace) -> int:
    key = load_key()
    state = json.loads((args.run_dir / "batch_state.json").read_text(encoding="utf-8"))
    batch = get_batch(key, state["batch_id"])
    if not batch.get("output_file_id"):
        print(json.dumps({"status": batch.get("status"), "request_counts": batch.get("request_counts")}, indent=2))
        return 3
    raw_output = args.run_dir / "batch_output_raw.jsonl"
    download_file(key, batch["output_file_id"], raw_output)
    if batch.get("error_file_id"):
        download_file(key, batch["error_file_id"], args.run_dir / "batch_errors.jsonl")
    parse_batch_output(args.run_dir, raw_output)
    return 0


def watch_batch(args: argparse.Namespace) -> int:
    key = load_key()
    state_path = args.run_dir / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch_id = state["batch_id"]
    log_path = args.run_dir / "batch_watch.jsonl"
    deadline = time.monotonic() + args.max_hours * 3600
    terminal = {"completed", "failed", "expired", "cancelled"}
    while time.monotonic() < deadline:
        try:
            batch = get_batch(key, batch_id)
            snapshot = {
                "checked_at": utc_now(),
                "batch_id": batch_id,
                "status": batch.get("status"),
                "request_counts": batch.get("request_counts"),
                "output_file_id": batch.get("output_file_id"),
                "error_file_id": batch.get("error_file_id"),
                "errors": batch.get("errors"),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            state.update(snapshot)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
            if batch.get("status") == "completed" and batch.get("output_file_id"):
                raw_output = args.run_dir / "batch_output_raw.jsonl"
                download_file(key, batch["output_file_id"], raw_output)
                if batch.get("error_file_id"):
                    download_file(key, batch["error_file_id"], args.run_dir / "batch_errors.jsonl")
                parse_batch_output(args.run_dir, raw_output)
                return 0
            if batch.get("status") in terminal:
                return 2
        except Exception as exc:
            error = {"checked_at": utc_now(), "watch_error": redact(f"{type(exc).__name__}: {exc}", key)}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(error, ensure_ascii=False) + "\n")
            print(json.dumps(error, ensure_ascii=False), flush=True)
        time.sleep(args.interval_seconds)
    return 4


def parse_batch_output(run_dir: Path, raw_output: Path) -> dict[str, Any]:
    manifest = {row["custom_id"]: row for row in jsonl_rows(run_dir / "batch_manifest.jsonl")}
    annotations_path = run_dir / "annotations_5000.jsonl"
    failures_path = run_dir / "annotation_failures.jsonl"
    counters: dict[str, Counter] = {
        "relevance": Counter(),
        "us_scope": Counter(),
        "overall_tone": Counter(),
        "primary_topic": Counter(),
        "needs_review": Counter(),
        "frames": Counter(),
        "validation_issues": Counter(),
    }
    usage = Counter()
    successes = 0
    failures = 0
    with annotations_path.open("w", encoding="utf-8") as good, failures_path.open(
        "w", encoding="utf-8"
    ) as bad:
        for record in jsonl_rows(raw_output):
            custom_id = record.get("custom_id")
            source = manifest.get(custom_id)
            try:
                response = record.get("response") or {}
                if response.get("status_code") != 200:
                    raise ValueError(f"status_code={response.get('status_code')} error={record.get('error')}")
                body = response["body"]
                annotation = json.loads(output_text(body))
                if source is None:
                    raise ValueError("custom_id missing from manifest")
                row = {
                    "article_id": source["article_id"],
                    "title": source["headline"],
                    "abstract": source["abstract"],
                }
                issues = validate_annotation(annotation, row)
                good.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            "annotation": annotation,
                            "validation_issues": issues,
                            "usage": body.get("usage") or {},
                            "response_id": body.get("id"),
                            "model": body.get("model"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                successes += 1
                counters["relevance"][annotation.get("relevance")] += 1
                counters["us_scope"][annotation.get("us_scope")] += 1
                counters["overall_tone"][str(annotation.get("overall_tone"))] += 1
                counters["primary_topic"][annotation.get("primary_topic")] += 1
                counters["needs_review"][str(annotation.get("needs_review"))] += 1
                counters["frames"].update(frame.get("frame") for frame in annotation.get("frames") or [])
                counters["validation_issues"].update(issues)
                body_usage = body.get("usage") or {}
                usage.update({key: body_usage.get(key, 0) for key in ("input_tokens", "output_tokens", "total_tokens")})
                usage["cached_tokens"] += (body_usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
                usage["cache_write_tokens"] += (body_usage.get("input_tokens_details") or {}).get(
                    "cache_write_tokens", 0
                )
            except Exception as exc:
                failures += 1
                bad.write(
                    json.dumps(
                        {"custom_id": custom_id, "error": f"{type(exc).__name__}: {exc}", "record": record},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    summary = {
        "generated_at": utc_now(),
        "successful": successes,
        "failed": failures,
        "model": MODEL,
        **{name: dict(counter) for name, counter in counters.items()},
        "usage": dict(usage),
    }
    (run_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label NYT title/abstract records with OpenAI Structured Outputs")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--prompt", type=Path, required=True)
    pilot.add_argument("--prompt-suffix", type=Path)
    pilot.add_argument("--count", type=int, default=100)
    pilot.add_argument("--production-count", type=int, default=5000)
    pilot.add_argument("--workers", type=int, default=5)
    pilot.set_defaults(func=run_pilot)

    prepare = subparsers.add_parser("prepare-batch")
    prepare.add_argument("--prompt", type=Path, required=True)
    prepare.add_argument("--prompt-suffix", type=Path)
    prepare.add_argument("--count", type=int, default=5000)
    prepare.set_defaults(func=prepare_batch)

    submit = subparsers.add_parser("submit-batch")
    submit.set_defaults(func=submit_batch)

    status = subparsers.add_parser("batch-status")
    status.set_defaults(func=batch_status)

    cancel = subparsers.add_parser("cancel-batch")
    cancel.set_defaults(func=cancel_batch)

    wait_cancel = subparsers.add_parser("wait-batch-cancelled")
    wait_cancel.add_argument("--interval-seconds", type=int, default=5)
    wait_cancel.add_argument("--max-minutes", type=float, default=10.0)
    wait_cancel.set_defaults(func=wait_batch_cancelled)

    download = subparsers.add_parser("download-batch")
    download.set_defaults(func=download_batch)

    watch = subparsers.add_parser("watch-batch")
    watch.add_argument("--interval-seconds", type=int, default=45)
    watch.add_argument("--max-hours", type=float, default=25.0)
    watch.set_defaults(func=watch_batch)

    standard = subparsers.add_parser("standard-run")
    standard.add_argument("--count", type=int, default=5000)
    standard.add_argument("--workers", type=int, default=20)
    standard.add_argument("--service-tier", choices=["default", "fast"], default="fast")
    standard.set_defaults(func=run_standard)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.dataset = args.dataset.resolve()
    args.run_dir = args.run_dir.resolve()
    if hasattr(args, "prompt"):
        args.prompt = args.prompt.resolve()
    if getattr(args, "prompt_suffix", None) is not None:
        args.prompt_suffix = args.prompt_suffix.resolve()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
