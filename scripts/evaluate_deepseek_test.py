from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "finetuning_compact_v2" / "test.jsonl"
PROMPT_PATH = ROOT / "data" / "annotations" / "gpt6_luna_compact_10000" / "prompt_used.txt"
OUTPUT_DIR = ROOT / "outputs" / "deepseek_test_compact_v2"
MODEL = "deepseek-flash"

TOPICS = [
    "cost_of_living_income",
    "labor_market",
    "household_economy",
    "macro_growth",
    "economic_policy",
]
TONES = ["-1", "0", "1"]
RELEVANCE = ["relevant", "irrelevant", "uncertain"]
REVIEW_FLAGS = [
    "missing_abstract",
    "truncated_or_corrupted",
    "unclear_relevance",
    "ambiguous_topic",
    "ambiguous_tone",
    "headline_abstract_conflict",
]


def object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


SCHEMA = object_schema(
    {
        "article_id": {"type": "string"},
        "relevance": {"type": "string", "enum": RELEVANCE},
        "primary_topic": {
            "anyOf": [{"type": "string", "enum": TOPICS}, {"type": "null"}]
        },
        "tone": {
            "anyOf": [{"type": "integer", "enum": [-1, 0, 1]}, {"type": "null"}]
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_text(body: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "".join(chunks)


def validate(row: dict[str, Any], annotation: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if annotation.get("article_id") != row["article_id"]:
        issues.append("article_id_mismatch")
    relevance = annotation.get("relevance")
    if relevance not in RELEVANCE:
        issues.append("invalid_relevance")
    topic = annotation.get("primary_topic")
    tone = annotation.get("tone")
    if relevance == "relevant":
        if topic not in TOPICS:
            issues.append("relevant_invalid_topic")
        if tone not in [-1, 0, 1]:
            issues.append("relevant_invalid_tone")
        source = f"{row['headline']}\n{row['abstract']}"
        if not annotation.get("topic_evidence") or annotation.get("topic_evidence") not in source:
            issues.append("invalid_topic_evidence")
        if not annotation.get("tone_evidence") or annotation.get("tone_evidence") not in source:
            issues.append("invalid_tone_evidence")
    elif relevance in {"irrelevant", "uncertain"}:
        if topic is not None or tone is not None:
            issues.append("nonrelevant_nonnull_label")
        if annotation.get("topic_evidence") != "" or annotation.get("tone_evidence") != "":
            issues.append("nonrelevant_nonempty_evidence")
    flags = annotation.get("review_flags")
    if not isinstance(flags, list) or any(flag not in REVIEW_FLAGS for flag in flags):
        issues.append("invalid_review_flags")
        flags = []
    if annotation.get("needs_review") != bool(flags):
        issues.append("review_boolean_mismatch")
    if relevance == "uncertain" and "unclear_relevance" not in flags:
        issues.append("uncertain_without_flag")
    return issues


def request_one(
    row: dict[str, Any], prompt: str, key: str, endpoint: str, max_attempts: int = 6
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "instructions": prompt,
        "input": json.dumps(
            {
                "article_id": row["article_id"],
                "headline": row["headline"],
                "abstract": row["abstract"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "reasoning": {"effort": "none"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "compact_economic_annotation",
                "schema": SCHEMA,
            }
        },
        "max_output_tokens": 512,
    }
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=(15, 180),
            )
            if response.status_code == 200:
                body = response.json()
                text = output_text(body)
                annotation = json.loads(text)
                return {
                    "article_id": row["article_id"],
                    "ok": True,
                    "annotation": annotation,
                    "validation_issues": validate(row, annotation),
                    "usage": body.get("usage") or {},
                    "response_id": body.get("id"),
                    "model": body.get("model", MODEL),
                }
            last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
            if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(30.0, (2 ** (attempt - 1)) + random.random()))
    return {"article_id": row["article_id"], "ok": False, "error": last_error}


def metric_block(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    indices = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    return {
        "effective_n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "invalid_prediction_count": int(sum(value < 0 for value in y_pred)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=indices).tolist(),
        "confusion_matrix_labels": labels,
    }


def build_report(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["article_id"]: row for row in rows}
    successful = [result for result in results if result.get("ok")]
    failed = [result for result in results if not result.get("ok")]
    evaluated = []
    topic_true: list[int] = []
    topic_pred: list[int] = []
    tone_true: list[int] = []
    tone_pred: list[int] = []
    relevance_correct = 0
    valid_semantics = 0
    relevance_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    for result in successful:
        row = by_id[result["article_id"]]
        ann = result["annotation"]
        relevance_counts[str(ann.get("relevance"))] += 1
        issue_counts.update(result["validation_issues"])
        is_relevant = ann.get("relevance") == "relevant"
        relevance_correct += int(is_relevant)
        valid_semantics += int(not result["validation_issues"])
        predicted_topic = TOPICS.index(ann["primary_topic"]) if is_relevant and ann.get("primary_topic") in TOPICS else -1
        predicted_tone = TONES.index(str(ann["tone"])) if is_relevant and str(ann.get("tone")) in TONES else -1
        true_topic = TOPICS.index(row["primary_topic"])
        true_tone = TONES.index(row["tone"])
        topic_true.append(true_topic)
        topic_pred.append(predicted_topic)
        tone_true.append(true_tone)
        tone_pred.append(predicted_tone)
        evaluated.append(
            {
                "article_id": row["article_id"],
                "headline": row["headline"],
                "gpt_relevance": "relevant",
                "deepseek_relevance": ann.get("relevance"),
                "gpt_primary_topic": row["primary_topic"],
                "deepseek_primary_topic": ann.get("primary_topic"),
                "gpt_tone": row["tone"],
                "deepseek_tone": None if ann.get("tone") is None else str(ann.get("tone")),
                "topic_match": predicted_topic == true_topic,
                "tone_match": predicted_tone == true_tone,
                "validation_issues": result["validation_issues"],
                "deepseek_annotation": ann,
            }
        )
    usage = {
        key: int(sum((result.get("usage") or {}).get(key, 0) for result in successful))
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    usage["cached_input_tokens"] = int(
        sum(
            ((result.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens", 0)
            for result in successful
        )
    )
    topic = metric_block(topic_true, topic_pred, TOPICS) if successful else {}
    tone = metric_block(tone_true, tone_pred, TONES) if successful else {}
    both_relevant = [row for row in evaluated if row["deepseek_relevance"] == "relevant"]
    conditional_topic = metric_block(
        [TOPICS.index(row["gpt_primary_topic"]) for row in both_relevant],
        [TOPICS.index(row["deepseek_primary_topic"]) for row in both_relevant],
        TOPICS,
    ) if both_relevant else {}
    conditional_tone = metric_block(
        [TONES.index(row["gpt_tone"]) for row in both_relevant],
        [TONES.index(row["deepseek_tone"]) for row in both_relevant],
        TONES,
    ) if both_relevant else {}
    uncached_input = usage["input_tokens"] - usage["cached_input_tokens"]
    cost = {
        "peak_usd": (
            uncached_input * 0.30
            + usage["cached_input_tokens"] * 0.006
            + usage["output_tokens"] * 1.20
        ) / 1_000_000,
        "off_peak_usd": (
            uncached_input * 0.15
            + usage["cached_input_tokens"] * 0.003
            + usage["output_tokens"] * 0.60
        ) / 1_000_000,
        "pricing_basis": "DeepSeek Flash listed USD per 1M tokens on 2026-10-15",
    }
    report = {
        "model": MODEL,
        "reference_labels": "GPT-generated compact v2 labels",
        "requested_rows": len(rows),
        "successful_rows": len(successful),
        "failed_rows": len(failed),
        "deepseek_relevance_counts": dict(relevance_counts),
        "relevance_agreement_accuracy": relevance_correct / len(successful) if successful else None,
        "json_parse_success_rate": len(successful) / len(results) if results else None,
        "strict_schema_and_semantic_valid_rate": valid_semantics / len(successful) if successful else None,
        "validation_issue_counts": dict(issue_counts),
        "primary_topic": topic,
        "overall_tone": tone,
        "selection_score": 0.5 * topic.get("macro_f1", 0) + 0.5 * tone.get("macro_f1", 0),
        "conditional_on_both_relevant": {
            "effective_n": len(both_relevant),
            "primary_topic": conditional_topic,
            "overall_tone": conditional_tone,
            "joint_exact_match": (
                sum(row["topic_match"] and row["tone_match"] for row in both_relevant) / len(both_relevant)
                if both_relevant else None
            ),
        },
        "all_600_joint_exact_match": (
            sum(row["topic_match"] and row["tone_match"] for row in evaluated) / len(rows)
            if rows else None
        ),
        "usage": usage,
        "estimated_api_cost": cost,
        "failed": failed,
        "caveat": "Agreement with GPT-generated labels, not accuracy against an independent human gold standard.",
    }
    return {"report": report, "evaluated": evaluated}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    endpoint = f"{base}/responses"
    prompt = PROMPT_PATH.read_text(encoding="utf-8-sig")
    rows = read_jsonl(TEST_PATH)[: args.limit]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / "raw_results.jsonl"
    existing = {row["article_id"]: row for row in read_jsonl(raw_path)} if raw_path.exists() else {}
    pending = [row for row in rows if row["article_id"] not in existing or not existing[row["article_id"]].get("ok")]
    lock = threading.Lock()
    completed = 0
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(request_one, row, prompt, key, endpoint): row["article_id"]
                for row in pending
            }
            for future in as_completed(futures):
                result = future.result()
                existing[result["article_id"]] = result
                append_jsonl(raw_path, result, lock)
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(json.dumps({"completed_now": completed, "pending_total": len(pending)}), flush=True)
    ordered = [existing[row["article_id"]] for row in rows if row["article_id"] in existing]
    artifacts = build_report(rows, ordered)
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(artifacts["report"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_DIR / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in artifacts["evaluated"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    errors = [row for row in artifacts["evaluated"] if not row["topic_match"] or not row["tone_match"] or row["validation_issues"]]
    with (OUTPUT_DIR / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for row in errors:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(artifacts["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
