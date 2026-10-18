from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/clean/relevant_only.jsonl"
T5 = ROOT / "outputs/all_relevant/t5/annotations.jsonl"
DEEPSEEK = ROOT / "outputs/all_relevant/deepseek/annotations.jsonl"
OUTPUT = ROOT / "outputs/all_relevant/delivery"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(SOURCE)
    t5_by_id = {row["article_id"]: row for row in read_jsonl(T5)}
    deepseek_by_id = {row["article_id"]: row for row in read_jsonl(DEEPSEEK)}
    if len(source_rows) != 41273 or len(t5_by_id) != len(source_rows) or len(deepseek_by_id) != len(source_rows):
        raise SystemExit(
            f"Incomplete inputs: source={len(source_rows)} t5={len(t5_by_id)} deepseek={len(deepseek_by_id)}"
        )

    merged = []
    for source in source_rows:
        article_id = str(source["article_id"])
        t5 = t5_by_id[article_id]
        deepseek = deepseek_by_id[article_id]
        annotation = deepseek.get("annotation") or {}
        fact = annotation.get("factual_calibration") or {}
        frames = annotation.get("frames") or []
        merged.append(
            {
                "article_id": article_id,
                "published_at": source.get("published_at"),
                "month": source.get("month") or str(source.get("published_at") or "")[:7],
                "headline": source.get("title") or source.get("headline") or "",
                "abstract": source.get("abstract") or "",
                "duplicate_group_id": source.get("duplicate_group_id"),
                "t5_primary_topic": t5.get("primary_topic"),
                "t5_tone": t5.get("tone"),
                "t5_format_valid": t5.get("format_valid"),
                "t5_raw_output": t5.get("raw_output"),
                "deepseek_relevance": annotation.get("relevance"),
                "deepseek_primary_topic": annotation.get("primary_topic"),
                "deepseek_tone": annotation.get("tone"),
                "deepseek_topic_evidence": annotation.get("topic_evidence"),
                "deepseek_tone_evidence": annotation.get("tone_evidence"),
                "deepseek_frames": "; ".join(frame.get("frame", "") for frame in frames),
                "deepseek_frame_evidence": json.dumps(frames, ensure_ascii=False, separators=(",", ":")),
                "factual_calibration": fact.get("status"),
                "factual_claim_evidence": fact.get("claim_evidence"),
                "factual_reference_matches": json.dumps(
                    fact.get("reference_matches") or [], ensure_ascii=False, separators=(",", ":")
                ),
                "factual_note": fact.get("note"),
                "deepseek_needs_review": annotation.get("needs_review"),
                "deepseek_review_flags": "; ".join(annotation.get("review_flags") or []),
                "deepseek_validation_issues": "; ".join(deepseek.get("validation_issues") or []),
            }
        )

    jsonl_path = OUTPUT / "all_relevant_annotations.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = OUTPUT / "all_relevant_annotations.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)

    relevant = [row for row in merged if row["deepseek_relevance"] == "relevant"]
    topic_comparable = [row for row in relevant if row["deepseek_primary_topic"] is not None]
    tone_comparable = [row for row in relevant if row["deepseek_tone"] is not None]
    summary = {
        "total_rule_screened_relevant": len(merged),
        "t5_format_valid": sum(bool(row["t5_format_valid"]) for row in merged),
        "deepseek_relevance": dict(Counter(row["deepseek_relevance"] for row in merged)),
        "t5_primary_topic": dict(Counter(row["t5_primary_topic"] for row in merged)),
        "t5_tone": dict(Counter(str(row["t5_tone"]) for row in merged)),
        "deepseek_primary_topic_relevant_only": dict(Counter(row["deepseek_primary_topic"] for row in relevant)),
        "deepseek_tone_relevant_only": dict(Counter(str(row["deepseek_tone"]) for row in relevant)),
        "deepseek_frames_relevant_only": dict(
            Counter(frame for row in relevant for frame in row["deepseek_frames"].split("; ") if frame)
        ),
        "factual_calibration_all": dict(Counter(row["factual_calibration"] for row in merged)),
        "deepseek_validation_issue_rows": sum(bool(row["deepseek_validation_issues"]) for row in merged),
        "t5_deepseek_topic_agreement_on_deepseek_relevant": {
            "n": len(topic_comparable),
            "count": sum(row["t5_primary_topic"] == row["deepseek_primary_topic"] for row in topic_comparable),
            "rate": sum(row["t5_primary_topic"] == row["deepseek_primary_topic"] for row in topic_comparable)
            / len(topic_comparable),
        },
        "t5_deepseek_tone_agreement_on_deepseek_relevant": {
            "n": len(tone_comparable),
            "count": sum(row["t5_tone"] == row["deepseek_tone"] for row in tone_comparable),
            "rate": sum(row["t5_tone"] == row["deepseek_tone"] for row in tone_comparable) / len(tone_comparable),
        },
        "method_notes": [
            "The 41,273 records are deterministic rule-screened relevant candidates, not GPT-confirmed relevant labels.",
            "DeepSeek topic and tone use the exact saved GPT prompt; framing and factual calibration are appended tasks.",
            "Factual calibration covers only indicators present in Consumer_Sentiment_Economic_Data_2010_2026.xlsx.",
            "The economic workbook contains current revised series, not a historical real-time vintage panel.",
            "T5 always emits topic and tone because it has no relevance output head.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
