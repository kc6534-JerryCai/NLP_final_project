from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "outputs/all_relevant/deepseek/annotations.jsonl"
ARTICLE_ID = "nyt://article/d0470c8e-4ad5-568b-b1c6-d6f6d6b35af9"
REPLACEMENT = "speech on homelessness"


def main() -> None:
    latest = None
    with PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["article_id"] == ARTICLE_ID:
                    latest = row
    if latest is None:
        raise SystemExit("Target article was not found")
    source = f"{latest['headline']}\n{latest['abstract']}"
    if REPLACEMENT not in source:
        raise SystemExit("Replacement is not an exact source substring")
    if latest.get("validation_issues") != ["invalid_topic_evidence"]:
        raise SystemExit(f"Unexpected validation state: {latest.get('validation_issues')}")
    repaired = json.loads(json.dumps(latest, ensure_ascii=False))
    repaired["annotation"]["topic_evidence"] = REPLACEMENT
    repaired["validation_issues"] = []
    repaired.setdefault("normalization_actions", []).append(
        "topic_evidence_replaced_with_exact_source_substring"
    )
    repaired["local_repair_note"] = (
        "DeepSeek omitted 'Bill' from the quoted source phrase twice; replaced only the topic evidence "
        "with a shorter exact substring. Labels and all other fields are unchanged."
    )
    repaired["usage"] = {}
    repaired["response_id"] = None
    with PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(repaired, ensure_ascii=False) + "\n")
    print(json.dumps({"article_id": ARTICLE_ID, "replacement": REPLACEMENT}, ensure_ascii=False))


if __name__ == "__main__":
    main()
