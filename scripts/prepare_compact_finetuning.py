from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from econmedia_finetuning.compact import (
    TOPIC_LABELS,
    TONE_LABELS,
    article_text,
    build_t5_input,
    build_t5_target,
)
from econmedia_finetuning.data import duplicate_groups
from econmedia_finetuning.utils import load_config, read_jsonl, write_json, write_jsonl


def label_vector(row: dict[str, Any]) -> list[int]:
    topic = [int(row["primary_topic"] == label) for label in TOPIC_LABELS]
    tone = [int(row["tone"] == label) for label in TONE_LABELS]
    joint = [
        int(row["primary_topic"] == topic_label and row["tone"] == tone_label)
        for topic_label in TOPIC_LABELS
        for tone_label in TONE_LABELS
    ]
    return topic + tone + joint


def group_stratified_split(rows, groups, sizes: dict[str, int], seed: int) -> dict[str, list[int]]:
    if sum(sizes.values()) != len(rows):
        raise ValueError("Split sizes must sum to row count")
    names = list(sizes)
    labels = [label_vector(row) for row in rows]
    totals = [sum(vector[index] for vector in labels) for index in range(len(labels[0]))]
    targets = {name: [total * sizes[name] / len(rows) for total in totals] for name in names}
    desired = {name: list(values) for name, values in targets.items()}
    assigned = {name: [] for name in names}
    remaining = dict(sizes)
    rng = random.Random(seed)

    group_data = []
    for group_id, group in enumerate(groups):
        vector = [sum(labels[row_index][label_index] for row_index in group) for label_index in range(len(totals))]
        rarity = sum(value / max(1, totals[index]) for index, value in enumerate(vector))
        group_data.append((group_id, group, vector, rarity, rng.random()))
    group_data.sort(key=lambda item: (-item[3], -len(item[1]), item[4]))

    for _, group, vector, _, _ in group_data:
        feasible = [name for name in names if remaining[name] >= len(group)]
        if not feasible:
            raise ValueError("Near-duplicate group constraints prevent exact split sizes; records were not copied")
        scores = {}
        for name in feasible:
            label_need = sum(
                vector[index] * desired[name][index] / max(1.0, targets[name][index])
                for index in range(len(totals))
            )
            capacity_need = remaining[name] / max(1, sizes[name])
            scores[name] = label_need + 0.01 * capacity_need
        best_score = max(scores.values())
        choices = [name for name in feasible if math.isclose(scores[name], best_score, abs_tol=1e-12)]
        chosen = max(choices, key=lambda name: (remaining[name] / max(1, sizes[name]), -names.index(name)))
        assigned[chosen].extend(group)
        remaining[chosen] -= len(group)
        desired[chosen] = [need - actual for need, actual in zip(desired[chosen], vector)]
    if any(remaining.values()):
        raise ValueError(f"Exact split sizes could not be met: {remaining}")
    return assigned


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "primary_topic": dict(sorted(Counter(row["primary_topic"] for row in rows).items())),
        "tone": dict(sorted(Counter(row["tone"] for row in rows).items())),
        "joint": dict(
            sorted(Counter(f"{row['primary_topic']}|{row['tone']}" for row in rows).items())
        ),
    }


def describe(lengths: list[int], candidates: list[int], max_fraction: float) -> dict[str, Any]:
    ordered = sorted(lengths)
    quantiles = {
        str(q): ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]
        for q in (0.5, 0.9, 0.95, 0.99, 1.0)
    }
    truncation = {str(length): sum(value > length for value in lengths) / len(lengths) for length in candidates}
    selected = next((length for length in candidates if truncation[str(length)] <= max_fraction), candidates[-1])
    return {
        "n": len(lengths),
        "quantiles": quantiles,
        "truncation_fraction": truncation,
        "selected_max_length": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/finetuning.compact.v2.json")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    source_path = ROOT / config["source_annotations"]
    source = read_jsonl(source_path)
    rows = []
    for item in source:
        annotation = item["annotation"]
        if annotation["relevance"] != "relevant":
            continue
        topic = annotation["primary_topic"]
        tone = str(annotation["tone"])
        if topic not in TOPIC_LABELS or tone not in TONE_LABELS:
            raise ValueError(f"Invalid compact label for {item['article_id']}: {topic}, {tone}")
        rows.append(
            {
                "article_id": item["article_id"],
                "headline": item.get("headline") or "",
                "abstract": item.get("abstract") or "",
                "published_at": item.get("published_at") or "",
                "month": item.get("month") or "",
                "duplicate_group_id": item.get("duplicate_group_id") or "",
                "primary_topic": topic,
                "tone": tone,
            }
        )
    if len(rows) != sum(config["split_sizes"].values()):
        raise ValueError(f"Expected {sum(config['split_sizes'].values())} relevant rows, found {len(rows)}")
    if len({row["article_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate article_id values found")

    duplicate_cfg = config["near_duplicate"]
    groups, duplicate_audit = duplicate_groups(
        rows,
        permutations=duplicate_cfg["minhash_permutations"],
        bands=duplicate_cfg["bands"],
        jaccard_threshold=duplicate_cfg["jaccard_threshold"],
        min_length_ratio=duplicate_cfg["min_length_ratio"],
    )
    assigned = group_stratified_split(rows, groups, config["split_sizes"], config["seed"])
    membership = {index: name for name, indices in assigned.items() for index in indices}
    no_leakage = all(len({membership[index] for index in group}) == 1 for group in groups)
    if not no_leakage:
        raise AssertionError("Near-duplicate leakage detected")

    data_dir = ROOT / config["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    split_ids = {}
    distributions = {"all_relevant": distribution(rows)}
    for name in config["split_sizes"]:
        split_rows = [rows[index] for index in sorted(assigned[name])]
        split_ids[name] = [row["article_id"] for row in split_rows]
        distributions[name] = distribution(split_rows)
        write_jsonl(data_dir / f"{name}.jsonl", split_rows)
    write_json(data_dir / "article_id_splits.json", split_ids)
    write_json(data_dir / "label_distribution.json", distributions)
    write_json(data_dir / "duplicate_audit.json", duplicate_audit)

    finbert_tokenizer = AutoTokenizer.from_pretrained(config["finbert_full"]["base_model"], local_files_only=True)
    t5_tokenizer = AutoTokenizer.from_pretrained(config["t5"]["base_model"], local_files_only=True)
    candidates = config["token_length"]["candidate_lengths"]
    max_fraction = config["token_length"]["max_truncation_fraction"]
    bert_lengths = [
        len(finbert_tokenizer(article_text(row["headline"], row["abstract"]), truncation=False)["input_ids"])
        for row in rows
    ]
    t5_lengths = [
        len(t5_tokenizer(build_t5_input(row["headline"], row["abstract"]), truncation=False)["input_ids"])
        for row in rows
    ]
    target_lengths = [
        len(t5_tokenizer(build_t5_target(row["primary_topic"], row["tone"]), truncation=False)["input_ids"])
        for row in rows
    ]
    token_audit = {
        "selection_rule": f"smallest configured length with truncation fraction <= {max_fraction}",
        "finbert": describe(bert_lengths, candidates, max_fraction),
        "t5_source": describe(t5_lengths, candidates, max_fraction),
        "t5_target": describe(target_lengths, [16, 24, 32], 0.0),
    }
    write_json(data_dir / "token_length_audit.json", token_audit)
    audit = {
        "created_from": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "seed": config["seed"],
        "relevant_rows": len(rows),
        "requested_split_sizes": config["split_sizes"],
        "actual_split_sizes": {name: len(indices) for name, indices in assigned.items()},
        "unique_article_ids": len({row["article_id"] for row in rows}),
        "near_duplicate_group_count": len(groups),
        "no_cross_split_near_duplicate_groups": no_leakage,
        "input_features": ["headline", "abstract"],
        "labels": {"primary_topic": TOPIC_LABELS, "tone": TONE_LABELS},
        "gpt_label_caveat": "Labels are GPT-generated and are not an independent human gold standard.",
    }
    write_json(data_dir / "data_audit.json", audit)
    print(json.dumps({"audit": audit, "distribution": distributions, "tokens": token_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
