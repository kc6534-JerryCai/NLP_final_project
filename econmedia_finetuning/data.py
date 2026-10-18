from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import PRIMARY_TOPIC_LABELS, TONE_LABELS, TOPIC_LABELS
from .utils import read_jsonl, write_json, write_jsonl


def normalized_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    return re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).strip()


def article_text(headline: str, abstract: str) -> str:
    return f"Title: {headline.strip()}\nAbstract: {abstract.strip()}"


def _shingles(text: str, width: int = 3) -> set[str]:
    tokens = normalized_text(text).split()
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else {""}
    return {" ".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def _stable_hash(value: str, salt: int) -> int:
    data = salt.to_bytes(2, "little") + value.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def duplicate_groups(
    rows: list[dict[str, Any]],
    permutations: int = 64,
    bands: int = 16,
    jaccard_threshold: float = 0.82,
    min_length_ratio: float = 0.7,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Group exact and high-overlap text using deterministic MinHash candidates."""
    if permutations % bands:
        raise ValueError("minhash_permutations must be divisible by bands")
    uf = UnionFind(len(rows))
    exact: dict[str, int] = {}
    shingle_sets: list[set[str]] = []
    signatures: list[tuple[int, ...]] = []
    for idx, row in enumerate(rows):
        text = normalized_text(row["headline"]) + "\n" + normalized_text(row["abstract"])
        if text in exact:
            uf.union(idx, exact[text])
        else:
            exact[text] = idx
        shingles = _shingles(text)
        shingle_sets.append(shingles)
        signatures.append(tuple(min(_stable_hash(s, p) for s in shingles) for p in range(permutations)))

    rows_per_band = permutations // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for idx, signature in enumerate(signatures):
        for band in range(bands):
            start = band * rows_per_band
            key = (band, signature[start : start + rows_per_band])
            for other in buckets[key]:
                candidates.add((other, idx))
            buckets[key].append(idx)

    near_pairs: list[dict[str, Any]] = []
    for a, b in sorted(candidates):
        left, right = shingle_sets[a], shingle_sets[b]
        ratio = min(len(left), len(right)) / max(len(left), len(right))
        if ratio < min_length_ratio:
            continue
        score = len(left & right) / max(1, len(left | right))
        if score >= jaccard_threshold:
            uf.union(a, b)
            near_pairs.append(
                {"left_article_id": rows[a]["article_id"], "right_article_id": rows[b]["article_id"], "jaccard": round(score, 6)}
            )

    grouped: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(rows)):
        grouped[uf.find(idx)].append(idx)
    groups = sorted(grouped.values(), key=lambda group: min(group))
    audit = {
        "method": "exact normalized title+abstract plus 64-permutation MinHash candidates and word-trigram Jaccard",
        "candidate_pair_count": len(candidates),
        "near_duplicate_pair_count": len(near_pairs),
        "multirow_group_count": sum(len(group) > 1 for group in groups),
        "rows_in_multirow_groups": sum(len(group) for group in groups if len(group) > 1),
        "largest_group_size": max(map(len, groups), default=0),
        "near_duplicate_pairs": near_pairs,
    }
    return groups, audit


def load_labeled_rows(manifest_path: Path, annotations_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_jsonl(manifest_path)
    results = read_jsonl(annotations_path)
    result_by_id = {row["article_id"]: row for row in results if row.get("ok")}
    if len(result_by_id) != len(results):
        raise ValueError("Annotation results contain failed or duplicate article IDs")
    topic_set, tone_set = set(TOPIC_LABELS), set(TONE_LABELS)
    rows: list[dict[str, Any]] = []
    unknown_topics: Counter[str] = Counter()
    unknown_tones: Counter[str] = Counter()
    for source in manifest:
        article_id = source["article_id"]
        if article_id not in result_by_id:
            raise ValueError(f"Missing annotation for {article_id}")
        annotation = result_by_id[article_id]["annotation"]
        relevance = annotation.get("relevance")
        topics = sorted(
            {item.get("topic") for item in annotation.get("topics", []) if item.get("topic") in topic_set},
            key=TOPIC_LABELS.index,
        )
        unknown_topics.update(
            item.get("topic") for item in annotation.get("topics", []) if item.get("topic") not in topic_set
        )
        tone = annotation.get("overall_tone")
        if tone is not None and tone not in tone_set:
            unknown_tones[tone] += 1

        primary_topic = annotation.get("primary_topic")
        primary_topic_valid = relevance == "relevant" and primary_topic in PRIMARY_TOPIC_LABELS
        # Empty topic lists created by relevance filtering are unknown, not all-negative.
        topic_valid = relevance == "relevant" and bool(topics)
        tone_valid = tone in tone_set
        rows.append(
            {
                "article_id": article_id,
                "headline": source.get("headline") or "",
                "abstract": source.get("abstract") or "",
                "topics": topics,
                "overall_tone": tone if tone_valid else None,
                "primary_topic": primary_topic if primary_topic_valid else None,
                "primary_topic_valid": primary_topic_valid,
                "topic_valid": topic_valid,
                "tone_valid": tone_valid,
                "joint_valid": topic_valid and tone_valid,
                "primary_joint_valid": primary_topic_valid and tone_valid,
                "label_relevance": relevance,
            }
        )
    if len(rows) != 5000 or len({row["article_id"] for row in rows}) != 5000:
        raise ValueError(f"Expected exactly 5,000 unique aligned rows, found {len(rows)}")
    audit = {
        "row_count": len(rows),
        "unique_article_ids": len({row["article_id"] for row in rows}),
        "unknown_topics": dict(unknown_topics),
        "unknown_tones": dict(unknown_tones),
        "topic_label_policy": (
            "topic_valid is true only for GPT-relevant rows with at least one topic; relevance-filtered empty lists are masked"
        ),
        "tone_label_policy": "overall_tone=None is masked and is never converted to neutral",
        "input_feature_policy": "model input is constructed only from headline and abstract",
    }
    return rows, audit


def _stratification_vector(row: dict[str, Any]) -> list[int]:
    vector = [int(row["topic_valid"] and label in row["topics"]) for label in TOPIC_LABELS]
    vector.extend(int(row["tone_valid"] and row["overall_tone"] == label) for label in TONE_LABELS)
    vector.extend([int(row["topic_valid"]), int(row["tone_valid"])])
    return vector


def stratified_group_split(
    rows: list[dict[str, Any]], groups: list[list[int]], sizes: dict[str, int], seed: int
) -> dict[str, list[int]]:
    """Iteratively allocate indivisible groups while matching rare-label marginals."""
    if sum(sizes.values()) != len(rows):
        raise ValueError("Requested split sizes do not sum to row count")
    names = ["train", "validation", "test"]
    labels = [_stratification_vector(row) for row in rows]
    totals = [sum(vector[j] for vector in labels) for j in range(len(labels[0]))]
    targets = {name: [value * sizes[name] / len(rows) for value in totals] for name in names}
    desired = {name: list(values) for name, values in targets.items()}
    assigned = {name: [] for name in names}
    remaining = dict(sizes)
    rng = random.Random(seed)

    group_info: dict[int, tuple[list[int], list[int], float]] = {}
    label_groups: list[set[int]] = [set() for _ in totals]
    for group_id, group in enumerate(groups):
        vector = [sum(labels[idx][j] for idx in group) for j in range(len(totals))]
        rarity = sum(value / max(1, totals[j]) for j, value in enumerate(vector))
        group_info[group_id] = (group, vector, rarity)
        for j, value in enumerate(vector):
            if value:
                label_groups[j].add(group_id)

    unassigned = set(group_info)
    label_remaining = list(totals)
    while unassigned:
        positive_labels = [j for j, count in enumerate(label_remaining) if count > 0]
        if positive_labels:
            rarest = min(positive_labels, key=lambda j: (label_remaining[j], j))
            candidates = list(label_groups[rarest] & unassigned)
            candidates.sort(
                key=lambda group_id: (
                    -group_info[group_id][2],
                    -sum(group_info[group_id][1]),
                    rng.random(),
                )
            )
            group_id = candidates[0]
        else:
            group_id = min(unassigned, key=lambda value: rng.random())
        group, vector, _ = group_info[group_id]
        feasible = [name for name in names if remaining[name] >= len(group)]
        if not feasible:
            raise ValueError("Group constraints prevent exact requested split sizes; no rows were duplicated")
        if positive_labels:
            # Score every label carried by this group, not just the currently
            # rarest one. This prevents disjoint rare labels from collectively
            # overfilling the smaller splits' general valid-label margins.
            need = {
                name: sum(
                    vector[j] * desired[name][j] / max(1.0, targets[name][j])
                    for j in range(len(totals))
                )
                for name in feasible
            }
            best_need = max(need.values())
            choices = [name for name in feasible if math.isclose(need[name], best_need, abs_tol=1e-12)]
        else:
            choices = feasible
        chosen = max(choices, key=lambda name: (remaining[name] / sizes[name], -names.index(name)))
        assigned[chosen].extend(group)
        remaining[chosen] -= len(group)
        desired[chosen] = [need - actual for need, actual in zip(desired[chosen], vector)]
        unassigned.remove(group_id)
        label_remaining = [need - actual for need, actual in zip(label_remaining, vector)]

    if any(remaining.values()):
        raise ValueError(f"Could not meet exact split sizes: {remaining}")
    return assigned


def distribution(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "rows": len(rows),
        "label_relevance": dict(Counter(row["label_relevance"] for row in rows)),
        "valid": {
            "topic": sum(row["topic_valid"] for row in rows),
            "primary_topic": sum(row["primary_topic_valid"] for row in rows),
            "tone": sum(row["tone_valid"] for row in rows),
            "joint": sum(row["joint_valid"] for row in rows),
            "primary_joint": sum(row["primary_joint_valid"] for row in rows),
        },
        "topics_positive_support": {label: sum(row["topic_valid"] and label in row["topics"] for row in rows) for label in TOPIC_LABELS},
        "overall_tone_support": {label: sum(row["tone_valid"] and row["overall_tone"] == label for row in rows) for label in TONE_LABELS},
        "primary_topic_support": {label: sum(row["primary_topic_valid"] and row["primary_topic"] == label for row in rows) for label in PRIMARY_TOPIC_LABELS},
    }


def prepare_data(config: dict[str, Any], root: Path) -> dict[str, Any]:
    source = config["source"]
    rows, audit = load_labeled_rows(root / source["manifest"], root / source["annotations"])
    duplicate_cfg = config["near_duplicate"]
    groups, duplicate_audit = duplicate_groups(
        rows,
        permutations=duplicate_cfg["minhash_permutations"],
        bands=duplicate_cfg["bands"],
        jaccard_threshold=duplicate_cfg["jaccard_threshold"],
        min_length_ratio=duplicate_cfg["min_length_ratio"],
    )
    assigned = stratified_group_split(rows, groups, config["split_sizes"], config["seed"])
    output_dir = root / config["data_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest: dict[str, list[str]] = {}
    distributions = {"all": distribution(rows)}
    for name in ["train", "validation", "test"]:
        split_rows = [rows[idx] for idx in sorted(assigned[name])]
        split_manifest[name] = [row["article_id"] for row in split_rows]
        distributions[name] = distribution(split_rows)
        write_jsonl(output_dir / f"{name}.jsonl", split_rows)
    write_json(output_dir / "article_id_splits.json", split_manifest)
    write_json(output_dir / "label_distribution.json", distributions)
    write_json(output_dir / "duplicate_audit.json", duplicate_audit)
    audit.update(
        {
            "seed": config["seed"],
            "requested_split_sizes": config["split_sizes"],
            "actual_split_sizes": {name: len(ids) for name, ids in split_manifest.items()},
            "original_4500_500_split_found": False,
            "duplicate_group_count": len(groups),
            "no_cross_split_duplicate_groups": True,
        }
    )
    membership = {idx: name for name, indices in assigned.items() for idx in indices}
    for group in groups:
        if len({membership[idx] for idx in group}) != 1:
            audit["no_cross_split_duplicate_groups"] = False
            break
    write_json(output_dir / "data_audit.json", audit)
    return {"audit": audit, "duplicates": duplicate_audit, "distribution": distributions}
