from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from econmedia_finetuning.data import article_text
from econmedia_finetuning.t5_format import build_t5_input, build_t5_target
from econmedia_finetuning.utils import load_config, read_jsonl, write_json


def describe(lengths: list[int], candidates: list[int], max_fraction: float) -> dict:
    ordered = sorted(lengths)
    quantiles = {
        str(q): ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]
        for q in (0.5, 0.9, 0.95, 0.99, 1.0)
    }
    truncation = {str(length): sum(value > length for value in lengths) / len(lengths) for length in candidates}
    selected = next((length for length in candidates if truncation[str(length)] <= max_fraction), candidates[-1])
    return {"n": len(lengths), "quantiles": quantiles, "truncation_fraction": truncation, "selected_max_length": selected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/finetuning.v1.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    rows = []
    for split in ("train", "validation", "test"):
        rows.extend(read_jsonl(ROOT / config["data_dir"] / f"{split}.jsonl"))
    if args.limit:
        rows = rows[: args.limit]
    bert_tokenizer = AutoTokenizer.from_pretrained(config["finbert"]["base_model"])
    t5_tokenizer = AutoTokenizer.from_pretrained(config["t5"]["base_model"])
    candidates = config["token_length"]["candidate_lengths"]
    max_fraction = config["token_length"]["max_truncation_fraction"]
    bert_lengths = [len(bert_tokenizer(article_text(row["headline"], row["abstract"]), truncation=False)["input_ids"]) for row in rows]
    t5_lengths = [len(t5_tokenizer(build_t5_input(row["headline"], row["abstract"]), truncation=False)["input_ids"]) for row in rows]
    target_rows = [row for row in rows if row["primary_joint_valid"]]
    target_lengths = [len(t5_tokenizer(build_t5_target(row["primary_topic"], row["overall_tone"]), truncation=False)["input_ids"]) for row in target_rows]
    report = {
        "selection_rule": f"smallest configured length with truncation fraction <= {max_fraction}",
        "finbert": describe(bert_lengths, candidates, max_fraction),
        "t5_source": describe(t5_lengths, candidates, max_fraction),
        "t5_target": describe(target_lengths, [32, 64, 96, 128], 0.0),
    }
    write_json(ROOT / config["data_dir"] / "token_length_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
