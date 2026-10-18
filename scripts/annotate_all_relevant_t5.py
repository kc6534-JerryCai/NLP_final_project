from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from econmedia_finetuning.compact import build_t5_input, parse_t5_output


class RelevantDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(
            build_t5_input(row["headline"], row["abstract"]),
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["row_index"] = index
        return encoded


def collate(tokenizer):
    def inner(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        indices = [item.pop("row_index") for item in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["row_index"] = torch.tensor(indices, dtype=torch.long)
        return batch

    return inner


def read_relevant(path: Path) -> list[dict[str, Any]]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/clean/relevant_only.jsonl")
    parser.add_argument("--model", type=Path, default=ROOT / "models/econmedia-t5-compact")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/all_relevant/t5")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_relevant(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "annotations.jsonl"
    progress_path = args.output_dir / "progress.json"
    summary_path = args.output_dir / "summary.json"

    completed: dict[str, dict[str, Any]] = {}
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    completed[item["article_id"]] = item
    pending = [row for row in rows if row["article_id"] not in completed]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = json.loads((args.model / "training_metadata.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=True).to(device).eval()
    if device.type == "cuda":
        model = model.half()
    model.config.use_cache = True

    dataset = RelevantDataset(pending, tokenizer, int(metadata["max_source_length"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate(tokenizer),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    started = time.time()
    mode = "a" if predictions_path.exists() else "w"
    with predictions_path.open(mode, encoding="utf-8") as output:
        for batch_number, batch in enumerate(tqdm(loader, desc="T5 all relevant"), start=1):
            indices = batch.pop("row_index").tolist()
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                generated = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for index, raw in zip(indices, texts):
                row = pending[index]
                parsed = parse_t5_output(raw)
                result = {
                    **row,
                    "model": "econmedia-t5-compact",
                    "primary_topic": parsed.primary_topic,
                    "tone": int(parsed.tone) if parsed.tone is not None else None,
                    "raw_output": raw,
                    "format_valid": parsed.valid,
                    "parse_errors": parsed.errors,
                    "unknown_labels": parsed.unknown_labels,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            if batch_number % 25 == 0 or batch_number == len(loader):
                done = len(completed) + min(batch_number * args.batch_size, len(pending))
                elapsed = time.time() - started
                progress_path.write_text(
                    json.dumps(
                        {
                            "status": "running" if done < len(rows) else "complete",
                            "total": len(rows),
                            "completed": done,
                            "remaining": len(rows) - done,
                            "elapsed_seconds": elapsed,
                            "rows_per_second": (done - len(completed)) / elapsed if elapsed else None,
                            "device": str(device),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

    final_rows = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        final_rows = [json.loads(line) for line in handle if line.strip()]
    topic_counts = Counter(row.get("primary_topic") for row in final_rows)
    tone_counts = Counter(str(row.get("tone")) for row in final_rows)
    invalid = sum(not row.get("format_valid", False) for row in final_rows)
    summary = {
        "status": "complete",
        "source": str(args.input.relative_to(ROOT)),
        "model": str(args.model.relative_to(ROOT)),
        "total": len(final_rows),
        "format_valid": len(final_rows) - invalid,
        "format_invalid": invalid,
        "topic_counts": dict(sorted(topic_counts.items(), key=lambda item: str(item[0]))),
        "tone_counts": dict(sorted(tone_counts.items())),
        "device": str(device),
        "elapsed_seconds": time.time() - started,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
