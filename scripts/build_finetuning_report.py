from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    out = ROOT / "outputs" / "finetuning"
    records = []
    validation_rows = []
    full = {}
    for key, directory in [("FinBERT", "finbert"), ("FLAN-T5", "t5")]:
        result = json.loads((out / directory / "experiment_results.json").read_text(encoding="utf-8"))
        selected = result["selected"]
        validation = selected["history"][selected["best_epoch"] - 1]["validation"]
        test = result["test"]
        record = {
            "model": key,
            "selected_learning_rate": selected.get("encoder_learning_rate", selected.get("learning_rate")),
            "best_epoch": selected["best_epoch"],
            "validation_selection_score": validation["selection_score"],
            "validation_primary_topic_accuracy": validation["primary_topic"]["accuracy"],
            "validation_primary_topic_macro_f1": validation["primary_topic"]["macro_f1"],
            "validation_tone_accuracy": validation["overall_tone"]["accuracy"],
            "validation_tone_macro_f1": validation["overall_tone"]["macro_f1"],
            "test_selection_score": test["selection_score"],
            "test_primary_topic_n": test["primary_topic"]["effective_n"],
            "test_primary_topic_accuracy": test["primary_topic"]["accuracy"],
            "test_primary_topic_macro_f1": test["primary_topic"]["macro_f1"],
            "test_tone_n": test["overall_tone"]["effective_n"],
            "test_tone_accuracy": test["overall_tone"]["accuracy"],
            "test_tone_macro_f1": test["overall_tone"]["macro_f1"],
            "test_format_valid_rate": test.get("output_format_valid_rate"),
        }
        records.append(record)
        full[key] = {"summary": record, "validation": validation, "test": test}
        for trial in result["trials"]:
            learning_rate = trial.get("encoder_learning_rate", trial.get("learning_rate"))
            for epoch in trial["history"]:
                metrics = epoch["validation"]
                validation_rows.append(
                    {
                        "model": key,
                        "learning_rate": learning_rate,
                        "epoch": epoch["epoch"],
                        "selection_score": metrics["selection_score"],
                        "primary_topic_accuracy": metrics["primary_topic"]["accuracy"],
                        "primary_topic_macro_f1": metrics["primary_topic"]["macro_f1"],
                        "tone_accuracy": metrics["overall_tone"]["accuracy"],
                        "tone_macro_f1": metrics["overall_tone"]["macro_f1"],
                        "format_valid_rate": metrics.get("output_format_valid_rate"),
                        "validation_loss": metrics.get("mean_loss", metrics.get("mean_validation_loss")),
                    }
                )
    with (out / "model_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (out / "validation_experiments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)
    (out / "model_comparison.json").write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "split_manifest_sha256": sha256(ROOT / "data" / "finetuning" / "article_id_splits.json"),
        "model_artifacts": {
            name: {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in directory.iterdir() if path.is_file()}
            for name, directory in {
                "econmediabert": ROOT / "models" / "econmediabert",
                "econmedia-t5": ROOT / "models" / "econmedia-t5",
            }.items()
        },
    }
    (out / "environment_and_artifacts.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    test_rows = {}
    with (ROOT / "data" / "finetuning" / "test.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            test_rows[row["article_id"]] = {"headline": row["headline"], "abstract": row["abstract"]}
    for directory in ("finbert", "t5"):
        for filename in ("test_predictions.jsonl", "test_errors.jsonl"):
            path = out / directory / filename
            enriched = []
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    enriched.append({**test_rows[row["article_id"]], **row})
            with path.open("w", encoding="utf-8") as handle:
                for row in enriched:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
