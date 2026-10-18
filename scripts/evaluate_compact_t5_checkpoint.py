from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from econmedia_finetuning.compact import T5Collator, T5Dataset
from econmedia_finetuning.utils import load_config, read_jsonl, write_json, write_jsonl
from scripts.train_compact_t5 import evaluate


def main() -> None:
    config = load_config(ROOT / "config/finetuning.compact.v2.json")
    cfg = config["t5"]
    data_dir = ROOT / config["data_dir"]
    output_root = ROOT / config["output_dir"] / "t5"
    source_dir = output_root / "lr_3e-05" / "best"
    final_dir = ROOT / config["models_dir"] / "econmedia-t5-compact"
    if not final_dir.exists():
        shutil.copytree(source_dir, final_dir)

    test_rows = read_jsonl(data_dir / "test.jsonl")
    token_audit = json.loads((data_dir / "token_length_audit.json").read_text(encoding="utf-8"))
    source_length = token_audit["t5_source"]["selected_max_length"]
    target_length = min(cfg["max_target_length"], token_audit["t5_target"]["selected_max_length"])
    tokenizer = AutoTokenizer.from_pretrained(final_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(final_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = T5Dataset(test_rows, tokenizer, source_length, target_length)
    loader = DataLoader(
        dataset,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        collate_fn=T5Collator(tokenizer),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    metrics, predictions = evaluate(
        model,
        loader,
        dataset.rows,
        tokenizer,
        device,
        cfg["generation_max_new_tokens"],
        use_amp,
        torch.bfloat16,
    )
    errors = [
        row
        for row in predictions
        if not row["format_valid"]
        or row["true_primary_topic"] != row["predicted_primary_topic"]
        or row["true_tone"] != row["predicted_tone"]
    ]
    history = json.loads((output_root / "lr_3e-05" / "history.json").read_text(encoding="utf-8"))
    report = {
        "model": "econmedia-t5-compact",
        "training_status": "selected from completed 3e-5 trial; 1e-4 trial stopped by user before epoch 1 completed",
        "selected": {
            "learning_rate": 3e-5,
            "best_epoch": 5,
            "best_selection_score": max(row["validation"]["selection_score"] for row in history),
            "epochs_completed": len(history),
        },
        "test": metrics,
        "gpt_label_caveat": "Metrics measure agreement with GPT-generated labels, not accuracy against an independent human gold standard.",
    }
    write_json(output_root / "partial_experiment_results.json", report)
    write_jsonl(output_root / "test_predictions.jsonl", predictions)
    write_jsonl(output_root / "test_errors.jsonl", errors)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
