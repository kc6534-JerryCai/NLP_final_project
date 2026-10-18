from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from econmedia_finetuning.compact import (
    TOPIC_LABELS,
    TONE_LABELS,
    T5Collator,
    T5Dataset,
    combined_metrics,
    parse_t5_output,
    t5_parse_counts,
)
from econmedia_finetuning.utils import load_config, read_jsonl, seed_everything, write_json, write_jsonl


@torch.inference_mode()
def evaluate(model, loader, rows, tokenizer, device, max_new_tokens, use_amp, amp_dtype):
    model.eval()
    topic_true, topic_pred, tone_true, tone_pred = [], [], [], []
    predictions, losses = [], []
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    for batch in tqdm(loader, desc="generate/evaluate", leave=False):
        row_indices = batch.pop("row_index").tolist()
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))
        generation_inputs = {key: value for key, value in batch.items() if key != "labels"}
        generated = model.generate(
            **generation_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row_index, text in zip(row_indices, texts):
            row = rows[row_index]
            parsed = parse_t5_output(text)
            true_topic = TOPIC_LABELS.index(row["primary_topic"])
            true_tone = TONE_LABELS.index(row["tone"])
            predicted_topic = TOPIC_LABELS.index(parsed.primary_topic) if parsed.valid else -1
            predicted_tone = TONE_LABELS.index(parsed.tone) if parsed.valid else -1
            topic_true.append(true_topic)
            topic_pred.append(predicted_topic)
            tone_true.append(true_tone)
            tone_pred.append(predicted_tone)
            predictions.append(
                {
                    "article_id": row["article_id"],
                    "raw_output": text,
                    "format_valid": parsed.valid,
                    "parse_errors": parsed.errors,
                    "unknown_labels": parsed.unknown_labels,
                    "true_primary_topic": row["primary_topic"],
                    "predicted_primary_topic": parsed.primary_topic,
                    "true_tone": row["tone"],
                    "predicted_tone": parsed.tone,
                }
            )
    model.config.use_cache = previous_cache
    metrics = combined_metrics(
        np.asarray(topic_true, dtype=np.int64),
        np.asarray(topic_pred, dtype=np.int64),
        np.asarray(tone_true, dtype=np.int64),
        np.asarray(tone_pred, dtype=np.int64),
        mean_loss=float(np.mean(losses)),
        **t5_parse_counts(predictions),
    )
    return metrics, predictions


def save_checkpoint(model, tokenizer, directory: Path, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    write_json(directory / "label_mapping.json", {"primary_topic": TOPIC_LABELS, "tone": TONE_LABELS})
    write_json(directory / "training_metadata.json", metadata)


def train_trial(config, train_rows, val_rows, tokenizer, source_length, target_length, learning_rate, epochs, output_dir, device):
    seed_everything(config["seed"])
    cfg = config["t5"]
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["base_model"], local_files_only=True).to(device)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    train_dataset = T5Dataset(train_rows, tokenizer, source_length, target_length)
    val_dataset = T5Dataset(val_rows, tokenizer, source_length, target_length)
    collator = T5Collator(tokenizer)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["per_device_batch_size"],
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg["weight_decay"])
    accumulation = max(1, math.ceil(cfg["effective_batch_size"] / cfg["per_device_batch_size"]))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_updates * cfg["warmup_ratio"]),
        num_training_steps=total_updates,
    )
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16
    history = []
    best_score, best_epoch, epochs_without_improvement = -1.0, 0, 0
    best_dir = output_dir / "best"
    started = time.time()
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        model.config.use_cache = False
        running_loss = 0.0
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"T5 lr={learning_rate:g} epoch={epoch}"), start=1
        ):
            batch.pop("row_index")
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model(**batch).loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite T5 loss")
            loss.backward()
            running_loss += float(loss.detach().cpu()) * accumulation
            if step % accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        val_metrics, _ = evaluate(
            model,
            val_loader,
            val_dataset.rows,
            tokenizer,
            device,
            cfg["generation_max_new_tokens"],
            use_amp,
            amp_dtype,
        )
        score = val_metrics["selection_score"]
        history.append(
            {
                "epoch": epoch,
                "train_mean_batch_loss": running_loss / len(train_loader),
                "validation": val_metrics,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "history.json", history)
        if score > best_score + 1e-12:
            best_score, best_epoch, epochs_without_improvement = score, epoch, 0
            save_checkpoint(
                model,
                tokenizer,
                best_dir,
                {
                    "base_model": cfg["base_model"],
                    "learning_rate": learning_rate,
                    "effective_batch_size": cfg["effective_batch_size"],
                    "per_device_batch_size": cfg["per_device_batch_size"],
                    "gradient_accumulation_steps": accumulation,
                    "epoch": epoch,
                    "max_source_length": source_length,
                    "max_target_length": target_length,
                    "trainable_parameters": trainable_parameters,
                    "validation_metrics": val_metrics,
                    "seed": config["seed"],
                },
            )
        else:
            epochs_without_improvement += 1
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "selection_score": score,
                    "topic_accuracy": val_metrics["primary_topic"]["accuracy"],
                    "tone_accuracy": val_metrics["overall_tone"]["accuracy"],
                    "format_valid_rate": val_metrics["output_format_valid_rate"],
                }
            ),
            flush=True,
        )
        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            break
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_selection_score": best_score,
        "best_checkpoint": str(best_dir),
        "trainable_parameters": trainable_parameters,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/finetuning.compact.v2.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    cfg = config["t5"]
    data_dir = ROOT / config["data_dir"]
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "validation.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    if args.smoke:
        train_rows, val_rows, test_rows = train_rows[:8], val_rows[:4], test_rows[:4]
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    if args.eval_limit:
        val_rows, test_rows = val_rows[: args.eval_limit], test_rows[: args.eval_limit]
    token_audit = json.loads((data_dir / "token_length_audit.json").read_text(encoding="utf-8"))
    source_length = token_audit["t5_source"]["selected_max_length"] if cfg["max_source_length"] == "auto" else int(cfg["max_source_length"])
    target_length = min(cfg["max_target_length"], token_audit["t5_target"]["selected_max_length"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    output_root = ROOT / config["output_dir"] / ("smoke_t5" if args.smoke else "t5")
    learning_rates = cfg["learning_rates"][:1] if args.smoke else cfg["learning_rates"]
    epochs = args.epochs or (1 if args.smoke else cfg["epochs"])
    trials = [
        train_trial(
            config,
            train_rows,
            val_rows,
            tokenizer,
            source_length,
            target_length,
            learning_rate,
            epochs,
            output_root / f"lr_{learning_rate:g}",
            device,
        )
        for learning_rate in learning_rates
    ]
    best = max(trials, key=lambda trial: trial["best_selection_score"])
    final_dir = output_root / "selected_model" if args.smoke else ROOT / config["models_dir"] / "econmedia-t5-compact"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(best["best_checkpoint"], final_dir)
    selected_model = AutoModelForSeq2SeqLM.from_pretrained(final_dir).to(device)
    test_dataset = T5Dataset(test_rows, tokenizer, source_length, target_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        collate_fn=T5Collator(tokenizer),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    test_metrics, predictions = evaluate(
        selected_model,
        test_loader,
        test_dataset.rows,
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
    report = {
        "model": final_dir.name,
        "training_mode": "full_parameter_seq2seq",
        "task": "five-class primary topic plus three-class tone",
        "device": str(device),
        "split_row_counts": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "trials": trials,
        "selected": best,
        "test": test_metrics,
        "gpt_label_caveat": "Metrics measure agreement with GPT-generated labels, not accuracy against an independent human gold standard.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "experiment_results.json", report)
    write_jsonl(output_root / "test_predictions.jsonl", predictions)
    write_jsonl(output_root / "test_errors.jsonl", errors)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
