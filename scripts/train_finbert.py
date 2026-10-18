from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from econmedia_finetuning.constants import PRIMARY_TOPIC_LABELS, TONE_LABELS
from econmedia_finetuning.datasets import FinBertCollator, FinBertDataset
from econmedia_finetuning.metrics import combined_primary_metrics
from econmedia_finetuning.modeling_econmediabert import EconMediaBertForJointClassification
from econmedia_finetuning.utils import load_config, read_jsonl, seed_everything, write_json, write_jsonl


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if key != "row_index"}


@torch.inference_mode()
def evaluate(
    model,
    loader,
    rows: list[dict[str, Any]],
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    primary_true, primary_pred, tone_true, tone_pred = [], [], [], []
    predictions = []
    losses = []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        row_indices = batch.pop("row_index").tolist()
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(**batch)
        if output.loss is not None:
            losses.append(float(output.loss.detach().cpu()))
        primary_prob = torch.softmax(output.primary_topic_logits, dim=-1).float().cpu().numpy()
        primary_hat = primary_prob.argmax(axis=-1)
        tone_prob = torch.softmax(output.tone_logits, dim=-1).float().cpu().numpy()
        tone_hat = tone_prob.argmax(axis=-1)
        for pos, row_index in enumerate(row_indices):
            row = rows[row_index]
            if row["primary_topic_valid"]:
                primary_true.append(PRIMARY_TOPIC_LABELS.index(row["primary_topic"]))
                primary_pred.append(int(primary_hat[pos]))
            if row["tone_valid"]:
                tone_true.append(TONE_LABELS.index(row["overall_tone"]))
                tone_pred.append(int(tone_hat[pos]))
            predictions.append(
                {
                    "article_id": row["article_id"],
                    "primary_topic_label_valid": row["primary_topic_valid"],
                    "true_primary_topic": row["primary_topic"] if row["primary_topic_valid"] else None,
                    "predicted_primary_topic": PRIMARY_TOPIC_LABELS[int(primary_hat[pos])],
                    "primary_topic_probabilities": {label: float(value) for label, value in zip(PRIMARY_TOPIC_LABELS, primary_prob[pos])},
                    "tone_label_valid": row["tone_valid"],
                    "true_overall_tone": row["overall_tone"] if row["tone_valid"] else None,
                    "predicted_overall_tone": TONE_LABELS[int(tone_hat[pos])],
                    "tone_probabilities": {label: float(value) for label, value in zip(TONE_LABELS, tone_prob[pos])},
                }
            )
    metrics = combined_primary_metrics(
        np.asarray(primary_true, dtype=np.int64),
        np.asarray(primary_pred, dtype=np.int64),
        np.asarray(tone_true, dtype=np.int64),
        np.asarray(tone_pred, dtype=np.int64),
        mean_loss=float(np.mean(losses)) if losses else None,
    )
    return metrics, predictions


def save_checkpoint(model, tokenizer, directory: Path, metadata: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    shutil.copy2(ROOT / "econmedia_finetuning" / "modeling_econmediabert.py", directory / "modeling_econmediabert.py")
    write_json(directory / "label_mapping.json", {"primary_topic": PRIMARY_TOPIC_LABELS, "overall_tone": TONE_LABELS})
    write_json(directory / "training_metadata.json", metadata)


def train_trial(
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    tokenizer,
    max_length: int,
    encoder_lr: float,
    epochs: int,
    per_device_batch: int,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(config["seed"])
    cfg = config["finbert"]
    model = EconMediaBertForJointClassification.from_finbert(
        cfg["base_model"],
        num_primary_topic_labels=len(PRIMARY_TOPIC_LABELS),
        num_tone_labels=len(TONE_LABELS),
        primary_topic_loss_weight=1.0,
        tone_loss_weight=1.0,
        primary_topic_labels=PRIMARY_TOPIC_LABELS,
        tone_labels=TONE_LABELS,
    ).to(device)
    collator = FinBertCollator(tokenizer)
    train_dataset = FinBertDataset(train_rows, tokenizer, max_length, supervised_only=True)
    val_dataset = FinBertDataset(val_rows, tokenizer, max_length, supervised_only=False)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["eval_batch_size"], shuffle=False, collate_fn=collator, num_workers=0)
    head_params = list(model.primary_topic_classifier.parameters()) + list(model.tone_classifier.parameters())
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": encoder_lr},
            {"params": head_params, "lr": cfg["head_learning_rate"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    accumulation = max(1, math.ceil(cfg["effective_batch_size"] / per_device_batch))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_updates * cfg["warmup_ratio"]),
        num_training_steps=total_updates,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    best_score = -1.0
    best_epoch = 0
    best_dir = output_dir / "best"
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"FinBERT lr={encoder_lr:g} epoch={epoch}"), start=1):
            batch.pop("row_index")
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(**batch)
                loss = output.loss / accumulation
            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * accumulation
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        val_metrics, _ = evaluate(model, val_loader, val_dataset.rows, device, use_amp)
        record = {"epoch": epoch, "train_mean_batch_loss": running_loss / len(train_loader), "validation": val_metrics}
        history.append(record)
        write_json(output_dir / "history.json", history)
        score = val_metrics["selection_score"] if val_metrics["selection_score"] is not None else -1.0
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(
                model,
                tokenizer,
                best_dir,
                {
                    "base_model": cfg["base_model"],
                    "encoder_learning_rate": encoder_lr,
                    "head_learning_rate": cfg["head_learning_rate"],
                    "effective_batch_size": cfg["effective_batch_size"],
                    "per_device_batch_size": per_device_batch,
                    "gradient_accumulation_steps": accumulation,
                    "epoch": epoch,
                    "max_length": max_length,
                    "validation_metrics": val_metrics,
                    "seed": config["seed"],
                },
            )
        print(json.dumps({"epoch": epoch, "encoder_lr": encoder_lr, "selection_score": score}, ensure_ascii=False))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "encoder_learning_rate": encoder_lr,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_checkpoint": str(best_dir),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-parameter joint fine-tuning of ProsusAI/finbert")
    parser.add_argument("--config", default="config/finetuning.v1.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--per-device-batch-size", type=int)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    cfg = config["finbert"]
    data_dir = ROOT / config["data_dir"]
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "validation.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    if args.smoke:
        train_rows = [row for row in train_rows if row["primary_joint_valid"]][:16]
        val_rows = [row for row in val_rows if row["primary_joint_valid"]][:8]
        test_rows = [row for row in test_rows if row["primary_joint_valid"]][:8]
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    if args.eval_limit:
        val_rows, test_rows = val_rows[: args.eval_limit], test_rows[: args.eval_limit]
    length_audit = json.loads((data_dir / "token_length_audit.json").read_text(encoding="utf-8"))
    max_length = length_audit["finbert"]["selected_max_length"] if cfg["max_length"] == "auto" else int(cfg["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = ROOT / config["output_dir"] / ("smoke_finbert" if args.smoke else "finbert")
    learning_rates = [cfg["encoder_learning_rates"][0]] if args.smoke else cfg["encoder_learning_rates"]
    epochs = args.epochs or (1 if args.smoke else cfg["epochs"])
    per_device_batch = args.per_device_batch_size or cfg["per_device_batch_size"]
    trials = []
    for lr in learning_rates:
        trials.append(
            train_trial(
                config,
                train_rows,
                val_rows,
                tokenizer,
                max_length,
                lr,
                epochs,
                per_device_batch,
                output_root / f"encoder_lr_{lr:g}",
                device,
            )
        )
    best = max(trials, key=lambda item: item["best_selection_score"])
    final_dir = output_root / "selected_model" if args.smoke else ROOT / config["models_dir"] / "econmediabert"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(best["best_checkpoint"], final_dir)
    selected_model = EconMediaBertForJointClassification.from_pretrained(final_dir).to(device)
    test_dataset = FinBertDataset(test_rows, tokenizer, max_length, supervised_only=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["eval_batch_size"], shuffle=False, collate_fn=FinBertCollator(tokenizer))
    test_metrics, predictions = evaluate(selected_model, test_loader, test_dataset.rows, device, device.type == "cuda")
    errors = [
        row
        for row in predictions
        if (row["primary_topic_label_valid"] and row["true_primary_topic"] != row["predicted_primary_topic"])
        or (row["tone_label_valid"] and row["true_overall_tone"] != row["predicted_overall_tone"])
    ]
    report = {"model": "econmediabert", "task": "primary topic single-label plus overall tone", "device": str(device), "trials": trials, "selected": best, "test": test_metrics, "gpt_label_caveat": "Metrics measure agreement with GPT-generated labels, not accuracy against an independent human gold standard."}
    write_json(output_root / "experiment_results.json", report)
    write_jsonl(output_root / "test_predictions.jsonl", predictions)
    write_jsonl(output_root / "test_errors.jsonl", errors)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
