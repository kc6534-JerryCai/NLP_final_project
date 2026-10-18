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

from econmedia_finetuning.compact import (
    TOPIC_LABELS,
    TONE_LABELS,
    FinBertCollator,
    FinBertDataset,
    combined_metrics,
)
from econmedia_finetuning.modeling_econmediabert import EconMediaBertForJointClassification
from econmedia_finetuning.utils import load_config, read_jsonl, seed_everything, write_json, write_jsonl


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if key != "row_index"}


@torch.inference_mode()
def evaluate(model, loader, rows: list[dict[str, Any]], device, use_amp: bool):
    model.eval()
    topic_true, topic_pred, tone_true, tone_pred = [], [], [], []
    predictions, losses = [], []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        row_indices = batch.pop("row_index").tolist()
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))
        topic_prob = torch.softmax(output.primary_topic_logits, dim=-1).float().cpu().numpy()
        tone_prob = torch.softmax(output.tone_logits, dim=-1).float().cpu().numpy()
        topic_hat = topic_prob.argmax(axis=-1)
        tone_hat = tone_prob.argmax(axis=-1)
        for position, row_index in enumerate(row_indices):
            row = rows[row_index]
            true_topic = TOPIC_LABELS.index(row["primary_topic"])
            true_tone = TONE_LABELS.index(row["tone"])
            topic_true.append(true_topic)
            topic_pred.append(int(topic_hat[position]))
            tone_true.append(true_tone)
            tone_pred.append(int(tone_hat[position]))
            predictions.append(
                {
                    "article_id": row["article_id"],
                    "true_primary_topic": row["primary_topic"],
                    "predicted_primary_topic": TOPIC_LABELS[int(topic_hat[position])],
                    "primary_topic_probabilities": {
                        label: float(value) for label, value in zip(TOPIC_LABELS, topic_prob[position])
                    },
                    "true_tone": row["tone"],
                    "predicted_tone": TONE_LABELS[int(tone_hat[position])],
                    "tone_probabilities": {
                        label: float(value) for label, value in zip(TONE_LABELS, tone_prob[position])
                    },
                }
            )
    metrics = combined_metrics(
        np.asarray(topic_true, dtype=np.int64),
        np.asarray(topic_pred, dtype=np.int64),
        np.asarray(tone_true, dtype=np.int64),
        np.asarray(tone_pred, dtype=np.int64),
        mean_loss=float(np.mean(losses)),
    )
    return metrics, predictions


def save_checkpoint(model, tokenizer, directory: Path, metadata: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    shutil.copy2(
        ROOT / "econmedia_finetuning" / "modeling_econmediabert.py",
        directory / "modeling_econmediabert.py",
    )
    write_json(directory / "label_mapping.json", {"primary_topic": TOPIC_LABELS, "tone": TONE_LABELS})
    write_json(directory / "training_metadata.json", metadata)


def train_trial(config, mode, train_rows, val_rows, tokenizer, max_length, learning_rate, epochs, output_dir, device):
    seed_everything(config["seed"])
    cfg = config[f"finbert_{mode}"]
    model = EconMediaBertForJointClassification.from_finbert(
        cfg["base_model"],
        local_files_only=True,
        num_primary_topic_labels=len(TOPIC_LABELS),
        num_tone_labels=len(TONE_LABELS),
        primary_topic_loss_weight=1.0,
        tone_loss_weight=1.0,
        primary_topic_labels=TOPIC_LABELS,
        tone_labels=TONE_LABELS,
    ).to(device)
    head_parameters = list(model.primary_topic_classifier.parameters()) + list(model.tone_classifier.parameters())
    if mode == "head":
        for parameter in model.bert.parameters():
            parameter.requires_grad = False
        optimizer = AdamW(head_parameters, lr=learning_rate, weight_decay=cfg["weight_decay"])
    else:
        optimizer = AdamW(
            [
                {"params": model.bert.parameters(), "lr": learning_rate},
                {"params": head_parameters, "lr": cfg["head_learning_rate"]},
            ],
            weight_decay=cfg["weight_decay"],
        )
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    train_dataset = FinBertDataset(train_rows, tokenizer, max_length)
    val_dataset = FinBertDataset(val_rows, tokenizer, max_length)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["per_device_batch_size"],
        shuffle=True,
        generator=generator,
        collate_fn=FinBertCollator(tokenizer),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        collate_fn=FinBertCollator(tokenizer),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    accumulation = max(1, math.ceil(cfg["effective_batch_size"] / cfg["per_device_batch_size"]))
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
    best_score, best_epoch, epochs_without_improvement = -1.0, 0, 0
    best_dir = output_dir / "best"
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        if mode == "head":
            model.bert.eval()
        running_loss = 0.0
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"FinBERT-{mode} lr={learning_rate:g} epoch={epoch}"), start=1
        ):
            batch.pop("row_index")
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(**batch)
                loss = output.loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite FinBERT loss")
            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * accumulation
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        val_metrics, _ = evaluate(model, val_loader, val_dataset.rows, device, use_amp)
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
                    "mode": "classification_heads_only" if mode == "head" else "full_parameter",
                    "base_model": cfg["base_model"],
                    "encoder_frozen": mode == "head",
                    "learning_rate": learning_rate,
                    "head_learning_rate": learning_rate if mode == "head" else cfg["head_learning_rate"],
                    "effective_batch_size": cfg["effective_batch_size"],
                    "per_device_batch_size": cfg["per_device_batch_size"],
                    "gradient_accumulation_steps": accumulation,
                    "epoch": epoch,
                    "max_length": max_length,
                    "trainable_parameters": trainable_parameters,
                    "total_parameters": total_parameters,
                    "validation_metrics": val_metrics,
                    "seed": config["seed"],
                },
            )
        else:
            epochs_without_improvement += 1
        print(
            json.dumps(
                {
                    "mode": mode,
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "selection_score": score,
                    "topic_accuracy": val_metrics["primary_topic"]["accuracy"],
                    "tone_accuracy": val_metrics["overall_tone"]["accuracy"],
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
        "total_parameters": total_parameters,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/finetuning.compact.v2.json")
    parser.add_argument("--mode", choices=["head", "full"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    cfg = config[f"finbert_{args.mode}"]
    data_dir = ROOT / config["data_dir"]
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "validation.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    if args.smoke:
        train_rows, val_rows, test_rows = train_rows[:16], val_rows[:8], test_rows[:8]
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    if args.eval_limit:
        val_rows, test_rows = val_rows[: args.eval_limit], test_rows[: args.eval_limit]
    token_audit = json.loads((data_dir / "token_length_audit.json").read_text(encoding="utf-8"))
    max_length = token_audit["finbert"]["selected_max_length"] if cfg["max_length"] == "auto" else int(cfg["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    output_root = ROOT / config["output_dir"] / (f"smoke_finbert_{args.mode}" if args.smoke else f"finbert_{args.mode}")
    learning_rates = (
        cfg["learning_rates"] if args.mode == "head" else cfg["encoder_learning_rates"]
    )
    if args.smoke:
        learning_rates = learning_rates[:1]
    epochs = args.epochs or (1 if args.smoke else cfg["epochs"])
    trials = [
        train_trial(
            config,
            args.mode,
            train_rows,
            val_rows,
            tokenizer,
            max_length,
            learning_rate,
            epochs,
            output_root / f"lr_{learning_rate:g}",
            device,
        )
        for learning_rate in learning_rates
    ]
    best = max(trials, key=lambda trial: trial["best_selection_score"])
    if args.smoke:
        final_dir = output_root / "selected_model"
    else:
        model_name = "econmediabert-head-only" if args.mode == "head" else "econmediabert-full"
        final_dir = ROOT / config["models_dir"] / model_name
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(best["best_checkpoint"], final_dir)
    selected_model = EconMediaBertForJointClassification.from_pretrained(final_dir).to(device)
    test_dataset = FinBertDataset(test_rows, tokenizer, max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        collate_fn=FinBertCollator(tokenizer),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_metrics, predictions = evaluate(selected_model, test_loader, test_dataset.rows, device, device.type == "cuda")
    errors = [
        row
        for row in predictions
        if row["true_primary_topic"] != row["predicted_primary_topic"]
        or row["true_tone"] != row["predicted_tone"]
    ]
    report = {
        "model": final_dir.name,
        "training_mode": "classification_heads_only" if args.mode == "head" else "full_parameter",
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
