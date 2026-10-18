from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset


TOPIC_LABELS = [
    "cost_of_living_income",
    "labor_market",
    "household_economy",
    "macro_growth",
    "economic_policy",
]
TONE_LABELS = ["-1", "0", "1"]
T5_INSTRUCTION = (
    "Classify this U.S. economic news article. Return exactly one line in this format: "
    "primary_topic: <label>; tone: <label>. Primary topic must be one of "
    "cost_of_living_income, labor_market, household_economy, macro_growth, economic_policy. "
    "Tone must be -1, 0, or 1. Do not explain."
)


def article_text(headline: str, abstract: str) -> str:
    return f"Title: {headline.strip()}\nAbstract: {abstract.strip()}"


def build_t5_input(headline: str, abstract: str) -> str:
    return f"{T5_INSTRUCTION}\n\n{article_text(headline, abstract)}"


def build_t5_target(topic: str, tone: str) -> str:
    return f"primary_topic: {topic}; tone: {tone}"


@dataclass
class ParsedT5:
    valid: bool
    primary_topic: str | None
    tone: str | None
    errors: list[str]
    unknown_labels: list[str]


def parse_t5_output(text: str) -> ParsedT5:
    stripped = text.strip()
    match = re.fullmatch(r"primary_topic: ([a-z_]+); tone: (-1|0|1)", stripped)
    if not match:
        return ParsedT5(False, None, None, ["format_error"], [])
    topic, tone = match.groups()
    unknown = [] if topic in TOPIC_LABELS else [topic]
    valid = not unknown
    return ParsedT5(valid, topic if valid else None, tone if valid else None, [] if valid else ["unknown_label"], unknown)


class FinBertDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(
            article_text(row["headline"], row["abstract"]),
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded.update(
            {
                "primary_topic_labels": TOPIC_LABELS.index(row["primary_topic"]),
                "tone_labels": TONE_LABELS.index(row["tone"]),
                "primary_topic_valid_mask": 1.0,
                "tone_valid_mask": 1.0,
                "row_index": index,
            }
        )
        return encoded


class FinBertCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        model_keys = ["input_ids", "attention_mask", "token_type_ids"]
        padded = self.tokenizer.pad(
            [{key: item[key] for key in model_keys if key in item} for item in features],
            padding=True,
            return_tensors="pt",
        )
        padded["primary_topic_labels"] = torch.tensor(
            [item["primary_topic_labels"] for item in features], dtype=torch.long
        )
        padded["tone_labels"] = torch.tensor([item["tone_labels"] for item in features], dtype=torch.long)
        padded["primary_topic_valid_mask"] = torch.ones(len(features), dtype=torch.float32)
        padded["tone_valid_mask"] = torch.ones(len(features), dtype=torch.float32)
        padded["row_index"] = torch.tensor([item["row_index"] for item in features], dtype=torch.long)
        return padded


class T5Dataset(Dataset):
    def __init__(self, rows, tokenizer, max_source_length: int, max_target_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source = self.tokenizer(
            build_t5_input(row["headline"], row["abstract"]),
            truncation=True,
            max_length=self.max_source_length,
            padding=False,
        )
        target = self.tokenizer(
            text_target=build_t5_target(row["primary_topic"], row["tone"]),
            truncation=True,
            max_length=self.max_target_length,
            padding=False,
        )
        source["labels"] = target["input_ids"]
        source["row_index"] = index
        return source


class T5Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [item["labels"] for item in features]
        row_indices = [item["row_index"] for item in features]
        inputs = self.tokenizer.pad(
            [{key: value for key, value in item.items() if key not in {"labels", "row_index"}} for item in features],
            padding=True,
            return_tensors="pt",
        )
        max_length = max(map(len, labels))
        inputs["labels"] = torch.tensor(
            [label + [-100] * (max_length - len(label)) for label in labels], dtype=torch.long
        )
        inputs["row_index"] = torch.tensor(row_indices, dtype=torch.long)
        return inputs


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    indices = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, average=None, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, average="macro", zero_division=0
    )
    _, _, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, average="micro", zero_division=0
    )
    return {
        "effective_n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "invalid_prediction_count": int(np.sum(y_pred < 0)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "undefined_zero_support": bool(support[index] == 0),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=indices).astype(int).tolist(),
        "confusion_matrix_labels": labels,
    }


def combined_metrics(topic_true, topic_pred, tone_true, tone_pred, **extra) -> dict[str, Any]:
    topic = classification_metrics(topic_true, topic_pred, TOPIC_LABELS)
    tone = classification_metrics(tone_true, tone_pred, TONE_LABELS)
    result = {
        "selection_score": 0.5 * topic["macro_f1"] + 0.5 * tone["macro_f1"],
        "primary_topic": topic,
        "overall_tone": tone,
    }
    result.update(extra)
    return result


def t5_parse_counts(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    errors = Counter(error for row in predictions for error in row["parse_errors"])
    unknown = Counter(label for row in predictions for label in row["unknown_labels"])
    valid = sum(row["format_valid"] for row in predictions)
    return {
        "output_format_valid_rate": valid / len(predictions) if predictions else None,
        "invalid_output_count": len(predictions) - valid,
        "invalid_output_reasons": dict(errors),
        "unknown_labels": dict(unknown),
    }
