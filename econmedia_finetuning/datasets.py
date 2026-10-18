from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from .constants import PRIMARY_TOPIC_LABELS, TONE_LABELS
from .data import article_text
from .t5_format import build_t5_input, build_t5_target


class FinBertDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int, supervised_only: bool = False):
        self.rows = [row for row in rows if not supervised_only or row["primary_topic_valid"] or row["tone_valid"]]
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
                "primary_topic_labels": PRIMARY_TOPIC_LABELS.index(row["primary_topic"]) if row["primary_topic_valid"] else -1,
                "tone_labels": TONE_LABELS.index(row["overall_tone"]) if row["tone_valid"] else -1,
                "primary_topic_valid_mask": float(row["primary_topic_valid"]),
                "tone_valid_mask": float(row["tone_valid"]),
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
        padded["primary_topic_labels"] = torch.tensor([item["primary_topic_labels"] for item in features], dtype=torch.long)
        padded["tone_labels"] = torch.tensor([item["tone_labels"] for item in features], dtype=torch.long)
        padded["primary_topic_valid_mask"] = torch.tensor([item["primary_topic_valid_mask"] for item in features], dtype=torch.float32)
        padded["tone_valid_mask"] = torch.tensor([item["tone_valid_mask"] for item in features], dtype=torch.float32)
        padded["row_index"] = torch.tensor([item["row_index"] for item in features], dtype=torch.long)
        return padded


class T5Dataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        max_source_length: int,
        max_target_length: int,
        supervised_only: bool = True,
    ):
        self.rows = [row for row in rows if not supervised_only or row["primary_joint_valid"]]
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
            text_target=build_t5_target(row["primary_topic"], row["overall_tone"]),
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
        rows = [item["row_index"] for item in features]
        model_inputs = self.tokenizer.pad(
            [{key: value for key, value in item.items() if key not in {"labels", "row_index"}} for item in features],
            padding=True,
            return_tensors="pt",
        )
        max_length = max(map(len, labels))
        padded_labels = [label + [-100] * (max_length - len(label)) for label in labels]
        model_inputs["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        model_inputs["row_index"] = torch.tensor(rows, dtype=torch.long)
        return model_inputs
