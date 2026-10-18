from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .constants import PRIMARY_TOPIC_LABELS, TONE_LABELS, TOPIC_LABELS


def _safe_number(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def topic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "effective_n": 0,
            "micro_f1": None,
            "macro_f1": None,
            "exact_match": None,
            "per_class": {label: {"precision": None, "recall": None, "f1": None, "support": 0, "undefined_zero_support": True} for label in TOPIC_LABELS},
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(TOPIC_LABELS))), average=None, zero_division=0
    )
    _, _, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "effective_n": int(len(y_true)),
        "micro_f1": _safe_number(micro_f1),
        "macro_f1": _safe_number(macro_f1),
        "exact_match": float(np.all(y_true == y_pred, axis=1).mean()),
        "per_class": {
            label: {
                "precision": _safe_number(precision[idx]),
                "recall": _safe_number(recall[idx]),
                "f1": _safe_number(f1[idx]),
                "support": int(support[idx]),
                "undefined_zero_support": bool(support[idx] == 0),
            }
            for idx, label in enumerate(TOPIC_LABELS)
        },
    }


def tone_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "effective_n": 0,
            "accuracy": None,
            "macro_f1": None,
            "per_class": {label: {"precision": None, "recall": None, "f1": None, "support": 0, "undefined_zero_support": True} for label in TONE_LABELS},
            "confusion_matrix": [],
            "confusion_matrix_labels": TONE_LABELS,
        }
    indices = list(range(len(TONE_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, average=None, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, average="macro", zero_division=0
    )
    return {
        "effective_n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": _safe_number(macro_f1),
        "per_class": {
            label: {
                "precision": _safe_number(precision[idx]),
                "recall": _safe_number(recall[idx]),
                "f1": _safe_number(f1[idx]),
                "support": int(support[idx]),
                "undefined_zero_support": bool(support[idx] == 0),
            }
            for idx, label in enumerate(TONE_LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=indices).astype(int).tolist(),
        "confusion_matrix_labels": TONE_LABELS,
    }


def primary_topic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "effective_n": 0,
            "accuracy": None,
            "macro_f1": None,
            "per_class": {label: {"precision": None, "recall": None, "f1": None, "support": 0, "undefined_zero_support": True} for label in PRIMARY_TOPIC_LABELS},
        }
    indices = list(range(len(PRIMARY_TOPIC_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, average=None, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, average="macro", zero_division=0
    )
    return {
        "effective_n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": _safe_number(macro_f1),
        "per_class": {
            label: {
                "precision": _safe_number(precision[idx]),
                "recall": _safe_number(recall[idx]),
                "f1": _safe_number(f1[idx]),
                "support": int(support[idx]),
                "undefined_zero_support": bool(support[idx] == 0),
            }
            for idx, label in enumerate(PRIMARY_TOPIC_LABELS)
        },
    }


def combined_primary_metrics(
    primary_true: np.ndarray,
    primary_pred: np.ndarray,
    tone_true: np.ndarray,
    tone_pred: np.ndarray,
    **extra: Any,
) -> dict[str, Any]:
    primary = primary_topic_metrics(primary_true, primary_pred)
    tone = tone_metrics(tone_true, tone_pred)
    score = None if primary["macro_f1"] is None or tone["macro_f1"] is None else 0.5 * primary["macro_f1"] + 0.5 * tone["macro_f1"]
    result = {"selection_score": score, "primary_topic": primary, "overall_tone": tone}
    result.update(extra)
    return result


def combined_metrics(
    topic_true: np.ndarray,
    topic_pred: np.ndarray,
    tone_true: np.ndarray,
    tone_pred: np.ndarray,
    **extra: Any,
) -> dict[str, Any]:
    topics = topic_metrics(topic_true, topic_pred)
    tone = tone_metrics(tone_true, tone_pred)
    topic_macro = topics["macro_f1"]
    tone_macro = tone["macro_f1"]
    score = None if topic_macro is None or tone_macro is None else 0.5 * topic_macro + 0.5 * tone_macro
    result = {"selection_score": score, "topics": topics, "overall_tone": tone}
    result.update(extra)
    return result
