from __future__ import annotations

import re
from dataclasses import dataclass

from .constants import PRIMARY_TOPIC_LABELS, T5_INSTRUCTION, TONE_LABELS


def build_t5_input(headline: str, abstract: str) -> str:
    return f"{T5_INSTRUCTION}\nTitle: {headline.strip()}\nAbstract: {abstract.strip()}"


def build_t5_target(primary_topic: str, tone: str) -> str:
    return f"primary_topic: {primary_topic}; tone: {tone}"


@dataclass
class ParsedT5:
    valid: bool
    primary_topic: str | None
    tone: str | None
    errors: list[str]
    unknown_labels: list[str]


def parse_t5_output(text: str) -> ParsedT5:
    stripped = text.strip()
    match = re.fullmatch(r"primary_topic: ([a-z_]+); tone: ([a-z_]+)", stripped)
    if not match:
        return ParsedT5(False, None, None, ["format_error"], [])
    primary_topic, tone = match.groups()
    unknown = [] if primary_topic in PRIMARY_TOPIC_LABELS else [primary_topic]
    if tone not in TONE_LABELS:
        unknown.append(tone)
    errors: list[str] = []
    if unknown:
        errors.append("unknown_label")
    valid = not errors
    return ParsedT5(valid, primary_topic if valid else None, tone if valid and tone in TONE_LABELS else None, errors, unknown)
