from __future__ import annotations

from pathlib import Path
from urllib.parse import quote, quote_plus


TEXT_SUFFIXES = {".jsonl", ".json", ".csv", ".md", ".txt", ".log"}


def scrub_secrets(root: Path, secrets: list[str]) -> int:
    """Replace configured secrets in generated text artifacts without printing them."""
    candidates = {value for secret in secrets if secret for value in (secret, quote(secret), quote_plus(secret))}
    if not candidates or not root.exists():
        return 0
    changed = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        scrubbed = text
        for secret in candidates:
            scrubbed = scrubbed.replace(secret, "<redacted>")
        if scrubbed != text:
            path.write_text(scrubbed, encoding="utf-8")
            changed += 1
    return changed
