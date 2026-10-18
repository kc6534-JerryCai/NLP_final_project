from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "keywords.v1.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class KeywordConfig:
    version: str
    groups: dict[str, dict]
    direct_sentiment: dict
    us_markers: list[str]
    us_geo_markers: list[str]
    foreign_markers: list[str]

    @property
    def retrieval_groups(self) -> dict[str, list[str]]:
        groups = {name: value["terms"] for name, value in self.groups.items()}
        groups["DIRECT"] = self.direct_sentiment["terms"]
        return groups


def load_keyword_config(path: Path = DEFAULT_CONFIG) -> KeywordConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return KeywordConfig(
        version=raw["version"],
        groups=raw["groups"],
        direct_sentiment=raw["direct_sentiment"],
        us_markers=raw["us_markers"],
        us_geo_markers=raw["us_geo_markers"],
        foreign_markers=raw["foreign_markers"],
    )


def load_environment() -> dict[str, str | None]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return {
        "nyt_api_key": os.getenv("NYT_API_KEY") or os.getenv("NYT_API_Key"),
        "guardian_api_key": os.getenv("GUARDIAN_API_KEY"),
        "nyt_interval": os.getenv("NYT_MIN_INTERVAL_SECONDS", "12.5"),
        "guardian_interval": os.getenv("GUARDIAN_MIN_INTERVAL_SECONDS", "1.05"),
    }


def month_intervals(start: date, end: date) -> Iterator[tuple[date, date]]:
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        yield max(start, cursor), min(end, next_month.fromordinal(next_month.toordinal() - 1))
        cursor = next_month


def pilot_intervals(cutoff: date) -> list[tuple[date, date]]:
    years = [2010, 2015, 2020, 2022, 2025, cutoff.year]
    selected: set[tuple[int, int]] = set()
    for year in years:
        if year > cutoff.year:
            continue
        months = [1, 7]
        if year == cutoff.year:
            months = [1, cutoff.month]
        for month in months:
            if date(year, month, 1) <= cutoff:
                selected.add((year, month))
    intervals: list[tuple[date, date]] = []
    for year, month in sorted(selected):
        start = date(year, month, 1)
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        end = min(cutoff, next_month.fromordinal(next_month.toordinal() - 1))
        intervals.append((start, end))
    return intervals


def ensure_data_dirs(data_dir: Path) -> None:
    for child in ("raw", "clean", "logs", "state", "reports"):
        (data_dir / child).mkdir(parents=True, exist_ok=True)


def quoted_query(terms: list[str]) -> str:
    def quote(term: str) -> str:
        escaped = term.replace('"', '\\"')
        if " " in term or "-" in term or "." in term:
            return f'"{escaped}"'
        return escaped

    return "(" + " OR ".join(quote(term) for term in terms) + ")"
