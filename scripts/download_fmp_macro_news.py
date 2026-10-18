#!/usr/bin/env python3
"""Download and filter FMP General News for three macroeconomic topics.

The script reads ``FMP_API`` from the project-level ``.env`` file.  It never
prints or writes the API key.  Results are deduplicated and written as both a
UTF-8-with-BOM CSV (convenient for Excel) and UTF-8 JSON Lines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://financialmodelingprep.com/stable/news/general-latest"

# A focused retrieval dictionary: it includes common names used in economic
# reporting, but deliberately avoids generic words such as "rates" by itself.
TOPIC_PATTERNS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "inflation_rate": (
        ("inflation", re.compile(r"\binflation(?:ary)?\b", re.I)),
        ("consumer price index", re.compile(r"\bconsumer price(?:s| index)?\b", re.I)),
        ("CPI", re.compile(r"\bCPI\b")),
        ("PCE price", re.compile(r"\bPCE(?: price(?: index)?)?\b", re.I)),
        ("price pressure", re.compile(r"\bprice pressures?\b", re.I)),
    ),
    "unemployment_rate": (
        ("unemployment", re.compile(r"\bunemployment(?: rate)?\b", re.I)),
        ("jobless rate", re.compile(r"\bjobless rate\b", re.I)),
        ("jobless claims", re.compile(r"\bjobless claims?\b", re.I)),
        ("unemployment claims", re.compile(r"\bunemployment claims?\b", re.I)),
    ),
    "interest_rate": (
        ("interest rate", re.compile(r"\binterest rates?\b", re.I)),
        ("federal funds rate", re.compile(r"\bfederal funds(?: target)? rate\b", re.I)),
        ("fed funds rate", re.compile(r"\bfed funds rate\b", re.I)),
        ("policy rate", re.compile(r"\bpolicy rates?\b", re.I)),
        ("rate cut", re.compile(r"\brate cuts?\b", re.I)),
        ("rate hike", re.compile(r"\brate hikes?\b", re.I)),
    ),
}

CSV_FIELDS = (
    "topics",
    "matched_keywords",
    "publishedDate",
    "publisher",
    "site",
    "title",
    "text",
    "url",
    "image",
    "symbol",
)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from exc


def article_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def month_windows(start: date, end: date) -> Iterable[tuple[date, date]]:
    cursor = start
    while cursor <= end:
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        window_end = min(end, next_month - timedelta(days=1))
        yield cursor, window_end
        cursor = window_end + timedelta(days=1)


def redact(value: str, api_key: str) -> str:
    sanitized = value.replace(api_key, "<redacted>") if api_key else value
    return re.sub(r"([?&]apikey=)[^&\s]+", r"\1<redacted>", sanitized, flags=re.I)


def get_page(
    session: requests.Session,
    api_key: str,
    start: date,
    end: date,
    page: int,
    limit: int,
    retries: int,
) -> list[dict[str, Any]]:
    params = {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "page": page,
        "limit": limit,
        "apikey": api_key,
    }
    for attempt in range(retries + 1):
        try:
            response = session.get(ENDPOINT, params=params, timeout=60)
        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
                continue
            # Suppress the requests exception context because it can contain the
            # fully prepared URL (and therefore the API key) in its traceback.
            raise RuntimeError(redact(f"FMP request failed: {exc}", api_key)) from None

        if response.status_code == 429 and attempt < retries:
            retry_after = response.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2**attempt
            time.sleep(min(max(wait, 1), 60))
            continue
        if response.status_code != 200:
            body = redact(response.text[:500], api_key)
            raise RuntimeError(f"FMP HTTP {response.status_code}: {body}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("FMP returned a non-JSON response") from exc
        if isinstance(payload, dict):
            message = redact(str(payload.get("Error Message") or payload.get("message") or payload), api_key)
            raise RuntimeError(f"FMP returned an error object: {message[:500]}")
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected FMP response type: {type(payload).__name__}")
        return [item for item in payload if isinstance(item, dict)]
    raise RuntimeError("FMP request exhausted retries")


def topic_matches(article: dict[str, Any]) -> tuple[list[str], list[str]]:
    searchable = "\n".join((str(article.get("title") or ""), str(article.get("text") or "")))
    topics: list[str] = []
    keywords: list[str] = []
    for topic, patterns in TOPIC_PATTERNS.items():
        topic_keywords = [label for label, pattern in patterns if pattern.search(searchable)]
        if topic_keywords:
            topics.append(topic)
            keywords.extend(topic_keywords)
    return topics, sorted(set(keywords), key=str.casefold)


def record_key(article: dict[str, Any]) -> str:
    url = str(article.get("url") or "").strip()
    if url:
        return "url:" + url
    basis = "\x1f".join(
        str(article.get(field) or "").strip().casefold()
        for field in ("publishedDate", "publisher", "title")
    )
    return "hash:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()


def normalized_record(article: dict[str, Any], topics: list[str], keywords: list[str]) -> dict[str, Any]:
    record = {field: article.get(field) for field in CSV_FIELDS if field not in {"topics", "matched_keywords"}}
    record["topics"] = "|".join(topics)
    record["matched_keywords"] = "|".join(keywords)
    return {field: record.get(field) for field in CSV_FIELDS}


def write_outputs(
    records: list[dict[str, Any]],
    output_dir: Path,
    start: date,
    end: date,
    summary: dict[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fmp_macro_news_{start.isoformat()}_to_{end.isoformat()}"
    csv_path = output_dir / f"{stem}.csv"
    jsonl_path = output_dir / f"{stem}.jsonl"
    summary_path = output_dir / f"{stem}_summary.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return csv_path, jsonl_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    today = date.today()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_date, default=today - timedelta(days=365))
    parser.add_argument("--end", type=parse_date, default=today)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "fmp_general_news")
    parser.add_argument("--limit", type=int, default=250, choices=range(1, 251), metavar="1..250")
    parser.add_argument("--max-pages", type=int, default=100, choices=range(1, 101), metavar="1..100")
    parser.add_argument("--delay", type=float, default=0.15, help="seconds between successful requests")
    parser.add_argument("--retries", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = (os.getenv("FMP_API") or "").strip()
    if not api_key:
        raise SystemExit(f"FMP_API is missing from {PROJECT_ROOT / '.env'}")

    session = requests.Session()
    session.headers.update({"User-Agent": "NLP-Final-Assignment-FMP-News/1.0"})
    unique: dict[str, dict[str, Any]] = {}
    fetched_items = 0
    fetched_pages = 0
    truncated_windows: list[str] = []

    windows = list(month_windows(args.start, args.end))
    for index, (window_start, window_end) in enumerate(windows, start=1):
        window_items = 0
        hit_page_limit = True
        for page in range(args.max_pages):
            batch = get_page(
                session,
                api_key,
                window_start,
                window_end,
                page,
                args.limit,
                args.retries,
            )
            fetched_pages += 1
            fetched_items += len(batch)
            window_items += len(batch)
            for article in batch:
                published = article_date(article.get("publishedDate"))
                if published is None or not (args.start <= published <= args.end):
                    continue
                topics, keywords = topic_matches(article)
                if not topics:
                    continue
                key = record_key(article)
                incoming = normalized_record(article, topics, keywords)
                existing = unique.get(key)
                if existing is None:
                    unique[key] = incoming
                else:
                    merged_topics = sorted(set(str(existing["topics"]).split("|") + topics))
                    merged_keywords = sorted(
                        set(str(existing["matched_keywords"]).split("|") + keywords), key=str.casefold
                    )
                    existing["topics"] = "|".join(filter(None, merged_topics))
                    existing["matched_keywords"] = "|".join(filter(None, merged_keywords))
            if len(batch) < args.limit:
                hit_page_limit = False
                break
            if args.delay > 0:
                time.sleep(args.delay)
        if hit_page_limit:
            truncated_windows.append(f"{window_start.isoformat()}..{window_end.isoformat()}")
        print(
            f"window {index}/{len(windows)} {window_start}..{window_end}: "
            f"fetched={window_items}, cumulative_matches={len(unique)}",
            flush=True,
        )

    records = sorted(
        unique.values(),
        key=lambda item: (str(item.get("publishedDate") or ""), str(item.get("title") or "")),
        reverse=True,
    )
    topic_counts: Counter[str] = Counter()
    for record in records:
        topic_counts.update(filter(None, str(record["topics"]).split("|")))
    dates = [article_date(record.get("publishedDate")) for record in records]
    valid_dates = [value for value in dates if value is not None]
    summary = {
        "endpoint": ENDPOINT,
        "requested_start": args.start.isoformat(),
        "requested_end": args.end.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fetched_pages": fetched_pages,
        "fetched_items_before_filtering": fetched_items,
        "unique_matching_articles": len(records),
        "topic_counts_including_multi_topic_articles": dict(sorted(topic_counts.items())),
        "earliest_matching_article": min(valid_dates).isoformat() if valid_dates else None,
        "latest_matching_article": max(valid_dates).isoformat() if valid_dates else None,
        "truncated_windows": truncated_windows,
        "topic_keywords": {topic: [label for label, _ in patterns] for topic, patterns in TOPIC_PATTERNS.items()},
        "notes": [
            "Filtering searches each article's title and text fields.",
            "An article can be assigned to more than one topic.",
            "The API key is not included in any output file.",
        ],
    }
    csv_path, jsonl_path, summary_path = write_outputs(
        records, args.output_dir.resolve(), args.start, args.end, summary
    )
    print(f"completed: {len(records)} unique matching articles")
    print(f"csv: {csv_path}")
    print(f"jsonl: {jsonl_path}")
    print(f"summary: {summary_path}")
    if truncated_windows:
        print("WARNING: one or more date windows reached the 100-page API ceiling", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
