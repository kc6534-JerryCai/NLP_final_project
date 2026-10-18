#!/usr/bin/env python3
"""Build official-event anchors and candidate FMP news for event detection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
FMP_ENDPOINT = "https://financialmodelingprep.com/stable/news/general-latest"
DEFAULT_EVENTS = ROOT / "config" / "official_macro_events_2025-10-16_to_2026-10-16.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "macro_event_identification"

EVENT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "CPI_RELEASE": (
        re.compile(r"\bCPI\b", re.I),
        re.compile(r"\bconsumer price(?:s| index)?\b", re.I),
        re.compile(r"\binflation\b", re.I),
    ),
    "PCE_RELEASE": (
        re.compile(r"\bPCE(?: price index| inflation)?\b", re.I),
        re.compile(r"\bpersonal consumption expenditures?\b", re.I),
        re.compile(r"\bpersonal income and outlays\b", re.I),
        re.compile(r"\bFed(?:eral Reserve)?['’]s preferred inflation\b", re.I),
    ),
    "EMPLOYMENT_RELEASE": (
        re.compile(r"\bjobs report\b", re.I),
        re.compile(r"\bemployment (?:report|situation)\b", re.I),
        re.compile(r"\bnonfarm payrolls?\b", re.I),
        re.compile(r"\bpayrolls?\b", re.I),
        re.compile(r"\bunemployment rate\b", re.I),
        re.compile(r"\bjobless rate\b", re.I),
    ),
    "FOMC_RATE_DECISION": (
        re.compile(r"\bFOMC\b", re.I),
        re.compile(r"\bFederal Reserve\b", re.I),
        re.compile(r"\bFed\b", re.I),
    ),
}

FORECAST_RE = re.compile(
    r"\b(expected?|expects?|forecast(?:s|ed)?|consensus|economists? (?:polled|surveyed)|"
    r"ahead of|what to expect|preview|likely|probability|odds|anticipat(?:e|es|ed|ion)|"
    r"project(?:s|ed|ion)|estimate(?:s|d)?)\b",
    re.I,
)
RESULT_RE = re.compile(
    r"\b(reported|released|rose|risen|fell|fallen|increased|decreased|declined|cooled|"
    r"accelerated|slowed|held|unchanged|added|lost|cut|lowered|raised|hiked|maintained|"
    r"left rates|target range|came in|prints? at)\b",
    re.I,
)
AGENCY_PATTERNS = {
    "CPI_RELEASE": re.compile(r"\b(BLS|Bureau of Labor Statistics|Labor Department)\b", re.I),
    "PCE_RELEASE": re.compile(r"\b(BEA|Bureau of Economic Analysis|Commerce Department)\b", re.I),
    "EMPLOYMENT_RELEASE": re.compile(r"\b(BLS|Bureau of Labor Statistics|Labor Department)\b", re.I),
    "FOMC_RATE_DECISION": re.compile(r"\b(FOMC|Federal Reserve|Fed)\b", re.I),
}
RATE_DECISION_RE = re.compile(
    r"\b(rate(?:s)?|target range|basis points?|bps|cut|cuts|hike|hikes|hold|held|maintain(?:ed|s)?)\b",
    re.I,
)
NUMBER_RE = re.compile(r"(?:[-+]?\d+(?:\.\d+)?\s*%|[-+]?\d[\d,]*(?:\.\d+)?\s*(?:jobs?|payrolls?|basis points?|bps))", re.I)


def redact(text: str, secret: str) -> str:
    value = text.replace(secret, "<redacted>") if secret else text
    return re.sub(r"([?&]apikey=)[^&\s]+", r"\1<redacted>", value, flags=re.I)


def parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def month_windows(start: date, end: date) -> Iterable[tuple[date, date]]:
    cursor = start
    while cursor <= end:
        next_month = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
        window_end = min(end, next_month - timedelta(days=1))
        yield cursor, window_end
        cursor = window_end + timedelta(days=1)


def get_json_page(
    session: requests.Session,
    api_key: str,
    start: date,
    end: date,
    page: int,
    limit: int = 250,
) -> list[dict[str, Any]]:
    params = {
        "from": start.isoformat(), "to": end.isoformat(), "page": page,
        "limit": limit, "apikey": api_key,
    }
    last_error = ""
    for attempt in range(5):
        try:
            response = session.get(FMP_ENDPOINT, params=params, timeout=60)
        except requests.RequestException as exc:
            last_error = redact(str(exc), api_key)
            if attempt < 4:
                time.sleep(min(2**attempt, 20))
                continue
            raise RuntimeError(f"FMP request failed: {last_error}") from None
        if response.status_code == 429 and attempt < 4:
            time.sleep(min(2**attempt + 1, 30))
            continue
        if response.status_code != 200:
            raise RuntimeError(f"FMP HTTP {response.status_code}: {redact(response.text[:500], api_key)}")
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected FMP payload: {redact(str(payload)[:500], api_key)}")
        return [row for row in payload if isinstance(row, dict)]
    raise RuntimeError(f"FMP retries exhausted: {last_error}")


def article_key(row: dict[str, Any]) -> str:
    if row.get("url"):
        return str(row["url"])
    basis = "\x1f".join(str(row.get(key) or "") for key in ("publishedDate", "publisher", "title"))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def download_raw_news(
    session: requests.Session,
    api_key: str,
    start: date,
    end: date,
    cache_path: Path,
) -> list[dict[str, Any]]:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        print(f"raw cache: {len(rows)} articles", flush=True)
        return rows

    unique: dict[str, dict[str, Any]] = {}
    windows = list(month_windows(start, end))
    for window_index, (window_start, window_end) in enumerate(windows, 1):
        count = 0
        for page in range(100):
            batch = get_json_page(session, api_key, window_start, window_end, page)
            count += len(batch)
            for row in batch:
                published = parse_dt(row.get("publishedDate"))
                if published and start <= published.date() <= end:
                    unique[article_key(row)] = row
            if len(batch) < 250:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"FMP 100-page ceiling reached for {window_start}..{window_end}")
        print(f"raw window {window_index}/{len(windows)}: {window_start}..{window_end} fetched={count}", flush=True)
    rows = sorted(unique.values(), key=lambda row: str(row.get("publishedDate") or ""))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def repair_daily_months(
    session: requests.Session,
    api_key: str,
    rows: list[dict[str, Any]],
    months: list[str],
    raw_path: Path,
    marker_path: Path,
    overall_start: date,
    overall_end: date,
) -> list[dict[str, Any]]:
    """Refetch selected months day by day when a broad FMP query is capped."""
    completed_months: set[str] = set()
    if marker_path.exists():
        completed_months.update(json.loads(marker_path.read_text(encoding="utf-8")).get("months", []))
    pending_months = [month for month in months if month not in completed_months]
    if not pending_months:
        return rows
    unique = {article_key(row): row for row in rows}
    added_before = len(unique)
    fetched = 0
    for month in pending_months:
        year, month_number = (int(part) for part in month.split("-"))
        first = date(year, month_number, 1)
        next_month = date(year + (month_number == 12), 1 if month_number == 12 else month_number + 1, 1)
        cursor = max(first, overall_start)
        month_end = min(next_month, overall_end + timedelta(days=1))
        while cursor < month_end:
            day_count = 0
            for page in range(100):
                batch = get_json_page(session, api_key, cursor, cursor, page)
                day_count += len(batch)
                for row in batch:
                    unique[article_key(row)] = row
                if len(batch) < 250:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"FMP 100-page ceiling reached for {cursor}")
            fetched += day_count
            print(f"daily repair {cursor}: fetched={day_count}", flush=True)
            cursor += timedelta(days=1)
        completed_months.add(month)
        repaired = sorted(unique.values(), key=lambda row: str(row.get("publishedDate") or ""))
        with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in repaired:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        marker_path.write_text(
            json.dumps({
                "months": sorted(completed_months),
                "last_completed_month": month,
                "last_run_fetched": fetched,
                "last_run_unique_added": len(unique) - added_before,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"daily repair checkpoint: {month} saved", flush=True)
    repaired = sorted(unique.values(), key=lambda row: str(row.get("publishedDate") or ""))
    print(f"daily repair complete: unique_added={len(unique) - added_before}", flush=True)
    return repaired


def clean_web_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def fetch_official_texts(
    session: requests.Session,
    events: list[dict[str, Any]],
    cache_path: Path,
) -> dict[str, str]:
    existing: dict[str, str] = {}
    if cache_path.exists():
        existing = json.loads(cache_path.read_text(encoding="utf-8"))
    for index, event in enumerate(events, 1):
        event_id = event["event_id"]
        if existing.get(event_id):
            continue
        try:
            response = session.get(event["official_url"], timeout=15)
            if response.status_code != 200:
                existing[event_id] = f"OFFICIAL PAGE FETCH FAILED: HTTP {response.status_code}"
            else:
                existing[event_id] = clean_web_text(response.text)
        except requests.RequestException as exc:
            existing[event_id] = f"OFFICIAL PAGE FETCH FAILED: {type(exc).__name__}"
        print(f"official page {index}/{len(events)}: {event_id}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return existing


def relevance_score(event_type: str, row: dict[str, Any], phase: str) -> tuple[int, list[str]]:
    text = "\n".join((str(row.get("title") or ""), str(row.get("text") or "")))
    reasons: list[str] = []
    hits = sum(bool(pattern.search(text)) for pattern in EVENT_PATTERNS[event_type])
    score = min(hits * 2, 6)
    if hits:
        reasons.append(f"topic_hits={hits}")
    if AGENCY_PATTERNS[event_type].search(text):
        score += 2
        reasons.append("agency")
    if NUMBER_RE.search(text):
        score += 2
        reasons.append("numeric_result")
    if FORECAST_RE.search(text):
        score += 2 if phase == "pre" else -2
        reasons.append("forecast_language")
    if RESULT_RE.search(text):
        score += 2 if phase == "post" else 0
        reasons.append("result_language")
    if event_type == "FOMC_RATE_DECISION":
        if RATE_DECISION_RE.search(text):
            score += 2
            reasons.append("rate_decision_language")
        else:
            score -= 3
    if event_type == "PCE_RELEASE" and not re.search(r"\b(PCE|personal consumption expenditures?|personal income and outlays)\b", text, re.I):
        score -= 3
    if event_type == "EMPLOYMENT_RELEASE" and re.search(r"\b(jobless claims?|initial claims?)\b", text, re.I) and not re.search(r"\b(payrolls?|jobs report|unemployment rate)\b", text, re.I):
        score -= 4
        reasons.append("claims_only")
    return score, reasons


def candidate_id(row: dict[str, Any]) -> str:
    return hashlib.sha256(article_key(row).encode("utf-8")).hexdigest()[:16]


def build_candidates(
    events: list[dict[str, Any]],
    news: list[dict[str, Any]],
    official_texts: dict[str, str],
) -> list[dict[str, Any]]:
    parsed_news = [(parse_dt(row.get("publishedDate")), row) for row in news]
    bundles: list[dict[str, Any]] = []
    for event in events:
        release_dt = datetime.fromisoformat(event["release_datetime_et"])
        pre_start = release_dt - timedelta(days=7)
        post_end = release_dt + timedelta(days=2)
        pre: list[dict[str, Any]] = []
        post: list[dict[str, Any]] = []
        for published, row in parsed_news:
            if published is None or published < pre_start or published > post_end:
                continue
            phase = "pre" if published < release_dt else "post"
            score, reasons = relevance_score(event["event_type"], row, phase)
            if score < 3:
                continue
            candidate = {
                "candidate_id": candidate_id(row),
                "phase": phase,
                "rule_score": score,
                "rule_reasons": reasons,
                "publishedDate": row.get("publishedDate"),
                "publisher": row.get("publisher"),
                "site": row.get("site"),
                "title": row.get("title"),
                "text": row.get("text"),
                "url": row.get("url"),
            }
            (pre if phase == "pre" else post).append(candidate)
        pre = sorted(pre, key=lambda row: (row["rule_score"], row["publishedDate"]), reverse=True)[:18]
        post = sorted(post, key=lambda row: row["publishedDate"])[:25]
        bundles.append({
            **event,
            "official_text": official_texts.get(event["event_id"], "")[:12000],
            "pre_candidates": pre,
            "post_candidates": post,
        })
        print(f"candidates {event['event_id']}: pre={len(pre)} post={len(post)}", flush=True)
    return bundles


def write_candidate_outputs(bundles: list[dict[str, Any]], output_dir: Path) -> None:
    payload_path = output_dir / "event_candidate_bundles.jsonl"
    audit_path = output_dir / "candidate_audit_rules.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with payload_path.open("w", encoding="utf-8", newline="\n") as handle:
        for bundle in bundles:
            handle.write(json.dumps(bundle, ensure_ascii=False) + "\n")
    fields = [
        "event_id", "event_type", "reference_period", "release_datetime_et", "candidate_id",
        "phase", "rule_score", "rule_reasons", "publishedDate", "publisher", "site", "title", "text", "url",
    ]
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bundle in bundles:
            for candidate in bundle["pre_candidates"] + bundle["post_candidates"]:
                writer.writerow({
                    "event_id": bundle["event_id"],
                    "event_type": bundle["event_type"],
                    "reference_period": bundle["reference_period"],
                    "release_datetime_et": bundle["release_datetime_et"],
                    **candidate,
                    "rule_reasons": "|".join(candidate["rule_reasons"]),
                })
    print(f"candidate bundles: {payload_path}")
    print(f"rule audit: {audit_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 10, 16))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 16))
    parser.add_argument(
        "--repair-daily-month",
        action="append",
        default=["2026-01", "2026-02", "2026-10"],
        help="month queried one day at a time to bypass broad-query coverage caps",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    api_key = (os.getenv("FMP_API") or "").strip()
    if not api_key:
        raise SystemExit("FMP_API is missing from .env")
    events = json.loads(args.events.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    session = requests.Session()
    session.headers.update({"User-Agent": "NLP-Final-Assignment-Macro-Event-Research/1.0"})
    raw_path = output_dir / f"fmp_general_news_raw_{args.start}_to_{args.end}.jsonl"
    news = download_raw_news(session, api_key, args.start, args.end, raw_path)
    news = repair_daily_months(
        session,
        api_key,
        news,
        args.repair_daily_month,
        raw_path,
        output_dir / "daily_repair_2026-01_2026-02.json",
        args.start,
        args.end,
    )
    official_texts = fetch_official_texts(session, events, output_dir / "official_release_texts.json")
    bundles = build_candidates(events, news, official_texts)
    write_candidate_outputs(bundles, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
