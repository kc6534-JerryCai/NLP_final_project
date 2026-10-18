#!/usr/bin/env python
"""Build CPI, Employment, PCE, and FOMC release anchors from official sites.

The generated JSON is compatible with ``build_macro_event_candidates.py``.
Historical BLS and BEA schedules supply CPI, Employment Situation, and
Personal Income and Outlays dates.  Federal Reserve statement links supply
FOMC decision dates.  A hand-audited JSON may be merged to preserve unusual
or delayed releases (for example, releases affected by a shutdown).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGE = ROOT / "config" / "official_macro_events_2025-10-16_to_2026-10-16.json"
DEFAULT_OUTPUT = ROOT / "config" / "official_macro_events_2023-10-16_to_2026-10-16.json"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 10, 16))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 16))
    parser.add_argument("--merge", type=Path, default=DEFAULT_MERGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--refresh-official", action="store_true",
        help="Re-scrape official calendars; otherwise use the verified historical seed plus --merge",
    )
    return parser.parse_args()


# Verified against BLS annual calendars, BEA Personal Income and Outlays
# archives, and Federal Reserve FOMC statement archives.  The seed stops just
# before the hand-audited one-year file begins.
BLS_CPI_SEED = [
    ("2023-10", "2023-11-14"), ("2023-11", "2023-12-12"),
    ("2023-12", "2024-01-11"), ("2024-01", "2024-02-13"),
    ("2024-02", "2024-03-12"), ("2024-03", "2024-04-10"),
    ("2024-04", "2024-05-15"), ("2024-05", "2024-06-12"),
    ("2024-06", "2024-07-11"), ("2024-07", "2024-08-14"),
    ("2024-08", "2024-09-11"), ("2024-09", "2024-10-10"),
    ("2024-10", "2024-11-13"), ("2024-11", "2024-12-11"),
    ("2024-12", "2025-01-15"), ("2025-01", "2025-02-12"),
    ("2025-02", "2025-03-12"), ("2025-03", "2025-04-10"),
    ("2025-04", "2025-05-13"), ("2025-05", "2025-06-11"),
    ("2025-06", "2025-07-15"), ("2025-07", "2025-08-12"),
    ("2025-08", "2025-09-11"),
]

BLS_EMPLOYMENT_SEED = [
    ("2023-10", "2023-11-03"), ("2023-11", "2023-12-08"),
    ("2023-12", "2024-01-05"), ("2024-01", "2024-02-02"),
    ("2024-02", "2024-03-08"), ("2024-03", "2024-04-05"),
    ("2024-04", "2024-05-03"), ("2024-05", "2024-06-07"),
    ("2024-06", "2024-07-05"), ("2024-07", "2024-08-02"),
    ("2024-08", "2024-09-06"), ("2024-09", "2024-10-04"),
    ("2024-10", "2024-11-01"), ("2024-11", "2024-12-06"),
    ("2024-12", "2025-01-10"), ("2025-01", "2025-02-07"),
    ("2025-02", "2025-03-07"), ("2025-03", "2025-04-04"),
    ("2025-04", "2025-05-02"), ("2025-05", "2025-06-06"),
    ("2025-06", "2025-07-03"), ("2025-07", "2025-08-01"),
    ("2025-08", "2025-09-05"),
]

BEA_PCE_SEED = [
    ("2023-09", "2023-10-27"), ("2023-10", "2023-11-30"),
    ("2023-11", "2023-12-22"), ("2023-12", "2024-01-26"),
    ("2024-01", "2024-02-28"), ("2024-02", "2024-03-29"),
    ("2024-03", "2024-04-26"), ("2024-04", "2024-05-31"),
    ("2024-05", "2024-06-28"), ("2024-06", "2024-07-26"),
    ("2024-07", "2024-08-30"), ("2024-08", "2024-09-27"),
    ("2024-09", "2024-10-31"), ("2024-10", "2024-11-27"),
    ("2024-11", "2024-12-20"), ("2024-12", "2025-01-31"),
    ("2025-01", "2025-02-28"), ("2025-02", "2025-03-28"),
    ("2025-03", "2025-04-30"), ("2025-04", "2025-05-30"),
    ("2025-05", "2025-06-27"), ("2025-06", "2025-07-31"),
    ("2025-07", "2025-08-29"), ("2025-08", "2025-09-26"),
]

FOMC_SEED = [
    "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17",
]


def historical_seed_events(start: date, end: date) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for ref, release_day in BLS_CPI_SEED:
        dt = datetime.fromisoformat(f"{release_day} 08:30:00")
        if start <= dt.date() <= end:
            events.append({
                "event_id": f"CPI_{ref}", "event_type": "CPI_RELEASE",
                "reference_period": ref, "release_datetime_et": dt.isoformat(sep=" "),
                "official_url": f"https://www.bls.gov/news.release/archives/cpi_{dt.strftime('%m%d%Y')}.htm",
            })
    for ref, release_day in BLS_EMPLOYMENT_SEED:
        dt = datetime.fromisoformat(f"{release_day} 08:30:00")
        if start <= dt.date() <= end:
            events.append({
                "event_id": f"EMPLOYMENT_{ref}", "event_type": "EMPLOYMENT_RELEASE",
                "reference_period": ref, "release_datetime_et": dt.isoformat(sep=" "),
                "official_url": f"https://www.bls.gov/news.release/archives/empsit_{dt.strftime('%m%d%Y')}.htm",
            })
    for ref, release_day in BEA_PCE_SEED:
        dt = datetime.fromisoformat(f"{release_day} 08:30:00")
        month_name = datetime.strptime(ref, "%Y-%m").strftime("%B").lower()
        ref_year = ref[:4]
        if start <= dt.date() <= end:
            events.append({
                "event_id": f"PCE_{ref}", "event_type": "PCE_RELEASE",
                "reference_period": ref, "release_datetime_et": dt.isoformat(sep=" "),
                "official_url": f"https://www.bea.gov/news/{dt.year}/personal-income-and-outlays-{month_name}-{ref_year}",
            })
    for release_day in FOMC_SEED:
        dt = datetime.fromisoformat(f"{release_day} 14:00:00")
        if start <= dt.date() <= end:
            events.append({
                "event_id": f"FOMC_{release_day}", "event_type": "FOMC_RATE_DECISION",
                "reference_period": release_day, "release_datetime_et": dt.isoformat(sep=" "),
                "official_url": (
                    "https://www.federalreserve.gov/newsevents/pressreleases/"
                    f"monetary{dt.strftime('%Y%m%d')}a.htm"
                ),
            })
    return events


def fetch(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def release_datetime(day_text: str, time_text: str) -> datetime:
    day_text = re.sub(r"^[A-Za-z]+,\s*", "", day_text.strip())
    return datetime.strptime(f"{day_text} {time_text.upper()}", "%B %d, %Y %I:%M %p")


def reference_period(text: str) -> str:
    match = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b",
        text,
        re.I,
    )
    if not match:
        raise ValueError(f"Reference period not found in: {text}")
    return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%B %Y").strftime("%Y-%m")


def bls_events(session: requests.Session, start: date, end: date) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    row_pattern = re.compile(
        r"(?P<day>[A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2},\s+20\d{2})\s+"
        r"(?P<time>\d{1,2}:\d{2}\s+[AP]M)\s+"
        r"(?P<title>(?:Consumer Price Index|Employment Situation)\s+for\s+"
        r"[A-Za-z]+\s+20\d{2})",
        re.I,
    )
    for year in range(start.year, end.year + 1):
        url = f"https://www.bls.gov/schedule/{year}/"
        soup = BeautifulSoup(fetch(session, url), "html.parser")
        for row in soup.select("tr"):
            text = " ".join(row.stripped_strings)
            match = row_pattern.search(text)
            if not match:
                continue
            dt = release_datetime(match.group("day"), match.group("time"))
            if not start <= dt.date() <= end:
                continue
            title = match.group("title")
            ref = reference_period(title)
            if title.lower().startswith("consumer price index"):
                event_type, prefix, archive = "CPI_RELEASE", "CPI", "cpi"
            else:
                event_type, prefix, archive = "EMPLOYMENT_RELEASE", "EMPLOYMENT", "empsit"
            events.append(
                {
                    "event_id": f"{prefix}_{ref}",
                    "event_type": event_type,
                    "reference_period": ref,
                    "release_datetime_et": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "official_url": (
                        "https://www.bls.gov/news.release/archives/"
                        f"{archive}_{dt.strftime('%m%d%Y')}.htm"
                    ),
                }
            )
    return events


def bea_events(session: requests.Session, start: date, end: date) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    seen: set[str] = set()
    for year in range(start.year, end.year + 1):
        url = f"https://www.bea.gov/news/schedule/full-{year}"
        soup = BeautifulSoup(fetch(session, url), "html.parser")
        for anchor in soup.find_all("a", href=True):
            title = " ".join(anchor.stripped_strings)
            if not title.lower().startswith("personal income and outlays"):
                continue
            official_url = urljoin(url, anchor["href"])
            if official_url in seen:
                continue
            container = anchor.find_parent("tr") or anchor.parent
            context = " ".join(container.stripped_strings)
            date_match = re.search(
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
                r"(\d{1,2})(?:,\s*(20\d{2}))?\b",
                context,
                re.I,
            )
            time_match = re.search(r"\b(\d{1,2}:\d{2}\s*[AP]M)\b", context, re.I)
            if not date_match or not time_match:
                continue
            release_year = int(date_match.group(3) or year)
            dt = datetime.strptime(
                f"{date_match.group(1)} {date_match.group(2)} {release_year} {time_match.group(1).upper()}",
                "%B %d %Y %I:%M %p",
            )
            if not start <= dt.date() <= end:
                continue
            ref = reference_period(title)
            event_id = f"PCE_{ref}"
            events.append(
                {
                    "event_id": event_id,
                    "event_type": "PCE_RELEASE",
                    "reference_period": ref,
                    "release_datetime_et": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "official_url": official_url,
                }
            )
            seen.add(official_url)
    return events


def fomc_events(session: requests.Session, start: date, end: date) -> list[dict[str, str]]:
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    soup = BeautifulSoup(fetch(session, url), "html.parser")
    events: list[dict[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(r"monetary(20\d{6})a\.htm$", re.I)
    for anchor in soup.find_all("a", href=True):
        official_url = urljoin(url, anchor["href"])
        match = pattern.search(official_url)
        if not match or official_url in seen:
            continue
        label = " ".join(anchor.stripped_strings).lower()
        if "statement" not in label:
            continue
        dt = datetime.strptime(match.group(1), "%Y%m%d").replace(hour=14)
        if not start <= dt.date() <= end:
            continue
        ref = dt.strftime("%Y-%m-%d")
        events.append(
            {
                "event_id": f"FOMC_{ref}",
                "event_type": "FOMC_RATE_DECISION",
                "reference_period": ref,
                "release_datetime_et": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "official_url": official_url,
            }
        )
        seen.add(official_url)
    return events


def main() -> None:
    args = parse_args()
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")
    events = historical_seed_events(args.start, args.end)
    if args.refresh_official:
        session = requests.Session()
        session.headers.update(HEADERS)
        for loader in (bls_events, bea_events, fomc_events):
            try:
                events.extend(loader(session, args.start, args.end))
            except requests.RequestException as exc:
                print(f"WARNING: {loader.__name__} refresh failed; using verified seed: {exc}")

    merged = {event["event_id"]: event for event in events}
    if args.merge and args.merge.exists():
        for event in json.loads(args.merge.read_text(encoding="utf-8")):
            release_day = datetime.fromisoformat(event["release_datetime_et"]).date()
            if args.start <= release_day <= args.end:
                merged[event["event_id"]] = event

    ordered = sorted(merged.values(), key=lambda row: (row["release_datetime_et"], row["event_type"]))
    counts: dict[str, int] = {}
    for event in ordered:
        counts[event["event_type"]] = counts.get(event["event_type"], 0) + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(ordered)} official event anchors to {args.output}")
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
