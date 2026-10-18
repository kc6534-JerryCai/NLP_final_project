#!/usr/bin/env python
"""Fetch official release dates and first-release U-3/PCE values for the IV study.

Only primary BLS and BEA archive pages are queried. The output preserves parse failures
instead of silently substituting revised observations.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests


USER_AGENT = "Mozilla/5.0 threshold-IV academic research; contact: local project"
MONTHS = {
    name: number
    for number, name in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}


class VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)

    @property
    def text(self) -> str:
        return " ".join(self.parts)


class TableRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            text = " ".join(data.split())
            if text:
                self.cell.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append(" ".join(self.cell))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None


def visible_text(html: str) -> str:
    parser = VisibleText()
    parser.feed(html)
    return parser.text


def get(session: requests.Session, url: str, retries: int = 3) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200:
                return response
            last = RuntimeError(f"HTTP {response.status_code}: {url}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(str(last))


def bls_schedule(session: requests.Session, start_year: int, end_year: int) -> dict[str, str]:
    result: dict[str, str] = {}
    weekdays = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    pattern = re.compile(
        rf"({weekdays},\s+[A-Za-z]+\s+\d{{1,2}},\s+\d{{4}})\s+08:30\s+AM\s+"
        r"Employment Situation for\s+([A-Za-z]+)\s+(\d{4})",
        re.I,
    )
    for year in range(start_year, end_year + 2):
        url = f"https://www.bls.gov/schedule/{year}/home.htm"
        try:
            text = visible_text(get(session, url).text)
        except RuntimeError:
            continue
        for date_text, reference_name, reference_year in pattern.findall(text):
            release = datetime.strptime(re.sub(r"^[A-Za-z]+,\s+", "", date_text), "%B %d, %Y")
            month_number = MONTHS[reference_name.title()]
            key = f"{int(reference_year):04d}-{month_number:02d}"
            result[key] = release.strftime("%Y-%m-%d")
    return result


def parse_u3(text: str) -> float | None:
    patterns = [
        r"unemployment rate\s+(?:was\s+(?:unchanged\s+)?at|remained\s+at|held\s+at|"
        r"changed\s+little\s+at|edged\s+(?:up|down)\s+to|rose\s+to|increased\s+to|"
        r"declined\s+to|decreased\s+to|fell\s+to)\s+([0-9]+(?:\.[0-9]+)?)\s+percent",
        r"unemployment rate.{0,90}?\b([0-9]+\.[0-9]+)\s+percent",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:30000], flags=re.I)
        if match:
            value = float(match.group(1))
            if 0 <= value <= 30:
                return value
    return None


def fetch_bls(session: requests.Session, months: pd.DatetimeIndex) -> list[dict]:
    schedule = bls_schedule(session, months.min().year - 1, months.max().year)
    rows: list[dict] = []
    for month in months:
        key = month.strftime("%Y-%m")
        release = schedule.get(key)
        if not release:
            rows.append({
                "indicator": "u3_unemployment_pct_sa",
                "reference_month": key,
                "release_date": "",
                "first_release_value": "",
                "source_url": "",
                "parse_status": "missing_release_date",
            })
            continue
        mmddyyyy = pd.Timestamp(release).strftime("%m%d%Y")
        url = f"https://www.bls.gov/news.release/archives/empsit_{mmddyyyy}.htm"
        try:
            response = get(session, url)
            value = parse_u3(visible_text(response.text))
            status = "official_archive" if value is not None else "value_parse_failed"
        except RuntimeError:
            value = None
            status = "archive_fetch_failed"
        rows.append({
            "indicator": "u3_unemployment_pct_sa",
            "reference_month": key,
            "release_date": release,
            "first_release_value": value if value is not None else "",
            "source_url": url,
            "parse_status": status,
        })
        time.sleep(0.04)
    return rows


def pce_url(month: pd.Timestamp) -> str:
    release_year = (month + pd.offsets.MonthBegin(1)).year
    month_name = month.strftime("%B").lower()
    return f"https://www.bea.gov/news/{release_year}/personal-income-and-outlays-{month_name}-{month.year}"


def parse_bea_release_date(text: str) -> str | None:
    match = re.search(
        r"EMBARGOED UNTIL RELEASE AT\s+8:30\s*(?:A\.?M\.?|a\.?m\.?).{0,20}?"
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*"
        r"([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        text,
        flags=re.I,
    )
    if not match:
        return None
    return datetime.strptime(match.group(1), "%B %d, %Y").strftime("%Y-%m-%d")


def parse_pce_yoy(text: str, month: pd.Timestamp, html: str = "") -> float | None:
    month_name = month.strftime("%B")
    patterns = [
        rf"PCE price index for\s+{month_name}\s+(increased|decreased)\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s+percent from one year ago",
        rf"price index for personal consumption expenditures.*?for\s+{month_name}\s+"
        r"(increased|decreased)\s+([0-9]+(?:\.[0-9]+)?)\s+percent from one year ago",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = float(match.group(2))
            return -value if match.group(1).lower() == "decreased" else value
    if html:
        parser = TableRows()
        parser.feed(html)
        in_yoy_section = False
        for row in parser.rows:
            joined = " ".join(row).replace("\xa0", " ")
            if re.search(r"(?:one year ago|same month one year ago)", joined, flags=re.I):
                in_yoy_section = True
            if in_yoy_section and row:
                label = re.sub(r"\s+", " ", row[0]).strip().casefold()
                if label in {"pce", "pce price index"}:
                    numbers = []
                    for cell in row[1:]:
                        numbers.extend(re.findall(r"-?[0-9]+(?:\.[0-9]+)?", cell))
                    if numbers:
                        return float(numbers[-1])
    return None


def fetch_bea(session: requests.Session, months: pd.DatetimeIndex) -> list[dict]:
    rows: list[dict] = []
    for month in months:
        url = pce_url(month)
        alternatives = [url, url.replace("/news/", "/index.php/news/")]
        text = ""
        html = ""
        used_url = url
        status = "archive_fetch_failed"
        for candidate in alternatives:
            try:
                response = get(session, candidate, retries=2)
                html = response.text
                text = visible_text(html)
                used_url = candidate
                status = "official_archive"
                break
            except RuntimeError:
                continue
        value = parse_pce_yoy(text, month, html) if text else None
        release = parse_bea_release_date(text) if text else None
        if status == "official_archive" and value is None:
            status = "value_parse_failed"
        if status == "official_archive" and release is None:
            status = "date_parse_failed"
        rows.append({
            "indicator": "pce_inflation_yoy_pct",
            "reference_month": month.strftime("%Y-%m"),
            "release_date": release or "",
            "first_release_value": value if value is not None else "",
            "source_url": used_url,
            "parse_status": status,
        })
        time.sleep(0.04)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2009-01")
    parser.add_argument("--end", default="2026-09")
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/official_release_snapshot_iv.csv")
    )
    parser.add_argument("--indicator", choices=["all", "u3", "pce"], default="all")
    args = parser.parse_args()
    months = pd.date_range(args.start, args.end, freq="MS")
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows: list[dict] = []
    if args.indicator in {"all", "u3"}:
        rows.extend(fetch_bls(session, months))
    if args.indicator in {"all", "pce"}:
        rows.extend(fetch_bea(session, months))
    if args.indicator != "all" and args.output.exists():
        existing = pd.read_csv(args.output, dtype=str).fillna("")
        replaced = {
            "u3": "u3_unemployment_pct_sa",
            "pce": "pce_inflation_yoy_pct",
        }[args.indicator]
        rows = existing.loc[existing["indicator"] != replaced].to_dict("records") + rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frame = pd.DataFrame(rows)
    print(frame.groupby(["indicator", "parse_status"]).size().to_string())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
