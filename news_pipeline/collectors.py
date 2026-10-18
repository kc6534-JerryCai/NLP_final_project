from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from .core import normalize_guardian, normalize_nyt, utc_now
from .settings import KeywordConfig, quoted_query


NYT_ARCHIVE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
NYT_SEARCH_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
GUARDIAN_SEARCH_URL = "https://content.guardianapis.com/search"


class CollectionError(RuntimeError):
    pass


class CallBudgetExceeded(CollectionError):
    pass


@dataclass
class RequestResult:
    payload: dict[str, Any]
    retrieved_at: str
    cached: bool


class CachedApiClient:
    def __init__(
        self,
        *,
        source: str,
        api_key: str,
        data_dir: Path,
        min_interval: float,
        max_calls: int | None,
        daily_limit: int | None = None,
        timeout: float = 120.0,
        max_retries: int = 5,
    ) -> None:
        self.source = source
        self.api_key = api_key
        self.data_dir = data_dir
        self.min_interval = min_interval
        self.max_calls = max_calls
        self.daily_limit = daily_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls = 0
        self._last_call = 0.0
        self.credential_id = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NLP-Final-Assignment-News-Collector/1.0"})

    def _daily_calls_used(self) -> int:
        path = self.data_dir / "state" / "api_calls.jsonl"
        if not path.exists():
            return 0
        today = datetime.now(timezone.utc).date().isoformat()
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if entry.get("source") != self.source or str(entry.get("at", ""))[:10] != today:
                    continue
                recorded_id = entry.get("credential_id")
                if recorded_id == self.credential_id:
                    count += 1
                elif recorded_id is None and self.source != "guardian":
                    # Pre-v1.0 ledger entries did not include credential IDs. The
                    # Guardian legacy entries belong to the replaced key; other
                    # source entries remain conservatively charged.
                    count += 1
        return count

    def _record_api_call(self) -> None:
        path = self.data_dir / "state" / "api_calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "source": self.source, "credential_id": self.credential_id, "at": utc_now(),
            }, sort_keys=True) + "\n")

    def _sanitize_error(self, value: str) -> str:
        sanitized = value.replace(self.api_key, "<redacted>") if self.api_key else value
        sanitized = re.sub(r"([?&]api-key=)[^&\s]+", r"\1<redacted>", sanitized, flags=re.IGNORECASE)
        return sanitized

    def _fingerprint(self, url: str, params: dict[str, Any]) -> str:
        redacted = {key: value for key, value in params.items() if key != "api-key"}
        canonical = json.dumps({"url": url, "params": redacted}, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

    def _cache_path(self, url: str, params: dict[str, Any], bucket: str) -> Path:
        path = self.data_dir / "raw" / self.source / bucket
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{self._fingerprint(url, params)}.json.gz"

    def _append_manifest(self, entry: dict[str, Any]) -> None:
        path = self.data_dir / "state" / "manifest.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def get(
        self,
        *,
        url: str,
        params: dict[str, Any],
        bucket: str,
        manifest: dict[str, Any],
        force_refresh: bool = False,
    ) -> RequestResult:
        cache_path = self._cache_path(url, params, bucket)
        if cache_path.exists() and not force_refresh:
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                stored = json.load(handle)
            self._append_manifest({**manifest, "status": "cached", "at": utc_now(), "cache": str(cache_path)})
            return RequestResult(stored["response"], stored["retrieved_at"], True)

        if self.max_calls is not None and self.calls >= self.max_calls:
            self._append_manifest({**manifest, "status": "deferred_call_budget", "at": utc_now()})
            raise CallBudgetExceeded(
                f"{self.source} uncached call budget ({self.max_calls}) reached; rerun to resume from cache."
            )
        if self.daily_limit is not None and self._daily_calls_used() >= self.daily_limit:
            self._append_manifest({**manifest, "status": "deferred_daily_budget", "at": utc_now()})
            raise CallBudgetExceeded(
                f"{self.source} recorded daily safety budget ({self.daily_limit}) reached; resume on the next UTC day."
            )

        request_params = dict(params)
        request_params["api-key"] = self.api_key
        redacted_params = {key: value for key, value in request_params.items() if key != "api-key"}
        error = ""
        for attempt in range(self.max_retries + 1):
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            try:
                response = self.session.get(url, params=request_params, timeout=self.timeout)
                self._last_call = time.monotonic()
                self.calls += 1
                self._record_api_call()
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait = (
                        min(float(retry_after), 120.0)
                        if retry_after and retry_after.isdigit()
                        else min(max(15, 2 ** attempt), 60)
                    )
                    error = f"HTTP 429; retrying after {wait}s"
                    if attempt < self.max_retries:
                        time.sleep(wait)
                        continue
                if response.status_code in {400, 401, 403}:
                    error = f"HTTP {response.status_code}: request rejected (credentials, API access, or query syntax)."
                    break
                response.raise_for_status()
                payload = response.json()
                retrieved_at = utc_now()
                stored = {
                    "request": {"url": url, "params": redacted_params},
                    "retrieved_at": retrieved_at,
                    "response": payload,
                }
                with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
                    json.dump(stored, handle, ensure_ascii=False)
                self._append_manifest({
                    **manifest, "status": "complete", "at": retrieved_at,
                    "cache": str(cache_path), "http_status": response.status_code,
                })
                return RequestResult(payload, retrieved_at, False)
            except (requests.RequestException, ValueError) as exc:
                error = self._sanitize_error(f"{type(exc).__name__}: {exc}")
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 60))
        self._append_manifest({**manifest, "status": "failed", "at": utc_now(), "error": error})
        raise CollectionError(f"{self.source} request failed after retries: {error}")


def _published_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None


def collect_nyt_archive(
    *,
    client: CachedApiClient,
    intervals: Iterable[tuple[date, date]],
    config: KeywordConfig,
    on_records: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen_months: set[tuple[int, int]] = set()
    for start, end in intervals:
        month_key = (start.year, start.month)
        if month_key in seen_months:
            continue
        seen_months.add(month_key)
        bucket = f"{start.year:04d}-{start.month:02d}"
        result = client.get(
            url=NYT_ARCHIVE_URL.format(year=start.year, month=start.month),
            params={},
            bucket=bucket,
            manifest={
                "source": "nyt", "endpoint": "archive", "start": start.isoformat(),
                "end": end.isoformat(), "group": "ALL_METADATA", "page": 1,
            },
        )
        docs = ((result.payload.get("response") or {}).get("docs") or [])
        month_records: list[dict[str, Any]] = []
        for doc in docs:
            published = _published_date(str(doc.get("pub_date") or ""))
            if not published or not (start <= published <= end):
                continue
            record = normalize_nyt(doc, config, result.retrieved_at)
            if record:
                month_records.append(record)
        collected.extend(month_records)
        if on_records:
            on_records(month_records)
    return collected


def _split_interval(start: date, end: date) -> tuple[tuple[date, date], tuple[date, date]]:
    midpoint = start + timedelta(days=(end - start).days // 2)
    return (start, midpoint), (midpoint + timedelta(days=1), end)


def collect_nyt_search(
    *,
    client: CachedApiClient,
    intervals: Iterable[tuple[date, date]],
    config: KeywordConfig,
    page_cap: int = 100,
    on_records: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []

    def collect_group(start: date, end: date, group: str, terms: list[str]) -> None:
        bucket = f"{start.year:04d}-{start.month:02d}"
        params = {
            "fq": quoted_query(terms),
            "begin_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "sort": "oldest",
            "page": 0,
        }
        manifest = {
            "source": "nyt", "endpoint": "article_search", "start": start.isoformat(),
            "end": end.isoformat(), "group": group, "page": 0,
        }
        first = client.get(url=NYT_SEARCH_URL, params=params, bucket=bucket, manifest=manifest)
        response = first.payload.get("response") or {}
        hits = int((response.get("meta") or {}).get("hits") or 0)
        if hits > 1000 and start < end:
            client._append_manifest({
                **manifest, "status": "split_for_pagination", "at": utc_now(), "hits": hits,
            })
            left, right = _split_interval(start, end)
            collect_group(*left, group, terms)
            collect_group(*right, group, terms)
            return
        if hits > 1000:
            client._append_manifest({
                **manifest, "status": "incomplete_api_cap", "at": utc_now(), "hits": hits,
                "reason": "NYT Article Search exposes at most 1,000 results per interval.",
            })

        pages = min(math.ceil(hits / 10), page_cap)
        page_results = [first]
        for page in range(1, pages):
            page_params = {**params, "page": page}
            page_results.append(client.get(
                url=NYT_SEARCH_URL,
                params=page_params,
                bucket=bucket,
                manifest={**manifest, "page": page},
            ))
        interval_records: list[dict[str, Any]] = []
        for request_result in page_results:
            for doc in (((request_result.payload.get("response") or {}).get("docs")) or []):
                record = normalize_nyt(doc, config, request_result.retrieved_at, retrieval_group=group)
                if record:
                    if group != "DIRECT" and group not in record["matched_query_groups"]:
                        record["matched_query_groups"].append(group)
                        record["matched_query_groups"].sort()
                    if group == "DIRECT":
                        record["direct_sentiment_coverage"] = True
                    interval_records.append(record)
        collected.extend(interval_records)
        if on_records:
            on_records(interval_records)

    for interval in intervals:
        for group, terms in config.retrieval_groups.items():
            collect_group(*interval, group, terms)
    return collected


def collect_guardian(
    *,
    client: CachedApiClient,
    intervals: Iterable[tuple[date, date]],
    config: KeywordConfig,
    page_size: int = 200,
    page_cap: int = 50,
    on_records: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []

    def collect_group(start: date, end: date, group: str, terms: list[str]) -> None:
        bucket = f"{start.year:04d}-{start.month:02d}"
        base_params = {
            "q": quoted_query(terms),
            "from-date": start.isoformat(),
            "to-date": end.isoformat(),
            "use-date": "published",
            "order-by": "oldest",
            "page-size": page_size,
            "show-fields": "headline,trailText,bodyText,firstPublicationDate,lastModified,shortUrl,byline",
            "show-tags": "all",
            "page": 1,
        }
        manifest = {
            "source": "guardian", "endpoint": "search", "start": start.isoformat(),
            "end": end.isoformat(), "group": group, "page": 1,
        }
        first = client.get(url=GUARDIAN_SEARCH_URL, params=base_params, bucket=bucket, manifest=manifest)
        response = first.payload.get("response") or {}
        pages = int(response.get("pages") or 0)
        total = int(response.get("total") or 0)
        if pages > page_cap and start < end:
            client._append_manifest({
                **manifest, "status": "split_for_pagination", "at": utc_now(), "hits": total,
                "pages": pages,
            })
            left, right = _split_interval(start, end)
            collect_group(*left, group, terms)
            collect_group(*right, group, terms)
            return
        if pages > page_cap:
            client._append_manifest({
                **manifest, "status": "incomplete_api_cap", "at": utc_now(), "hits": total,
                "reason": f"Single-day result requires {pages} pages, above configured cap {page_cap}.",
            })
            pages = page_cap

        page_results = [first]
        for page in range(2, pages + 1):
            page_results.append(client.get(
                url=GUARDIAN_SEARCH_URL,
                params={**base_params, "page": page},
                bucket=bucket,
                manifest={**manifest, "page": page},
            ))
        interval_records: list[dict[str, Any]] = []
        for request_result in page_results:
            for item in (((request_result.payload.get("response") or {}).get("results")) or []):
                record = normalize_guardian(item, config, request_result.retrieved_at, retrieval_group=group)
                if record:
                    if group != "DIRECT" and group not in record["matched_query_groups"]:
                        record["matched_query_groups"].append(group)
                        record["matched_query_groups"].sort()
                    if group == "DIRECT":
                        record["direct_sentiment_coverage"] = True
                    interval_records.append(record)
        collected.extend(interval_records)
        if on_records:
            on_records(interval_records)

    for interval in intervals:
        for group, terms in config.retrieval_groups.items():
            collect_group(*interval, group, terms)
    return collected
