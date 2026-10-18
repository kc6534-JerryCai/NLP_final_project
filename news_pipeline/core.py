from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from .settings import KeywordConfig


OUTPUT_FIELDS = [
    "article_id", "source", "url", "title", "abstract", "body", "text_scope",
    "published_at", "updated_at", "retrieved_at", "section", "keywords",
    "geographic_tags", "matched_query_groups", "us_relevance", "relevance_reason",
    "direct_sentiment_coverage", "article_type", "duplicate_group_id",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_url(value: str) -> str:
    if not value:
        return ""
    parts = urlsplit(value.strip())
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+", "/", parts.path).rstrip("/")
    return urlunsplit((parts.scheme.lower() or "https", host, path, "", ""))


def _searchable(value: str) -> str:
    value = value.casefold().replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def contains_term(text: str, term: str) -> bool:
    normalized_text = _searchable(text)
    return _contains_normalized(normalized_text, term)


def _contains_normalized(normalized_text: str, term: str) -> bool:
    normalized_term = _searchable(term)
    return bool(normalized_term and f" {normalized_term} " in f" {normalized_text} ")


def matched_groups(text: str, config: KeywordConfig) -> list[str]:
    normalized_text = _searchable(text)
    found = []
    for group, terms in config.retrieval_groups.items():
        if any(_contains_normalized(normalized_text, term) for term in terms):
            found.append(group)
    return found


def classify_relevance(
    *, source: str, title: str, abstract: str, body: str, section: str,
    keywords: list[str], geographic_tags: list[str], config: KeywordConfig,
) -> tuple[str, str]:
    combined = " ".join([title, abstract, body, section, *keywords, *geographic_tags])
    normalized = _searchable(combined)

    promotional = ("paid content", "advertisement", "sponsored content", "promotion")
    if any(marker in normalized for marker in promotional):
        return "irrelevant", "Excluded promotional or sponsored content."

    strong_us = [marker for marker in config.us_markers if _contains_normalized(normalized, marker)]
    tag_text = " ".join(keywords + geographic_tags).casefold()
    us_tag = any(marker in tag_text for marker in ("world/usa", "us-news", "united states", "u.s."))
    if strong_us or us_tag:
        evidence = strong_us[:3] or ["US-specific API tag"]
        return "relevant", "Explicit U.S. economic/geographic evidence: " + ", ".join(evidence)

    economic_context = any(
        _contains_normalized(normalized, term)
        for group in config.groups.values()
        for term in group["terms"]
    )
    geo_hits = [marker for marker in config.us_geo_markers if _contains_normalized(normalized, marker)]
    if geo_hits and economic_context:
        return "relevant", "U.S. state/city evidence combined with economic context: " + ", ".join(geo_hits[:3])

    market_only = any(
        phrase in normalized
        for phrase in ("stock pick", "share price", "stock price", "quarterly earnings", "earnings report")
    )
    household_or_macro = any(
        phrase in normalized
        for phrase in ("economy", "economic", "consumer", "household", "employment", "inflation", "wage", "housing")
    )
    if market_only and not household_or_macro:
        return "irrelevant", "Company/market item without substantive macroeconomic or household context."

    foreign_hits = [marker for marker in config.foreign_markers if _contains_normalized(normalized, marker)]
    if foreign_hits and not geo_hits:
        return "irrelevant", "Foreign focus with no detected U.S. economic link: " + ", ".join(foreign_hits[:3])

    return "uncertain", (
        "Economic retrieval match, but deterministic metadata/text rules found neither sufficient "
        "U.S. evidence nor a conclusive non-U.S. exclusion. Retained for review."
    )


def classify_article_type(source: str, raw_type: str, section: str, tags: list[str]) -> str:
    text = " ".join([raw_type, section, *tags]).casefold()
    if "editorial" in text:
        return "editorial"
    if any(value in text for value in ("op-ed", "opinion", "commentisfree", "column", "letter")):
        return "opinion"
    if any(value in text for value in ("video", "podcast", "audio", "interactive")):
        return "other"
    return "news"


def normalize_nyt(
    doc: dict[str, Any], config: KeywordConfig, retrieved_at: str,
    retrieval_group: str | None = None,
) -> dict[str, Any] | None:
    headline = doc.get("headline") or {}
    title = clean_text(headline.get("main") or headline.get("print_headline"))
    abstract_parts = [doc.get("abstract"), doc.get("snippet"), doc.get("lead_paragraph")]
    abstract = clean_text(" ".join(str(value) for value in abstract_parts if value))
    keyword_objects = doc.get("keywords") or []
    keywords = sorted({clean_text(item.get("value")) for item in keyword_objects if item.get("value")})
    geo = sorted({
        clean_text(item.get("value")) for item in keyword_objects
        if str(item.get("name", "")).casefold() in {"glocations", "locations"} and item.get("value")
    })
    searchable = " ".join([title, abstract, *keywords, *geo])
    groups = matched_groups(searchable, config)
    if retrieval_group and retrieval_group not in groups:
        groups.append(retrieval_group)
    if not groups:
        return None
    section = clean_text(doc.get("section_name") or doc.get("news_desk"))
    relevance, reason = classify_relevance(
        source="nyt", title=title, abstract=abstract, body="", section=section,
        keywords=keywords, geographic_tags=geo, config=config,
    )
    raw_type = clean_text(doc.get("type_of_material") or doc.get("document_type"))
    article_id = clean_text(doc.get("_id") or doc.get("uri") or doc.get("web_url"))
    return {
        "article_id": article_id,
        "source": "nyt",
        "url": normalized_url(clean_text(doc.get("web_url"))),
        "title": title,
        "abstract": abstract,
        "body": "",
        "text_scope": "headline_abstract_lead" if abstract else "headline_only",
        "published_at": clean_text(doc.get("pub_date")),
        "updated_at": "",
        "retrieved_at": retrieved_at,
        "section": section,
        "keywords": keywords,
        "geographic_tags": geo,
        "matched_query_groups": [group for group in groups if group != "DIRECT"],
        "us_relevance": relevance,
        "relevance_reason": reason,
        "direct_sentiment_coverage": "DIRECT" in groups,
        "article_type": classify_article_type("nyt", raw_type, section, keywords),
        "duplicate_group_id": "",
    }


def normalize_guardian(
    item: dict[str, Any], config: KeywordConfig, retrieved_at: str,
    retrieval_group: str | None = None,
) -> dict[str, Any] | None:
    fields = item.get("fields") or {}
    tags = item.get("tags") or []
    title = clean_text(fields.get("headline") or item.get("webTitle"))
    abstract = clean_text(fields.get("trailText"))
    body = clean_text(fields.get("bodyText"))
    keywords = sorted({clean_text(tag.get("id") or tag.get("webTitle")) for tag in tags if tag})
    geo = sorted({
        clean_text(tag.get("webTitle") or tag.get("id")) for tag in tags
        if str(tag.get("type", "")).casefold() == "keyword"
        and any(token in str(tag.get("id", "")).casefold() for token in ("usa", "us-news", "states", "cities"))
    })
    searchable = " ".join([title, abstract, body, *keywords])
    groups = matched_groups(searchable, config)
    if retrieval_group and retrieval_group not in groups:
        groups.append(retrieval_group)
    if not groups:
        return None
    section = clean_text(item.get("sectionName") or item.get("sectionId"))
    relevance, reason = classify_relevance(
        source="guardian", title=title, abstract=abstract, body=body, section=section,
        keywords=keywords, geographic_tags=geo, config=config,
    )
    raw_type = clean_text(item.get("type"))
    return {
        "article_id": clean_text(item.get("id") or item.get("apiUrl") or item.get("webUrl")),
        "source": "guardian",
        "url": normalized_url(clean_text(item.get("webUrl") or fields.get("shortUrl"))),
        "title": title,
        "abstract": abstract,
        "body": body,
        "text_scope": "full_body" if body else ("headline_abstract" if abstract else "headline_only"),
        "published_at": clean_text(fields.get("firstPublicationDate") or item.get("webPublicationDate")),
        "updated_at": clean_text(fields.get("lastModified")),
        "retrieved_at": retrieved_at,
        "section": section,
        "keywords": keywords,
        "geographic_tags": geo,
        "matched_query_groups": [group for group in groups if group != "DIRECT"],
        "us_relevance": relevance,
        "relevance_reason": reason,
        "direct_sentiment_coverage": "DIRECT" in groups,
        "article_type": classify_article_type("guardian", raw_type, section, keywords),
        "duplicate_group_id": "",
    }


def _merge_record(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    target["matched_query_groups"] = sorted(set(target["matched_query_groups"]) | set(incoming["matched_query_groups"]))
    target["direct_sentiment_coverage"] = bool(
        target["direct_sentiment_coverage"] or incoming["direct_sentiment_coverage"]
    )
    for field in ("title", "abstract", "body", "updated_at", "section"):
        if len(str(incoming.get(field, ""))) > len(str(target.get(field, ""))):
            target[field] = incoming[field]
    target["keywords"] = sorted(set(target.get("keywords", [])) | set(incoming.get("keywords", [])))
    target["geographic_tags"] = sorted(
        set(target.get("geographic_tags", [])) | set(incoming.get("geographic_tags", []))
    )


def deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    exact: dict[str, dict[str, Any]] = {}
    url_to_key: dict[str, str] = {}
    for record in records:
        source = record["source"]
        key = f"{source}::id::{record['article_id']}"
        url = normalized_url(record.get("url", ""))
        if url and f"{source}::{url}" in url_to_key:
            key = url_to_key[f"{source}::{url}"]
        if key in exact:
            _merge_record(exact[key], record)
        else:
            exact[key] = record
            if url:
                url_to_key[f"{source}::{url}"] = key

    output = list(exact.values())
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in output:
        day = str(record.get("published_at", ""))[:10]
        title_tokens = _searchable(record.get("title", "")).split()
        block = " ".join(title_tokens[:3])
        buckets[(record["source"], day, block)].append(record)

    for (source, day, _block), items in buckets.items():
        representatives: list[tuple[str, str]] = []
        for record in sorted(items, key=lambda value: _searchable(value.get("title", ""))):
            title = _searchable(record.get("title", ""))
            group_id = ""
            # Blocking plus a bounded deterministic candidate window keeps this
            # near-linear on large corpora while still grouping adjacent, highly
            # similar titles. Exact ID/URL duplicates were already merged above.
            for representative, candidate_id in representatives[-25:]:
                if title and SequenceMatcher(None, title, representative).ratio() >= 0.94:
                    group_id = candidate_id
                    break
            if not group_id:
                digest = hashlib.sha1(f"{source}|{day}|{title}".encode("utf-8")).hexdigest()[:16]
                group_id = f"{source}-{digest}"
                representatives.append((title, group_id))
            record["duplicate_group_id"] = group_id
    return sorted(output, key=lambda value: (value.get("published_at", ""), value["source"], value["article_id"]))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_clean_outputs(records: list[dict[str, Any]], data_dir: Path) -> None:
    clean_dir = data_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = clean_dir / "news_articles.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with (clean_dir / "news_articles.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for field in ("keywords", "geographic_tags", "matched_query_groups"):
                row[field] = json.dumps(row.get(field, []), ensure_ascii=False)
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})

    headline_fields = [
        "article_id", "source", "url", "title", "published_at", "updated_at", "retrieved_at",
        "section", "matched_query_groups", "us_relevance", "relevance_reason",
        "direct_sentiment_coverage", "article_type", "duplicate_group_id",
    ]
    with (clean_dir / "headlines_only.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headline_fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in headline_fields}
            row["matched_query_groups"] = json.dumps(row["matched_query_groups"], ensure_ascii=False)
            writer.writerow(row)

    with (clean_dir / "full_text_subset.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            if record["text_scope"] == "full_body" and record["body"]:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with (clean_dir / "analysis_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            if record["us_relevance"] in {"relevant", "uncertain"}:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with (clean_dir / "relevant_only.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            if record["us_relevance"] == "relevant":
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def generate_reports(records: list[dict[str, Any]], data_dir: Path, config_version: str) -> None:
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    monthly: dict[tuple[str, str], Counter] = defaultdict(Counter)
    topics: Counter = Counter()
    relevant_topics: Counter = Counter()
    candidate_topics: Counter = Counter()
    overlaps: Counter = Counter()
    for record in records:
        month = str(record.get("published_at", ""))[:7] or "unknown"
        source = record["source"]
        counter = monthly[(month, source)]
        counter["total_articles"] += 1
        counter[record["us_relevance"]] += 1
        if record["text_scope"] == "full_body" and record["body"]:
            counter["full_body_count"] += 1
        for group in record["matched_query_groups"]:
            topics[(source, group)] += 1
            if record["us_relevance"] == "relevant":
                relevant_topics[(source, group)] += 1
            if record["us_relevance"] in {"relevant", "uncertain"}:
                candidate_topics[(source, group)] += 1
        signature = "+".join(record["matched_query_groups"]) or "DIRECT_ONLY"
        overlaps[(source, signature)] += 1

    manifest_path = data_dir / "state" / "manifest.jsonl"
    latest_manifest: dict[tuple, dict[str, Any]] = {}
    if manifest_path.exists():
        for entry in load_jsonl(manifest_path):
            key = (entry.get("source"), entry.get("start"), entry.get("end"), entry.get("group"), entry.get("page"))
            latest_manifest[key] = entry
    incomplete_by_month_source: Counter = Counter()
    request_status_by_month_source: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for entry in latest_manifest.values():
        month_source = (str(entry.get("start", ""))[:7], entry.get("source", ""))
        request_status_by_month_source[month_source][str(entry.get("status", "unknown"))] += 1
        if entry.get("status") not in {"complete", "cached", "split_for_pagination"}:
            incomplete_by_month_source[month_source] += 1

    status_fields = ["source", "endpoint", "start", "end", "group", "page", "status", "at", "error", "reason"]
    with (reports_dir / "collection_status.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=status_fields)
        writer.writeheader()
        for entry in sorted(
            latest_manifest.values(),
            key=lambda value: (
                str(value.get("source", "")), str(value.get("start", "")),
                str(value.get("group", "")), int(value.get("page") or 0),
            ),
        ):
            writer.writerow({field: entry.get(field, "") for field in status_fields})

    monthly_fields = [
        "month", "source", "total_articles", "full_body_count", "full_body_rate",
        "relevant", "irrelevant", "uncertain", "analysis_candidate_count",
        "request_units", "complete_request_units",
        "incomplete_intervals",
    ]
    with (reports_dir / "monthly_source_coverage.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=monthly_fields)
        writer.writeheader()
        month_sources = set(monthly) | set(request_status_by_month_source)
        for month, source in sorted(month_sources):
            counts = monthly[(month, source)]
            statuses = request_status_by_month_source[(month, source)]
            total = counts["total_articles"]
            writer.writerow({
                "month": month,
                "source": source,
                "total_articles": total,
                "full_body_count": counts["full_body_count"],
                "full_body_rate": round(counts["full_body_count"] / total, 4) if total else 0,
                "relevant": counts["relevant"],
                "irrelevant": counts["irrelevant"],
                "uncertain": counts["uncertain"],
                "analysis_candidate_count": counts["relevant"] + counts["uncertain"],
                "request_units": sum(statuses.values()),
                "complete_request_units": sum(
                    statuses[value] for value in ("complete", "cached", "split_for_pagination")
                ),
                "incomplete_intervals": incomplete_by_month_source[(month, source)],
            })

    with (reports_dir / "topic_counts.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "group", "all_retrieval_hits", "relevant_hits", "analysis_candidate_hits"])
        for (source, group), count in sorted(topics.items()):
            writer.writerow([
                source, group, count, relevant_topics[(source, group)], candidate_topics[(source, group)],
            ])

    with (reports_dir / "topic_overlap.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "groups_signature", "article_count"])
        for (source, signature), count in sorted(overlaps.items()):
            writer.writerow([source, signature, count])

    review_fields = [
        "source", "month", "article_id", "title", "abstract", "body_excerpt", "url",
        "matched_query_groups", "automatic_us_relevance", "manual_tone_positive_negative_neutral",
        "manual_us_relevance", "review_notes",
    ]
    review_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        review_buckets[(record["source"], str(record.get("published_at", ""))[:7])].append(record)
    with (reports_dir / "manual_review_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for (source, month), items in sorted(review_buckets.items()):
            deterministic = sorted(
                items,
                key=lambda value: hashlib.sha1(
                    f"{value.get('article_id', '')}|{value.get('title', '')}".encode("utf-8")
                ).hexdigest(),
            )[:5]
            for record in deterministic:
                writer.writerow({
                    "source": source,
                    "month": month,
                    "article_id": record["article_id"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "body_excerpt": record["body"][:500],
                    "url": record["url"],
                    "matched_query_groups": json.dumps(record["matched_query_groups"], ensure_ascii=False),
                    "automatic_us_relevance": record["us_relevance"],
                    "manual_tone_positive_negative_neutral": "",
                    "manual_us_relevance": "",
                    "review_notes": "",
                })

    diagnostics = generate_raw_diagnostics(data_dir)
    archive_rows = [
        row for row in diagnostics
        if row["source"] == "nyt" and row["endpoint"] == "archive" and row["bucket"] != "probe"
    ]
    archive_counts = [int(row["result_count"]) for row in archive_rows if str(row["result_count"]).isdigit()]
    median_archive_count = int(statistics.median(archive_counts)) if archive_counts else 0
    anomalous_archive_months = [
        f"{row['bucket']} ({row['result_count']} raw records)"
        for row in archive_rows
        if median_archive_count and str(row["result_count"]).isdigit()
        and int(row["result_count"]) < 0.35 * median_archive_count
        and row["bucket"] != max((item["bucket"] for item in archive_rows), default="")
    ]
    latest_published = max((str(record.get("published_at", "")) for record in records), default="")
    requested_end = ""
    last_run_path = data_dir / "state" / "last_run.json"
    if last_run_path.exists():
        try:
            requested_end = str(json.loads(last_run_path.read_text(encoding="utf-8")).get("end", ""))
        except ValueError:
            requested_end = ""
    sources = sorted({record["source"] for record in records})
    incomplete_units = sum(incomplete_by_month_source.values())
    quality = f"""# Collection quality note

- Generated: {utc_now()}
- Keyword dictionary version: {config_version}
- Cleaned unique articles: {len(records)}
- Relevant-only articles: {sum(record['us_relevance'] == 'relevant' for record in records)}
- Analysis candidates (relevant + uncertain): {sum(record['us_relevance'] in {'relevant', 'uncertain'} for record in records)}
- Sources present: {', '.join(sources) if sources else 'none'}
- NYT Archive months cached: {len(archive_rows)}
- Incomplete request units recorded: {incomplete_units}
- Requested cutoff date: {requested_end or 'not recorded'}
- Latest publication timestamp returned: {latest_published or 'none'}
- Median NYT Archive raw records per cached month: {median_archive_count}
- Low-volume non-cutoff Archive anomalies: {', '.join(anomalous_archive_months) if anomalous_archive_months else 'none detected'}
- NYT text scope: headline/abstract/snippet/lead metadata only; it is never labelled full text.
- U.S. relevance is a conservative deterministic screen. `uncertain` records are retained for human review.
- `duplicate_group_id` identifies exact or very similar same-source, same-day titles. Cross-source event coverage is not merged.
- API completeness is only claimed for intervals whose manifest status is complete/cached. Rate limits, API caps,
  historical metadata gaps, later article edits, and retrieval-time metadata prevent a guarantee that the dataset
  reproduces the exact originally published article version.
- A successful Archive response does not prove content completeness. In particular, a latest publication date before
  the requested cutoff and unusually low monthly raw counts must be treated as unresolved API coverage gaps.
- Topic counts are multi-label and may sum to more than 100% of the source-month denominator.
"""
    (reports_dir / "collection_quality.md").write_text(quality, encoding="utf-8")


def generate_raw_diagnostics(data_dir: Path) -> list[dict[str, Any]]:
    fields = [
        "source", "bucket", "endpoint", "from_date", "to_date", "query_sha1",
        "total_hits", "pages", "current_page", "result_count", "retrieved_at", "raw_file",
    ]
    rows: list[dict[str, Any]] = []
    raw_root = data_dir / "raw"
    if raw_root.exists():
        for path in raw_root.rglob("*.json.gz"):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    stored = json.load(handle)
            except (OSError, ValueError):
                continue
            request = stored.get("request") or {}
            params = request.get("params") or {}
            payload = stored.get("response") or {}
            url = str(request.get("url") or "")
            source = path.relative_to(raw_root).parts[0] if path.is_relative_to(raw_root) else ""
            endpoint = "unknown"
            total_hits: Any = ""
            pages: Any = ""
            current_page: Any = params.get("page", "")
            results: list[Any] = []
            if "/archive/" in url:
                endpoint = "archive"
                results = ((payload.get("response") or {}).get("docs") or [])
                total_hits = len(results)
                pages = 1
            elif "articlesearch" in url:
                endpoint = "article_search"
                response = payload.get("response") or {}
                results = response.get("docs") or []
                total_hits = (response.get("meta") or {}).get("hits", "")
                pages = math.ceil(int(total_hits) / 10) if str(total_hits).isdigit() else ""
            elif "guardianapis.com/search" in url:
                endpoint = "guardian_search"
                response = payload.get("response") or {}
                results = response.get("results") or []
                total_hits = response.get("total", "")
                pages = response.get("pages", "")
                current_page = response.get("currentPage", current_page)
            query = str(params.get("q") or params.get("fq") or "")
            rows.append({
                "source": source,
                "bucket": path.parent.name,
                "endpoint": endpoint,
                "from_date": params.get("from-date") or params.get("begin_date") or "",
                "to_date": params.get("to-date") or params.get("end_date") or "",
                "query_sha1": hashlib.sha1(query.encode("utf-8")).hexdigest()[:12] if query else "",
                "total_hits": total_hits,
                "pages": pages,
                "current_page": current_page,
                "result_count": len(results),
                "retrieved_at": stored.get("retrieved_at", ""),
                "raw_file": str(path),
            })
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "raw_response_diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["source"], row["bucket"], row["endpoint"], row["query_sha1"], str(row["current_page"]))))
    return rows
