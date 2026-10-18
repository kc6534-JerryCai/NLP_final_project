from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

from .collectors import (
    GUARDIAN_SEARCH_URL,
    NYT_ARCHIVE_URL,
    NYT_SEARCH_URL,
    CachedApiClient,
    CallBudgetExceeded,
    CollectionError,
    collect_guardian,
    collect_nyt_archive,
    collect_nyt_search,
)
from .core import deduplicate, generate_reports, load_jsonl, utc_now, write_clean_outputs
from .settings import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    ensure_data_dirs,
    load_environment,
    load_keyword_config,
    month_intervals,
    pilot_intervals,
    quoted_query,
)
from .security import scrub_secrets


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Date must be YYYY-MM-DD") from exc


def _log(data_dir: Path, event: str, **details: Any) -> None:
    path = data_dir / "logs" / "collection.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"at": utc_now(), "event": event, **details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _persist(records: list[dict[str, Any]], data_dir: Path, config_version: str) -> list[dict[str, Any]]:
    existing = load_jsonl(data_dir / "clean" / "news_articles.jsonl")
    merged = deduplicate([*existing, *records])
    write_clean_outputs(merged, data_dir)
    generate_reports(merged, data_dir, config_version)
    return merged


def _write_run_metadata(
    *, data_dir: Path, command: str, start: date, end: date, source: str,
    nyt_mode: str, config_path: Path,
) -> None:
    config_bytes = config_path.read_bytes()
    metadata = {
        "started_at": utc_now(),
        "command": command,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "source": source,
        "nyt_mode": nyt_mode,
        "keyword_config": str(config_path),
        "keyword_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "query_scope": {
            "nyt_archive": "monthly complete API metadata, locally matched over headline/abstract/snippet/lead/keywords",
            "nyt_article_search": "optional supplement; unfielded Lucene filter can match body/headline/byline but returns metadata only",
            "guardian": "q query over Guardian content index; bodyText requested with show-fields",
        },
    }
    path = data_dir / "state" / "last_run.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def validate(args: argparse.Namespace) -> int:
    config = load_keyword_config(args.config)
    env = load_environment()
    print(f"Keyword dictionary: v{config.version}")
    print(f"Topical groups: {len(config.groups)}; direct-sentiment terms: {len(config.direct_sentiment['terms'])}")
    print(f"NYT_API_KEY present: {bool(env['nyt_api_key'])}")
    print(f"GUARDIAN_API_KEY present: {bool(env['guardian_api_key'])}")
    for name, terms in config.retrieval_groups.items():
        query = quoted_query(terms)
        print(f"{name}: {len(terms)} terms; query characters={len(query)}")
    return 0


def report(args: argparse.Namespace) -> int:
    config = load_keyword_config(args.config)
    ensure_data_dirs(args.data_dir)
    records = deduplicate(load_jsonl(args.data_dir / "clean" / "news_articles.jsonl"))
    write_clean_outputs(records, args.data_dir)
    generate_reports(records, args.data_dir, config.version)
    print(f"Reports regenerated for {len(records)} unique articles in {args.data_dir / 'reports'}")
    return 0


def scrub(args: argparse.Namespace) -> int:
    env = load_environment()
    count = scrub_secrets(
        args.data_dir,
        [str(env.get("nyt_api_key") or ""), str(env.get("guardian_api_key") or "")],
    )
    print(f"Scrubbed secrets from {count} generated text artifact(s).")
    return 0


def purge_source(args: argparse.Namespace) -> int:
    source = args.source
    data_dir = args.data_dir.resolve()
    ensure_data_dirs(data_dir)
    config = load_keyword_config(args.config)
    clean_path = data_dir / "clean" / "news_articles.jsonl"
    existing = load_jsonl(clean_path)
    kept = [record for record in existing if record.get("source") != source]
    removed_records = len(existing) - len(kept)

    raw_dir = (data_dir / "raw" / source).resolve()
    expected_parent = (data_dir / "raw").resolve()
    removed_raw_files = 0
    if raw_dir.parent != expected_parent:
        raise RuntimeError(f"Refusing to remove unexpected path: {raw_dir}")
    if raw_dir.exists():
        removed_raw_files = sum(1 for path in raw_dir.rglob("*") if path.is_file())
        shutil.rmtree(raw_dir)

    for state_name in ("manifest.jsonl", "api_calls.jsonl"):
        path = data_dir / "state" / state_name
        if not path.exists():
            continue
        retained_lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                retained_lines.append(line)
                continue
            if entry.get("source") != source:
                retained_lines.append(line)
        path.write_text("".join(value + "\n" for value in retained_lines), encoding="utf-8")

    last_run_path = data_dir / "state" / "last_run.json"
    if last_run_path.exists():
        try:
            last_run = json.loads(last_run_path.read_text(encoding="utf-8"))
        except ValueError:
            last_run = {}
        if last_run.get("source") == source:
            last_run_path.unlink()

    log_path = data_dir / "logs" / "collection.jsonl"
    if log_path.exists():
        retained_lines = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                retained_lines.append(line)
                continue
            if entry.get("source") != source:
                retained_lines.append(line)
        log_path.write_text("".join(value + "\n" for value in retained_lines), encoding="utf-8")

    kept = deduplicate(kept)
    write_clean_outputs(kept, data_dir)
    generate_reports(kept, data_dir, config.version)
    print(f"Purged source={source}: {removed_records} cleaned articles and {removed_raw_files} raw files removed.")
    print(f"Remaining cleaned articles: {len(kept)}")
    return 0


def probe(args: argparse.Namespace) -> int:
    env = load_environment()
    ensure_data_dirs(args.data_dir)
    checks: list[tuple[str, bool, str]] = []
    if args.source in {"all", "nyt"} and env["nyt_api_key"]:
        client = CachedApiClient(
            source="nyt", api_key=str(env["nyt_api_key"]), data_dir=args.data_dir,
            min_interval=float(str(env["nyt_interval"])), max_calls=3, daily_limit=1900,
            max_retries=0,
        )
        for label, url, params in (
            ("NYT Archive", NYT_ARCHIVE_URL.format(year=2010, month=1), {}),
            ("NYT Article Search", NYT_SEARCH_URL, {
                "q": "inflation", "begin_date": "20100101", "end_date": "20100101", "page": 0,
            }),
        ):
            try:
                client.get(
                    url=url, params=params, bucket="probe",
                    manifest={
                        "source": "nyt", "endpoint": "probe", "start": "2010-01-01",
                        "end": "2010-01-01", "group": label, "page": 0,
                    }, force_refresh=True,
                )
                checks.append((label, True, "ok"))
            except CollectionError as exc:
                checks.append((label, False, str(exc)))
    elif args.source in {"all", "nyt"}:
        checks.append(("NYT", False, "NYT_API_KEY missing"))

    if args.source in {"all", "guardian"} and env["guardian_api_key"]:
        client = CachedApiClient(
            source="guardian", api_key=str(env["guardian_api_key"]), data_dir=args.data_dir,
            min_interval=float(str(env["guardian_interval"])), max_calls=1, daily_limit=480,
            max_retries=0,
        )
        try:
            client.get(
                url=GUARDIAN_SEARCH_URL,
                params={
                    "q": "inflation", "from-date": "2010-01-01", "to-date": "2010-01-01",
                    "page-size": 1, "show-fields": "headline,bodyText", "page": 1,
                }, force_refresh=True,
                bucket="probe",
                manifest={
                    "source": "guardian", "endpoint": "probe", "start": "2010-01-01",
                    "end": "2010-01-01", "group": "Guardian Search", "page": 1,
                },
            )
            checks.append(("Guardian Search", True, "ok"))
        except CollectionError as exc:
            checks.append(("Guardian Search", False, str(exc)))
    elif args.source in {"all", "guardian"}:
        checks.append(("Guardian", False, "GUARDIAN_API_KEY missing"))

    for label, ok, detail in checks:
        print(f"{label}: {'OK' if ok else 'FAILED'} - {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def collect(args: argparse.Namespace) -> int:
    config = load_keyword_config(args.config)
    env = load_environment()
    ensure_data_dirs(args.data_dir)
    end = args.end or date.today()
    start = args.start
    if start > end:
        raise SystemExit("--start must not be after --end")
    intervals = pilot_intervals(end) if args.command == "pilot" else list(month_intervals(start, end))

    needs_nyt = args.source in {"all", "nyt"}
    needs_guardian = args.source in {"all", "guardian"}
    missing = []
    if needs_nyt and not env["nyt_api_key"]:
        missing.append("NYT_API_KEY")
    if needs_guardian and not env["guardian_api_key"]:
        missing.append("GUARDIAN_API_KEY")
    if missing:
        print(
            "Missing required environment variable(s): " + ", ".join(missing)
            + ". Add them to the project .env file; no API request was made.",
            file=sys.stderr,
        )
        return 2

    _write_run_metadata(
        data_dir=args.data_dir, command=args.command, start=start, end=end,
        source=args.source, nyt_mode=args.nyt_mode, config_path=args.config,
    )
    _log(
        args.data_dir, "run_started", command=args.command, source=args.source,
        start=start.isoformat(), end=end.isoformat(), intervals=len(intervals),
        config_version=config.version,
    )
    new_records: list[dict[str, Any]] = []
    deferred = False
    errors: list[str] = []
    if needs_nyt:
        try:
            nyt_client = CachedApiClient(
                source="nyt", api_key=str(env["nyt_api_key"]), data_dir=args.data_dir,
                min_interval=float(str(env["nyt_interval"])), max_calls=args.nyt_max_calls,
                daily_limit=args.nyt_daily_limit,
            )
            if args.nyt_mode in {"archive", "archive+search"}:
                collect_nyt_archive(
                    client=nyt_client, intervals=intervals, config=config,
                    on_records=new_records.extend,
                )
            if args.nyt_mode in {"search", "archive+search"}:
                collect_nyt_search(
                    client=nyt_client, intervals=intervals, config=config,
                    page_cap=args.nyt_page_cap, on_records=new_records.extend,
                )
        except CallBudgetExceeded as exc:
            deferred = True
            _log(args.data_dir, "source_deferred", source="nyt", reason=str(exc), records_in_memory=len(new_records))
            print(str(exc), file=sys.stderr)
        except CollectionError as exc:
            errors.append(str(exc))
            _log(args.data_dir, "source_failed", source="nyt", reason=str(exc), records_in_memory=len(new_records))
            print(str(exc), file=sys.stderr)

    if needs_guardian:
        try:
            guardian_client = CachedApiClient(
                source="guardian", api_key=str(env["guardian_api_key"]), data_dir=args.data_dir,
                min_interval=float(str(env["guardian_interval"])), max_calls=args.guardian_max_calls,
                daily_limit=args.guardian_daily_limit,
            )
            collect_guardian(
                client=guardian_client, intervals=intervals, config=config,
                page_size=args.guardian_page_size, page_cap=args.guardian_page_cap,
                on_records=new_records.extend,
            )
        except CallBudgetExceeded as exc:
            deferred = True
            _log(args.data_dir, "source_deferred", source="guardian", reason=str(exc), records_in_memory=len(new_records))
            print(str(exc), file=sys.stderr)
        except CollectionError as exc:
            errors.append(str(exc))
            _log(args.data_dir, "source_failed", source="guardian", reason=str(exc), records_in_memory=len(new_records))
            print(str(exc), file=sys.stderr)

    merged = _persist(new_records, args.data_dir, config.version)
    _log(
        args.data_dir, "run_failed" if errors else ("run_finished" if not deferred else "run_checkpointed"),
        new_records=len(new_records), total_unique_records=len(merged), deferred=deferred, errors=errors,
    )
    print(f"New normalized records: {len(new_records)}")
    print(f"Total unique cleaned records: {len(merged)}")
    print(f"Clean data: {args.data_dir / 'clean' / 'news_articles.jsonl'}")
    print(f"Reports: {args.data_dir / 'reports'}")
    if errors:
        return 1
    return 3 if deferred else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect reproducible U.S.-economy news from NYT and Guardian APIs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Validate configuration and report whether API keys are present.")
    subparsers.add_parser("report", help="Regenerate cleaned exports and reports from existing data.")
    subparsers.add_parser("scrub-secrets", help="Remove configured API keys from generated text artifacts.")
    probe_parser = subparsers.add_parser("probe", help="Make minimal requests to configured API endpoints.")
    probe_parser.add_argument("--source", choices=["nyt"], default="nyt")
    purge_parser = subparsers.add_parser("purge-source", help="Delete one source's raw data, cleaned articles, and state entries.")
    purge_parser.add_argument("source", choices=["nyt", "guardian"])

    for name in ("pilot", "collect"):
        command = subparsers.add_parser(name)
        command.add_argument("--source", choices=["nyt"], default="nyt")
        command.add_argument("--start", type=_date, default=date(2010, 1, 1))
        command.add_argument("--end", type=_date, default=None)
        command.add_argument(
            "--nyt-mode", choices=["archive", "search", "archive+search"],
            default="archive+search" if name == "pilot" else "archive",
        )
        command.add_argument("--nyt-max-calls", type=int, default=450 if name == "pilot" else 1900)
        command.add_argument("--guardian-max-calls", type=int, default=450)
        command.add_argument("--nyt-daily-limit", type=int, default=1900)
        command.add_argument("--guardian-daily-limit", type=int, default=480)
        command.add_argument("--nyt-page-cap", type=int, default=100)
        command.add_argument("--guardian-page-size", type=int, default=200)
        command.add_argument("--guardian-page-cap", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        return validate(args)
    if args.command == "report":
        return report(args)
    if args.command == "scrub-secrets":
        return scrub(args)
    if args.command == "probe":
        return probe(args)
    if args.command == "purge-source":
        return purge_source(args)
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
