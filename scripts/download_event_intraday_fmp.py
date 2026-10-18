#!/usr/bin/env python
"""Download FMP 1-minute prices around identifiable macro-news events.

For each event date, the output contains the regular US trading sessions T-1,
T, and T+1. Events whose equity-market direction is labelled ``无法判断`` are
skipped. The API key is read from ``FMP_API`` in the environment or ``.env``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import certifi
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "outputs/macro_event_identification/macro_event_surprises_by_release_date.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/macro_event_identification"
FMP_URL = "https://financialmodelingprep.com/stable/historical-chart/1min"
ET = "America/New_York"


def load_env_value(name: str, env_file: Path) -> str | None:
    """Read one value without echoing or otherwise exposing secrets."""
    value = os.environ.get(name)
    if value:
        return value.strip()
    if not env_file.exists():
        return None
    for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, candidate = line.split("=", 1)
        if key.strip() == name:
            return candidate.strip().strip('"').strip("'")
    return None


def parse_api_payload(response: requests.Response) -> list[dict[str, Any]]:
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        message = (
            payload.get("Error Message")
            or payload.get("error")
            or payload.get("message")
            or json.dumps(payload, ensure_ascii=False)[:500]
        )
        raise RuntimeError(f"FMP returned an error: {message}")
    raise RuntimeError(f"Unexpected FMP response type: {type(payload).__name__}")


def fetch_range(
    session: requests.Session,
    api_key: str,
    symbol: str,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
    extended: bool,
) -> list[dict[str, Any]]:
    mode = "extended" if extended else "regular"
    cache_path = cache_dir / f"{symbol}_{mode}_{date_from:%Y-%m-%d}_{date_to:%Y-%m-%d}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    params = {
        "symbol": symbol,
        "from": date_from.strftime("%Y-%m-%d"),
        "to": date_to.strftime("%Y-%m-%d"),
        "extended": str(extended).lower(),
        "apikey": api_key,
    }
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(
                FMP_URL,
                params=params,
                timeout=60,
                verify=certifi.where(),
            )
            rows = parse_api_payload(response)
            cache_path.write_text(
                json.dumps(rows, ensure_ascii=False), encoding="utf-8"
            )
            return rows
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == 4:
                break
            time.sleep(2**attempt)
    raise RuntimeError(
        f"Failed to download {symbol} from {date_from:%Y-%m-%d} "
        f"to {date_to:%Y-%m-%d}: {last_error}"
    )


def prepare_prices(rows: list[dict[str, Any]], extended: bool) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    prices = pd.DataFrame(rows)
    if "date" not in prices.columns:
        raise RuntimeError("FMP response is missing the 'date' field")

    # FMP documents intraday timestamps in the exchange's local timezone.
    prices["datetime_et"] = pd.to_datetime(prices["date"], errors="coerce")
    if prices["datetime_et"].dt.tz is None:
        prices["datetime_et"] = prices["datetime_et"].dt.tz_localize(
            ET, ambiguous="NaT", nonexistent="shift_forward"
        )
    else:
        prices["datetime_et"] = prices["datetime_et"].dt.tz_convert(ET)

    for column in ["open", "high", "low", "close", "volume"]:
        if column in prices.columns:
            prices[column] = pd.to_numeric(prices[column], errors="coerce")
        else:
            prices[column] = np.nan

    prices = prices.dropna(subset=["datetime_et", "close"]).copy()
    local_time = prices["datetime_et"].dt.time
    market_open = pd.Timestamp("04:00" if extended else "09:30").time()
    market_close = pd.Timestamp("20:00" if extended else "16:00").time()
    prices = prices.loc[(local_time >= market_open) & (local_time <= market_close)].copy()
    prices["trading_date"] = prices["datetime_et"].dt.date
    return prices.sort_values("datetime_et").drop_duplicates("datetime_et", keep="last")


def fetch_event_prices(
    session: requests.Session,
    api_key: str,
    symbol: str,
    event_date: pd.Timestamp,
    cache_dir: Path,
    refresh: bool,
    extended: bool,
) -> pd.DataFrame:
    """Download exactly T-1, T and T+1, scanning past closed-market dates."""

    def one_day(day: pd.Timestamp) -> pd.DataFrame:
        rows = fetch_range(
            session, api_key, symbol, day, day, cache_dir, refresh, extended
        )
        return prepare_prices(rows, extended)

    event_prices = one_day(event_date)

    previous_prices = pd.DataFrame()
    previous_day = event_date - pd.Timedelta(days=1)
    for _ in range(7):
        previous_prices = one_day(previous_day)
        if not previous_prices.empty:
            break
        previous_day -= pd.Timedelta(days=1)

    next_prices = pd.DataFrame()
    next_day = event_date + pd.Timedelta(days=1)
    for _ in range(7):
        next_prices = one_day(next_day)
        if not next_prices.empty:
            break
        next_day += pd.Timedelta(days=1)

    if previous_prices.empty or next_prices.empty:
        return pd.DataFrame()
    frames = [previous_prices]
    if not event_prices.empty:
        frames.append(event_prices)
    frames.append(next_prices)
    return pd.concat(frames, ignore_index=True).sort_values("datetime_et").drop_duplicates(
        "datetime_et", keep="last"
    )


def parse_new_york_timestamp(value: Any) -> pd.Timestamp:
    """Interpret naive project timestamps as New York local time, not UTC."""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(ET, ambiguous="raise", nonexistent="raise")
    return timestamp.tz_convert(ET)


def event_window(
    event: pd.Series, prices: pd.DataFrame, symbol: str
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    release_date = pd.Timestamp(event["release_date"]).date()
    status: dict[str, Any] = {
        "release_date": str(release_date),
        "event_ids": event.get("event_ids"),
        "event_types": event.get("event_types"),
        "equity_label": event.get("equity_label"),
        "symbol": symbol,
        "status": "ok",
        "message": "",
        "t_minus_1_date": None,
        "event_trading_date": None,
        "t_plus_1_date": None,
        "minute_rows": 0,
    }
    if prices.empty:
        status.update(status="skipped", message="No minute bars returned")
        return None, status

    sessions = sorted(prices["trading_date"].dropna().unique())
    before = [session_date for session_date in sessions if session_date < release_date]
    after = [session_date for session_date in sessions if session_date > release_date]
    if not before or not after:
        status.update(status="skipped", message="Adjacent trading session unavailable")
        return None, status

    previous_session = before[-1]
    next_session = after[0]
    event_is_trading_day = release_date in sessions
    if event_is_trading_day:
        chosen = [previous_session, release_date, next_session]
        role_map = {chosen[0]: "T-1", chosen[1]: "T", chosen[2]: "T+1"}
    else:
        chosen = [previous_session, next_session]
        role_map = {chosen[0]: "T-1", chosen[1]: "T+1"}
    window = prices.loc[prices["trading_date"].isin(chosen)].copy()
    window["session_role"] = window["trading_date"].map(role_map)
    role_order = pd.CategoricalDtype(["T-1", "T", "T+1"], ordered=True)
    window["session_role"] = window["session_role"].astype(role_order)
    window = window.sort_values(["session_role", "datetime_et"]).reset_index(drop=True)

    official_dt = parse_new_york_timestamp(event.get("official_release_datetime_et"))
    news_dt = parse_new_york_timestamp(event.get("earliest_news_datetime_et"))

    metadata_columns = [
        "release_date",
        "event_ids",
        "event_types",
        "official_release_datetime_et",
        "earliest_news_datetime_et",
        "earliest_news_lag_minutes",
        "event_surprise_mean",
        "event_surprise_pre_mean",
        "event_surprise_unit",
        "equity_direction_score",
        "equity_label",
        "equity_direction_pre_score",
        "equity_label_pre",
        "equity_label_basis",
    ]
    for column in metadata_columns:
        window[column] = event.get(column)
    window["official_release_datetime_et"] = official_dt
    window["earliest_news_datetime_et"] = news_dt
    window["symbol"] = symbol
    if not pd.isna(news_dt):
        news_minute = news_dt.floor("min")
        window["news_minute_et"] = news_minute
        window["minutes_from_earliest_news_exact"] = (
            window["datetime_et"] - news_dt
        ).dt.total_seconds() / 60
        window["relative_event_minute"] = (
            (window["datetime_et"] - news_minute).dt.total_seconds() / 60
        ).astype(int)
        window["after_earliest_news"] = window["datetime_et"] >= news_dt
        window["is_news_release_minute"] = window["datetime_et"].eq(news_minute)
    else:
        window["news_minute_et"] = pd.NaT
        window["minutes_from_earliest_news_exact"] = np.nan
        window["relative_event_minute"] = pd.NA
        window["after_earliest_news"] = pd.NA
        window["is_news_release_minute"] = False

    # Backward-compatible alias, now correctly measured from exact New York time.
    window["minutes_from_earliest_news"] = window["minutes_from_earliest_news_exact"]

    previous_close = window.groupby("trading_date", observed=True)["close"].shift(1)
    minute_reference_price = previous_close.fillna(window["open"])
    window["price_change"] = window["close"] - minute_reference_price
    window["minute_return"] = window["close"] / minute_reference_price - 1
    window["minute_return_pct"] = 100 * window["minute_return"]
    window["log_return"] = np.log(window["close"] / minute_reference_price)
    session_open = window.groupby("trading_date", observed=True)["open"].transform("first")
    window["session_return_from_open_pct"] = 100 * (window["close"] / session_open - 1)
    pre_event_close = window.loc[window["session_role"] == "T-1", "close"].iloc[-1]
    window["return_from_pre_event_close_pct"] = 100 * (
        window["close"] / pre_event_close - 1
    )
    if not pd.isna(news_dt):
        pre_news_rows = window.loc[window["datetime_et"] < news_dt.floor("min")]
        if not pre_news_rows.empty:
            pre_news_reference = pre_news_rows.iloc[-1]
            pre_news_price = float(pre_news_reference["close"])
            window["pre_news_reference_datetime_et"] = pre_news_reference["datetime_et"]
            window["pre_news_reference_price"] = pre_news_price
            window["return_from_pre_news_price_pct"] = 100 * (
                window["close"] / pre_news_price - 1
            )
        else:
            window["pre_news_reference_datetime_et"] = pd.NaT
            window["pre_news_reference_price"] = np.nan
            window["return_from_pre_news_price_pct"] = np.nan
    else:
        window["pre_news_reference_datetime_et"] = pd.NaT
        window["pre_news_reference_price"] = np.nan
        window["return_from_pre_news_price_pct"] = np.nan

    status.update(
        t_minus_1_date=str(chosen[0]),
        event_trading_date=str(release_date) if event_is_trading_day else None,
        t_plus_1_date=str(next_session),
        event_is_trading_day=event_is_trading_day,
        news_minute_available=bool(window["is_news_release_minute"].any()),
        earliest_news_datetime_et=str(news_dt),
        news_minute_et=str(news_dt.floor("min")) if not pd.isna(news_dt) else None,
        minute_rows=len(window),
    )
    return window, status


def clean_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in [
        "datetime_et",
        "official_release_datetime_et",
        "earliest_news_datetime_et",
        "news_minute_et",
        "pre_news_reference_datetime_et",
    ]:
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="QQQ", help="US equity ticker (default: QQQ)")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh", action="store_true", help="Ignore cached FMP responses")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include 04:00-20:00 ET pre-market and after-hours bars",
    )
    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    api_key = load_env_value("FMP_API", ROOT / ".env")
    if not api_key:
        raise RuntimeError("FMP_API is missing from the environment and .env")

    events = pd.read_csv(args.events)
    required = {"release_date", "equity_label"}
    missing = required.difference(events.columns)
    if missing:
        raise RuntimeError(f"Event file is missing columns: {sorted(missing)}")
    events["release_date"] = pd.to_datetime(events["release_date"], errors="coerce")
    invalid_labels = {"无法判断", "鏃犳硶鍒ゆ柇"}
    usable = (
        events["release_date"].notna()
        & events["equity_label"].notna()
        & ~events["equity_label"].isin(invalid_labels)
    )
    if "earliest_news_datetime_et" in events.columns:
        usable &= events["earliest_news_datetime_et"].notna()
    events = events.loc[usable].sort_values("release_date")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "fmp_1min_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    http = requests.Session()
    http.headers.update({"User-Agent": "NLP-Final-Assignment/1.0"})

    windows: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    total = len(events)
    for number, (_, event) in enumerate(events.iterrows(), start=1):
        event_date = pd.Timestamp(event["release_date"])
        print(f"[{number:02d}/{total:02d}] {event_date:%Y-%m-%d} {symbol}", flush=True)
        try:
            prices = fetch_event_prices(
                http, api_key, symbol, event_date, cache_dir, args.refresh, args.extended
            )
            window, status = event_window(event, prices, symbol)
            if window is not None:
                windows.append(window)
            statuses.append(status)
        except Exception as exc:  # preserve progress and give a per-event audit trail
            statuses.append(
                {
                    "release_date": event_date.strftime("%Y-%m-%d"),
                    "event_ids": event.get("event_ids"),
                    "event_types": event.get("event_types"),
                    "equity_label": event.get("equity_label"),
                    "symbol": symbol,
                    "status": "error",
                    "message": str(exc),
                    "minute_rows": 0,
                }
            )
            print(f"  ERROR: {exc}", flush=True)

    status_df = pd.DataFrame(statuses)
    suffix = "_extended" if args.extended else ""
    summary_path = args.output_dir / f"{symbol}_event_intraday_summary{suffix}.csv"
    status_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if not windows:
        raise RuntimeError(
            f"No usable minute windows were downloaded. See {summary_path} for details."
        )

    minute_df = pd.concat(windows, ignore_index=True)
    minute_df = minute_df.drop_duplicates(
        subset=["release_date", "symbol", "session_role", "datetime_et"], keep="last"
    )
    leading = [
        "release_date",
        "event_ids",
        "event_types",
        "official_release_datetime_et",
        "earliest_news_datetime_et",
        "earliest_news_lag_minutes",
        "equity_label",
        "equity_direction_score",
        "equity_label_pre",
        "equity_direction_pre_score",
        "event_surprise_mean",
        "event_surprise_pre_mean",
        "event_surprise_unit",
        "symbol",
        "session_role",
        "trading_date",
        "datetime_et",
        "news_minute_et",
        "relative_event_minute",
        "minutes_from_earliest_news_exact",
        "minutes_from_earliest_news",
        "after_earliest_news",
        "is_news_release_minute",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "price_change",
        "minute_return",
        "minute_return_pct",
        "log_return",
        "session_return_from_open_pct",
        "return_from_pre_event_close_pct",
        "pre_news_reference_datetime_et",
        "pre_news_reference_price",
        "return_from_pre_news_price_pct",
    ]
    ordered = [column for column in leading if column in minute_df.columns]
    ordered += [column for column in minute_df.columns if column not in ordered and column != "date"]
    minute_df = minute_df[ordered].sort_values(
        ["release_date", "session_role", "datetime_et"]
    )
    output_path = args.output_dir / f"{symbol}_event_intraday_1min{suffix}.csv"
    clean_for_csv(minute_df).to_csv(output_path, index=False, encoding="utf-8-sig")

    ok_count = int((status_df["status"] == "ok").sum())
    error_count = int((status_df["status"] == "error").sum())
    skipped_count = int((status_df["status"] == "skipped").sum())
    print(
        f"Saved {len(minute_df):,} minute rows for {ok_count}/{total} events "
        f"({skipped_count} skipped, {error_count} errors)."
    )
    print(output_path)
    print(summary_path)


if __name__ == "__main__":
    main()
