#!/usr/bin/env python
"""Exploratory integer-threshold IV study for media tone and consumer sentiment.

The specification is frozen in outputs/threshold_iv_analysis/design_spec.md. This script
retains all prespecified results and never selects thresholds using second-stage p-values.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


SEED = 42
PRIMARY_MEDIA = "news_tone_index_deepseek"
SECONDARY_MEDIA = "news_tone_index_t5"
OUTCOMES = [
    "michigan_sentiment",
    "michigan_current_conditions",
    "michigan_expectations",
]
BASE_CONTROLS = [
    "u3_release_value",
    "d_u3_release_value",
    "pce_release_value",
    "d_pce_release_value",
    "gdp_latest_release_value",
    "time_trend",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path(
            "outputs/monthly_tone_index/"
            "Consumer_Sentiment_Economic_Data_2010_2026_with_tone_indices.xlsx"
        ),
    )
    parser.add_argument(
        "--articles",
        type=Path,
        default=Path("outputs/all_relevant/delivery/all_relevant_annotations.csv"),
    )
    parser.add_argument(
        "--release-data",
        type=Path,
        default=Path("data/raw/official_release_snapshot_iv.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/threshold_iv_analysis")
    )
    return parser.parse_args()


def round_half_up(value: float) -> float:
    if pd.isna(value):
        return math.nan
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def bh_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if not len(pv):
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    out[np.where(valid)[0]] = restored
    return out


def hac_meat(moment_rows: np.ndarray, lag: int) -> np.ndarray:
    n = len(moment_rows)
    centered = moment_rows - moment_rows.mean(axis=0, keepdims=True)
    meat = centered.T @ centered / n
    for ell in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - ell / (lag + 1.0)
        gamma = centered[ell:].T @ centered[:-ell] / n
        meat += weight * (gamma + gamma.T)
    return meat


@dataclass
class OLSResult:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    pvalue: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    residual: np.ndarray
    fitted: np.ndarray
    covariance: np.ndarray
    index: pd.Index
    nobs: int
    r2: float


def fit_ols(frame: pd.DataFrame, y_name: str, x_names: Sequence[str], lag: int = 3) -> OLSResult:
    columns = [y_name, *x_names]
    data = frame.dropna(subset=columns).copy()
    y = data[y_name].to_numpy(float)
    x = np.column_stack([np.ones(len(data)), data[list(x_names)].to_numpy(float)])
    names = ["const", *x_names]
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    residual = y - fitted
    n = len(y)
    moment = x * residual[:, None]
    meat = hac_meat(moment, lag) * n
    covariance = xtx_inv @ meat @ xtx_inv * n / max(n - x.shape[1], 1)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvalue = 2 * stats.t.sf(np.abs(stat), max(n - x.shape[1], 1))
    crit = stats.t.ppf(0.975, max(n - x.shape[1], 1))
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(residual**2) / tss if tss > 0 else math.nan
    return OLSResult(
        names, beta, se, pvalue, beta - crit * se, beta + crit * se,
        residual, fitted, covariance, data.index, n, r2,
    )


def ols_joint_test(result: OLSResult, terms: Sequence[str]) -> tuple[float, int, float]:
    positions = [result.names.index(term) for term in terms]
    r = np.zeros((len(positions), len(result.names)))
    for row, position in enumerate(positions):
        r[row, position] = 1.0
    value = r @ result.beta
    covariance = r @ result.covariance @ r.T
    wald = float(value.T @ np.linalg.pinv(covariance) @ value)
    return wald, len(positions), float(stats.chi2.sf(wald, len(positions)))


@dataclass
class IVResult:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    pvalue: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    covariance: np.ndarray
    residual: np.ndarray
    index: pd.Index
    nobs: int
    j_stat: float
    j_df: int
    j_pvalue: float


def fit_iv(
    frame: pd.DataFrame,
    y_name: str,
    media_name: str,
    controls: Sequence[str],
    instruments: Sequence[str],
    lag: int = 3,
) -> IVResult:
    columns = [y_name, media_name, *controls, *instruments]
    data = frame.dropna(subset=columns).copy()
    y = data[y_name].to_numpy(float)
    d = np.column_stack(
        [np.ones(len(data)), data[[media_name, *controls]].to_numpy(float)]
    )
    h = np.column_stack(
        [np.ones(len(data)), data[[*instruments, *controls]].to_numpy(float)]
    )
    names = ["const", media_name, *controls]
    n = len(y)
    q_dh = d.T @ h / n
    w = np.linalg.pinv(h.T @ h / n)
    bread = q_dh @ w @ q_dh.T
    bread_inv = np.linalg.pinv(bread)
    beta = bread_inv @ q_dh @ w @ (h.T @ y / n)
    residual = y - d @ beta
    s = hac_meat(h * residual[:, None], lag)
    covariance = bread_inv @ q_dh @ w @ s @ w @ q_dh.T @ bread_inv / n
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    dof = max(n - d.shape[1], 1)
    pvalue = 2 * stats.t.sf(np.abs(stat), dof)
    crit = stats.t.ppf(0.975, dof)

    x = np.column_stack([np.ones(n), data[list(controls)].to_numpy(float)])
    z = data[list(instruments)].to_numpy(float)
    z_resid = z - x @ (np.linalg.pinv(x.T @ x) @ x.T @ z)
    gbar = (z_resid * residual[:, None]).mean(axis=0)
    sz = hac_meat(z_resid * residual[:, None], lag)
    j_stat = float(n * gbar.T @ np.linalg.pinv(sz) @ gbar)
    j_df = max(len(instruments) - 1, 0)
    j_pvalue = float(stats.chi2.sf(j_stat, j_df)) if j_df else math.nan
    return IVResult(
        names, beta, se, pvalue, beta - crit * se, beta + crit * se,
        covariance, residual, data.index, n, j_stat, j_df, j_pvalue,
    )


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    return values - controls @ (np.linalg.pinv(controls.T @ controls) @ controls.T @ values)


def first_stage_diagnostics(
    frame: pd.DataFrame,
    media: str,
    controls: Sequence[str],
    instruments: Sequence[str],
    lag: int,
) -> tuple[OLSResult, dict]:
    required = [media, *controls, *instruments]
    data = frame.dropna(subset=required).copy()
    unrestricted = fit_ols(data, media, [*controls, *instruments], lag)
    wald, df, pvalue = ols_joint_test(unrestricted, instruments)
    y = data[media].to_numpy(float)
    x = np.column_stack([np.ones(len(data)), data[list(controls)].to_numpy(float)])
    z = data[list(instruments)].to_numpy(float)
    y_resid = residualize(y, x)
    z_resid = residualize(z, x)
    fitted = z_resid @ (np.linalg.pinv(z_resid.T @ z_resid) @ z_resid.T @ y_resid)
    denominator = float(y_resid.T @ y_resid)
    partial_r2 = float(fitted.T @ fitted / denominator) if denominator > 0 else math.nan
    return unrestricted, {
        "joint_hac_wald": wald,
        "joint_df": df,
        "joint_p_value": pvalue,
        "wald_f_type": wald / df,
        "partial_r_squared": partial_r2,
        "nobs": len(data),
    }


def ar_test(
    frame: pd.DataFrame,
    y_name: str,
    media: str,
    controls: Sequence[str],
    instruments: Sequence[str],
    beta0: float,
    lag: int,
) -> tuple[float, int, float]:
    required = [y_name, media, *controls, *instruments]
    data = frame.dropna(subset=required).copy()
    temp = data.copy()
    temp["ar_adjusted_outcome"] = temp[y_name] - beta0 * temp[media]
    result = fit_ols(temp, "ar_adjusted_outcome", [*controls, *instruments], lag)
    return ols_joint_test(result, instruments)


def ar_confidence_grid(
    frame: pd.DataFrame,
    y_name: str,
    media: str,
    controls: Sequence[str],
    instruments: Sequence[str],
    iv: IVResult,
    lag: int,
    points: int = 401,
) -> pd.DataFrame:
    media_position = iv.names.index(media)
    center = float(iv.beta[media_position])
    width = max(5.0, 12 * float(iv.se[media_position]))
    grid = np.linspace(center - width, center + width, points)
    rows = []
    for beta0 in grid:
        wald, df, pvalue = ar_test(frame, y_name, media, controls, instruments, beta0, lag)
        rows.append({"beta0": beta0, "ar_wald": wald, "ar_df": df, "ar_p_value": pvalue, "accepted_5pct": pvalue >= 0.05})
    return pd.DataFrame(rows)


def load_release_archive(path: Path) -> pd.DataFrame:
    columns = [
        "indicator", "reference_month", "release_date", "first_release_value",
        "source_url", "parse_status",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    data = pd.read_csv(path)
    for column in columns:
        if column not in data:
            data[column] = np.nan
    data["reference_month"] = pd.to_datetime(data["reference_month"], errors="coerce")
    data["release_date"] = pd.to_datetime(data["release_date"], errors="coerce")
    data["first_release_value"] = pd.to_numeric(data["first_release_value"], errors="coerce")
    return data[columns]


def load_articles(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "article_id", "published_at", "headline", "abstract", "duplicate_group_id",
        "t5_primary_topic", "t5_tone", "t5_format_valid", "deepseek_relevance",
        "deepseek_primary_topic", "deepseek_tone",
    ]
    articles = pd.read_csv(path, usecols=columns, low_memory=False)
    articles["published_at_parsed"] = pd.to_datetime(
        articles["published_at"], errors="coerce", utc=True, format="mixed"
    )
    articles = articles.dropna(subset=["published_at_parsed"]).copy()
    articles = articles.sort_values(["published_at_parsed", "article_id"])
    articles = articles.drop_duplicates("article_id", keep="first")
    articles["date"] = articles["published_at_parsed"].dt.tz_localize(None).dt.normalize()
    articles["month"] = articles["date"].dt.to_period("M").dt.to_timestamp()
    relevance = articles["deepseek_relevance"].fillna("").str.strip().str.casefold()
    eligible = articles.loc[~relevance.eq("irrelevant")].copy()
    eligible["deepseek_tone_valid"] = eligible["deepseek_tone"].isin([-1, 0, 1])
    eligible["t5_tone_valid"] = (
        eligible["t5_tone"].isin([-1, 0, 1])
        & eligible["t5_format_valid"].fillna(False).astype(bool)
    )

    monthly_rows: list[dict] = []
    for month, part in eligible.groupby("month", sort=True):
        row: dict[str, object] = {
            "month": month,
            "article_count": len(part),
        }
        for prefix, tone, valid, topic in [
            ("deepseek", "deepseek_tone", "deepseek_tone_valid", "deepseek_primary_topic"),
            ("t5", "t5_tone", "t5_tone_valid", "t5_primary_topic"),
        ]:
            valid_part = part.loc[part[valid]]
            row[f"{prefix}_valid_tone_count"] = len(valid_part)
            row[f"{prefix}_tone_index_article_rebuild"] = (
                100 * valid_part[tone].astype(float).mean() if len(valid_part) else math.nan
            )
            for topic_name, short in [
                ("labor_market", "labor"),
                ("cost_of_living_income", "cost_living"),
                ("macro_growth", "macro_growth"),
            ]:
                topic_part = part.loc[part[topic] == topic_name]
                topic_valid = topic_part.loc[topic_part[valid]]
                row[f"{prefix}_{short}_article_count"] = len(topic_part)
                row[f"{prefix}_{short}_tone_index"] = (
                    100 * topic_valid[tone].astype(float).mean() if len(topic_valid) else math.nan
                )
        monthly_rows.append(row)
    return eligible, pd.DataFrame(monthly_rows)


def build_release_series(
    monthly: pd.DataFrame,
    archive: pd.DataFrame,
    indicator: str,
) -> pd.DataFrame:
    source = monthly[["month", indicator]].rename(
        columns={"month": "reference_month", indicator: "latest_revised_value"}
    )
    source["reference_month"] = pd.to_datetime(source["reference_month"])
    archived = archive.loc[archive["indicator"] == indicator].copy()
    source = source.merge(archived, on="reference_month", how="left")
    source["public_value"] = source["first_release_value"].combine_first(
        source["latest_revised_value"]
    )
    source["value_source"] = np.where(
        source["first_release_value"].notna(), "official_first_release", "latest_revised_fallback"
    )
    source["public_value"] = source["public_value"].map(round_half_up)
    source["release_month"] = source["reference_month"] + pd.offsets.MonthBegin(1)
    source["release_date_source"] = np.where(
        source["release_date"].notna(), "official_archive", "release_month_only"
    )
    return source.sort_values("reference_month").reset_index(drop=True)


def threshold_crossings(
    series: pd.DataFrame,
    indicator: str,
    thresholds: Sequence[float],
    rarity_windows: Sequence[int | None] = (24, 12, None),
    label: str = "integer",
) -> pd.DataFrame:
    rows: list[dict] = []
    values = series["public_value"].to_numpy(float)
    for threshold in thresholds:
        threshold_events: list[int] = []
        for i in range(1, len(series)):
            previous, current = values[i - 1], values[i]
            if not np.isfinite(previous) or not np.isfinite(current):
                continue
            direction = None
            if previous < threshold <= current:
                direction = "up"
            elif previous >= threshold > current:
                direction = "down"
            if direction is None:
                continue
            months_since = i - threshold_events[-1] if threshold_events else math.nan
            threshold_events.append(i)
            for rarity in rarity_windows:
                complete_lookback = rarity is None or i >= rarity
                rare = rarity is None or (
                    complete_lookback
                    and (not np.isfinite(months_since) or months_since > rarity)
                )
                rows.append(
                    {
                        "indicator": indicator,
                        "threshold_family": label,
                        "threshold": threshold,
                        "rarity_months": "none" if rarity is None else rarity,
                        "reference_month": series.loc[i, "reference_month"],
                        "release_month": series.loc[i, "release_month"],
                        "release_date": series.loc[i, "release_date"],
                        "release_date_source": series.loc[i, "release_date_source"],
                        "previous_value": previous,
                        "current_value": current,
                        "change": current - previous,
                        "direction": direction,
                        "distance_to_threshold": current - threshold,
                        "months_since_prior_same_threshold": months_since,
                        "complete_lookback": complete_lookback,
                        "qualifies_rarity": rare,
                        "current_value_source": series.loc[i, "value_source"],
                        "previous_value_source": series.loc[i - 1, "value_source"],
                        "revision_misclassification_risk": (
                            series.loc[i, "value_source"] != "official_first_release"
                            or series.loc[i - 1, "value_source"] != "official_first_release"
                        ),
                        "source_url": series.loc[i, "source_url"],
                    }
                )
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    counts = (
        events.groupby(["indicator", "threshold_family", "rarity_months", "release_month"])
        ["threshold"].transform("count")
    )
    events["thresholds_crossed_same_release"] = counts
    return events.sort_values(
        ["threshold_family", "rarity_months", "release_month", "threshold"]
    ).reset_index(drop=True)


def add_event_instruments(
    analysis: pd.DataFrame,
    events: pd.DataFrame,
    prefix: str,
    rarity: int | None,
    favorable_direction: str = "down",
) -> pd.DataFrame:
    rarity_value = "none" if rarity is None else rarity
    chosen = events.loc[
        (events["rarity_months"].astype(str) == str(rarity_value))
        & events["qualifies_rarity"].astype(bool)
    ].copy()
    collapsed_rows = []
    for month, part in chosen.groupby("release_month"):
        directions = sorted(part["direction"].unique())
        direction = directions[0] if len(directions) == 1 else "mixed"
        signed = 1 if direction == favorable_direction else -1 if direction != "mixed" else 0
        collapsed_rows.append(
            {
                "month": pd.Timestamp(month),
                f"z_{prefix}_signed_r{rarity_value}": signed,
                f"z_{prefix}_any_r{rarity_value}": 1,
                f"z_{prefix}_up_r{rarity_value}": int(direction == "up"),
                f"z_{prefix}_down_r{rarity_value}": int(direction == "down"),
                f"{prefix}_crossed_thresholds_r{rarity_value}": ",".join(
                    str(x) for x in sorted(part["threshold"].unique())
                ),
            }
        )
    event_frame = pd.DataFrame(collapsed_rows)
    columns = [
        f"z_{prefix}_signed_r{rarity_value}",
        f"z_{prefix}_any_r{rarity_value}",
        f"z_{prefix}_up_r{rarity_value}",
        f"z_{prefix}_down_r{rarity_value}",
    ]
    if len(event_frame):
        analysis = analysis.merge(event_frame, on="month", how="left")
    for column in columns:
        if column not in analysis:
            analysis[column] = 0.0
        analysis[column] = analysis[column].fillna(0.0)
    if rarity is not None:
        start = analysis["month"].min() + pd.DateOffset(months=rarity)
        analysis.loc[analysis["month"] < start, columns] = np.nan
    return analysis


def construct_data(
    workbook: Path, articles_path: Path, release_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    monthly = pd.read_excel(workbook, sheet_name="Monthly")
    quarterly = pd.read_excel(workbook, sheet_name="Quarterly GDP")
    monthly["month"] = pd.to_datetime(monthly["month"], utc=True).dt.tz_localize(None)
    monthly = monthly.sort_values("month").reset_index(drop=True)
    expected = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    if not monthly["month"].equals(pd.Series(expected)):
        raise ValueError("Monthly source is duplicated or not a complete calendar sequence.")
    archive = load_release_archive(release_path)
    articles, article_monthly = load_articles(articles_path)

    u3 = build_release_series(monthly, archive, "u3_unemployment_pct_sa")
    pce = build_release_series(monthly, archive, "pce_inflation_yoy_pct")
    u3_events = threshold_crossings(u3, "u3_unemployment_pct_sa", list(range(4, 11)))
    pce_events = threshold_crossings(pce, "pce_inflation_yoy_pct", list(range(3, 8)))
    pce2_events = threshold_crossings(pce, "pce_inflation_yoy_pct", [2], label="policy_2pct")
    u3_half = threshold_crossings(
        u3, "u3_unemployment_pct_sa", [x + 0.5 for x in range(4, 10)], label="half_integer"
    )
    pce_half = threshold_crossings(
        pce, "pce_inflation_yoy_pct", [x + 0.5 for x in range(2, 8)], label="half_integer"
    )
    all_events = pd.concat(
        [u3_events, pce_events, pce2_events, u3_half, pce_half], ignore_index=True
    )

    analysis = monthly.merge(article_monthly, on="month", how="left")
    u3_release = u3[["release_month", "public_value", "value_source"]].rename(
        columns={"release_month": "month", "public_value": "u3_release_value", "value_source": "u3_value_source"}
    )
    pce_release = pce[["release_month", "public_value", "value_source"]].rename(
        columns={"release_month": "month", "public_value": "pce_release_value", "value_source": "pce_value_source"}
    )
    analysis = analysis.merge(u3_release, on="month", how="left").merge(pce_release, on="month", how="left")
    analysis["d_u3_release_value"] = analysis["u3_release_value"].diff()
    analysis["d_pce_release_value"] = analysis["pce_release_value"].diff()

    quarterly["quarter_start"] = pd.to_datetime(
        quarterly["quarter_start"], utc=True
    ).dt.tz_localize(None)
    gdp = quarterly[["quarter_start", "real_gdp_qoq_annualized_official_pct"]].copy()
    gdp["release_month"] = gdp["quarter_start"] + pd.offsets.MonthBegin(3)
    gdp = gdp.rename(columns={"real_gdp_qoq_annualized_official_pct": "gdp_latest_release_value"})
    analysis = pd.merge_asof(
        analysis.sort_values("month"),
        gdp[["release_month", "gdp_latest_release_value"]].sort_values("release_month"),
        left_on="month", right_on="release_month", direction="backward",
    ).drop(columns="release_month")
    analysis["time_trend"] = np.arange(len(analysis), dtype=float)
    analysis["u3_release_value_sq"] = analysis["u3_release_value"] ** 2
    analysis["pce_release_value_sq"] = analysis["pce_release_value"] ** 2
    for month_number in range(2, 13):
        analysis[f"calendar_month_{month_number:02d}"] = (
            analysis["month"].dt.month == month_number
        ).astype(int)
    for outcome in OUTCOMES:
        analysis[f"{outcome}_lag1"] = analysis[outcome].shift(1)
        analysis[f"{outcome}_lead1"] = analysis[outcome].shift(-1)
        analysis[f"d_{outcome}"] = analysis[outcome].diff()
    analysis[f"{PRIMARY_MEDIA}_lag1"] = analysis[PRIMARY_MEDIA].shift(1)

    for rarity in (24, 12, None):
        analysis = add_event_instruments(analysis, u3_events, "u3", rarity)
        analysis = add_event_instruments(analysis, pce_events, "pce", rarity)
    analysis = add_event_instruments(analysis, pce2_events, "pce2", 24)
    analysis = add_event_instruments(analysis, u3_half, "u3half", 24)
    analysis = add_event_instruments(analysis, pce_half, "pcehalf", 24)

    gdp_events = []
    for i in range(1, len(gdp)):
        prior = gdp.iloc[i - 1]["gdp_latest_release_value"]
        current = gdp.iloc[i]["gdp_latest_release_value"]
        if (prior < 0 <= current) or (prior >= 0 > current):
            gdp_events.append(
                {
                    "month": gdp.iloc[i]["release_month"],
                    "z_gdp_zero_signed": 1 if current >= 0 else -1,
                    "gdp_prior": prior,
                    "gdp_current": current,
                }
            )
    analysis = analysis.merge(pd.DataFrame(gdp_events), on="month", how="left")
    analysis["z_gdp_zero_signed"] = analysis["z_gdp_zero_signed"].fillna(0)

    primary_events = all_events.loc[
        (all_events["threshold_family"] == "integer")
        & (all_events["rarity_months"].astype(str) == "24")
        & all_events["qualifies_rarity"].astype(bool)
    ].copy()
    quality = {
        "monthly_start": monthly["month"].min().strftime("%Y-%m"),
        "monthly_end": monthly["month"].max().strftime("%Y-%m"),
        "monthly_rows": len(monthly),
        "eligible_article_rows": len(articles),
        "article_date_start": articles["date"].min().strftime("%Y-%m-%d"),
        "article_date_end": articles["date"].max().strftime("%Y-%m-%d"),
        "official_u3_values": int((u3["value_source"] == "official_first_release").sum()),
        "official_pce_values": int((pce["value_source"] == "official_first_release").sum()),
        "u3_primary_event_months": int(primary_events.loc[primary_events["indicator"] == "u3_unemployment_pct_sa", "release_month"].nunique()),
        "pce_primary_event_months": int(primary_events.loc[primary_events["indicator"] == "pce_inflation_yoy_pct", "release_month"].nunique()),
        "main_media_definition": "100 times mean DeepSeek article score; -1 negative, 0 valid zero, +1 positive; larger is more positive",
        "vintage_limitation": "U-3 archive access was blocked to automated retrieval; PCE first-release coverage is partial. Latest revised fallbacks are flagged row by row.",
    }
    return analysis, all_events, primary_events, articles, quality


def event_window_mechanism(
    primary_events: pd.DataFrame, articles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_months = primary_events.drop_duplicates(["indicator", "release_month"]).copy()
    rows: list[dict] = []
    snippets: list[dict] = []
    phrase = re.compile(
        r"\b(?:above|below|hits?|cross(?:es|ed)?|falls?|fell|drops?|dropped|"
        r"rises?|rose|climbs?|first time|percent|percentage point)\b",
        flags=re.I,
    )
    topic_map = {
        "u3_unemployment_pct_sa": "labor_market",
        "pce_inflation_yoy_pct": "cost_of_living_income",
    }
    for event in event_months.itertuples():
        release_date = pd.Timestamp(event.release_date) if pd.notna(event.release_date) else pd.NaT
        if pd.isna(release_date):
            rows.append(
                {
                    "indicator": event.indicator,
                    "release_month": event.release_month,
                    "release_date": pd.NaT,
                    "window": "unavailable",
                    "n_articles": math.nan,
                    "n_topic_articles": math.nan,
                    "tone_index": math.nan,
                    "topic_tone_index": math.nan,
                    "threshold_phrase_articles": math.nan,
                    "timing_note": "Exact official release date unavailable",
                }
            )
            continue
        for label, start, end in [
            ("pre_7_days", release_date - pd.Timedelta(days=7), release_date - pd.Timedelta(days=1)),
            ("release_plus_7_days", release_date, release_date + pd.Timedelta(days=7)),
        ]:
            part = articles.loc[articles["date"].between(start, end)].copy()
            valid = part.loc[part["deepseek_tone_valid"]]
            topic = topic_map[event.indicator]
            topic_part = part.loc[part["deepseek_primary_topic"] == topic]
            topic_valid = topic_part.loc[topic_part["deepseek_tone_valid"]]
            combined = (part["headline"].fillna("") + " " + part["abstract"].fillna(""))
            phrase_mask = combined.str.contains(phrase, na=False)
            rows.append(
                {
                    "indicator": event.indicator,
                    "release_month": event.release_month,
                    "release_date": release_date,
                    "window": label,
                    "n_articles": len(part),
                    "n_topic_articles": len(topic_part),
                    "tone_index": 100 * valid["deepseek_tone"].mean() if len(valid) else math.nan,
                    "topic_tone_index": 100 * topic_valid["deepseek_tone"].mean() if len(topic_valid) else math.nan,
                    "threshold_phrase_articles": int(phrase_mask.sum()),
                    "timing_note": "Official date from archive",
                }
            )
            if label == "release_plus_7_days":
                for row in part.loc[phrase_mask].head(10).itertuples():
                    snippets.append(
                        {
                            "indicator": event.indicator,
                            "release_date": release_date,
                            "published_at": row.published_at,
                            "article_id": row.article_id,
                            "headline": row.headline,
                            "primary_topic": row.deepseek_primary_topic,
                            "tone": row.deepseek_tone,
                            "note": "Keyword fragment from headline/abstract; not verified full article body",
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(snippets)


def build_specs() -> list[dict]:
    month_fe = [f"calendar_month_{month:02d}" for month in range(2, 13)]
    primary_z = ["z_u3_signed_r24", "z_pce_signed_r24"]
    specs = [
        {
            "id": "PRIMARY_R24_HAC3",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": primary_z, "hac": 3,
            "family": "primary", "primary": True,
        },
        {
            "id": "PRIMARY_R24_HAC6",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": primary_z, "hac": 6,
            "family": "hac", "primary": False,
        },
        {
            "id": "PRIMARY_R24_HAC12",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": primary_z, "hac": 12,
            "family": "hac", "primary": False,
        },
        {
            "id": "LEAD1_OUTCOME",
            "outcome": f"{OUTCOMES[0]}_lead1", "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": primary_z, "hac": 3,
            "family": "timing", "primary": False,
        },
        {
            "id": "DYNAMIC_LAGGED_Y",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": [f"{OUTCOMES[0]}_lag1", *BASE_CONTROLS], "z": primary_z, "hac": 3,
            "family": "dynamic", "primary": False,
        },
        {
            "id": "QUADRATIC_CONTROLS",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": [*BASE_CONTROLS, "u3_release_value_sq", "pce_release_value_sq"],
            "z": primary_z, "hac": 3, "family": "functional_form", "primary": False,
        },
        {
            "id": "CALENDAR_MONTH_FE",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": [*BASE_CONTROLS, *month_fe], "z": primary_z, "hac": 3,
            "family": "seasonality", "primary": False,
        },
        {
            "id": "RARITY_12",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": ["z_u3_signed_r12", "z_pce_signed_r12"], "hac": 3,
            "family": "rarity", "primary": False,
        },
        {
            "id": "NO_RARITY",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": ["z_u3_signed_rnone", "z_pce_signed_rnone"], "hac": 3,
            "family": "rarity", "primary": False,
        },
        {
            "id": "ADD_PCE_2PCT",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": [*primary_z, "z_pce2_signed_r24"], "hac": 3,
            "family": "high_exclusion_risk", "primary": False,
        },
        {
            "id": "ALTERNATE_T5_MEDIA",
            "outcome": OUTCOMES[0], "media": SECONDARY_MEDIA,
            "controls": BASE_CONTROLS, "z": primary_z, "hac": 3,
            "family": "measurement", "primary": False,
        },
        {
            "id": "CURRENT_CONDITIONS",
            "outcome": OUTCOMES[1], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": primary_z, "hac": 3,
            "family": "secondary_outcome", "primary": False,
        },
        {
            "id": "EXPECTATIONS",
            "outcome": OUTCOMES[2], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": primary_z, "hac": 3,
            "family": "secondary_outcome", "primary": False,
        },
        {
            "id": "U3_ONLY",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": ["z_u3_signed_r24"], "hac": 3,
            "family": "instrument_subset", "primary": False,
        },
        {
            "id": "PCE_ONLY",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": ["z_pce_signed_r24"], "hac": 3,
            "family": "instrument_subset", "primary": False,
        },
        {
            "id": "HALF_INTEGER_PLACEBO",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": ["z_u3half_signed_r24", "z_pcehalf_signed_r24"], "hac": 3,
            "family": "placebo", "primary": False,
        },
        {
            "id": "GDP_ZERO_HIGH_RISK",
            "outcome": OUTCOMES[0], "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS, "z": ["z_gdp_zero_signed"], "hac": 3,
            "family": "high_exclusion_risk", "primary": False,
        },
    ]
    return specs


def safe_fit_spec(analysis: pd.DataFrame, spec: dict) -> dict:
    required = [spec["outcome"], spec["media"], *spec["controls"], *spec["z"]]
    data = analysis.dropna(subset=required).copy()
    if len(data) <= len(required) + 5:
        raise ValueError(f"Too few observations for {spec['id']}")
    if any(data[z].nunique() < 2 for z in spec["z"]):
        raise ValueError(f"No within-sample variation in at least one instrument for {spec['id']}")
    ols = fit_ols(data, spec["outcome"], [spec["media"], *spec["controls"]], spec["hac"])
    first, weak = first_stage_diagnostics(
        data, spec["media"], spec["controls"], spec["z"], spec["hac"]
    )
    reduced = fit_ols(data, spec["outcome"], [*spec["controls"], *spec["z"]], spec["hac"])
    reduced_joint = ols_joint_test(reduced, spec["z"])
    iv = fit_iv(data, spec["outcome"], spec["media"], spec["controls"], spec["z"], spec["hac"])
    ar0 = ar_test(data, spec["outcome"], spec["media"], spec["controls"], spec["z"], 0.0, spec["hac"])
    media_i_ols = ols.names.index(spec["media"])
    media_i_iv = iv.names.index(spec["media"])
    return {
        "data": data, "ols": ols, "first": first, "weak": weak,
        "reduced": reduced, "reduced_joint": reduced_joint, "iv": iv, "ar0": ar0,
        "summary": {
            "model_id": spec["id"], "family": spec["family"], "primary": spec["primary"],
            "outcome": spec["outcome"], "media": spec["media"],
            "instruments": " | ".join(spec["z"]), "controls": " | ".join(spec["controls"]),
            "hac_lag": spec["hac"], "nobs": len(data),
            "sample_start": data["month"].min().strftime("%Y-%m"),
            "sample_end": data["month"].max().strftime("%Y-%m"),
            "ols_media_estimate": ols.beta[media_i_ols],
            "ols_media_se": ols.se[media_i_ols], "ols_media_p_value": ols.pvalue[media_i_ols],
            "iv_media_estimate": iv.beta[media_i_iv],
            "iv_media_se": iv.se[media_i_iv], "iv_media_p_value": iv.pvalue[media_i_iv],
            "iv_ci_95_low": iv.ci_low[media_i_iv], "iv_ci_95_high": iv.ci_high[media_i_iv],
            "first_stage_joint_hac_wald": weak["joint_hac_wald"],
            "first_stage_joint_df": weak["joint_df"],
            "first_stage_joint_p_value": weak["joint_p_value"],
            "first_stage_wald_f_type": weak["wald_f_type"],
            "first_stage_partial_r_squared": weak["partial_r_squared"],
            "reduced_form_joint_wald": reduced_joint[0],
            "reduced_form_joint_p_value": reduced_joint[2],
            "anderson_rubin_beta0_wald": ar0[0],
            "anderson_rubin_beta0_p_value": ar0[2],
            "overidentification_j": iv.j_stat,
            "overidentification_df": iv.j_df,
            "overidentification_p_value": iv.j_pvalue,
            "u3_event_months": int((data.get("z_u3_any_r24", 0) == 1).sum()),
            "pce_event_months": int((data.get("z_pce_any_r24", 0) == 1).sum()),
        },
    }


def run_specs(analysis: pd.DataFrame, specs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    summaries: list[dict] = []
    first_rows: list[dict] = []
    reduced_rows: list[dict] = []
    fitted: dict[str, dict] = {}
    for spec in specs:
        try:
            result = safe_fit_spec(analysis, spec)
        except Exception as exc:
            summaries.append(
                {
                    "model_id": spec["id"], "family": spec["family"],
                    "primary": spec["primary"], "outcome": spec["outcome"],
                    "media": spec["media"], "instruments": " | ".join(spec["z"]),
                    "controls": " | ".join(spec["controls"]), "hac_lag": spec["hac"],
                    "status": f"not_estimated: {exc}",
                }
            )
            continue
        fitted[spec["id"]] = result
        summary = result["summary"]
        summary["status"] = "estimated"
        summaries.append(summary)
        first = result["first"]
        reduced = result["reduced"]
        for instrument in spec["z"]:
            i = first.names.index(instrument)
            first_rows.append(
                {
                    "model_id": spec["id"], "instrument": instrument,
                    "estimate": first.beta[i], "hac_std_error": first.se[i],
                    "p_value": first.pvalue[i], "ci_95_low": first.ci_low[i],
                    "ci_95_high": first.ci_high[i], "hac_lag": spec["hac"],
                    "nobs": first.nobs,
                }
            )
            j = reduced.names.index(instrument)
            reduced_rows.append(
                {
                    "model_id": spec["id"], "instrument": instrument,
                    "estimate": reduced.beta[j], "hac_std_error": reduced.se[j],
                    "p_value": reduced.pvalue[j], "ci_95_low": reduced.ci_low[j],
                    "ci_95_high": reduced.ci_high[j], "hac_lag": spec["hac"],
                    "nobs": reduced.nobs,
                }
            )
    summary_frame = pd.DataFrame(summaries)
    estimated = summary_frame["status"].eq("estimated")
    exploratory = estimated & ~summary_frame["primary"].fillna(False).astype(bool)
    summary_frame["iv_bh_q_value"] = np.nan
    summary_frame.loc[exploratory, "iv_bh_q_value"] = bh_adjust(
        summary_frame.loc[exploratory, "iv_media_p_value"].to_numpy(float)
    )
    summary_frame["ar_bh_q_value"] = np.nan
    summary_frame.loc[exploratory, "ar_bh_q_value"] = bh_adjust(
        summary_frame.loc[exploratory, "anderson_rubin_beta0_p_value"].to_numpy(float)
    )
    return summary_frame, pd.DataFrame(first_rows), pd.DataFrame(reduced_rows), fitted


def mechanism_tests(analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    settings = [
        (
            "u3", "z_u3_any_r24",
            ["article_count", "deepseek_labor_article_count", "deepseek_labor_tone_index", PRIMARY_MEDIA],
        ),
        (
            "pce", "z_pce_any_r24",
            ["article_count", "deepseek_cost_living_article_count", "deepseek_cost_living_tone_index", PRIMARY_MEDIA],
        ),
    ]
    for indicator, event, outcomes in settings:
        for outcome in outcomes:
            if outcome not in analysis:
                continue
            result = fit_ols(analysis, outcome, [*BASE_CONTROLS, event], lag=3)
            i = result.names.index(event)
            rows.append(
                {
                    "indicator": indicator, "event_indicator": event,
                    "mechanism_outcome": outcome, "estimate": result.beta[i],
                    "hac_std_error": result.se[i], "p_value": result.pvalue[i],
                    "ci_95_low": result.ci_low[i], "ci_95_high": result.ci_high[i],
                    "nobs": result.nobs,
                }
            )
    output = pd.DataFrame(rows)
    output["bh_q_value"] = bh_adjust(output["p_value"].to_numpy(float))
    return output


def pretrend_tests(analysis: pd.DataFrame) -> pd.DataFrame:
    work = analysis.copy()
    work["pre_media"] = work[PRIMARY_MEDIA].shift(1)
    work["pre_sentiment_change"] = work[f"d_{OUTCOMES[0]}"].shift(1)
    instruments = ["z_u3_signed_r24", "z_pce_signed_r24"]
    rows = []
    for outcome in ["pre_media", "pre_sentiment_change"]:
        result = fit_ols(work, outcome, [*BASE_CONTROLS, *instruments], lag=3)
        wald, df, pvalue = ols_joint_test(result, instruments)
        rows.append(
            {
                "outcome": outcome, "joint_hac_wald": wald, "df": df,
                "p_value": pvalue, "nobs": result.nobs,
                "note": "Lead/placebo check; failure to reject does not prove no anticipation",
            }
        )
    output = pd.DataFrame(rows)
    output["bh_q_value"] = bh_adjust(output["p_value"].to_numpy(float))
    return output


def leave_one_event_out(primary_result: dict, spec: dict) -> pd.DataFrame:
    data = primary_result["data"]
    event_mask = data[spec["z"]].abs().sum(axis=1) > 0
    event_months = data.loc[event_mask, "month"].sort_values().unique()
    rows = []
    for month in event_months:
        subset = data.loc[data["month"] != month].copy()
        try:
            first, weak = first_stage_diagnostics(
                subset, spec["media"], spec["controls"], spec["z"], spec["hac"]
            )
            iv = fit_iv(
                subset, spec["outcome"], spec["media"], spec["controls"], spec["z"], spec["hac"]
            )
            ar = ar_test(
                subset, spec["outcome"], spec["media"], spec["controls"], spec["z"], 0, spec["hac"]
            )
            i = iv.names.index(spec["media"])
            rows.append(
                {
                    "excluded_event_month": pd.Timestamp(month), "iv_media_estimate": iv.beta[i],
                    "iv_media_std_error": iv.se[i], "iv_media_p_value": iv.pvalue[i],
                    "iv_ci_95_low": iv.ci_low[i], "iv_ci_95_high": iv.ci_high[i],
                    "first_stage_wald_f_type": weak["wald_f_type"],
                    "first_stage_partial_r_squared": weak["partial_r_squared"],
                    "anderson_rubin_beta0_p_value": ar[2], "nobs": iv.nobs,
                    "status": "estimated",
                }
            )
        except Exception as exc:
            rows.append({"excluded_event_month": pd.Timestamp(month), "status": str(exc)})
    return pd.DataFrame(rows)


def event_summary(primary_events: pd.DataFrame) -> pd.DataFrame:
    if primary_events.empty:
        return pd.DataFrame()
    rows = []
    for (indicator, direction), part in primary_events.groupby(["indicator", "direction"]):
        rows.append(
            {
                "indicator": indicator, "direction": direction,
                "threshold_crossing_records": len(part),
                "independent_release_months": part["release_month"].nunique(),
                "first_release_month": part["release_month"].min(),
                "last_release_month": part["release_month"].max(),
                "revised_fallback_records": int(part["revision_misclassification_risk"].sum()),
                "multiple_threshold_records": int((part["thresholds_crossed_same_release"] > 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def svg_line_plot(path: Path, title: str, rows: list[tuple[str, float, float, float]], y_label: str) -> None:
    width, height = 1100, 620
    left, right, top, bottom = 100, 40, 70, 120
    inner_w, inner_h = width - left - right, height - top - bottom
    finite = [value for _, value, low, high in rows for value in (value, low, high) if np.isfinite(value)]
    if not finite:
        finite = [-1, 1]
    lo, hi = min(finite + [0]), max(finite + [0])
    margin = max((hi - lo) * 0.12, 0.1)
    lo, hi = lo - margin, hi + margin
    def y(value: float) -> float:
        return top + (hi - value) / (hi - lo) * inner_h
    step = inner_w / max(len(rows), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="{left}" y="35" font-family="Arial" font-size="22" font-weight="bold" fill="#17365D">{html.escape(title)}</text>',
        f'<line x1="{left}" x2="{width-right}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#8A94A6" stroke-dasharray="5,5"/>',
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top+inner_h}" stroke="#333"/>',
        f'<line x1="{left}" x2="{width-right}" y1="{top+inner_h}" y2="{top+inner_h}" stroke="#333"/>',
    ]
    for tick in np.linspace(lo, hi, 6):
        yy = y(float(tick))
        parts.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>')
    for index, (label, value, low, high) in enumerate(rows):
        xx = left + step * (index + 0.5)
        if np.isfinite(value):
            parts.extend([
                f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y(high):.1f}" y2="{y(low):.1f}" stroke="#4472C4" stroke-width="3"/>',
                f'<line x1="{xx-7:.1f}" x2="{xx+7:.1f}" y1="{y(high):.1f}" y2="{y(high):.1f}" stroke="#4472C4" stroke-width="2"/>',
                f'<line x1="{xx-7:.1f}" x2="{xx+7:.1f}" y1="{y(low):.1f}" y2="{y(low):.1f}" stroke="#4472C4" stroke-width="2"/>',
                f'<circle cx="{xx:.1f}" cy="{y(value):.1f}" r="5" fill="#C00000"/>',
            ])
        parts.append(f'<text transform="translate({xx:.1f},{top+inner_h+18}) rotate(50)" font-family="Arial" font-size="11">{html.escape(label)}</text>')
    parts.append(f'<text transform="translate(22,{top+inner_h/2}) rotate(-90)" font-family="Arial" font-size="13">{html.escape(y_label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_event_timeline(path: Path, events: pd.DataFrame) -> None:
    months = pd.date_range(events["release_month"].min(), events["release_month"].max(), freq="MS")
    width, height = 1200, 420
    left, right, top, bottom = 90, 40, 60, 70
    inner_w = width - left - right
    def x(month: pd.Timestamp) -> float:
        return left + ((pd.Timestamp(month) - months.min()).days / max((months.max() - months.min()).days, 1)) * inner_w
    ymap = {"u3_unemployment_pct_sa": 150, "pce_inflation_yoy_pct": 280}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="90" y="34" font-family="Arial" font-size="22" font-weight="bold" fill="#17365D">Rare integer-threshold event timing</text>',
    ]
    for indicator, yy in ymap.items():
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy}" y2="{yy}" stroke="#B7C9E2"/>')
        label = "U-3 unemployment" if indicator.startswith("u3") else "PCE inflation"
        parts.append(f'<text x="{left-10}" y="{yy+4}" text-anchor="end" font-family="Arial" font-size="12">{label}</text>')
    unique = events.drop_duplicates(["indicator", "release_month"])
    for row in unique.itertuples():
        xx, yy = x(row.release_month), ymap[row.indicator]
        color = "#C00000" if row.direction == "up" else "#548235"
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy}" r="7" fill="{color}"/>')
    for year in range(months.min().year, months.max().year + 1, 2):
        date = pd.Timestamp(year=year, month=1, day=1)
        xx = x(date)
        parts.append(f'<text x="{xx:.1f}" y="355" text-anchor="middle" font-family="Arial" font-size="11">{year}</text>')
    parts.append('<text x="90" y="395" font-family="Arial" font-size="11" fill="#555">Red = upward crossing; green = downward crossing. Multiple thresholds in one release month are shown once.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_data_quality(path: Path, quality: dict, events: pd.DataFrame) -> None:
    risky = int(events["revision_misclassification_risk"].sum()) if len(events) else 0
    text = f"""# Data quality and feasibility

- Monthly range: {quality['monthly_start']} to {quality['monthly_end']} ({quality['monthly_rows']} months).
- Relevance-eligible, date-valid article rows after article-ID deduplication: {quality['eligible_article_rows']:,}.
- Article dates: {quality['article_date_start']} to {quality['article_date_end']}.
- Primary media index: {quality['main_media_definition']}.
- Official first-release U-3 observations recovered: {quality['official_u3_values']}.
- Official first-release PCE observations recovered: {quality['official_pce_values']}.
- Rare-24 independent U-3 event months: {quality['u3_primary_event_months']}.
- Rare-24 independent PCE event months: {quality['pce_primary_event_months']}.
- Threshold-level rare-24 records exposed to revision misclassification: {risky}.

## Timing and vintage limitation

{quality['vintage_limitation']} The analysis uses the official release month and exact release
date only where recovered. Unavailable dates are not fabricated. Crossings based on a revised
fallback are explicitly marked in the event table. GDP uses revised quarterly growth and a
proxy release month, so GDP-zero results are high-risk exploratory evidence only.

Michigan survey interviews occur through the month, while the monthly media index covers the
whole month. A same-month model can therefore mix articles before and after an economic
release and can overlap the survey field period. The one-month-ahead specification is reported
but does not by itself solve endogeneity.

## Article limitations

The project retains article IDs, timestamps, headlines, abstracts/leads, primary topic, and
article-level tone. It does not contain a verified complete body-text archive. Keyword snippets
are therefore based on headlines and abstracts and are descriptive only.
"""
    path.write_text(text, encoding="utf-8")


def write_exclusion_risks(path: Path) -> None:
    text = """# Exclusion-restriction assessment

## Conditions partly testable with the current data

- Whether rare integer crossings predict the overall media-tone index (first stage).
- Whether they predict article volume, indicator-topic volume, and indicator-topic tone.
- Whether the instruments predict pre-event media tone or prior sentiment changes.
- Whether a conventional 2SLS estimate disappears when individual event months are removed.
- Whether half-integer placebos show comparable predictive power.

## Conditions not fully testable

1. Consumers may react directly to a salient official integer, without the measured news
   articles mediating that reaction.
2. Inflation, unemployment, and GDP thresholds can change policy expectations, market prices,
   or business decisions. The PCE 2% threshold and GDP zero crossing are especially exposed.
3. A crossing can coincide with a large economic shock. April 2020 is the clearest example;
   multiple unemployment thresholds crossed in one release and the event is not replicated as
   independent observations.
4. The threshold may primarily change attention, volume, framing, or other news channels
   rather than the average -1/0/+1 tone index. Even a first stage for average tone would not
   establish that tone is the only media channel.
5. Television, social media, other news outlets, and interpersonal communication are omitted.
6. The national monthly design lacks the cross-state variation central to Garz's design and
   has very few independent threshold events.
7. Incomplete first-release histories can misclassify crossings and rarity. Row-level flags
   expose this problem but cannot eliminate it.

A non-rejection in an overidentification test does not prove these assumptions. HAC standard
errors address serial correlation and heteroskedasticity; they do not make the exclusion
restriction true and do not remove bias from a lagged dependent variable.
"""
    path.write_text(text, encoding="utf-8")


def write_report(
    path: Path,
    summary: pd.DataFrame,
    first: pd.DataFrame,
    mechanism: pd.DataFrame,
    leave_out: pd.DataFrame,
    quality: dict,
    ar_grid: pd.DataFrame,
) -> None:
    primary = summary.loc[summary["model_id"] == "PRIMARY_R24_HAC3"].iloc[0]
    primary_first = first.loc[first["model_id"] == "PRIMARY_R24_HAC3"]
    accepted = ar_grid.loc[ar_grid["accepted_5pct"]]
    if accepted.empty:
        ar_text = "The evaluated Anderson-Rubin grid contains no accepted value."
    else:
        bounded = not (accepted.index.min() == ar_grid.index.min() or accepted.index.max() == ar_grid.index.max())
        ar_text = (
            f"The 95% Anderson-Rubin acceptance set over the evaluated grid runs from "
            f"{accepted['beta0'].min():.3f} to {accepted['beta0'].max():.3f}"
            + ("." if bounded else " and reaches a grid boundary, so it is not a bounded confidence interval.")
        )
    overall_mechanism = mechanism.loc[mechanism["mechanism_outcome"] == PRIMARY_MEDIA]
    volume_mechanism = mechanism.loc[mechanism["mechanism_outcome"].str.contains("article_count")]
    first_supported = bool(primary["first_stage_joint_p_value"] < 0.05)
    ar_supported = bool(primary["anderson_rubin_beta0_p_value"] < 0.05)
    loo_estimated = leave_out.loc[leave_out["status"] == "estimated"] if len(leave_out) else leave_out
    loo_sign_flip = (
        bool((np.sign(loo_estimated["iv_media_estimate"]) != np.sign(primary["iv_media_estimate"])).any())
        if len(loo_estimated) else True
    )
    identification = first_supported and ar_supported and not loo_sign_flip
    conclusion = (
        "The diagnostics meet the predeclared statistical conditions for a conditional causal interpretation, subject to unresolved exclusion and timing risks."
        if identification
        else "The available national monthly data do not reliably identify a causal effect of average media sentiment. Any conventional significant coefficient should be treated as exploratory."
    )
    fs_lines = "\n".join(
        f"- {row.instrument}: coefficient {row.estimate:.3f}, HAC p={row.p_value:.4f}."
        for row in primary_first.itertuples()
    ) or "- No first-stage coefficient was estimable."
    mech_lines = "\n".join(
        f"- {row.indicator} -> {row.mechanism_outcome}: coefficient {row.estimate:.3f}, p={row.p_value:.4f}, BH q={row.bh_q_value:.4f}."
        for row in mechanism.itertuples()
    )
    pce2 = summary.loc[summary["model_id"] == "ADD_PCE_2PCT"].iloc[0]
    dynamic = summary.loc[summary["model_id"] == "DYNAMIC_LAGGED_Y"].iloc[0]
    text = f"""# Integer-threshold IV study report

## Bottom line

{conclusion}

The primary 2SLS estimate is {primary['iv_media_estimate']:.3f} Michigan-sentiment points per
one point of the DeepSeek media index (HAC(3) SE {primary['iv_media_se']:.3f}, conventional
p={primary['iv_media_p_value']:.4f}, 95% CI [{primary['iv_ci_95_low']:.3f},
{primary['iv_ci_95_high']:.3f}]). The weak-instrument-robust Anderson-Rubin test of zero has
p={primary['anderson_rubin_beta0_p_value']:.4f}. {ar_text}

## 1. Do integer thresholds change tone or only reporting volume?

{mech_lines}

The volume and tone tests use distinct outcomes. A volume response without a first stage for
`{PRIMARY_MEDIA}` does not validate average tone as the endogenous media variable.

## 2. First-stage support

The primary excluded-instrument HAC Wald statistic is
{primary['first_stage_joint_hac_wald']:.3f} on {int(primary['first_stage_joint_df'])} restrictions
(p={primary['first_stage_joint_p_value']:.4f}). Its F-type scaling is
{primary['first_stage_wald_f_type']:.3f}, and the excluded-instrument partial R-squared is
{primary['first_stage_partial_r_squared']:.4f}. This F-type statistic is not labeled
Kleibergen-Paap and no universal F>10 rule is applied.

{fs_lines}

## 3. Conventional 2SLS versus weak-instrument-robust inference

- OLS media coefficient on the identical sample: {primary['ols_media_estimate']:.3f}
  (p={primary['ols_media_p_value']:.4f}).
- Conventional 2SLS: {primary['iv_media_estimate']:.3f}
  (p={primary['iv_media_p_value']:.4f}).
- Anderson-Rubin test of a zero media coefficient: p={primary['anderson_rubin_beta0_p_value']:.4f}.
- HAC moment-based overidentification test: p={primary['overidentification_p_value']:.4f};
  non-rejection would not establish validity.

### Apparent significance in high-risk sensitivities

Adding the policy-salient PCE 2% threshold produces a conventional 2SLS coefficient of
{pce2['iv_media_estimate']:.3f} with p={pce2['iv_media_p_value']:.4f}, but the first-stage
joint p-value is {pce2['first_stage_joint_p_value']:.4f}, the HAC Wald/F-type statistic is only
{pce2['first_stage_wald_f_type']:.3f}, and its BH-adjusted conventional IV q-value is
{pce2['iv_bh_q_value']:.4f}. The 2% threshold can directly change policy expectations, so this
is not promoted to a causal result.

In the lagged-outcome sensitivity the Anderson-Rubin zero test rejects
(p={dynamic['anderson_rubin_beta0_p_value']:.6f}) while conventional 2SLS does not
(p={dynamic['iv_media_p_value']:.4f}) and the first stage remains absent
(p={dynamic['first_stage_joint_p_value']:.4f}). This pattern means the instruments predict the
outcome conditional on the controls without predicting media tone; it raises an exclusion or
specification concern rather than identifying a media-tone effect.

## 4. Event concentration

The primary design has {quality['u3_primary_event_months']} independent U-3 event months and
{quality['pce_primary_event_months']} independent PCE event months. Leave-one-event-out
estimates are stored in full. Sign reversal after an event deletion: {loo_sign_flip}.

## 5. Unresolved exclusion risks

Direct psychological responses to round official numbers, policy and market reactions,
coincident macro shocks, omitted media channels, and effects operating through attention,
volume, or framing remain plausible. April 2020 creates extreme concentration. Partial
first-release coverage creates event-classification risk. See `exclusion_restriction_assessment.md`.

## 6. Causal interpretation

This is a national monthly adaptation of a state-panel design. Article counts do not increase
the number of independent macro observations. With few threshold months, incomplete vintages,
and overlapping media/survey timing, the study is weaker than the original design. The
appropriate conclusion follows the predeclared decision rule rather than whether one 2SLS
p-value is below 0.05.

## Reproducibility

Run `python scripts/run_threshold_iv_analysis.py` from the repository root after optionally
refreshing official release data with `python scripts/fetch_iv_release_data.py`. The frozen
specification is in `design_spec.md`; every prespecified sensitivity remains in
`iv_model_summary.csv`.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    analysis, all_events, primary_events, articles, quality = construct_data(
        args.workbook, args.articles, args.release_data
    )
    specs = build_specs()
    summary, first, reduced, fitted = run_specs(analysis, specs)
    if "PRIMARY_R24_HAC3" not in fitted:
        raise RuntimeError("Primary IV specification could not be estimated.")
    primary_spec = next(spec for spec in specs if spec["id"] == "PRIMARY_R24_HAC3")
    primary_fit = fitted["PRIMARY_R24_HAC3"]
    ar_grid = ar_confidence_grid(
        primary_fit["data"], primary_spec["outcome"], primary_spec["media"],
        primary_spec["controls"], primary_spec["z"], primary_fit["iv"], 3,
    )
    mechanism = mechanism_tests(analysis)
    pretrend = pretrend_tests(analysis)
    leave_out = leave_one_event_out(primary_fit, primary_spec)
    window, snippets = event_window_mechanism(primary_events, articles)
    events_summary = event_summary(primary_events)

    all_events.to_csv(out / "threshold_event_table.csv", index=False, encoding="utf-8-sig")
    events_summary.to_csv(out / "event_summary.csv", index=False, encoding="utf-8-sig")
    analysis.to_csv(out / "aligned_iv_dataset.csv", index=False, encoding="utf-8-sig")
    first.to_csv(out / "first_stage_results.csv", index=False, encoding="utf-8-sig")
    reduced.to_csv(out / "reduced_form_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "iv_model_summary.csv", index=False, encoding="utf-8-sig")
    mechanism.to_csv(out / "mechanism_results.csv", index=False, encoding="utf-8-sig")
    pretrend.to_csv(out / "pretrend_placebo_results.csv", index=False, encoding="utf-8-sig")
    leave_out.to_csv(out / "leave_one_event_out.csv", index=False, encoding="utf-8-sig")
    ar_grid.to_csv(out / "anderson_rubin_confidence_grid.csv", index=False, encoding="utf-8-sig")
    window.to_csv(out / "event_window_mechanism.csv", index=False, encoding="utf-8-sig")
    snippets.to_csv(out / "news_threshold_snippets.csv", index=False, encoding="utf-8-sig")

    write_data_quality(out / "data_quality_report.md", quality, primary_events)
    write_exclusion_risks(out / "exclusion_restriction_assessment.md")
    write_report(out / "research_report.md", summary, first, mechanism, leave_out, quality, ar_grid)

    fs_plot = first.loc[first["model_id"] == "PRIMARY_R24_HAC3"]
    svg_line_plot(
        out / "first_stage_coefficients.svg", "First-stage threshold coefficients",
        [(row.instrument, row.estimate, row.ci_95_low, row.ci_95_high) for row in fs_plot.itertuples()],
        "Effect on DeepSeek media-tone index",
    )
    plot_summary = summary.loc[summary["status"] == "estimated"].head(13)
    svg_line_plot(
        out / "iv_coefficients.svg", "2SLS media-sentiment coefficients",
        [(row.model_id, row.iv_media_estimate, row.iv_ci_95_low, row.iv_ci_95_high) for row in plot_summary.itertuples()],
        "Effect on Michigan sentiment index",
    )
    if len(primary_events):
        svg_event_timeline(out / "threshold_event_timeline.svg", primary_events)

    metadata = {
        "random_seed": SEED,
        "design_frozen_before_estimation": True,
        "primary_model": "PRIMARY_R24_HAC3",
        "primary_media": PRIMARY_MEDIA,
        "primary_hac_lag": 3,
        "quality": quality,
        "official_release_data_path": str(args.release_data),
        "specification_ids": [spec["id"] for spec in specs],
        "notes": {
            "weak_stat": "HAC joint Wald divided by number of exclusions; not Kleibergen-Paap",
            "ar": "Anderson-Rubin test and inverted grid conditional on instrument exogeneity",
            "multiple_testing": "BH adjustment applied to non-primary IV and AR p-values; mechanism family adjusted separately",
        },
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    primary = summary.loc[summary["model_id"] == "PRIMARY_R24_HAC3"].iloc[0]
    print(
        json.dumps(
            {
                "primary_first_stage_p": primary["first_stage_joint_p_value"],
                "primary_first_stage_f_type": primary["first_stage_wald_f_type"],
                "primary_iv_estimate": primary["iv_media_estimate"],
                "primary_iv_p": primary["iv_media_p_value"],
                "primary_ar_p": primary["anderson_rubin_beta0_p_value"],
                "u3_event_months": quality["u3_primary_event_months"],
                "pce_event_months": quality["pce_primary_event_months"],
                "output_dir": str(out.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
