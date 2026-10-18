#!/usr/bin/env python
"""Reproducible monthly media-sentiment regression and forecasting analysis.

The specification family is declared before model estimation in ``build_specs``.
The script reports every declared media test and controls false discoveries across
the exploratory family. It does not search specifications and then retain only
significant results.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


SEED = 42
PRIMARY_MEDIA = "news_tone_index_deepseek"
SECONDARY_MEDIA = "news_tone_index_t5"
OUTCOMES = [
    "michigan_sentiment",
    "michigan_current_conditions",
    "michigan_expectations",
]
ECON = [
    "u3_unemployment_pct_sa",
    "pce_inflation_yoy_pct",
    "real_gdp_qoq_annualized_official_pct",
]
CONTROLS = ["real_income_yoy_pct", "pce_cumulative_since_jan2010_pct"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path(
            "outputs/monthly_tone_index/"
            "Consumer_Sentiment_Economic_Data_2010_2026_with_tone_indices.xlsx"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/monthly_regression_analysis")
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    return parser.parse_args()


def clean_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_localize(None)


def load_and_align(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    monthly = pd.read_excel(path, sheet_name="Monthly")
    quarterly = pd.read_excel(path, sheet_name="Quarterly GDP")
    sources = pd.read_excel(path, sheet_name="Sources and notes")
    monthly["month"] = clean_date(monthly["month"])
    quarterly["quarter_start"] = clean_date(quarterly["quarter_start"])
    monthly = monthly.sort_values("month").reset_index(drop=True)
    quarterly = quarterly.sort_values("quarter_start").reset_index(drop=True)

    duplicates = int(monthly["month"].duplicated().sum())
    expected = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    missing_calendar_months = sorted(set(expected) - set(monthly["month"]))
    invalid_months = int(monthly["month"].isna().sum())
    if duplicates or invalid_months or missing_calendar_months:
        raise ValueError(
            f"Monthly calendar invalid: duplicates={duplicates}, invalid={invalid_months}, "
            f"missing={[x.strftime('%Y-%m') for x in missing_calendar_months]}"
        )

    numeric_cols = [
        c for c in monthly.columns if c not in {"month", "survey_status"}
    ]
    for col in numeric_cols:
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce")

    # Retrospective association: the quarterly observation is held constant for
    # the three months of its quarter. This is not monthly interpolation.
    monthly["quarter_start"] = monthly["month"].dt.to_period("Q").dt.start_time
    analysis = monthly.merge(quarterly, on="quarter_start", how="left", validate="m:1")

    # Economically motivated additional controls declared before estimation.
    analysis["real_income_yoy_pct"] = 100 * (
        analysis["real_disposable_income_bn2017usd_saar"]
        / analysis["real_disposable_income_bn2017usd_saar"].shift(12)
        - 1
    )
    analysis["u6_minus_u3_pct"] = (
        analysis["u6_underutilization_pct_sa"] - analysis["u3_unemployment_pct_sa"]
    )
    for outcome in OUTCOMES:
        analysis[f"{outcome}_lag1"] = analysis[outcome].shift(1)
        analysis[f"d_{outcome}"] = analysis[outcome].diff()
    for col in ECON + CONTROLS + [PRIMARY_MEDIA, SECONDARY_MEDIA]:
        analysis[f"d_{col}"] = analysis[col].diff()
    for lag in (1, 2, 3):
        analysis[f"{PRIMARY_MEDIA}_lag{lag}"] = analysis[PRIMARY_MEDIA].shift(lag)
    analysis[f"{SECONDARY_MEDIA}_lag1"] = analysis[SECONDARY_MEDIA].shift(1)
    for month_no in range(2, 13):
        analysis[f"calendar_month_{month_no:02d}"] = (
            analysis["month"].dt.month == month_no
        ).astype(float)

    # Forecast information set. Revised monthly data are conservatively lagged
    # two calendar months because t-1 U-3/PCE/income are generally unavailable at
    # the end-of-t-1 cutoff. GDP uses a documented proxy release date: the last
    # calendar day of the month after quarter end. No vintage data are available.
    forecast = analysis.copy()
    forecast["forecast_cutoff"] = forecast["month"] - pd.Timedelta(days=1)
    for source_col, target_col in [
        ("u3_unemployment_pct_sa", "u3_available"),
        ("pce_inflation_yoy_pct", "pce_inflation_available"),
        ("real_income_yoy_pct", "real_income_yoy_available"),
        ("pce_cumulative_since_jan2010_pct", "price_level_available"),
    ]:
        forecast[target_col] = forecast[source_col].shift(2)
    forecast["sentiment_lag1"] = forecast["michigan_sentiment"].shift(1)
    forecast["current_conditions_lag1"] = forecast[
        "michigan_current_conditions"
    ].shift(1)
    forecast["expectations_lag1"] = forecast["michigan_expectations"].shift(1)
    forecast["media_deepseek_available"] = forecast[PRIMARY_MEDIA].shift(1)
    forecast["media_t5_available"] = forecast[SECONDARY_MEDIA].shift(1)

    q_release = quarterly[
        [
            "quarter_start",
            "quarter",
            "real_gdp_qoq_annualized_official_pct",
            "real_gdp_yoy_pct",
        ]
    ].copy()
    q_release["gdp_release_date_proxy"] = q_release["quarter_start"] + pd.offsets.MonthEnd(4)
    q_release = q_release.rename(
        columns={
            "quarter": "gdp_underlying_quarter",
            "real_gdp_qoq_annualized_official_pct": "gdp_qoq_available",
            "real_gdp_yoy_pct": "gdp_yoy_available",
        }
    )
    forecast = pd.merge_asof(
        forecast.sort_values("forecast_cutoff"),
        q_release.sort_values("gdp_release_date_proxy"),
        left_on="forecast_cutoff",
        right_on="gdp_release_date_proxy",
        direction="backward",
        suffixes=("", "_release"),
    ).sort_values("month").reset_index(drop=True)

    missing_detail = {
        col: analysis.loc[analysis[col].isna(), "month"].dt.strftime("%Y-%m").tolist()
        for col in monthly.columns
        if col != "month" and analysis[col].isna().any()
    }
    quality = {
        "workbook": str(path.resolve()),
        "monthly_rows": int(len(monthly)),
        "quarterly_rows": int(len(quarterly)),
        "month_start": monthly["month"].min().strftime("%Y-%m"),
        "month_end": monthly["month"].max().strftime("%Y-%m"),
        "duplicate_months": duplicates,
        "invalid_months": invalid_months,
        "missing_calendar_months": [x.strftime("%Y-%m") for x in missing_calendar_months],
        "missing_by_variable": missing_detail,
        "media_zero_rule": (
            "Monthly media indices are valid aggregates of saved -1/0/+1 scores. "
            "Zero-score subtypes cannot be reconstructed; no missing value is filled with zero."
        ),
        "gdp_timing_limitation": (
            "Quarterly release dates and real-time vintages are unavailable. Association models "
            "use current-quarter revised GDP held constant within the quarter. Forecast models "
            "use revised GDP with a conservative proxy release date."
        ),
    }
    return analysis, forecast, sources, quality


@dataclass
class OLSResult:
    names: list[str]
    beta: np.ndarray
    residuals: np.ndarray
    fitted: np.ndarray
    x: np.ndarray
    y: np.ndarray
    cov: np.ndarray
    se: np.ndarray
    stat: np.ndarray
    pvalue: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    nobs: int
    k: int
    r2: float
    adj_r2: float
    aic: float
    bic: float
    df_resid: int
    index: pd.Index


def newey_west_cov(x: np.ndarray, residuals: np.ndarray, lag: int) -> np.ndarray:
    n, k = x.shape
    bread = np.linalg.pinv(x.T @ x)
    xu = x * residuals[:, None]
    meat = xu.T @ xu
    for ell in range(1, min(lag, n - 1) + 1):
        weight = 1 - ell / (lag + 1)
        gamma = xu[ell:].T @ xu[:-ell]
        meat += weight * (gamma + gamma.T)
    cov = bread @ meat @ bread
    if n > k:
        cov *= n / (n - k)
    return cov


def fit_ols(
    frame: pd.DataFrame,
    outcome: str,
    predictors: Sequence[str],
    hac_lag: int = 3,
) -> OLSResult:
    cols = [outcome, *predictors]
    data = frame[cols].dropna()
    if len(data) <= len(predictors) + 3:
        raise ValueError(f"Insufficient observations for {outcome} ~ {predictors}")
    y = data[outcome].to_numpy(float)
    x = np.column_stack([np.ones(len(data)), data[list(predictors)].to_numpy(float)])
    names = ["const", *predictors]
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    fitted = x @ beta
    residuals = y - fitted
    n, k = x.shape
    sse = float(residuals @ residuals)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else math.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else math.nan
    sigma2_mle = max(sse / n, 1e-300)
    aic = n * math.log(sigma2_mle) + 2 * k
    bic = n * math.log(sigma2_mle) + math.log(n) * k
    cov = newey_west_cov(x, residuals, hac_lag)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    stat_values = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    df_resid = n - k
    pvalue = 2 * stats.t.sf(np.abs(stat_values), df=df_resid)
    critical = stats.t.ppf(0.975, df=df_resid)
    ci_low = beta - critical * se
    ci_high = beta + critical * se
    return OLSResult(
        names,
        beta,
        residuals,
        fitted,
        x,
        y,
        cov,
        se,
        stat_values,
        pvalue,
        ci_low,
        ci_high,
        n,
        k,
        r2,
        adj_r2,
        aic,
        bic,
        df_resid,
        data.index,
    )


def linear_test(result: OLSResult, terms: Sequence[str], sum_test: bool = False) -> dict:
    idx = [result.names.index(term) for term in terms]
    if sum_test:
        r = np.zeros((1, result.k))
        r[0, idx] = 1
    else:
        r = np.zeros((len(idx), result.k))
        for row, col in enumerate(idx):
            r[row, col] = 1
    effect = r @ result.beta
    vcov = r @ result.cov @ r.T
    inv = np.linalg.pinv(vcov)
    wald = float(effect.T @ inv @ effect)
    df = r.shape[0]
    pvalue = float(stats.chi2.sf(wald, df))
    out = {"wald_stat": wald, "wald_df": df, "wald_p_value": pvalue}
    if sum_test:
        se = float(math.sqrt(max(vcov[0, 0], 0)))
        t_stat = float(effect[0] / se) if se > 0 else math.nan
        p_t = float(2 * stats.t.sf(abs(t_stat), result.df_resid)) if se > 0 else math.nan
        critical = stats.t.ppf(0.975, result.df_resid)
        out.update(
            {
                "sum_effect": float(effect[0]),
                "sum_se": se,
                "sum_t_stat": t_stat,
                "sum_p_value": p_t,
                "sum_ci_low": float(effect[0] - critical * se),
                "sum_ci_high": float(effect[0] + critical * se),
            }
        )
    return out


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    vals = p[valid]
    if not len(vals):
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1)
    out[np.where(valid)[0]] = restored
    return out


def build_specs() -> list[dict]:
    """Declare the complete association specification family before estimation."""
    month_fe = [f"calendar_month_{m:02d}" for m in range(2, 13)]
    specs = [
        dict(id="H_M1_econ", outcome=OUTCOMES[0], x=ECON, group="required", primary=False),
        dict(
            id="H_M2_econ_media",
            outcome=OUTCOMES[0],
            x=ECON + [PRIMARY_MEDIA],
            group="required",
            primary=True,
        ),
        dict(
            id="H_M3_controls",
            outcome=OUTCOMES[0],
            x=ECON + [PRIMARY_MEDIA] + CONTROLS,
            group="required",
            primary=False,
        ),
        dict(
            id="H_DYN_M1_econ",
            outcome=OUTCOMES[0],
            x=[f"{OUTCOMES[0]}_lag1"] + ECON,
            group="dynamic",
            primary=False,
        ),
        dict(
            id="H_DYN_M2_media",
            outcome=OUTCOMES[0],
            x=[f"{OUTCOMES[0]}_lag1"] + ECON + [PRIMARY_MEDIA],
            group="dynamic",
            primary=False,
        ),
        dict(
            id="H_DYN_M3_controls",
            outcome=OUTCOMES[0],
            x=[f"{OUTCOMES[0]}_lag1"] + ECON + [PRIMARY_MEDIA] + CONTROLS,
            group="dynamic",
            primary=False,
        ),
        dict(
            id="H_SEASONFE_M2",
            outcome=OUTCOMES[0],
            x=ECON + [PRIMARY_MEDIA] + month_fe,
            group="seasonality",
            primary=False,
        ),
        dict(
            id="H_CONTROL_income",
            outcome=OUTCOMES[0],
            x=ECON + [PRIMARY_MEDIA, "real_income_yoy_pct"],
            group="control_variation",
            primary=False,
        ),
        dict(
            id="H_CONTROL_pricelevel",
            outcome=OUTCOMES[0],
            x=ECON + [PRIMARY_MEDIA, "pce_cumulative_since_jan2010_pct"],
            group="control_variation",
            primary=False,
        ),
        dict(
            id="H_CONTROL_u6_replace",
            outcome=OUTCOMES[0],
            x=[
                "u6_underutilization_pct_sa",
                "pce_inflation_yoy_pct",
                "real_gdp_qoq_annualized_official_pct",
                PRIMARY_MEDIA,
            ],
            group="control_variation",
            primary=False,
        ),
        dict(
            id="H_CONTROL_labor_spread",
            outcome=OUTCOMES[0],
            x=ECON + [PRIMARY_MEDIA, "u6_minus_u3_pct"],
            group="control_variation",
            primary=False,
        ),
        dict(
            id="H_MEDIA_lag1",
            outcome=OUTCOMES[0],
            x=ECON + [f"{PRIMARY_MEDIA}_lag1"],
            media_terms=[f"{PRIMARY_MEDIA}_lag1"],
            group="media_timing",
            primary=False,
        ),
        dict(
            id="H_MEDIA_lag3",
            outcome=OUTCOMES[0],
            x=ECON + [f"{PRIMARY_MEDIA}_lag3"],
            media_terms=[f"{PRIMARY_MEDIA}_lag3"],
            group="media_timing",
            primary=False,
        ),
        dict(
            id="H_MEDIA_distributed_0_3",
            outcome=OUTCOMES[0],
            x=ECON + [PRIMARY_MEDIA] + [f"{PRIMARY_MEDIA}_lag{x}" for x in (1, 2, 3)],
            media_terms=[PRIMARY_MEDIA] + [f"{PRIMARY_MEDIA}_lag{x}" for x in (1, 2, 3)],
            group="media_timing",
            primary=False,
        ),
        dict(
            id="H_DIFF_M2",
            outcome=f"d_{OUTCOMES[0]}",
            x=[f"d_{x}" for x in ECON] + [f"d_{PRIMARY_MEDIA}"],
            media_terms=[f"d_{PRIMARY_MEDIA}"],
            group="first_difference",
            primary=False,
        ),
        dict(
            id="H_T5_M2",
            outcome=OUTCOMES[0],
            x=ECON + [SECONDARY_MEDIA],
            media_terms=[SECONDARY_MEDIA],
            group="alternate_media",
            primary=False,
        ),
        dict(
            id="H_T5_M3",
            outcome=OUTCOMES[0],
            x=ECON + [SECONDARY_MEDIA] + CONTROLS,
            media_terms=[SECONDARY_MEDIA],
            group="alternate_media",
            primary=False,
        ),
    ]
    for prefix, outcome in [("C", OUTCOMES[1]), ("E", OUTCOMES[2])]:
        specs.extend(
            [
                dict(
                    id=f"{prefix}_M2_econ_media",
                    outcome=outcome,
                    x=ECON + [PRIMARY_MEDIA],
                    group="subindex",
                    primary=False,
                ),
                dict(
                    id=f"{prefix}_M3_controls",
                    outcome=outcome,
                    x=ECON + [PRIMARY_MEDIA] + CONTROLS,
                    group="subindex",
                    primary=False,
                ),
                dict(
                    id=f"{prefix}_DYN_M2_media",
                    outcome=outcome,
                    x=[f"{outcome}_lag1"] + ECON + [PRIMARY_MEDIA],
                    group="subindex_dynamic",
                    primary=False,
                ),
            ]
        )
    for spec in specs:
        spec.setdefault(
            "media_terms",
            [x for x in spec["x"] if x in {PRIMARY_MEDIA, SECONDARY_MEDIA}],
        )
    return specs


def common_samples(analysis: pd.DataFrame, specs: list[dict]) -> dict[str, pd.DataFrame]:
    static_union = set([OUTCOMES[0], *ECON, PRIMARY_MEDIA, *CONTROLS])
    dynamic_union = static_union | {f"{OUTCOMES[0]}_lag1"}
    return {
        "required": analysis.dropna(subset=sorted(static_union)),
        "dynamic": analysis.dropna(subset=sorted(dynamic_union)),
    }


def regression_tables(
    analysis: pd.DataFrame, specs: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, OLSResult]]:
    common = common_samples(analysis, specs)
    coefficients: list[dict] = []
    summaries: list[dict] = []
    media_rows: list[dict] = []
    results: dict[str, OLSResult] = {}
    for spec in specs:
        if spec["group"] == "required":
            frame = common["required"]
        elif spec["group"] == "dynamic":
            frame = common["dynamic"]
        else:
            frame = analysis
        result = fit_ols(frame, spec["outcome"], spec["x"], hac_lag=3)
        results[spec["id"]] = result
        data_months = analysis.loc[result.index, "month"]
        media_terms = spec["media_terms"]
        media_test = linear_test(result, media_terms) if media_terms else None
        media_sum = (
            linear_test(result, media_terms, sum_test=True)
            if len(media_terms) > 1
            else None
        )
        summaries.append(
            {
                "model_id": spec["id"],
                "outcome": spec["outcome"],
                "method": "OLS with HAC inference",
                "group": spec["group"],
                "primary_hypothesis": spec["primary"],
                "predictors": " | ".join(spec["x"]),
                "nobs": result.nobs,
                "sample_start": data_months.min().strftime("%Y-%m"),
                "sample_end": data_months.max().strftime("%Y-%m"),
                "r_squared": result.r2,
                "adjusted_r_squared": result.adj_r2,
                "aic": result.aic,
                "bic": result.bic,
                "hac_lag": 3,
                "joint_media_wald": media_test["wald_stat"] if media_test else math.nan,
                "joint_media_df": media_test["wald_df"] if media_test else math.nan,
                "joint_media_p_value": media_test["wald_p_value"] if media_test else math.nan,
            }
        )
        for i, term in enumerate(result.names):
            coefficients.append(
                {
                    "model_id": spec["id"],
                    "outcome": spec["outcome"],
                    "group": spec["group"],
                    "term": term,
                    "estimate": result.beta[i],
                    "hac_std_error": result.se[i],
                    "test_statistic": result.stat[i],
                    "p_value": result.pvalue[i],
                    "ci_95_low": result.ci_low[i],
                    "ci_95_high": result.ci_high[i],
                    "significant_5pct": bool(result.pvalue[i] < 0.05),
                    "nobs": result.nobs,
                    "hac_lag": 3,
                }
            )
        if media_terms:
            if len(media_terms) == 1:
                i = result.names.index(media_terms[0])
                estimate = float(result.beta[i])
                se = float(result.se[i])
                pvalue = float(result.pvalue[i])
                ci_low = float(result.ci_low[i])
                ci_high = float(result.ci_high[i])
                test_type = "single coefficient"
            else:
                estimate = media_sum["sum_effect"]
                se = media_sum["sum_se"]
                pvalue = media_test["wald_p_value"]
                ci_low = media_sum["sum_ci_low"]
                ci_high = media_sum["sum_ci_high"]
                test_type = "joint Wald; estimate is sum of lags"
            raw_media = (
                analysis.loc[result.index, PRIMARY_MEDIA]
                if any(PRIMARY_MEDIA in term for term in media_terms)
                else analysis.loc[result.index, SECONDARY_MEDIA]
            )
            media_rows.append(
                {
                    "model_id": spec["id"],
                    "outcome": spec["outcome"],
                    "method": "OLS-HAC",
                    "group": spec["group"],
                    "primary_hypothesis": spec["primary"],
                    "media_terms": " | ".join(media_terms),
                    "test_type": test_type,
                    "estimate": estimate,
                    "std_error": se,
                    "raw_p_value": pvalue,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "media_sample_sd": float(raw_media.std(ddof=1)),
                    "effect_of_1sd_media": estimate * float(raw_media.std(ddof=1)),
                    "nobs": result.nobs,
                    "significant_raw_5pct": bool(pvalue < 0.05),
                }
            )
    media = pd.DataFrame(media_rows)
    exploratory = ~media["primary_hypothesis"]
    media["bh_q_value"] = np.nan
    media.loc[exploratory, "bh_q_value"] = bh_adjust(
        media.loc[exploratory, "raw_p_value"].to_numpy(float)
    )
    media["significant_bh_5pct"] = media["bh_q_value"] < 0.05
    return pd.DataFrame(coefficients), pd.DataFrame(summaries), media, results


def hac_sensitivity(
    analysis: pd.DataFrame, specs: list[dict]
) -> pd.DataFrame:
    chosen = {"H_M2_econ_media", "H_M3_controls", "H_DYN_M2_media", "H_T5_M2"}
    spec_map = {s["id"]: s for s in specs}
    common = common_samples(analysis, specs)
    rows = []
    for model_id in sorted(chosen):
        spec = spec_map[model_id]
        frame = common[spec["group"]] if spec["group"] in common else analysis
        for lag in (3, 6, 12):
            result = fit_ols(frame, spec["outcome"], spec["x"], hac_lag=lag)
            term = spec["media_terms"][0]
            i = result.names.index(term)
            rows.append(
                {
                    "model_id": model_id,
                    "media_term": term,
                    "hac_lag": lag,
                    "estimate": result.beta[i],
                    "std_error": result.se[i],
                    "test_statistic": result.stat[i],
                    "p_value": result.pvalue[i],
                    "ci_95_low": result.ci_low[i],
                    "ci_95_high": result.ci_high[i],
                    "significant_5pct": bool(result.pvalue[i] < 0.05),
                    "nobs": result.nobs,
                }
            )
    return pd.DataFrame(rows)


def huber_fit(x: np.ndarray, y: np.ndarray, c: float = 1.345) -> np.ndarray:
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    for _ in range(100):
        residuals = y - x @ beta
        center = np.median(residuals)
        scale = 1.4826 * np.median(np.abs(residuals - center))
        if not np.isfinite(scale) or scale < 1e-10:
            break
        u = residuals / (c * scale)
        weights = np.ones_like(u)
        outside = np.abs(u) > 1
        weights[outside] = 1 / np.abs(u[outside])
        beta_new = np.linalg.pinv(x.T @ (weights[:, None] * x)) @ x.T @ (weights * y)
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return beta


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    while sum(len(piece) for piece in pieces) < n:
        start = int(rng.integers(0, max(1, n - block + 1)))
        pieces.append(np.arange(start, min(start + block, n)))
    return np.concatenate(pieces)[:n]


def huber_bootstrap(
    analysis: pd.DataFrame, reps: int
) -> tuple[pd.DataFrame, dict]:
    predictors = ECON + [PRIMARY_MEDIA] + CONTROLS
    data = analysis[[OUTCOMES[0], *predictors, "month"]].dropna()
    y = data[OUTCOMES[0]].to_numpy(float)
    raw_x = data[predictors].to_numpy(float)
    means = raw_x.mean(axis=0)
    scales = raw_x.std(axis=0, ddof=1)
    scales[scales == 0] = 1
    x = np.column_stack([np.ones(len(data)), (raw_x - means) / scales])
    names = ["const", *predictors]
    beta_std = huber_fit(x, y)
    media_pos = names.index(PRIMARY_MEDIA)
    estimate = beta_std[media_pos] / scales[predictors.index(PRIMARY_MEDIA)]
    rng = np.random.default_rng(SEED)
    boot = np.empty(reps)
    for rep in range(reps):
        idx = block_indices(len(data), 6, rng)
        b = huber_fit(x[idx], y[idx])
        boot[rep] = b[media_pos] / scales[predictors.index(PRIMARY_MEDIA)]
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    pvalue = min(
        1.0,
        2
        * min(
            (1 + np.sum(boot <= 0)) / (reps + 1),
            (1 + np.sum(boot >= 0)) / (reps + 1),
        ),
    )
    row = pd.DataFrame(
        [
            {
                "model_id": "H_HUBER_M3",
                "outcome": OUTCOMES[0],
                "method": "Huber M-estimator with moving-block bootstrap",
                "media_term": PRIMARY_MEDIA,
                "estimate": estimate,
                "bootstrap_std_error": float(np.std(boot, ddof=1)),
                "raw_p_value": pvalue,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "significant_raw_5pct": bool(pvalue < 0.05),
                "block_length": 6,
                "bootstrap_reps": reps,
                "nobs": len(data),
            }
        ]
    )
    detail = {
        "estimate": estimate,
        "p_value": pvalue,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "nobs": len(data),
    }
    return row, detail


def autocorrelation(values: np.ndarray, max_lag: int = 12) -> np.ndarray:
    x = np.asarray(values, float)
    x = x - x.mean()
    denom = float(x @ x)
    return np.array(
        [float(x[lag:] @ x[:-lag]) / denom for lag in range(1, max_lag + 1)]
    )


def ljung_box(residuals: np.ndarray, lags: int, k_params: int) -> tuple[float, float, int]:
    n = len(residuals)
    acf = autocorrelation(residuals, lags)
    q = n * (n + 2) * sum(acf[k - 1] ** 2 / (n - k) for k in range(1, lags + 1))
    df = max(1, lags - k_params + 1)
    return float(q), float(stats.chi2.sf(q, df)), df


def breusch_pagan(result: OLSResult) -> tuple[float, float, int]:
    z = result.x
    u2 = result.residuals**2
    beta = np.linalg.pinv(z.T @ z) @ z.T @ u2
    fitted = z @ beta
    sst = ((u2 - u2.mean()) ** 2).sum()
    r2 = 1 - ((u2 - fitted) ** 2).sum() / sst if sst > 0 else 0
    lm = result.nobs * r2
    df = result.k - 1
    return float(lm), float(stats.chi2.sf(lm, df)), df


def vif_table(frame: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    data = frame[list(predictors)].dropna()
    rows = []
    for target in predictors:
        others = [x for x in predictors if x != target]
        y = data[target].to_numpy(float)
        x = np.column_stack([np.ones(len(data)), data[others].to_numpy(float)])
        beta = np.linalg.pinv(x.T @ x) @ x.T @ y
        resid = y - x @ beta
        sst = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (resid @ resid) / sst if sst > 0 else 1
        vif = math.inf if 1 - r2 < 1e-12 else 1 / (1 - r2)
        rows.append({"term": target, "vif": vif})
    return pd.DataFrame(rows)


def cooks_distance(result: OLSResult) -> np.ndarray:
    hat = np.sum((result.x @ np.linalg.pinv(result.x.T @ result.x)) * result.x, axis=1)
    mse = (result.residuals @ result.residuals) / result.df_resid
    return (result.residuals**2 / (result.k * mse)) * hat / np.maximum((1 - hat) ** 2, 1e-12)


def adf_test(values: pd.Series, lag: int = 3, trend: bool = False) -> dict:
    x = values.dropna().to_numpy(float)
    dx = np.diff(x)
    y = dx[lag:]
    columns = [np.ones(len(y))]
    if trend:
        columns.append(np.arange(1, len(y) + 1, dtype=float))
    columns.append(x[lag:-1])
    for i in range(1, lag + 1):
        columns.append(dx[lag - i : len(dx) - i])
    design = np.column_stack(columns)
    beta = np.linalg.pinv(design.T @ design) @ design.T @ y
    resid = y - design @ beta
    df = len(y) - design.shape[1]
    cov = (resid @ resid / df) * np.linalg.pinv(design.T @ design)
    level_pos = 2 if trend else 1
    stat_value = beta[level_pos] / math.sqrt(max(cov[level_pos, level_pos], 1e-300))
    critical = -3.41 if trend else -2.86
    return {
        "test": "ADF",
        "statistic": float(stat_value),
        "critical_5pct_approx": critical,
        "reject_5pct": bool(stat_value < critical),
        "deterministic": "constant+trend" if trend else "constant",
        "lag": lag,
        "note": "Approximate 5% critical value; no mechanical specification selection.",
    }


def kpss_test(values: pd.Series, lag: int = 6, trend: bool = False) -> dict:
    x = values.dropna().to_numpy(float)
    design = np.column_stack([np.ones(len(x)), np.arange(len(x))]) if trend else np.ones((len(x), 1))
    beta = np.linalg.pinv(design.T @ design) @ design.T @ x
    residuals = x - design @ beta
    partial = np.cumsum(residuals)
    eta = np.sum(partial**2) / len(x) ** 2
    lrv = np.sum(residuals**2) / len(x)
    for ell in range(1, min(lag, len(x) - 1) + 1):
        weight = 1 - ell / (lag + 1)
        lrv += 2 * weight * np.sum(residuals[ell:] * residuals[:-ell]) / len(x)
    stat_value = eta / lrv
    critical = 0.146 if trend else 0.463
    return {
        "test": "KPSS",
        "statistic": float(stat_value),
        "critical_5pct_approx": critical,
        "reject_5pct": bool(stat_value > critical),
        "deterministic": "constant+trend" if trend else "constant",
        "lag": lag,
        "note": "Approximate 5% critical value; no mechanical specification selection.",
    }


def diagnostics(
    analysis: pd.DataFrame, primary: OLSResult
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q12, q12_p, q12_df = ljung_box(primary.residuals, 12, primary.k - 1)
    bp, bp_p, bp_df = breusch_pagan(primary)
    acf = autocorrelation(primary.residuals, 12)
    diagnostics_rows = [
        {
            "model_id": "H_M2_econ_media",
            "diagnostic": "Ljung-Box residual autocorrelation",
            "statistic": q12,
            "df": q12_df,
            "p_value": q12_p,
            "interpretation": "Residual autocorrelation remains" if q12_p < 0.05 else "No rejection at 5%",
        },
        {
            "model_id": "H_M2_econ_media",
            "diagnostic": "Breusch-Pagan heteroskedasticity",
            "statistic": bp,
            "df": bp_df,
            "p_value": bp_p,
            "interpretation": "Heteroskedasticity detected" if bp_p < 0.05 else "No rejection at 5%",
        },
    ]
    acf_frame = pd.DataFrame({"lag": np.arange(1, 13), "residual_acf": acf})
    vif = vif_table(analysis, ECON + [PRIMARY_MEDIA] + CONTROLS)
    cooks = cooks_distance(primary)
    influence = pd.DataFrame(
        {
            "month": analysis.loc[primary.index, "month"].to_numpy(),
            "residual": primary.residuals,
            "cooks_distance": cooks,
        }
    ).sort_values("cooks_distance", ascending=False)
    return pd.DataFrame(diagnostics_rows), acf_frame, vif, influence.head(12)


def stationarity_table(analysis: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "michigan_sentiment",
        "michigan_current_conditions",
        "michigan_expectations",
        "u3_unemployment_pct_sa",
        "pce_inflation_yoy_pct",
        "real_gdp_qoq_annualized_official_pct",
        "real_disposable_income_bn2017usd_saar",
        "pce_cumulative_since_jan2010_pct",
        PRIMARY_MEDIA,
        SECONDARY_MEDIA,
    ]
    rows = []
    trending = {
        "real_disposable_income_bn2017usd_saar",
        "pce_cumulative_since_jan2010_pct",
    }
    for variable in variables:
        trend = variable in trending
        for test in (adf_test(analysis[variable], trend=trend), kpss_test(analysis[variable], trend=trend)):
            rows.append({"variable": variable, **test, "nobs": int(analysis[variable].notna().sum())})
    return pd.DataFrame(rows)


def stability_table(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictors = ECON + [PRIMARY_MEDIA]
    for label, mask in [
        ("2010-2019", analysis["month"] < pd.Timestamp("2020-01-01")),
        ("2020-2026", analysis["month"] >= pd.Timestamp("2020-01-01")),
    ]:
        result = fit_ols(analysis.loc[mask], OUTCOMES[0], predictors, 3)
        pos = result.names.index(PRIMARY_MEDIA)
        rows.append(
            {
                "window": label,
                "nobs": result.nobs,
                "media_estimate": result.beta[pos],
                "hac_std_error": result.se[pos],
                "p_value": result.pvalue[pos],
                "ci_95_low": result.ci_low[pos],
                "ci_95_high": result.ci_high[pos],
            }
        )
    rolling = []
    clean = analysis[["month", OUTCOMES[0], *predictors]].dropna().reset_index(drop=True)
    for end in range(60, len(clean) + 1):
        window = clean.iloc[end - 60 : end]
        result = fit_ols(window, OUTCOMES[0], predictors, 3)
        pos = result.names.index(PRIMARY_MEDIA)
        rolling.append(
            {
                "window_end": window["month"].iloc[-1],
                "window_start": window["month"].iloc[0],
                "media_estimate": result.beta[pos],
                "ci_95_low": result.ci_low[pos],
                "ci_95_high": result.ci_high[pos],
                "p_value": result.pvalue[pos],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(rolling)


def fit_predict_ols(train: pd.DataFrame, row: pd.Series, y_col: str, x_cols: Sequence[str]) -> float:
    y = train[y_col].to_numpy(float)
    x = np.column_stack([np.ones(len(train)), train[list(x_cols)].to_numpy(float)])
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    target = np.r_[1.0, row[list(x_cols)].to_numpy(float)]
    return float(target @ beta)


def fit_predict_ridge(
    train: pd.DataFrame,
    row: pd.Series,
    y_col: str,
    x_cols: Sequence[str],
    alpha: float,
) -> float:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[list(x_cols)].to_numpy(float))
    x_target = scaler.transform(row[list(x_cols)].to_numpy(float).reshape(1, -1))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_train, train[y_col].to_numpy(float))
    return float(model.predict(x_target)[0])


def choose_ridge_alpha(train: pd.DataFrame, y_col: str, x_cols: Sequence[str]) -> float:
    alphas = [0.1, 1.0, 10.0, 100.0]
    start = min(60, max(24, len(train) // 2))
    scores = {}
    for alpha in alphas:
        errors = []
        for i in range(start, len(train)):
            pred = fit_predict_ridge(train.iloc[:i], train.iloc[i], y_col, x_cols, alpha)
            errors.append((train[y_col].iloc[i] - pred) ** 2)
        scores[alpha] = math.sqrt(float(np.mean(errors)))
    return min(scores, key=scores.get)


def bootstrap_loss_difference(
    loss_econ: np.ndarray,
    loss_other: np.ndarray,
    reps: int,
    block: int = 6,
) -> dict:
    difference = np.asarray(loss_econ) - np.asarray(loss_other)
    rng = np.random.default_rng(SEED)
    boot = np.empty(reps)
    for rep in range(reps):
        idx = block_indices(len(difference), block, rng)
        boot[rep] = difference[idx].mean()
    lower, upper = np.quantile(boot, [0.025, 0.975])
    pvalue = min(
        1.0,
        2
        * min(
            (1 + np.sum(boot <= 0)) / (reps + 1),
            (1 + np.sum(boot >= 0)) / (reps + 1),
        ),
    )
    return {
        "mean_squared_loss_improvement": float(difference.mean()),
        "bootstrap_ci_low": float(lower),
        "bootstrap_ci_high": float(upper),
        "bootstrap_p_value": float(pvalue),
        "block_length": block,
        "bootstrap_reps": reps,
    }


def conformal_radius(errors: Sequence[float], coverage: float = 0.95) -> float:
    values = np.abs(np.asarray(errors, float))
    quantile = min(1.0, math.ceil((len(values) + 1) * coverage) / len(values))
    return float(np.quantile(values, quantile, method="higher"))


def forecast_outcome(
    forecast: pd.DataFrame,
    outcome: str,
    lag_col: str,
    reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    econ = [lag_col, "u3_available", "pce_inflation_available", "gdp_qoq_available"]
    media = econ + ["media_deepseek_available"]
    full = media + ["real_income_yoy_available", "price_level_available"]
    required = ["month", outcome, *full]
    data = forecast[required].dropna().sort_values("month").reset_index(drop=True)
    split = int(math.floor(0.8 * len(data)))
    if split < 60 or len(data) - split < 24:
        raise ValueError(f"Forecast sample too short for {outcome}: {len(data)}")
    initial_train = data.iloc[:split]
    ridge_alphas = {
        "ridge_econ": choose_ridge_alpha(initial_train, outcome, econ),
        "ridge_media": choose_ridge_alpha(initial_train, outcome, media),
        "ridge_full": choose_ridge_alpha(initial_train, outcome, full),
    }

    # Training-only prequential errors initialize prediction-interval calibration.
    prequential_errors = []
    for i in range(60, split):
        pred = fit_predict_ols(data.iloc[:i], data.iloc[i], outcome, econ)
        prequential_errors.append(float(data[outcome].iloc[i] - pred))

    rows = []
    interval_errors = list(prequential_errors)
    for i in range(split, len(data)):
        train = data.iloc[:i]
        target = data.iloc[i]
        predictions = {
            "naive": float(target[lag_col]),
            "ols_econ": fit_predict_ols(train, target, outcome, econ),
            "ols_media": fit_predict_ols(train, target, outcome, media),
            "ols_full": fit_predict_ols(train, target, outcome, full),
            "ridge_econ": fit_predict_ridge(
                train, target, outcome, econ, ridge_alphas["ridge_econ"]
            ),
            "ridge_media": fit_predict_ridge(
                train, target, outcome, media, ridge_alphas["ridge_media"]
            ),
            "ridge_full": fit_predict_ridge(
                train, target, outcome, full, ridge_alphas["ridge_full"]
            ),
        }
        radius = conformal_radius(interval_errors)
        observed = float(target[outcome])
        econ_error = observed - predictions["ols_econ"]
        rows.append(
            {
                "outcome": outcome,
                "month": target["month"],
                "observed": observed,
                **predictions,
                "econ_prediction_interval_low": predictions["ols_econ"] - radius,
                "econ_prediction_interval_high": predictions["ols_econ"] + radius,
                "econ_prediction_interval_radius": radius,
                "sentiment_gap": econ_error,
                "interval_covered": bool(abs(econ_error) <= radius),
                "training_n": len(train),
            }
        )
        interval_errors.append(econ_error)
    results = pd.DataFrame(rows)

    metrics = []
    for model in [
        "naive",
        "ols_econ",
        "ols_media",
        "ols_full",
        "ridge_econ",
        "ridge_media",
        "ridge_full",
    ]:
        error = results["observed"] - results[model]
        method = "ridge" if model.startswith("ridge") else "ols_or_naive"
        econ_model = "ridge_econ" if model.startswith("ridge") else "ols_econ"
        econ_error = results["observed"] - results[econ_model]
        econ_rmse = math.sqrt(float(np.mean(econ_error**2)))
        rmse = math.sqrt(float(np.mean(error**2)))
        metrics.append(
            {
                "outcome": outcome,
                "model": model,
                "method_family": method,
                "n_forecasts": len(results),
                "mae": float(np.mean(np.abs(error))),
                "rmse": rmse,
                "rmse_improvement_vs_econ_pct": 100 * (econ_rmse - rmse) / econ_rmse,
                "initial_training_n": split,
                "evaluation_start": results["month"].min().strftime("%Y-%m"),
                "evaluation_end": results["month"].max().strftime("%Y-%m"),
                "ridge_alpha": ridge_alphas.get(model, math.nan),
            }
        )
    metrics_frame = pd.DataFrame(metrics)

    tests = []
    for method, econ_model, alternatives in [
        ("OLS", "ols_econ", ["ols_media", "ols_full"]),
        ("Ridge", "ridge_econ", ["ridge_media", "ridge_full"]),
    ]:
        econ_loss = (results["observed"] - results[econ_model]) ** 2
        for alternative in alternatives:
            other_loss = (results["observed"] - results[alternative]) ** 2
            tests.append(
                {
                    "outcome": outcome,
                    "method": method,
                    "comparison": f"{alternative} versus {econ_model}",
                    **bootstrap_loss_difference(
                        econ_loss.to_numpy(), other_loss.to_numpy(), reps
                    ),
                }
            )
    tests_frame = pd.DataFrame(tests)
    interval_info = {
        "outcome": outcome,
        "coverage": float(results["interval_covered"].mean()),
        "n_forecasts": len(results),
        "initial_calibration_errors": len(prequential_errors),
        "method": "expanding prior one-step absolute-error conformal calibration",
        "ridge_alphas": ridge_alphas,
        "eligible_rows": len(data),
        "split_index": split,
    }
    return results, metrics_frame, tests_frame, interval_info


def variable_dictionary() -> pd.DataFrame:
    rows = [
        ("month", "Monthly", "Calendar month", "month", "First day of month"),
        ("michigan_sentiment", "Monthly", "Michigan Index of Consumer Sentiment", "index", "Primary outcome"),
        ("michigan_current_conditions", "Monthly", "Michigan Current Economic Conditions", "index", "Secondary outcome"),
        ("michigan_expectations", "Monthly", "Michigan Consumer Expectations", "index", "Secondary outcome"),
        ("u3_unemployment_pct_sa", "Monthly", "U-3 unemployment rate, seasonally adjusted", "percent", "Required economic predictor"),
        ("pce_inflation_yoy_pct", "Monthly", "Headline PCE inflation, year over year", "percent", "Required economic predictor"),
        ("real_gdp_qoq_annualized_official_pct", "Quarterly GDP", "Official real GDP quarter-over-quarter annualized growth", "percent SAAR", "Held constant within quarter for association"),
        ("real_gdp_yoy_pct", "Quarterly GDP", "Real GDP year-over-year growth", "percent", "Sensitivity/availability field"),
        ("real_disposable_income_bn2017usd_saar", "Monthly", "Real disposable personal income", "billions chained 2017 USD SAAR", "Used to derive year-over-year growth control"),
        ("real_income_yoy_pct", "Derived", "Real disposable personal income growth, year over year", "percent", "Additional control"),
        ("pce_cumulative_since_jan2010_pct", "Monthly", "Cumulative PCE price increase since January 2010", "percent", "Additional price-level control"),
        ("u6_underutilization_pct_sa", "Monthly", "U-6 labor underutilization rate, seasonally adjusted", "percent", "Alternative labor control"),
        ("u6_minus_u3_pct", "Derived", "U-6 less U-3 labor slack spread", "percentage points", "Additional labor stress control"),
        (PRIMARY_MEDIA, "Monthly", "DeepSeek monthly overall news tone index", "-100 to 100", "Primary media variable; higher is more positive"),
        (SECONDARY_MEDIA, "Monthly", "T5 monthly overall news tone index", "-100 to 100", "Robustness media variable; higher is more positive"),
        ("gdp_qoq_available", "Derived", "Latest revised quarterly GDP growth available under proxy release calendar", "percent SAAR", "Pseudo-forecast only; no real-time vintage"),
        ("media_deepseek_available", "Derived", "Prior-calendar-month DeepSeek tone", "-100 to 100", "Forecast cutoff information"),
        ("sentiment_gap", "Derived", "Observed sentiment minus economic-only held-out forecast", "index points", "Negative means more pessimistic than model prediction"),
    ]
    return pd.DataFrame(rows, columns=["variable", "source_sheet", "definition", "units", "analysis_use"])


def svg_line_chart(
    path: Path,
    x_labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    title: str,
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
    band: tuple[Sequence[float], Sequence[float], str] | None = None,
) -> None:
    width, height = 1500, 720
    left, right, top, bottom = 105, 45, 78, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [float(v) for _, values, _ in series for v in values if pd.notna(v)]
    if band:
        all_values.extend(float(v) for v in band[0] if pd.notna(v))
        all_values.extend(float(v) for v in band[1] if pd.notna(v))
    lo = min(all_values) if y_min is None else y_min
    hi = max(all_values) if y_max is None else y_max
    padding = max((hi - lo) * 0.08, 1)
    lo = lo - padding if y_min is None else lo
    hi = hi + padding if y_max is None else hi
    xp = lambda i: left + plot_w * i / max(1, len(x_labels) - 1)
    yp = lambda v: top + plot_h * (hi - float(v)) / max(1e-12, hi - lo)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFAF7"/>',
        f'<text x="{left}" y="40" font-family="Arial" font-size="26" font-weight="700" fill="#1F2937">{html.escape(title)}</text>',
    ]
    for tick in np.linspace(lo, hi, 6):
        yy = yp(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#D1D5DB"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#4B5563">{tick:.1f}</text>')
    if lo <= 0 <= hi:
        yy = yp(0)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#6B7280" stroke-width="1.2"/>')
    if band:
        low, high, color = band
        upper = [(xp(i), yp(v)) for i, v in enumerate(high) if pd.notna(v)]
        lower = [(xp(i), yp(v)) for i, v in reversed(list(enumerate(low))) if pd.notna(v)]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
        parts.append(f'<polygon points="{points}" fill="{color}" opacity="0.16"/>')
    for s_no, (name, values, color) in enumerate(series):
        points = " ".join(
            f"{xp(i):.1f},{yp(v):.1f}" for i, v in enumerate(values) if pd.notna(v)
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        lx = left + 18 + s_no * 250
        parts.append(f'<line x1="{lx}" y1="{top+18}" x2="{lx+38}" y2="{top+18}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+48}" y="{top+23}" font-family="Arial" font-size="14" fill="#374151">{html.escape(name)}</text>')
    step = max(1, len(x_labels) // 10)
    for i in range(0, len(x_labels), step):
        parts.append(f'<text x="{xp(i):.1f}" y="{top+plot_h+28}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4B5563">{html.escape(x_labels[i])}</text>')
    parts.append(f'<text x="25" y="{top+plot_h/2}" transform="rotate(-90 25 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="15" fill="#374151">{html.escape(y_label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_forest(path: Path, media: pd.DataFrame) -> None:
    plot = media[media["test_type"] == "single coefficient"].copy()
    plot = plot.sort_values(["primary_hypothesis", "raw_p_value"], ascending=[False, True]).head(18)
    width = 1500
    height = 90 + 38 * len(plot)
    left, right, top, bottom = 420, 50, 65, 45
    values = np.r_[plot["ci_95_low"].to_numpy(), plot["ci_95_high"].to_numpy(), 0]
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * 0.08, 0.02)
    lo, hi = lo - pad, hi + pad
    xp = lambda v: left + (width - left - right) * (float(v) - lo) / (hi - lo)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFAF7"/>',
        f'<text x="30" y="36" font-family="Arial" font-size="25" font-weight="700" fill="#1F2937">Media coefficients and HAC 95% intervals</text>',
        f'<line x1="{xp(0):.1f}" y1="{top}" x2="{xp(0):.1f}" y2="{height-bottom}" stroke="#6B7280"/>',
    ]
    for row_no, (_, row) in enumerate(plot.iterrows()):
        y = top + 22 + row_no * 38
        color = "#DC2626" if row["primary_hypothesis"] else "#2563EB"
        label = f"{row['model_id']} ({row['outcome']})"
        parts.append(f'<text x="{left-15}" y="{y+5}" text-anchor="end" font-family="Arial" font-size="13" fill="#374151">{html.escape(label)}</text>')
        parts.append(f'<line x1="{xp(row.ci_95_low):.1f}" y1="{y}" x2="{xp(row.ci_95_high):.1f}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{xp(row.estimate):.1f}" cy="{y}" r="5" fill="{color}"/>')
    for tick in np.linspace(lo, hi, 7):
        parts.append(f'<text x="{xp(tick):.1f}" y="{height-15}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4B5563">{tick:.3f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_quality_report(path: Path, quality: dict, analysis: pd.DataFrame) -> None:
    missing_lines = "\n".join(
        f"- `{key}`: {', '.join(value) if value else 'none'}"
        for key, value in quality["missing_by_variable"].items()
    ) or "- No missing values in reported source variables."
    text = f"""# Data quality and timing report

## Coverage and structure

- Monthly coverage: {quality['month_start']} to {quality['month_end']} ({quality['monthly_rows']} rows).
- Quarterly GDP rows: {quality['quarterly_rows']}.
- Duplicate months: {quality['duplicate_months']}.
- Invalid dates: {quality['invalid_months']}.
- Missing calendar months: {quality['missing_calendar_months'] or 'none'}.
- Higher media-index values indicate more positive overall news tone. The index is not a validated measure of media bias.

## Missing source values

{missing_lines}

Missing values are never replaced with zero. The complete calendar is retained before lags and differences are constructed.

## Timing assumptions

Association models are retrospective. Full-month media tone can include articles published after some survey interviews in the same month. Current-quarter GDP is repeated within its quarter without interpolation.

Forecasts use the last calendar day of month t-1 as the cutoff. Prior-month media and prior-month Michigan sentiment are treated as known. U-3, PCE inflation, real income growth, and the cumulative price level are lagged two calendar months. GDP uses a proxy release date equal to the last day of the month after quarter end and is held until the next proxy release.

{quality['gdp_timing_limitation']} The forecast exercise is therefore pseudo-out-of-sample using revised data, not a real-time backtest.

## Media zero limitation

{quality['media_zero_rule']}

## Survey-method limitation

The workbook notes a transition in the Michigan survey toward web collection but does not provide a dated method-change series. No unverified break dummy is introduced.
"""
    path.write_text(text, encoding="utf-8")


def write_findings(
    path: Path,
    coefficients: pd.DataFrame,
    media: pd.DataFrame,
    hac: pd.DataFrame,
    huber: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    forecast_tests: pd.DataFrame,
    interval_info: list[dict],
    stability: pd.DataFrame,
) -> None:
    primary = media.loc[media["primary_hypothesis"]].iloc[0]
    primary_sig = primary.raw_p_value < 0.05
    m3 = media.loc[media["model_id"] == "H_M3_controls"].iloc[0]
    dynamic = media.loc[media["model_id"] == "H_DYN_M2_media"].iloc[0]
    diff = media.loc[media["model_id"] == "H_DIFF_M2"].iloc[0]
    t5 = media.loc[media["model_id"] == "H_T5_M2"].iloc[0]
    exploratory_media = media.loc[~media["primary_hypothesis"]]
    bh_count = int(exploratory_media["significant_bh_5pct"].fillna(False).sum())
    raw_count = int(exploratory_media["significant_raw_5pct"].sum())
    econ_coeffs = coefficients[
        (coefficients["model_id"] == "H_M3_controls")
        & coefficients["term"].isin(ECON + CONTROLS)
    ].copy()
    significant_econ = econ_coeffs[econ_coeffs["p_value"] < 0.05]
    econ_text = (
        ", ".join(
            f"{row.term} (coefficient={row.estimate:.3f}, p={row.p_value:.4g})"
            for row in significant_econ.itertuples()
        )
        if len(significant_econ)
        else "none at the 5% level in the full-control model"
    )
    headline_metrics = forecast_metrics[forecast_metrics["outcome"] == OUTCOMES[0]].set_index("model")
    ols_econ = headline_metrics.loc["ols_econ"]
    ols_media = headline_metrics.loc["ols_media"]
    media_loss = forecast_tests[
        (forecast_tests["outcome"] == OUTCOMES[0])
        & (forecast_tests["comparison"] == "ols_media versus ols_econ")
    ].iloc[0]
    coverage = next(x for x in interval_info if x["outcome"] == OUTCOMES[0])
    conclusion = (
        "Media tone is statistically significant in the prespecified primary association model."
        if primary_sig
        else "There is insufficient evidence of a media-tone association in the prespecified primary model."
    )
    text = f"""# Monthly media sentiment regression findings

## Bottom line

{conclusion} The primary DeepSeek coefficient is {primary.estimate:.4f} Michigan-sentiment points per one media-index point (HAC(3) SE {primary.std_error:.4f}, p={primary.raw_p_value:.6f}, 95% CI [{primary.ci_95_low:.4f}, {primary.ci_95_high:.4f}]). A one-standard-deviation increase in media tone is associated with {primary.effect_of_1sd_media:.2f} index points in this retrospective specification.

This is an observational association. It is not a causal effect and does not establish media bias.

## Answers to the research questions

1. **Economic variables.** In the full-control headline model, significant economic coefficients are: {econ_text}. Coefficients are conditional associations and are sensitive to persistence and nonstationarity.

2. **Media significance after economic controls.** Primary model p={primary.raw_p_value:.6f}. With income growth and cumulative price-level controls, p={m3.raw_p_value:.6f}. The analysis reports all coefficients, not only significant terms.

3. **Robustness.** The dynamic model with lagged sentiment has p={dynamic.raw_p_value:.6f}; the first-difference model has p={diff.raw_p_value:.6f}; the alternate T5 index has p={t5.raw_p_value:.6f}. Huber block-bootstrap p={float(huber.raw_p_value.iloc[0]):.6f}. Across {len(exploratory_media)} exploratory OLS tests, {raw_count} are raw-significant and {bh_count} survive Benjamini-Hochberg correction. This is the relevant evidence against specification-driven cherry-picking. Pre/post-2020 estimates are reported in the stability table.

4. **Out-of-sample prediction.** Headline OLS economic-only RMSE is {ols_econ.rmse:.3f}; adding media produces RMSE {ols_media.rmse:.3f}, an improvement of {ols_media.rmse_improvement_vs_econ_pct:.2f}%. The moving-block bootstrap mean squared-loss improvement is {media_loss.mean_squared_loss_improvement:.3f}, 95% CI [{media_loss.bootstrap_ci_low:.3f}, {media_loss.bootstrap_ci_high:.3f}], p={media_loss.bootstrap_p_value:.4f}. A significant in-sample coefficient does not by itself imply forecast improvement.

5. **Headline versus subindexes.** Separate coefficient and forecast tables cover the headline index, Current Economic Conditions, and Consumer Expectations. Their signs and significance should be compared using the complete tables rather than a single selected outcome.

6. **Evidence weakening a media explanation.** Dynamic controls absorb persistent sentiment; first-difference estimates test whether month-to-month media changes align with sentiment changes; HAC(6/12), T5, Huber, pre/post-2020, and out-of-sample results show how dependent the finding is on specification and measurement. The difference between raw and BH-adjusted significance quantifies multiple-testing risk.

7. **Why this is not causal.** Media tone and consumer sentiment may respond jointly to omitted events, survey timing overlaps the article month, the media index is model-generated, and economic series are revised. No instrument, natural experiment, or validated exposure measure is available.

## Forecast and sentiment-gap limitations

The forecast is pseudo-out-of-sample using revised data because release dates and real-time vintages are absent. The headline economic-only prediction interval uses only earlier one-step errors and covers {coverage['coverage']:.1%} of {coverage['n_forecasts']} holdout months. The sentiment gap equals observed sentiment minus the held-out economic-only prediction; a negative gap means unexplained pessimism under that model, not irrationality or bias.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    analysis, forecast, sources, quality = load_and_align(args.workbook)
    specs = build_specs()
    coefficients, model_summary, media, results = regression_tables(analysis, specs)
    hac = hac_sensitivity(analysis, specs)
    huber, huber_info = huber_bootstrap(analysis, args.bootstrap_reps)

    primary_result = results["H_M2_econ_media"]
    diag, residual_acf, vif, influence = diagnostics(analysis, primary_result)
    stationarity = stationarity_table(analysis)
    stability, rolling = stability_table(analysis)

    all_forecasts = []
    all_metrics = []
    all_forecast_tests = []
    intervals = []
    lag_map = {
        OUTCOMES[0]: "sentiment_lag1",
        OUTCOMES[1]: "current_conditions_lag1",
        OUTCOMES[2]: "expectations_lag1",
    }
    for outcome in OUTCOMES:
        f_results, f_metrics, f_tests, f_interval = forecast_outcome(
            forecast, outcome, lag_map[outcome], args.bootstrap_reps
        )
        all_forecasts.append(f_results)
        all_metrics.append(f_metrics)
        all_forecast_tests.append(f_tests)
        intervals.append(f_interval)
    forecast_results = pd.concat(all_forecasts, ignore_index=True)
    forecast_metrics = pd.concat(all_metrics, ignore_index=True)
    forecast_tests = pd.concat(all_forecast_tests, ignore_index=True)
    gap = forecast_results[forecast_results["outcome"] == OUTCOMES[0]][
        [
            "month",
            "observed",
            "ols_econ",
            "sentiment_gap",
            "econ_prediction_interval_low",
            "econ_prediction_interval_high",
            "interval_covered",
        ]
    ].copy()

    # Primary/economic control sample flags in the aligned dataset.
    analysis["association_primary_sample"] = analysis[
        [OUTCOMES[0], *ECON, PRIMARY_MEDIA]
    ].notna().all(axis=1)
    analysis["association_full_control_sample"] = analysis[
        [OUTCOMES[0], *ECON, PRIMARY_MEDIA, *CONTROLS]
    ].notna().all(axis=1)
    forecast_columns = [
        "month",
        "forecast_cutoff",
        "gdp_underlying_quarter",
        "gdp_release_date_proxy",
        "gdp_qoq_available",
        "u3_available",
        "pce_inflation_available",
        "real_income_yoy_available",
        "price_level_available",
        "media_deepseek_available",
        "media_t5_available",
    ]
    aligned = analysis.merge(
        forecast[forecast_columns], on="month", how="left", validate="1:1"
    )

    dictionary = variable_dictionary()
    coefficients.to_csv(out / "regression_coefficients.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(out / "model_summary.csv", index=False, encoding="utf-8-sig")
    media.to_csv(out / "media_significance_summary.csv", index=False, encoding="utf-8-sig")
    hac.to_csv(out / "hac_sensitivity.csv", index=False, encoding="utf-8-sig")
    huber.to_csv(out / "huber_robust_regression.csv", index=False, encoding="utf-8-sig")
    diag.to_csv(out / "model_diagnostics.csv", index=False, encoding="utf-8-sig")
    residual_acf.to_csv(out / "residual_acf.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(out / "vif.csv", index=False, encoding="utf-8-sig")
    influence.to_csv(out / "influential_observations.csv", index=False, encoding="utf-8-sig")
    stationarity.to_csv(out / "stationarity_diagnostics.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(out / "coefficient_stability.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(out / "rolling_media_coefficients.csv", index=False, encoding="utf-8-sig")
    forecast_results.to_csv(out / "forecast_results.csv", index=False, encoding="utf-8-sig")
    forecast_metrics.to_csv(out / "forecast_metrics.csv", index=False, encoding="utf-8-sig")
    forecast_tests.to_csv(out / "forecast_loss_tests.csv", index=False, encoding="utf-8-sig")
    gap.to_csv(out / "sentiment_gap.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(out / "variable_dictionary.csv", index=False, encoding="utf-8-sig")
    aligned.to_csv(out / "aligned_analysis_dataset.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "random_seed": SEED,
        "bootstrap_reps": args.bootstrap_reps,
        "primary_media_selected_before_estimation": PRIMARY_MEDIA,
        "primary_model": "H_M2_econ_media",
        "primary_hac_bandwidth": 3,
        "hac_sensitivity_bandwidths": [6, 12],
        "control_selection": {
            "required": ECON,
            "additional_primary": CONTROLS,
            "alternatives": ["u6_underutilization_pct_sa", "u6_minus_u3_pct"],
            "selection_rule": "economic motivation declared before estimation, not significance",
        },
        "forecast_intervals": intervals,
        "quality": quality,
        "huber": huber_info,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_quality_report(out / "data_quality_report.md", quality, analysis)
    write_findings(
        out / "findings.md",
        coefficients,
        media,
        hac,
        huber,
        forecast_metrics,
        forecast_tests,
        intervals,
        stability,
    )

    x_labels = analysis["month"].dt.strftime("%Y-%m").tolist()
    svg_line_chart(
        out / "time_series_overview.svg",
        x_labels,
        [
            ("Michigan sentiment", analysis[OUTCOMES[0]].tolist(), "#1D4ED8"),
            ("DeepSeek media tone", analysis[PRIMARY_MEDIA].tolist(), "#D97706"),
        ],
        "Consumer sentiment and media tone",
        "Index value",
    )
    svg_forest(out / "media_coefficients.svg", media)
    headline_forecast = forecast_results[forecast_results["outcome"] == OUTCOMES[0]].copy()
    forecast_labels = headline_forecast["month"].dt.strftime("%Y-%m").tolist()
    svg_line_chart(
        out / "observed_vs_predicted.svg",
        forecast_labels,
        [
            ("Observed", headline_forecast["observed"].tolist(), "#111827"),
            ("Economic only", headline_forecast["ols_econ"].tolist(), "#2563EB"),
            ("Economic + media", headline_forecast["ols_media"].tolist(), "#D97706"),
        ],
        "Held-out headline sentiment forecasts",
        "Michigan sentiment index",
        band=(
            headline_forecast["econ_prediction_interval_low"].tolist(),
            headline_forecast["econ_prediction_interval_high"].tolist(),
            "#2563EB",
        ),
    )
    svg_line_chart(
        out / "sentiment_gap.svg",
        forecast_labels,
        [("Observed − economic forecast", headline_forecast["sentiment_gap"].tolist(), "#7C3AED")],
        "Held-out sentiment gap",
        "Index points",
    )
    svg_line_chart(
        out / "rolling_media_coefficient.svg",
        rolling["window_end"].dt.strftime("%Y-%m").tolist(),
        [("60-month media coefficient", rolling["media_estimate"].tolist(), "#0F766E")],
        "Rolling 60-month media coefficient",
        "Coefficient",
        band=(rolling["ci_95_low"].tolist(), rolling["ci_95_high"].tolist(), "#0F766E"),
    )
    svg_line_chart(
        out / "residual_acf.svg",
        residual_acf["lag"].astype(str).tolist(),
        [("Residual ACF", residual_acf["residual_acf"].tolist(), "#DC2626")],
        "Primary-model residual autocorrelation",
        "Autocorrelation",
        y_min=-1,
        y_max=1,
    )

    primary = media.loc[media["primary_hypothesis"]].iloc[0]
    print(
        json.dumps(
            {
                "primary_media_estimate": primary["estimate"],
                "primary_media_p_value": primary["raw_p_value"],
                "primary_media_ci": [primary["ci_95_low"], primary["ci_95_high"]],
                "primary_media_1sd_effect": primary["effect_of_1sd_media"],
                "raw_significant_exploratory": int(
                    media.loc[~media["primary_hypothesis"], "significant_raw_5pct"].sum()
                ),
                "bh_significant_exploratory": int(media["significant_bh_5pct"].sum()),
                "output_dir": str(out.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
