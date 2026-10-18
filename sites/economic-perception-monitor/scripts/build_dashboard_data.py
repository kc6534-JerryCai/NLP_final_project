"""Build the static data bundle for the Economic Perception Monitor.

The script intentionally uses only the Python standard library so the dashboard can be
refreshed from the repository's analysis outputs without installing a web framework.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SITE_ROOT.parents[1]
DIST = SITE_ROOT / "dist"


def as_float(value: str | None):
    if value is None or value == "":
        return None
    return float(value)


def as_int(value: str | None):
    if value is None or value == "":
        return None
    return int(float(value))


def as_bool(value: str | None):
    if value is None or value == "":
        return None
    return value.lower() == "true"


def month_key(value: str) -> str:
    return value[:7]


def read_csv(relative_path: str):
    path = REPO_ROOT / relative_path
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(relative_path: str):
    with (REPO_ROOT / relative_path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


timeline = []
for row in read_csv("outputs/monthly_regression_analysis/aligned_analysis_dataset.csv"):
    timeline.append(
        {
            "month": month_key(row["month"]),
            "sentiment": as_float(row["michigan_sentiment"]),
            "current": as_float(row["michigan_current_conditions"]),
            "expectations": as_float(row["michigan_expectations"]),
            "surveyStatus": row["survey_status"] or None,
            "unemployment": as_float(row["u3_unemployment_pct_sa"]),
            "pceInflation": as_float(row["pce_inflation_yoy_pct"]),
            "gdpQoq": as_float(row["real_gdp_qoq_annualized_official_pct"]),
            "gdpYoy": as_float(row["real_gdp_yoy_pct"]),
            "toneIndex": as_float(row["news_tone_index_deepseek"]),
            "realIncomeYoy": as_float(row["real_income_yoy_pct"]),
            "cumulativePrices": as_float(row["pce_cumulative_since_jan2010_pct"]),
        }
    )


forecast_fields = {
    "observed": "observed",
    "econ": "ols_econ",
    "media": "ols_media",
    "full": "ols_full",
    "low": "econ_prediction_interval_low",
    "high": "econ_prediction_interval_high",
    "gap": "sentiment_gap",
}
forecasts = defaultdict(list)
for row in read_csv("outputs/monthly_regression_analysis/forecast_results.csv"):
    item = {"month": month_key(row["month"])}
    for output_name, input_name in forecast_fields.items():
        item[output_name] = as_float(row[input_name])
    item["covered"] = as_bool(row["interval_covered"])
    forecasts[row["outcome"]].append(item)


forecast_metrics = []
for row in read_csv("outputs/monthly_regression_analysis/forecast_metrics.csv"):
    forecast_metrics.append(
        {
            "outcome": row["outcome"],
            "model": row["model"],
            "method": row["method_family"],
            "n": as_int(row["n_forecasts"]),
            "mae": as_float(row["mae"]),
            "rmse": as_float(row["rmse"]),
            "improvement": as_float(row["rmse_improvement_vs_econ_pct"]),
            "start": month_key(row["evaluation_start"]),
            "end": month_key(row["evaluation_end"]),
        }
    )


topics = [
    "cost_of_living_income",
    "labor_market",
    "household_economy",
    "macro_growth",
    "economic_policy",
]
frames = [
    "household_hardship",
    "economic_resilience",
    "unequal_outcomes",
    "macro_household_disconnect",
    "future_risk",
]
calibration_states = ["consistent", "mixed", "inconsistent", "not_assessable"]
media_rollup = defaultdict(
    lambda: {
        "n": 0,
        "tone": Counter(),
        "topics": Counter(),
        "frames": Counter(),
        "calibration": Counter(),
    }
)
delivery_path = REPO_ROOT / "outputs/all_relevant/delivery/all_relevant_annotations.csv"
with delivery_path.open("r", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["deepseek_relevance"] != "relevant":
            continue
        month = row["month"]
        bucket = media_rollup[month]
        bucket["n"] += 1
        if row["deepseek_tone"]:
            bucket["tone"][row["deepseek_tone"]] += 1
        if row["deepseek_primary_topic"]:
            bucket["topics"][row["deepseek_primary_topic"]] += 1
        raw_frames = row["deepseek_frames"].replace(",", ";")
        for frame in {part.strip() for part in raw_frames.split(";") if part.strip()}:
            bucket["frames"][frame] += 1
        if row["factual_calibration"]:
            bucket["calibration"][row["factual_calibration"]] += 1


media = []
for month in sorted(media_rollup):
    bucket = media_rollup[month]
    n = bucket["n"]
    positive = bucket["tone"]["1"]
    negative = bucket["tone"]["-1"]
    neutral = bucket["tone"]["0"]
    assessed = sum(bucket["calibration"][key] for key in calibration_states[:3])
    media.append(
        {
            "month": month,
            "nRelevant": n,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "toneIndex": 100 * (positive - negative) / n if n else None,
            "negativeShare": negative / n if n else None,
            "topicShares": {key: bucket["topics"][key] / n for key in topics},
            "frameShares": {key: bucket["frames"][key] / n for key in frames},
            "calibration": {
                key: bucket["calibration"][key] for key in calibration_states
            },
            "calibrationAssessableShare": assessed / n if n else None,
            "calibrationConsistentShare": (
                bucket["calibration"]["consistent"] / assessed if assessed else None
            ),
        }
    )


coverage = []
for row in read_csv("data/reports/monthly_source_coverage.csv"):
    coverage.append(
        {
            "month": row["month"],
            "source": row["source"],
            "total": as_int(row["total_articles"]),
            "fullBodyRate": as_float(row["full_body_rate"]),
            "ruleRelevant": as_int(row["relevant"]),
            "uncertain": as_int(row["uncertain"]),
            "incompleteIntervals": as_int(row["incomplete_intervals"]),
        }
    )


rare_events = []
seen_release_events = set()
for row in read_csv("outputs/threshold_iv_analysis/threshold_event_table.csv"):
    if (
        row["threshold_family"] != "integer"
        or row["rarity_months"] != "24"
        or row["qualifies_rarity"] != "True"
        or row["release_month"] < "2023-01-01"
    ):
        continue
    event_key = (row["indicator"], row["release_month"], row["direction"])
    if event_key in seen_release_events:
        continue
    seen_release_events.add(event_key)
    label = (
        "U-3 unemployment"
        if row["indicator"] == "u3_unemployment_pct_sa"
        else "Headline PCE inflation"
    )
    rare_events.append(
        {
            "month": month_key(row["release_month"]),
            "date": row["release_date"] or row["release_month"],
            "type": "labor" if row["indicator"].startswith("u3") else "prices",
            "title": f"{label} crossed {row['threshold']}% {row['direction']}",
            "detail": f"First-release value moved from {row['previous_value']}% to {row['current_value']}%.",
            "revisionRisk": as_bool(row["revision_misclassification_risk"]),
            "url": row["source_url"] or None,
        }
    )


regression_coefficients = read_csv(
    "outputs/monthly_regression_analysis/regression_coefficients.csv"
)


def coefficient(model_id: str, term: str):
    row = next(
        item
        for item in regression_coefficients
        if item["model_id"] == model_id and item["term"] == term
    )
    return {
        "estimate": as_float(row["estimate"]),
        "se": as_float(row["hac_std_error"]),
        "p": as_float(row["p_value"]),
        "low": as_float(row["ci_95_low"]),
        "high": as_float(row["ci_95_high"]),
        "n": as_int(row["nobs"]),
    }


run_metadata = read_json("outputs/monthly_regression_analysis/run_metadata.json")
delivery_summary = read_json("outputs/all_relevant/delivery/summary.json")
t5_results = read_json("outputs/finetuning_compact_v2/t5/partial_experiment_results.json")
collection_quality = (REPO_ROOT / "data/reports/collection_quality.md").read_text(
    encoding="utf-8-sig"
)


def markdown_metric(prefix: str):
    for line in collection_quality.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


latest = timeline[-1]
latest_media = media[-1]
headline_forecasts = forecasts["michigan_sentiment"]
latest_forecast = headline_forecasts[-1]
headline_ols_econ = next(
    row
    for row in forecast_metrics
    if row["outcome"] == "michigan_sentiment" and row["model"] == "ols_econ"
)
headline_ols_media = next(
    row
    for row in forecast_metrics
    if row["outcome"] == "michigan_sentiment" and row["model"] == "ols_media"
)


dashboard = {
    "meta": {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "analysisStart": timeline[0]["month"],
        "analysisEnd": timeline[-1]["month"],
        "collectionCutoff": markdown_metric("- Requested cutoff date"),
        "latestPublication": markdown_metric("- Latest publication timestamp returned"),
        "sourceScope": "New York Times headline, abstract, snippet, and lead metadata",
        "surveyNote": "October 2026 Michigan values are preliminary.",
    },
    "latest": {
        "sentiment": latest["sentiment"],
        "sentimentStatus": latest["surveyStatus"],
        "econPrediction": latest_forecast["econ"],
        "predictionLow": latest_forecast["low"],
        "predictionHigh": latest_forecast["high"],
        "gap": latest_forecast["gap"],
        "toneIndex": latest_media["toneIndex"],
        "negativeShare": latest_media["negativeShare"],
        "mediaN": latest_media["nRelevant"],
        "unemployment": latest["unemployment"],
        "pceInflation": latest["pceInflation"],
        "gdpQoq": latest["gdpQoq"],
    },
    "timeline": timeline,
    "forecasts": dict(forecasts),
    "forecastMetrics": forecast_metrics,
    "media": media,
    "coverage": coverage,
    "events": rare_events,
    "findings": {
        "primaryMedia": coefficient("H_M2_econ_media", "news_tone_index_deepseek"),
        "fullControlMedia": coefficient("H_M3_controls", "news_tone_index_deepseek"),
        "econRmse": headline_ols_econ["rmse"],
        "mediaRmse": headline_ols_media["rmse"],
        "mediaRmseImprovement": headline_ols_media["improvement"],
        "intervalCoverage": run_metadata["forecast_intervals"][0]["coverage"],
        "ivEstimate": 1.506,
        "ivP": 0.7103,
        "andersonRubinP": 0.4208,
        "causalConclusion": "Unable to identify reliably",
    },
    "aiPerformance": {
        "testN": t5_results["test"]["primary_topic"]["effective_n"],
        "topicAccuracy": t5_results["test"]["primary_topic"]["accuracy"],
        "topicMacroF1": t5_results["test"]["primary_topic"]["macro_f1"],
        "toneAccuracy": t5_results["test"]["overall_tone"]["accuracy"],
        "toneMacroF1": t5_results["test"]["overall_tone"]["macro_f1"],
        "gptLabelCaveat": t5_results["gpt_label_caveat"],
        "modelTopicAgreement": delivery_summary[
            "t5_deepseek_topic_agreement_on_deepseek_relevant"
        ]["rate"],
        "modelToneAgreement": delivery_summary[
            "t5_deepseek_tone_agreement_on_deepseek_relevant"
        ]["rate"],
        "humanValidation": "Pending - no two-annotator, 300-item gold sample found in current outputs.",
    },
    "corpus": {
        "cleanedArticles": as_int(markdown_metric("- Cleaned unique articles")),
        "ruleRelevant": delivery_summary["total_rule_screened_relevant"],
        "modelRelevant": sum(delivery_summary["deepseek_primary_topic_relevant_only"].values()),
        "sources": [{"name": "The New York Times", "share": 1.0}],
        "socialPlatforms": [],
        "fullBodyRate": 0,
    },
    "limitations": [
        "Association is not causation: media tone and sentiment can respond to the same shocks.",
        "The current corpus contains one national news source and no social-media platform.",
        "Historical macroeconomic series are revised, not a complete real-time vintage panel.",
        "Media text is metadata-level, not verified full article bodies or measured exposure.",
        "Human two-annotator validation remains outstanding; current classifier metrics use GPT-generated labels.",
        "October 2026 media coverage is partial through October 4 despite a collection cutoff of October 14.",
    ],
}


# Reconcile our monthly aggregation with the published DeepSeek tone table.
published_tone = {
    row["month"]: as_float(row["tone_index"])
    for row in read_csv("outputs/monthly_tone_index/monthly_tone_index_deepseek.csv")
}
max_tone_difference = max(
    abs(item["toneIndex"] - published_tone[item["month"]])
    for item in media
    if published_tone.get(item["month"]) is not None
)
if max_tone_difference > 1e-9:
    raise RuntimeError(f"Tone aggregation mismatch: {max_tone_difference}")


DIST.mkdir(parents=True, exist_ok=True)
payload = json.dumps(dashboard, ensure_ascii=False, separators=(",", ":"))
(DIST / "data.js").write_text(
    "window.DASHBOARD_DATA=" + payload + ";\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            "output": str(DIST / "data.js"),
            "timeline_rows": len(timeline),
            "forecast_rows": sum(len(rows) for rows in forecasts.values()),
            "media_months": len(media),
            "tone_reconciliation_max_abs_diff": max_tone_difference,
        },
        indent=2,
    )
)
