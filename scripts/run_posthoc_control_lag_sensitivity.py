#!/usr/bin/env python
"""Post-hoc control-variable and timing sensitivity for expanded thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_threshold_iv_analysis import (
    BASE_CONTROLS,
    OUTCOMES,
    PRIMARY_MEDIA,
    add_event_instruments,
    construct_data,
    run_specs,
)


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
        "--output-dir",
        type=Path,
        default=Path("outputs/threshold_iv_analysis/posthoc_expanded_grid"),
    )
    return parser.parse_args()


def build_specs() -> list[dict]:
    instruments = ["z_u3_signed_r24", "z_pcegrid_up_r24", "z_pcegrid_down_r24"]
    month_fe = [f"calendar_month_{month:02d}" for month in range(2, 13)]
    shared = {
        "media": PRIMARY_MEDIA,
        "z": instruments,
        "hac": 3,
        "family": "posthoc_control_timing",
        "primary": False,
    }
    return [
        {"id": "BASE_CONTEMP", "outcome": OUTCOMES[0], "controls": BASE_CONTROLS, **shared},
        {
            "id": "DROP_GDP",
            "outcome": OUTCOMES[0],
            "controls": [c for c in BASE_CONTROLS if c != "gdp_latest_release_value"],
            **shared,
        },
        {
            "id": "LEVELS_ONLY",
            "outcome": OUTCOMES[0],
            "controls": ["u3_release_value", "pce_release_value", "gdp_latest_release_value", "time_trend"],
            **shared,
        },
        {
            "id": "CHANGES_ONLY",
            "outcome": OUTCOMES[0],
            "controls": ["d_u3_release_value", "d_pce_release_value", "gdp_latest_release_value", "time_trend"],
            **shared,
        },
        {
            "id": "NO_TREND",
            "outcome": OUTCOMES[0],
            "controls": [c for c in BASE_CONTROLS if c != "time_trend"],
            **shared,
        },
        {
            "id": "QUADRATIC_MACRO",
            "outcome": OUTCOMES[0],
            "controls": [*BASE_CONTROLS, "u3_release_value_sq", "pce_release_value_sq"],
            **shared,
        },
        {
            "id": "QUADRATIC_TREND",
            "outcome": OUTCOMES[0],
            "controls": [*BASE_CONTROLS, "time_trend_sq"],
            **shared,
        },
        {
            "id": "CALENDAR_MONTH_FE",
            "outcome": OUTCOMES[0],
            "controls": [*BASE_CONTROLS, *month_fe],
            **shared,
        },
        {
            "id": "LAGGED_OUTCOME",
            "outcome": OUTCOMES[0],
            "controls": [f"{OUTCOMES[0]}_lag1", *BASE_CONTROLS],
            **shared,
        },
        {
            "id": "OUTCOME_LEAD1",
            "outcome": f"{OUTCOMES[0]}_lead1",
            "controls": BASE_CONTROLS,
            **shared,
        },
        {
            "id": "OUTCOME_LEAD2",
            "outcome": f"{OUTCOMES[0]}_lead2",
            "controls": BASE_CONTROLS,
            **shared,
        },
        {
            "id": "OUTCOME_CHANGE",
            "outcome": f"d_{OUTCOMES[0]}",
            "controls": BASE_CONTROLS,
            **shared,
        },
    ]


def write_report(summary: pd.DataFrame, first: pd.DataFrame, output: Path) -> None:
    estimated = summary.loc[summary["status"] == "estimated"].copy()
    ordered = estimated.sort_values("iv_media_p_value")
    best = ordered.iloc[0]
    significant = ordered.loc[ordered["iv_media_p_value"] < 0.05]
    if significant.empty:
        scan = "No conventional 2SLS estimate reaches p<0.05."
    else:
        scan = "\n".join(
            f"- {row.model_id}: coefficient={row.iv_media_estimate:.3f}, p={row.iv_media_p_value:.4f}, "
            f"BH q={row.iv_bh_q_value:.4f}, AR p={row.anderson_rubin_beta0_p_value:.4f}, "
            f"AR BH q={row.ar_bh_q_value:.4f}."
            for row in significant.itertuples()
        )
    base_first = first.loc[first["model_id"] == "BASE_CONTEMP"]
    first_lines = "\n".join(
        f"- {row.instrument}: coefficient={row.estimate:.3f}, HAC p={row.p_value:.4g}."
        for row in base_first.itertuples()
    )
    text = f"""# Post-hoc control and timing sensitivity results

This analysis was requested after the expanded-threshold results were observed. The fixed instrument set is the signed U-3 rare-event instrument plus separate PCE upward and downward expanded-grid instruments.

## Best conventional p-value in the fixed menu

- Model: {best['model_id']}.
- Outcome: {best['outcome']}.
- 2SLS coefficient={best['iv_media_estimate']:.3f}, HAC p={best['iv_media_p_value']:.4f}, 95% CI [{best['iv_ci_95_low']:.3f}, {best['iv_ci_95_high']:.3f}].
- BH-adjusted q={best['iv_bh_q_value']:.4f} across the twelve conventional tests.
- First-stage joint p={best['first_stage_joint_p_value']:.4g}; F-type={best['first_stage_wald_f_type']:.3f}; partial R-squared={best['first_stage_partial_r_squared']:.4f}.
- Anderson-Rubin p={best['anderson_rubin_beta0_p_value']:.4f}; BH-adjusted AR q={best['ar_bh_q_value']:.4f}.

## Significance scan

{scan}

## Baseline first-stage directions

{first_lines}

## Interpretation

These models vary controls or timing after earlier results were inspected. An isolated raw p-value is not a confirmatory result. The PCE 2% threshold remains part of the expanded grid and may directly affect policy expectations and consumer confidence. Lagged-outcome models additionally require sequential exogeneity, and HAC standard errors do not solve lagged-dependent-variable endogeneity.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    analysis, all_events, _, _, _ = construct_data(
        args.workbook, args.articles, args.release_data
    )
    grid = np.arange(2.0, 7.0 + 0.001, 0.5)
    expanded = all_events.loc[
        (all_events["indicator"] == "pce_inflation_yoy_pct")
        & all_events["threshold"].astype(float).round(1).isin(grid.round(1))
    ].copy()
    expanded = expanded.drop_duplicates(
        ["threshold", "rarity_months", "release_month", "direction"]
    )
    analysis = add_event_instruments(analysis, expanded, "pcegrid", 24)
    analysis["time_trend_sq"] = analysis["time_trend"] ** 2
    analysis[f"{OUTCOMES[0]}_lead2"] = analysis[OUTCOMES[0]].shift(-2)

    specs = build_specs()
    summary, first, reduced, _ = run_specs(analysis, specs)
    summary.to_csv(out / "control_lag_iv_model_summary.csv", index=False, encoding="utf-8-sig")
    first.to_csv(out / "control_lag_first_stage_results.csv", index=False, encoding="utf-8-sig")
    reduced.to_csv(out / "control_lag_reduced_form_results.csv", index=False, encoding="utf-8-sig")
    write_report(summary, first, out / "control_lag_sensitivity_report.md")
    metadata = {
        "posthoc": True,
        "instrument_set": ["z_u3_signed_r24", "z_pcegrid_up_r24", "z_pcegrid_down_r24"],
        "specification_ids": [spec["id"] for spec in specs],
    }
    (out / "control_lag_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
