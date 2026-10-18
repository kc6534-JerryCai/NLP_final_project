#!/usr/bin/env python
"""Post-hoc expanded PCE threshold analysis with upward/downward instruments."""

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
    SECONDARY_MEDIA,
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
    return [
        {
            "id": "PCE_GRID_SIGNED_R24",
            "outcome": OUTCOMES[0],
            "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": ["z_pcegrid_signed_r24"],
            "hac": 3,
            "family": "requested_posthoc",
            "primary": False,
        },
        {
            "id": "U3_SIGNED_PCE_GRID_SIGNED_R24",
            "outcome": OUTCOMES[0],
            "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": ["z_u3_signed_r24", "z_pcegrid_signed_r24"],
            "hac": 3,
            "family": "requested_posthoc",
            "primary": False,
        },
        {
            "id": "PCE_GRID_SPLIT_R24",
            "outcome": OUTCOMES[0],
            "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": ["z_pcegrid_up_r24", "z_pcegrid_down_r24"],
            "hac": 3,
            "family": "requested_posthoc",
            "primary": False,
        },
        {
            "id": "U3_SIGNED_PCE_GRID_SPLIT_R24",
            "outcome": OUTCOMES[0],
            "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": ["z_u3_signed_r24", "z_pcegrid_up_r24", "z_pcegrid_down_r24"],
            "hac": 3,
            "family": "requested_posthoc",
            "primary": False,
        },
        {
            "id": "ALL_SPLIT_R24",
            "outcome": OUTCOMES[0],
            "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": [
                "z_u3_up_r24",
                "z_u3_down_r24",
                "z_pcegrid_up_r24",
                "z_pcegrid_down_r24",
            ],
            "hac": 3,
            "family": "requested_posthoc",
            "primary": False,
        },
        {
            "id": "U3_SIGNED_PCE_GRID_SIGNED_R12",
            "outcome": OUTCOMES[0],
            "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": ["z_u3_signed_r12", "z_pcegrid_signed_r12"],
            "hac": 3,
            "family": "rarity_sensitivity",
            "primary": False,
        },
        {
            "id": "U3_SIGNED_PCE_GRID_SIGNED_NONE",
            "outcome": OUTCOMES[0],
            "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": ["z_u3_signed_rnone", "z_pcegrid_signed_rnone"],
            "hac": 3,
            "family": "rarity_sensitivity",
            "primary": False,
        },
        {
            "id": "U3_SIGNED_PCE_GRID_SPLIT_R12",
            "outcome": OUTCOMES[0],
            "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": ["z_u3_signed_r12", "z_pcegrid_up_r12", "z_pcegrid_down_r12"],
            "hac": 3,
            "family": "rarity_sensitivity",
            "primary": False,
        },
        {
            "id": "U3_SIGNED_PCE_GRID_SPLIT_NONE",
            "outcome": OUTCOMES[0],
            "media": PRIMARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": ["z_u3_signed_rnone", "z_pcegrid_up_rnone", "z_pcegrid_down_rnone"],
            "hac": 3,
            "family": "rarity_sensitivity",
            "primary": False,
        },
        {
            "id": "PCE_GRID_SIGNED_R24_T5",
            "outcome": OUTCOMES[0],
            "media": SECONDARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": ["z_pcegrid_signed_r24"],
            "hac": 3,
            "family": "measurement_sensitivity",
            "primary": False,
        },
        {
            "id": "PCE_GRID_SPLIT_R24_T5",
            "outcome": OUTCOMES[0],
            "media": SECONDARY_MEDIA,
            "controls": BASE_CONTROLS,
            "z": ["z_pcegrid_up_r24", "z_pcegrid_down_r24"],
            "hac": 3,
            "family": "measurement_sensitivity",
            "primary": False,
        },
    ]


def event_counts(analysis: pd.DataFrame, rarity: str) -> dict[str, int]:
    return {
        "pce_up_event_months": int((analysis[f"z_pcegrid_up_r{rarity}"] == 1).sum()),
        "pce_down_event_months": int((analysis[f"z_pcegrid_down_r{rarity}"] == 1).sum()),
        "u3_up_event_months": int((analysis[f"z_u3_up_r{rarity}"] == 1).sum()),
        "u3_down_event_months": int((analysis[f"z_u3_down_r{rarity}"] == 1).sum()),
    }


def write_report(summary: pd.DataFrame, first: pd.DataFrame, counts: pd.DataFrame, output: Path) -> None:
    estimated = summary.loc[summary["status"] == "estimated"].copy()
    requested = estimated.loc[estimated["model_id"] == "U3_SIGNED_PCE_GRID_SIGNED_R24"].iloc[0]
    pce_only = estimated.loc[estimated["model_id"] == "PCE_GRID_SPLIT_R24"].iloc[0]
    pce_signed = estimated.loc[estimated["model_id"] == "PCE_GRID_SIGNED_R24"].iloc[0]
    first_requested = first.loc[first["model_id"] == "U3_SIGNED_PCE_GRID_SIGNED_R24"]
    first_lines = "\n".join(
        f"- {row.instrument}: coefficient {row.estimate:.3f}, HAC p={row.p_value:.4f}."
        for row in first_requested.itertuples()
    )
    significant = estimated.loc[estimated["iv_media_p_value"] < 0.05, [
        "model_id", "iv_media_estimate", "iv_media_p_value", "iv_bh_q_value",
        "first_stage_joint_p_value", "first_stage_wald_f_type",
        "first_stage_partial_r_squared", "anderson_rubin_beta0_p_value",
        "ar_bh_q_value",
    ]]
    if significant.empty:
        significant_text = "No conventional 2SLS specification has p<0.05."
    else:
        significant_text = "\n".join(
            f"- {row.model_id}: 2SLS={row.iv_media_estimate:.3f}, p={row.iv_media_p_value:.4f}, "
            f"BH q={row.iv_bh_q_value:.4f}; first-stage p={row.first_stage_joint_p_value:.4f}, "
            f"F-type={row.first_stage_wald_f_type:.3f}, partial R-squared={row.first_stage_partial_r_squared:.4f}; "
            f"AR p={row.anderson_rubin_beta0_p_value:.4f}, AR BH q={row.ar_bh_q_value:.4f}."
            for row in significant.itertuples()
        )
    report = f"""# Post-hoc expanded PCE threshold results

This analysis was requested after the original results were observed. It is exploratory and does not replace the frozen primary specification.

## Requested single signed instrument, 24-month rarity

The unified PCE grid contains 2.0%, 2.5%, 3.0%, 3.5%, 4.0%, 4.5%, 5.0%, 5.5%, 6.0%, 6.5%, and 7.0%. The PCE instrument equals -1 for an upward crossing, +1 for a downward crossing, and 0 otherwise. The requested combined model also retains the original signed U-3 instrument.

- Sample: {int(requested['nobs'])} months.
- PCE upward event months: {int(counts.loc[counts['rarity'] == '24', 'pce_up_event_months'].iloc[0])}.
- PCE downward event months: {int(counts.loc[counts['rarity'] == '24', 'pce_down_event_months'].iloc[0])}.
- First-stage joint p={requested['first_stage_joint_p_value']:.4f}; HAC Wald/F-type={requested['first_stage_wald_f_type']:.3f}; partial R-squared={requested['first_stage_partial_r_squared']:.4f}.
- Conventional 2SLS media coefficient={requested['iv_media_estimate']:.3f}, p={requested['iv_media_p_value']:.4f}, 95% CI [{requested['iv_ci_95_low']:.3f}, {requested['iv_ci_95_high']:.3f}], BH q={requested['iv_bh_q_value']:.4f}.
- Anderson-Rubin zero-effect p={requested['anderson_rubin_beta0_p_value']:.4f}; BH q={requested['ar_bh_q_value']:.4f}.

First-stage coefficients:

{first_lines}

PCE signed instrument alone:

- First-stage joint p={pce_signed['first_stage_joint_p_value']:.4f}; HAC Wald/F-type={pce_signed['first_stage_wald_f_type']:.3f}; partial R-squared={pce_signed['first_stage_partial_r_squared']:.4f}.
- Conventional 2SLS media coefficient={pce_signed['iv_media_estimate']:.3f}, p={pce_signed['iv_media_p_value']:.4f}, BH q={pce_signed['iv_bh_q_value']:.4f}.
- Anderson-Rubin p={pce_signed['anderson_rubin_beta0_p_value']:.4f}, BH q={pce_signed['ar_bh_q_value']:.4f}.

## PCE-only direction-split model

- First-stage joint p={pce_only['first_stage_joint_p_value']:.4f}; HAC Wald/F-type={pce_only['first_stage_wald_f_type']:.3f}; partial R-squared={pce_only['first_stage_partial_r_squared']:.4f}.
- Conventional 2SLS media coefficient={pce_only['iv_media_estimate']:.3f}, p={pce_only['iv_media_p_value']:.4f}, BH q={pce_only['iv_bh_q_value']:.4f}.
- Anderson-Rubin p={pce_only['anderson_rubin_beta0_p_value']:.4f}, BH q={pce_only['ar_bh_q_value']:.4f}.

## Conventional significance scan

{significant_text}

## Interpretation

The expanded grid is post-hoc, half-integers do not automatically have news value, and PCE 2% can directly alter policy expectations and consumer beliefs. A significant second stage is not credible evidence of a media-tone channel unless the expanded instruments materially predict media tone and the exclusion restriction is defensible. Direction splitting also raises the instrument count in a small monthly sample.
"""
    output.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    analysis, all_events, _, _, quality = construct_data(
        args.workbook, args.articles, args.release_data
    )

    grid = np.arange(2.0, 7.0 + 0.001, 0.5)
    expanded = all_events.loc[
        (all_events["indicator"] == "pce_inflation_yoy_pct")
        & all_events["threshold"].astype(float).round(1).isin(grid.round(1))
    ].copy()
    expanded["threshold_family"] = "expanded_half_point_grid_2_to_7"
    expanded = expanded.drop_duplicates(
        ["threshold", "rarity_months", "release_month", "direction"]
    ).sort_values(["rarity_months", "release_month", "threshold"])

    for rarity in (24, 12, None):
        analysis = add_event_instruments(analysis, expanded, "pcegrid", rarity)

    specs = build_specs()
    summary, first, reduced, _ = run_specs(analysis, specs)

    count_rows = []
    for rarity in ("24", "12", "none"):
        row = {"rarity": rarity, **event_counts(analysis, rarity)}
        count_rows.append(row)
    counts = pd.DataFrame(count_rows)

    for rarity in ("24", "12", "none"):
        mask = summary["model_id"].str.endswith(f"R{rarity.upper()}")
        if rarity == "none":
            mask = summary["model_id"].str.endswith("NONE")
        for column, value in counts.loc[counts["rarity"] == rarity].iloc[0].items():
            if column != "rarity":
                summary.loc[mask, column] = value
    t5_mask = summary["model_id"].isin(
        ["PCE_GRID_SIGNED_R24_T5", "PCE_GRID_SPLIT_R24_T5"]
    )
    count_24 = counts.loc[counts["rarity"] == "24"].iloc[0]
    for column in [
        "pce_up_event_months",
        "pce_down_event_months",
        "u3_up_event_months",
        "u3_down_event_months",
    ]:
        summary.loc[t5_mask, column] = count_24[column]

    qualifying = expanded.loc[expanded["qualifies_rarity"].astype(bool)].copy()
    summary.to_csv(out / "expanded_iv_model_summary.csv", index=False, encoding="utf-8-sig")
    first.to_csv(out / "expanded_first_stage_results.csv", index=False, encoding="utf-8-sig")
    reduced.to_csv(out / "expanded_reduced_form_results.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(out / "expanded_event_counts.csv", index=False, encoding="utf-8-sig")
    expanded.to_csv(out / "expanded_pce_threshold_events.csv", index=False, encoding="utf-8-sig")
    qualifying.to_csv(out / "expanded_pce_qualifying_events.csv", index=False, encoding="utf-8-sig")
    keep = [
        "month", PRIMARY_MEDIA, SECONDARY_MEDIA, *OUTCOMES, *BASE_CONTROLS,
        "z_u3_signed_r24", "z_u3_up_r24", "z_u3_down_r24",
        "z_pcegrid_signed_r24", "z_pcegrid_up_r24", "z_pcegrid_down_r24", "pcegrid_crossed_thresholds_r24",
        "z_pcegrid_signed_r12", "z_pcegrid_up_r12", "z_pcegrid_down_r12", "pcegrid_crossed_thresholds_r12",
        "z_pcegrid_signed_rnone", "z_pcegrid_up_rnone", "z_pcegrid_down_rnone", "pcegrid_crossed_thresholds_rnone",
    ]
    analysis[keep].to_csv(out / "expanded_aligned_dataset.csv", index=False, encoding="utf-8-sig")
    write_report(summary, first, counts, out / "expanded_threshold_report.md")
    metadata = {
        "posthoc": True,
        "threshold_grid": grid.tolist(),
        "specification_ids": [spec["id"] for spec in specs],
        "quality": quality,
        "output_files": sorted(path.name for path in out.iterdir() if path.is_file()),
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
