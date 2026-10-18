#!/usr/bin/env python
"""Aggregate completed article-level tone predictions into monthly indices.

This script is deliberately inference-free: it reads saved predictions, performs
deterministic cleaning, and writes monthly tables, topic tables, SVG charts, and
an audit report. It never trains a model or calls an external API.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


SCORES = {-1, 0, 1}
KNOWN_TOPICS = [
    "macro_growth",
    "labor_market",
    "cost_of_living_income",
    "household_economy",
    "economic_policy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/all_relevant/delivery/all_relevant_annotations.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/monthly_tone_index")
    )
    parser.add_argument("--requested-start", default="2010-01-01")
    parser.add_argument("--requested-cutoff", default="2026-10-14")
    parser.add_argument(
        "--known-anomaly-month",
        action="append",
        default=["2026-01"],
        help="Month known to have an unresolved collection anomaly; repeatable.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return pd.DataFrame(rows)


def normalized_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def normalized_url(value: object) -> str:
    text = normalized_text(value)
    if not text:
        return ""
    text = re.sub(r"#.*$", "", text)
    text = re.sub(r"[?&](utm_[^=&]+|gclid|fbclid)=[^&]*", "", text)
    text = text.rstrip("?&/")
    return text


def build_dedupe_key(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    keys: list[str] = []
    sources: list[str] = []
    url_col = next(
        (col for col in ("normalized_url", "url", "web_url") if col in df.columns),
        None,
    )
    for row_no, row in df.iterrows():
        article_id = normalized_text(row.get("article_id"))
        url = normalized_url(row.get(url_col)) if url_col else ""
        if article_id:
            keys.append(f"id:{article_id}")
            sources.append("article_id")
        elif url:
            keys.append(f"url:{url}")
            sources.append("normalized_url")
        else:
            headline = normalized_text(row.get("headline"))
            abstract = normalized_text(row.get("abstract"))
            published = normalized_text(row.get("published_at"))
            if headline or abstract or published:
                keys.append(f"exact:{published}\x1f{headline}\x1f{abstract}")
                sources.append("headline_abstract_date")
            else:
                keys.append(f"row:{row_no}")
                sources.append("unique_row_fallback")
    return pd.Series(keys, index=df.index), pd.Series(sources, index=df.index)


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def score_validity(
    df: pd.DataFrame, score_col: str, required_valid_col: str | None = None
) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(df[score_col], errors="coerce")
    valid = numeric.isin(SCORES)
    if required_valid_col and required_valid_col in df.columns:
        valid &= parse_bool(df[required_valid_col])
    return numeric, valid


def coverage_fields(
    month: str,
    first_month: str,
    last_month: str,
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
    requested_start: pd.Timestamp | None,
    requested_cutoff: pd.Timestamp | None,
    anomaly_months: set[str],
) -> tuple[str, str]:
    if month in anomaly_months:
        return "known_collection_anomaly", "项目质量报告记录该月原始文章量异常偏低"
    if month == first_month:
        if first_date.day > 1 or (
            requested_start is not None
            and requested_start.strftime("%Y-%m") == month
            and requested_start.day > 1
        ):
            return "partial_start", f"样本从 {first_date.date()} 开始"
        return "observed_from_month_start", f"最早文章日期为 {first_date.date()}"
    if month == last_month:
        month_end = last_date.tz_localize(None).to_period("M").end_time.date()
        cutoff_date = requested_cutoff.date() if requested_cutoff is not None else None
        if last_date.date() < month_end or (cutoff_date and cutoff_date < month_end):
            return "partial_end", f"最晚返回文章日期为 {last_date.date()}"
    return "interior_month", "月份位于样本首尾之间；不等同于逐篇收录完整性保证"


def aggregate_monthly(
    df: pd.DataFrame,
    score_col: str,
    valid_col: str,
    months: list[str],
    model: str,
    coverage: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for month in months:
        part = df[df["month"] == month]
        valid_part = part[part[valid_col]]
        values = valid_part[score_col]
        n_articles = len(part)
        n_valid = len(valid_part)
        n_positive = int((values == 1).sum())
        n_negative = int((values == -1).sum())
        n_zero = int((values == 0).sum())
        positive_share = n_positive / n_valid if n_valid else math.nan
        negative_share = n_negative / n_valid if n_valid else math.nan
        zero_share = n_zero / n_valid if n_valid else math.nan
        tone_index = 100 * (n_positive - n_negative) / n_valid if n_valid else math.nan
        status, note = coverage[month]
        rows.append(
            {
                "model": model,
                "month": month,
                "n_articles": n_articles,
                "n_valid": n_valid,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "n_zero": n_zero,
                "positive_share": positive_share,
                "negative_share": negative_share,
                "zero_share": zero_share,
                "n_excluded_tone": n_articles - n_valid,
                "tone_index": tone_index,
                "n_neutral": pd.NA,
                "n_mixed": pd.NA,
                "n_insufficient_information": pd.NA,
                "coverage_status": status,
                "coverage_note": note,
            }
        )
    return pd.DataFrame(rows)


def aggregate_topics(
    df: pd.DataFrame,
    topic_col: str,
    score_col: str,
    valid_col: str,
    months: list[str],
    model: str,
) -> pd.DataFrame:
    observed = [
        str(x)
        for x in df[topic_col].dropna().unique()
        if normalized_text(x) not in {"", "none", "null"}
    ]
    topics = [topic for topic in KNOWN_TOPICS if topic in observed]
    topics.extend(sorted(set(observed) - set(topics)))
    rows = []
    for month in months:
        month_part = df[df["month"] == month]
        for topic in topics:
            part = month_part[month_part[topic_col] == topic]
            valid_part = part[part[valid_col]]
            values = valid_part[score_col]
            n_articles = len(part)
            n_valid = len(valid_part)
            n_positive = int((values == 1).sum())
            n_negative = int((values == -1).sum())
            n_zero = int((values == 0).sum())
            rows.append(
                {
                    "model": model,
                    "month": month,
                    "primary_topic": topic,
                    "n_articles": n_articles,
                    "n_valid": n_valid,
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "n_zero": n_zero,
                    "positive_share": n_positive / n_valid if n_valid else math.nan,
                    "negative_share": n_negative / n_valid if n_valid else math.nan,
                    "zero_share": n_zero / n_valid if n_valid else math.nan,
                    "n_excluded_tone": n_articles - n_valid,
                    "tone_index": (
                        100 * (n_positive - n_negative) / n_valid
                        if n_valid
                        else math.nan
                    ),
                    "n_neutral": pd.NA,
                    "n_mixed": pd.NA,
                    "n_insufficient_information": pd.NA,
                }
            )
    return pd.DataFrame(rows)


def validate_table(df: pd.DataFrame, label: str) -> dict[str, object]:
    counts_ok = bool(
        (
            df["n_positive"] + df["n_negative"] + df["n_zero"]
            == df["n_valid"]
        ).all()
    )
    active = df["n_valid"] > 0
    share_sum = df.loc[active, ["positive_share", "negative_share", "zero_share"]].sum(
        axis=1
    )
    shares_ok = bool(((share_sum - 1).abs() < 1e-10).all())
    expected = 100 * (df.loc[active, "positive_share"] - df.loc[active, "negative_share"])
    formula_ok = bool(((df.loc[active, "tone_index"] - expected).abs() < 1e-10).all())
    range_ok = bool(df["tone_index"].dropna().between(-100, 100).all())
    result = {
        "table": label,
        "count_identity_ok": counts_ok,
        "share_sum_ok": shares_ok,
        "tone_formula_ok": formula_ok,
        "index_range_ok": range_ok,
        "rows": int(len(df)),
        "active_rows": int(active.sum()),
        "sum_n_articles": int(df["n_articles"].sum()),
        "sum_n_valid": int(df["n_valid"].sum()),
    }
    if not all((counts_ok, shares_ok, formula_ok, range_ok)):
        raise AssertionError(f"Validation failed for {label}: {result}")
    return result


def common_sample_table(
    eligible: pd.DataFrame,
    t5_monthly: pd.DataFrame,
    deepseek_monthly: pd.DataFrame,
    months: list[str],
) -> pd.DataFrame:
    rows = []
    for month in months:
        part = eligible[eligible["month"] == month]
        both = part[part["t5_tone_valid"] & part["deepseek_tone_valid"]]
        row: dict[str, object] = {
            "month": month,
            "n_articles": len(part),
            "t5_n_valid_main": int(part["t5_tone_valid"].sum()),
            "deepseek_n_valid_main": int(part["deepseek_tone_valid"].sum()),
            "n_common_valid": len(both),
            "t5_only_valid": int((part["t5_tone_valid"] & ~part["deepseek_tone_valid"]).sum()),
            "deepseek_only_valid": int((~part["t5_tone_valid"] & part["deepseek_tone_valid"]).sum()),
            "neither_valid": int((~part["t5_tone_valid"] & ~part["deepseek_tone_valid"]).sum()),
        }
        for prefix, score_col in (("t5", "t5_tone_score"), ("deepseek", "deepseek_tone_score")):
            values = both[score_col]
            n = len(values)
            pos = int((values == 1).sum())
            neg = int((values == -1).sum())
            zero = int((values == 0).sum())
            row.update(
                {
                    f"{prefix}_n_positive": pos,
                    f"{prefix}_n_negative": neg,
                    f"{prefix}_n_zero": zero,
                    f"{prefix}_positive_share": pos / n if n else math.nan,
                    f"{prefix}_negative_share": neg / n if n else math.nan,
                    f"{prefix}_zero_share": zero / n if n else math.nan,
                    f"{prefix}_tone_index": 100 * (pos - neg) / n if n else math.nan,
                }
            )
        if len(both):
            row["tone_index_difference_t5_minus_deepseek"] = (
                row["t5_tone_index"] - row["deepseek_tone_index"]
            )
        else:
            row["tone_index_difference_t5_minus_deepseek"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def svg_line_chart(
    path: Path,
    series: list[tuple[str, list[float], str, str]],
    months: list[str],
    title: str,
    y_label: str,
    y_min: float,
    y_max: float,
) -> None:
    width, height = 1600, 820
    left, right, top, bottom = 105, 50, 80, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    x = lambda i: left + (plot_w * i / max(1, len(months) - 1))
    y = lambda v: top + plot_h * (y_max - v) / (y_max - y_min)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFAF7"/>',
        f'<text x="{left}" y="42" font-family="Arial" font-size="27" font-weight="700" fill="#1F2937">{html.escape(title)}</text>',
    ]
    for tick in range(int(y_min), int(y_max) + 1, 20):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#D1D5DB" stroke-width="1"/>')
        parts.append(f'<text x="{left-14}" y="{yy+5:.1f}" text-anchor="end" font-family="Arial" font-size="14" fill="#4B5563">{tick}</text>')
    for i, month in enumerate(months):
        if month.endswith("-01") or i in {0, len(months) - 1}:
            xx = x(i)
            parts.append(f'<line x1="{xx:.1f}" y1="{top+plot_h}" x2="{xx:.1f}" y2="{top+plot_h+7}" stroke="#6B7280"/>')
            parts.append(f'<text x="{xx:.1f}" y="{top+plot_h+32}" text-anchor="middle" font-family="Arial" font-size="13" fill="#4B5563">{html.escape(month)}</text>')
    zero_y = y(0)
    parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left+plot_w}" y2="{zero_y:.1f}" stroke="#6B7280" stroke-width="1.5"/>')
    legend_x = left + 20
    for s_no, (name, values, color, dash) in enumerate(series):
        chunks: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for i, value in enumerate(values):
            if value is None or pd.isna(value):
                if current:
                    chunks.append(current)
                    current = []
            else:
                current.append((x(i), y(float(value))))
        if current:
            chunks.append(current)
        for chunk in chunks:
            points = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in chunk)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6"{dash_attr}/>' )
        lx = legend_x + (s_no % 2) * 360
        ly = top + 22 + (s_no // 2) * 28
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+45}" y2="{ly}" stroke="{color}" stroke-width="3"{dash_attr}/>' )
        parts.append(f'<text x="{lx+55}" y="{ly+5}" font-family="Arial" font-size="15" fill="#374151">{html.escape(name)}</text>')
    parts.append(f'<text x="28" y="{top+plot_h/2}" transform="rotate(-90 28 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="16" fill="#374151">{html.escape(y_label)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_bar_chart(
    path: Path,
    months: list[str],
    t5_values: list[int],
    deepseek_values: list[int],
) -> None:
    width, height = 1600, 820
    left, right, top, bottom = 105, 50, 80, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = max(t5_values + deepseek_values + [1])
    y_max = int(math.ceil(max_value / 100) * 100)
    x_step = plot_w / max(1, len(months))
    bar_w = max(1.2, x_step * 0.40)
    y = lambda v: top + plot_h * (y_max - v) / y_max
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFAF7"/>',
        f'<text x="{left}" y="42" font-family="Arial" font-size="27" font-weight="700" fill="#1F2937">Monthly valid article counts</text>',
    ]
    tick_step = max(100, int(math.ceil(y_max / 5 / 100) * 100))
    for tick in range(0, y_max + 1, tick_step):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#D1D5DB"/>')
        parts.append(f'<text x="{left-14}" y="{yy+5:.1f}" text-anchor="end" font-family="Arial" font-size="14" fill="#4B5563">{tick}</text>')
    for i, month in enumerate(months):
        center = left + (i + 0.5) * x_step
        t5_top = y(t5_values[i])
        ds_top = y(deepseek_values[i])
        parts.append(f'<rect x="{center-bar_w:.2f}" y="{t5_top:.2f}" width="{bar_w:.2f}" height="{top+plot_h-t5_top:.2f}" fill="#2563EB" opacity="0.76"/>')
        parts.append(f'<rect x="{center:.2f}" y="{ds_top:.2f}" width="{bar_w:.2f}" height="{top+plot_h-ds_top:.2f}" fill="#D97706" opacity="0.76"/>')
        if month.endswith("-01") or i in {0, len(months) - 1}:
            parts.append(f'<text x="{center:.1f}" y="{top+plot_h+32}" text-anchor="middle" font-family="Arial" font-size="13" fill="#4B5563">{html.escape(month)}</text>')
    parts.extend(
        [
            f'<rect x="{left+20}" y="{top+16}" width="24" height="12" fill="#2563EB" opacity="0.76"/><text x="{left+52}" y="{top+27}" font-family="Arial" font-size="15" fill="#374151">T5 main sample</text>',
            f'<rect x="{left+205}" y="{top+16}" width="24" height="12" fill="#D97706" opacity="0.76"/><text x="{left+237}" y="{top+27}" font-family="Arial" font-size="15" fill="#374151">DeepSeek main sample</text>',
            f'<text x="28" y="{top+plot_h/2}" transform="rotate(-90 28 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="16" fill="#374151">n_valid</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_method_note(path: Path, audit: dict[str, object]) -> None:
    text = f"""# 月度经济新闻语气指数：方法说明

## 输入与字段映射

- 输入文件：`{audit['input_file']}`。
- 唯一标识：`article_id`；本次数据没有 URL 字段。若未来记录缺少 ID，脚本依次使用规范化 URL，或发布日期、标题、摘要的规范化完全匹配键。
- 日期：`published_at`，统一解析为 UTC 后按日历月 `YYYY-MM` 归属。
- 文本：`headline` 与 `abstract`（仅用于识别无可靠 ID/URL 时的完全重复，不用于重新分类）。
- T5：`t5_tone`、`t5_primary_topic`，并要求 `t5_format_valid=true`。
- DeepSeek：`deepseek_tone`、`deepseek_primary_topic`；显式相关性字段为 `deepseek_relevance`。
- 全量文件中没有 econmediabert 的文章级预测；现有两个 FinBERT 预测文件各只有 600 条测试集记录，未用于月度长序列。

## 清理、相关性与分数

- 原始行数：{audit['input_rows']:,}；去重删除：{audit['duplicates_removed']:,}；日期无法解析：{audit['invalid_dates']:,}。
- 显式标记为 DeepSeek `irrelevant` 的 {audit['irrelevant_excluded']:,} 条不进入任一主指数。`relevant` 与 `uncertain` 保留在 tone 有效性筛选前的文章数中；`uncertain` 若没有合法 tone，则计入 `n_excluded_tone`。
- 合法分数仅为 −1、0、+1。缺失、非法值和格式失败不会填成 0。
- 输入未保留 Neutral、Mixed、Insufficient information 的原始分类标签。因此所有已经保存为合法 0 的分数都参与平均，但无法进一步命名为 Neutral 或 Mixed；输出中的三项原始类别计数保持缺失。
- 不使用 `duplicate_group_id` 做近似去重，避免把同一事件的不同报道误删。

## 指数定义

每个月直接对文章等权平均：`tone_index = 100 × mean(tone_score)`，等价于 `100 × (n_positive − n_negative) / n_valid`。不先平均每日值，不按长度、置信度或新闻量加权，不平滑、不标准化。总指数从文章直接计算，不由主题指数平均得到。

主题表使用各模型保存的单一 `primary_topic`；它表示“涉及该主要主题的文章的整体语气”，不是主题专属 tone。当前输出不是多标签主题，因此每个模型的一篇文章最多进入一个主题分项。

## 模型可比性与覆盖

- T5 主样本有效文章：{audit['t5_valid_total']:,}。
- DeepSeek 主样本有效文章：{audit['deepseek_valid_total']:,}。
- 双方共同有效文章：{audit['common_valid_total']:,}。共同样本表只用这些文章重新计算两套指数，并列出仅单方有效的覆盖差异。
- 月份范围：{audit['month_start']} 至 {audit['month_end']}。起始月最早记录为 {audit['first_publication']}；末月最晚返回记录为 {audit['last_publication']}，因此末月标记为部分覆盖。
- 项目既有采集质量报告指出 2026-01 原始记录量异常偏低，且 2026-10 的最晚返回日期早于请求截止日；这些是数据覆盖限制，不通过插值修复。

## 输出与核验

所有月表补齐样本起止之间的日历月。无有效文章时，计数为 0，比例和指数为空。验证报告检查计数恒等式、占比和、指数公式、取值范围，并核对月度汇总与清理后文章级记录。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_jsonl(args.input)
    input_rows = len(df)

    required = {
        "article_id",
        "published_at",
        "headline",
        "abstract",
        "t5_tone",
        "t5_primary_topic",
        "deepseek_relevance",
        "deepseek_tone",
        "deepseek_primary_topic",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Required columns missing: {missing}")

    df["dedupe_key"], df["dedupe_key_source"] = build_dedupe_key(df)
    duplicate_mask = df.duplicated("dedupe_key", keep="first")
    duplicate_rows = df.loc[duplicate_mask].copy()
    duplicates_removed = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()

    df["published_at_parsed"] = pd.to_datetime(
        df["published_at"], errors="coerce", utc=True, format="mixed"
    )
    invalid_dates = df[df["published_at_parsed"].isna()].copy()
    invalid_dates["date_parse_issue"] = "published_at could not be parsed"
    invalid_dates[
        ["article_id", "published_at", "headline", "abstract", "date_parse_issue"]
    ].to_csv(output_dir / "date_parse_failures.csv", index=False, encoding="utf-8-sig")
    valid_date = df[df["published_at_parsed"].notna()].copy()
    valid_date["month"] = valid_date["published_at_parsed"].dt.strftime("%Y-%m")

    if valid_date.empty:
        raise ValueError("No parseable publication dates.")
    first_date = valid_date["published_at_parsed"].min()
    last_date = valid_date["published_at_parsed"].max()
    period_range = pd.period_range(
        first_date.tz_localize(None).to_period("M"),
        last_date.tz_localize(None).to_period("M"),
        freq="M",
    )
    months = [str(period) for period in period_range]

    requested_start = pd.to_datetime(args.requested_start, errors="coerce", utc=True)
    requested_cutoff = pd.to_datetime(args.requested_cutoff, errors="coerce", utc=True)
    requested_start = None if pd.isna(requested_start) else requested_start
    requested_cutoff = None if pd.isna(requested_cutoff) else requested_cutoff
    anomaly_months = set(args.known_anomaly_month)
    coverage = {
        month: coverage_fields(
            month,
            months[0],
            months[-1],
            first_date,
            last_date,
            requested_start,
            requested_cutoff,
            anomaly_months,
        )
        for month in months
    }

    relevance = valid_date["deepseek_relevance"].fillna("").astype(str).str.strip().str.casefold()
    irrelevant_mask = relevance.eq("irrelevant")
    eligible = valid_date.loc[~irrelevant_mask].copy()
    eligible["deepseek_relevance_normalized"] = relevance.loc[~irrelevant_mask]

    eligible["t5_tone_score"], eligible["t5_tone_valid"] = score_validity(
        eligible, "t5_tone", "t5_format_valid"
    )
    eligible["deepseek_tone_score"], eligible["deepseek_tone_valid"] = score_validity(
        eligible, "deepseek_tone"
    )

    t5_monthly = aggregate_monthly(
        eligible, "t5_tone_score", "t5_tone_valid", months, "t5", coverage
    )
    deepseek_monthly = aggregate_monthly(
        eligible,
        "deepseek_tone_score",
        "deepseek_tone_valid",
        months,
        "deepseek",
        coverage,
    )
    t5_topics = aggregate_topics(
        eligible,
        "t5_primary_topic",
        "t5_tone_score",
        "t5_tone_valid",
        months,
        "t5",
    )
    deepseek_topics = aggregate_topics(
        eligible,
        "deepseek_primary_topic",
        "deepseek_tone_score",
        "deepseek_tone_valid",
        months,
        "deepseek",
    )
    common = common_sample_table(eligible, t5_monthly, deepseek_monthly, months)

    t5_monthly.to_csv(output_dir / "monthly_tone_index_t5.csv", index=False, encoding="utf-8-sig")
    deepseek_monthly.to_csv(
        output_dir / "monthly_tone_index_deepseek.csv", index=False, encoding="utf-8-sig"
    )
    t5_topics.to_csv(output_dir / "monthly_topic_tone_index_t5.csv", index=False, encoding="utf-8-sig")
    deepseek_topics.to_csv(
        output_dir / "monthly_topic_tone_index_deepseek.csv", index=False, encoding="utf-8-sig"
    )
    common.to_csv(output_dir / "monthly_tone_index_common_sample.csv", index=False, encoding="utf-8-sig")
    duplicate_rows[
        [col for col in ("article_id", "published_at", "headline", "abstract", "dedupe_key_source") if col in duplicate_rows.columns]
    ].to_csv(output_dir / "duplicate_records_excluded.csv", index=False, encoding="utf-8-sig")

    validations = [
        validate_table(t5_monthly, "monthly_tone_index_t5"),
        validate_table(deepseek_monthly, "monthly_tone_index_deepseek"),
        validate_table(t5_topics, "monthly_topic_tone_index_t5"),
        validate_table(deepseek_topics, "monthly_topic_tone_index_deepseek"),
    ]
    common_checks = {}
    for prefix in ("t5", "deepseek"):
        active = common["n_common_valid"] > 0
        count_ok = bool(
            (
                common[f"{prefix}_n_positive"]
                + common[f"{prefix}_n_negative"]
                + common[f"{prefix}_n_zero"]
                == common["n_common_valid"]
            ).all()
        )
        shares = common.loc[
            active,
            [f"{prefix}_positive_share", f"{prefix}_negative_share", f"{prefix}_zero_share"],
        ].sum(axis=1)
        share_ok = bool(((shares - 1).abs() < 1e-10).all())
        formula = 100 * (
            common.loc[active, f"{prefix}_positive_share"]
            - common.loc[active, f"{prefix}_negative_share"]
        )
        formula_ok = bool(
            ((common.loc[active, f"{prefix}_tone_index"] - formula).abs() < 1e-10).all()
        )
        range_ok = bool(common[f"{prefix}_tone_index"].dropna().between(-100, 100).all())
        common_checks[prefix] = {
            "count_identity_ok": count_ok,
            "share_sum_ok": share_ok,
            "tone_formula_ok": formula_ok,
            "index_range_ok": range_ok,
        }
        if not all((count_ok, share_ok, formula_ok, range_ok)):
            raise AssertionError(f"Common-sample validation failed for {prefix}")

    article_month_counts = eligible.groupby("month").size().reindex(months, fill_value=0)
    reconciliation = {
        "eligible_article_rows": int(len(eligible)),
        "sum_monthly_article_counts_t5": int(t5_monthly["n_articles"].sum()),
        "sum_monthly_article_counts_deepseek": int(deepseek_monthly["n_articles"].sum()),
        "direct_month_count_sum": int(article_month_counts.sum()),
    }
    if len(set(reconciliation.values())) != 1:
        raise AssertionError(f"Monthly reconciliation failed: {reconciliation}")

    common_valid_total = int((eligible["t5_tone_valid"] & eligible["deepseek_tone_valid"]).sum())
    audit = {
        "input_file": str(args.input.resolve()),
        "input_rows": input_rows,
        "dedupe_key_source_counts": df["dedupe_key_source"].value_counts(dropna=False).to_dict(),
        "duplicates_removed": duplicates_removed,
        "invalid_dates": int(len(invalid_dates)),
        "valid_date_rows": int(len(valid_date)),
        "irrelevant_excluded": int(irrelevant_mask.sum()),
        "eligible_rows_before_tone_filter": int(len(eligible)),
        "relevance_counts_before_exclusion": relevance.value_counts(dropna=False).to_dict(),
        "t5_score_counts_all_input": pd.to_numeric(df["t5_tone"], errors="coerce").value_counts(dropna=False).to_dict(),
        "deepseek_score_counts_all_input": pd.to_numeric(df["deepseek_tone"], errors="coerce").value_counts(dropna=False).to_dict(),
        "t5_valid_total": int(eligible["t5_tone_valid"].sum()),
        "deepseek_valid_total": int(eligible["deepseek_tone_valid"].sum()),
        "common_valid_total": common_valid_total,
        "month_start": months[0],
        "month_end": months[-1],
        "n_calendar_months": len(months),
        "first_publication": first_date.isoformat(),
        "last_publication": last_date.isoformat(),
        "raw_tone_label_columns_present": [],
        "raw_label_limitation": "Only numeric -1/0/+1 tones remain; zero subtypes cannot be reconstructed.",
        "topic_structure": "single primary topic per model, not multilabel",
        "finbert_full_corpus_available": False,
        "finbert_note": "Saved FinBERT predictions cover only the 600-row test set and were not aggregated.",
        "validations": validations,
        "common_sample_validations": common_checks,
        "monthly_reconciliation": reconciliation,
    }
    with (output_dir / "aggregation_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, default=str)
    write_method_note(output_dir / "methodology.md", audit)

    svg_line_chart(
        output_dir / "monthly_tone_index_lines.svg",
        [
            ("T5 — main sample", t5_monthly["tone_index"].tolist(), "#2563EB", ""),
            ("DeepSeek — main sample", deepseek_monthly["tone_index"].tolist(), "#D97706", ""),
            ("T5 — common sample", common["t5_tone_index"].tolist(), "#60A5FA", "8 5"),
            ("DeepSeek — common sample", common["deepseek_tone_index"].tolist(), "#F59E0B", "8 5"),
        ],
        months,
        "Monthly economic news tone index",
        "Tone index (−100 to 100)",
        -100,
        100,
    )
    svg_bar_chart(
        output_dir / "monthly_n_valid.svg",
        months,
        t5_monthly["n_valid"].astype(int).tolist(),
        deepseek_monthly["n_valid"].astype(int).tolist(),
    )

    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    print("\nT5 monthly head:")
    print(t5_monthly.head().to_string(index=False))
    print("\nDeepSeek monthly head:")
    print(deepseek_monthly.head().to_string(index=False))


if __name__ == "__main__":
    main()
