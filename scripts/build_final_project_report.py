from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "tmp" / "pdfs" / "economic_perception_report"
OUT_DIR = ROOT / "output" / "pdf"
PDF_PATH = OUT_DIR / "Economic_Perception_Monitor_Final_Report_Revised.pdf"

NAVY = colors.HexColor("#15374B")
TEAL = colors.HexColor("#0E766E")
BLUE = colors.HexColor("#3B6D8C")
ORANGE = colors.HexColor("#C77A28")
RED = colors.HexColor("#B94D45")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5D6B75")
PALE = colors.HexColor("#EDF3F5")
PALE_BLUE = colors.HexColor("#E7F0F4")
PALE_ORANGE = colors.HexColor("#F8EFE1")
GRID = colors.HexColor("#CBD6DB")
WHITE = colors.white


def register_fonts() -> None:
    fonts = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Georgia", str(fonts / "georgia.ttf")))
    pdfmetrics.registerFont(TTFont("Georgia-Bold", str(fonts / "georgiab.ttf")))
    pdfmetrics.registerFont(TTFont("Arial", str(fonts / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(fonts / "arialbd.ttf")))
    pdfmetrics.registerFontFamily("Georgia", normal="Georgia", bold="Georgia-Bold")
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")


register_fonts()

styles = getSampleStyleSheet()
BODY = ParagraphStyle(
    "Body",
    fontName="Georgia",
    fontSize=8.45,
    leading=11.35,
    textColor=INK,
    spaceAfter=5.5,
)
BODY_SMALL = ParagraphStyle(
    "BodySmall",
    parent=BODY,
    fontSize=7.55,
    leading=9.8,
    spaceAfter=4,
)
BODY_TINY = ParagraphStyle(
    "BodyTiny",
    parent=BODY,
    fontSize=6.7,
    leading=8.5,
    spaceAfter=2,
)
TITLE = ParagraphStyle(
    "Title",
    fontName="Georgia-Bold",
    fontSize=23,
    leading=27,
    alignment=TA_CENTER,
    textColor=NAVY,
    spaceAfter=10,
)
SUBTITLE = ParagraphStyle(
    "Subtitle",
    fontName="Arial",
    fontSize=10.5,
    leading=14,
    alignment=TA_CENTER,
    textColor=MUTED,
    spaceAfter=15,
)
AUTHOR = ParagraphStyle(
    "Author",
    fontName="Arial",
    fontSize=8.6,
    leading=12,
    alignment=TA_CENTER,
    textColor=INK,
)
STUDENT_INFO = ParagraphStyle(
    "StudentInfo",
    fontName="Arial",
    fontSize=8.2,
    leading=11,
    alignment=TA_LEFT,
    textColor=MUTED,
    spaceAfter=14,
)
H1 = ParagraphStyle(
    "H1",
    fontName="Georgia-Bold",
    fontSize=16,
    leading=19,
    textColor=NAVY,
    spaceAfter=8,
)
H2 = ParagraphStyle(
    "H2",
    fontName="Arial-Bold",
    fontSize=9.6,
    leading=12,
    textColor=TEAL,
    spaceBefore=5,
    spaceAfter=4,
)
CAPTION = ParagraphStyle(
    "Caption",
    fontName="Arial",
    fontSize=6.9,
    leading=8.8,
    textColor=MUTED,
    alignment=TA_LEFT,
    spaceBefore=3,
    spaceAfter=5,
)
BOX = ParagraphStyle(
    "Box",
    fontName="Georgia",
    fontSize=8.0,
    leading=10.5,
    textColor=INK,
)
BOX_BOLD = ParagraphStyle(
    "BoxBold",
    fontName="Arial-Bold",
    fontSize=8.2,
    leading=10.5,
    textColor=NAVY,
)
LABEL = ParagraphStyle(
    "Label",
    fontName="Arial-Bold",
    fontSize=6.8,
    leading=8,
    textColor=TEAL,
    tracking=1.1,
)


def P(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def page_title(number: str, title: str, kicker: str | None = None):
    items = []
    if kicker:
        items.append(P(kicker.upper(), LABEL))
        items.append(Spacer(1, 3))
    items.append(P(f"{number}. {title}", H1))
    items.append(HRFlowable(width="100%", thickness=0.7, color=GRID, spaceAfter=8))
    return items


def callout(title: str, text: str, bg=PALE_BLUE, border=BLUE):
    content = [[P(title, BOX_BOLD)], [P(text, BOX)]]
    tbl = Table(content, colWidths=[6.82 * inch], hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.7, border),
                ("LINEBELOW", (0, 0), (-1, 0), 0.4, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def bullet(text: str, style: ParagraphStyle = BODY_SMALL):
    return P(f"<font color='#0E766E'>&#8226;</font>&nbsp;&nbsp;{text}", style)


def make_table(data, widths, header=True, font_size=7.2, row_bgs=True, aligns=None, highlight_rows=None):
    rendered = []
    for r, row in enumerate(data):
        rendered_row = []
        for cell in row:
            style = ParagraphStyle(
                f"table_{r}_{len(rendered_row)}",
                fontName="Arial-Bold" if header and r == 0 else "Arial",
                fontSize=font_size,
                leading=font_size + 2,
                textColor=WHITE if header and r == 0 else INK,
            )
            rendered_row.append(P(str(cell), style))
        rendered.append(rendered_row)
    tbl = Table(rendered, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if row_bgs:
        start = 1 if header else 0
        for i in range(start, len(data)):
            if (i - start) % 2 == 1:
                commands.append(("BACKGROUND", (0, i), (-1, i), PALE))
    if aligns:
        for col, align in aligns.items():
            commands.append(("ALIGN", (col, 1 if header else 0), (col, -1), align))
    if highlight_rows:
        for row in highlight_rows:
            commands.extend(
                [
                    ("BACKGROUND", (0, row), (-1, row), PALE_ORANGE),
                    ("LINEABOVE", (0, row), (-1, row), 0.8, ORANGE),
                    ("LINEBELOW", (0, row), (-1, row), 0.8, ORANGE),
                ]
            )
    tbl.setStyle(TableStyle(commands))
    return tbl


def add_image(path: Path, width: float, height: float | None = None):
    img = Image(str(path), width=width)
    if height is not None:
        img.drawHeight = height
        img.drawWidth = width
    else:
        img.drawHeight = img.imageHeight * width / img.imageWidth
        img.drawWidth = width
    return img


def save_fig(fig, name: str) -> Path:
    path = ASSET_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def chart_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB9C0")
    ax.tick_params(colors="#5D6B75", labelsize=8)
    ax.grid(axis="y", color="#DDE5E8", linewidth=0.7)
    ax.set_axisbelow(True)


def build_charts() -> dict[str, Path]:
    assets: dict[str, Path] = {}
    common = pd.read_csv(ROOT / "outputs/monthly_tone_index/monthly_tone_index_common_sample.csv")
    common["month"] = pd.to_datetime(common["month"])

    fig, ax = plt.subplots(figsize=(9.1, 4.0))
    ax.plot(common.month, common.deepseek_tone_index, color="#E13D2D", lw=1.9, label="Direct LLM (DeepSeek)")
    ax.plot(common.month, common.t5_tone_index, color="#165DFF", lw=1.65, ls="--", alpha=.95, label="Fine-tuned T5")
    ax.axhline(0, color="#8798A0", lw=.8)
    ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-01"), color="#C77A28", alpha=.10)
    ax.set_ylabel("Tone index (−100 to +100)", fontsize=8.5, color="#5D6B75")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    chart_style(ax)
    fig.tight_layout()
    assets["tone_index"] = save_fig(fig, "tone_index.png")

    with open(ROOT / "outputs/deepseek_test_compact_v2/report.json", encoding="utf-8") as f:
        llm = json.load(f)
    with open(ROOT / "outputs/finetuning_compact_v2/finbert_head/experiment_results.json", encoding="utf-8") as f:
        finbert = json.load(f)["test"]
    with open(ROOT / "outputs/finetuning_compact_v2/finbert_full/experiment_results.json", encoding="utf-8") as f:
        econbert = json.load(f)["test"]
    with open(ROOT / "outputs/finetuning_compact_v2/t5/partial_experiment_results.json", encoding="utf-8") as f:
        t5 = json.load(f)["test"]
    benchmark = pd.DataFrame(
        [
            ["Direct LLM", llm["primary_topic"]["accuracy"], llm["primary_topic"]["macro_f1"], llm["overall_tone"]["accuracy"], llm["overall_tone"]["macro_f1"]],
            ["Direct LLM\n(relevant-only)", llm["conditional_on_both_relevant"]["primary_topic"]["accuracy"], llm["conditional_on_both_relevant"]["primary_topic"]["macro_f1"], llm["conditional_on_both_relevant"]["overall_tone"]["accuracy"], llm["conditional_on_both_relevant"]["overall_tone"]["macro_f1"]],
            ["FinBERT", finbert["primary_topic"]["accuracy"], finbert["primary_topic"]["macro_f1"], finbert["overall_tone"]["accuracy"], finbert["overall_tone"]["macro_f1"]],
            ["EconMediaBERT", econbert["primary_topic"]["accuracy"], econbert["primary_topic"]["macro_f1"], econbert["overall_tone"]["accuracy"], econbert["overall_tone"]["macro_f1"]],
            ["FLAN-T5", t5["primary_topic"]["accuracy"], t5["primary_topic"]["macro_f1"], t5["overall_tone"]["accuracy"], t5["overall_tone"]["macro_f1"]],
        ],
        columns=["model", "topic_accuracy", "topic_f1", "tone_accuracy", "tone_f1"],
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.65))
    x = np.arange(len(benchmark))
    w = .19
    series = [
        ("topic_accuracy", "Topic accuracy", "#15374B"),
        ("topic_f1", "Topic macro-F1", "#3B6D8C"),
        ("tone_accuracy", "Tone accuracy", "#0E766E"),
        ("tone_f1", "Tone macro-F1", "#C77A28"),
    ]
    for j, (col, label, color) in enumerate(series):
        pos = x + (j - 1.5) * w
        bars = ax.bar(pos, benchmark[col], w, color=color, label=label)
        for bar, value in zip(bars, benchmark[col]):
            ax.text(bar.get_x() + bar.get_width()/2, value + .013, f"{value:.2f}", ha="center", fontsize=6.6, color="#1F2933")
    ax.set_xticks(x, benchmark.model)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Held-out score", fontsize=8.5, color="#5D6B75")
    ax.legend(frameon=False, fontsize=7.4, ncol=4, loc="upper center")
    chart_style(ax)
    fig.tight_layout()
    assets["model_perf"] = save_fig(fig, "model_performance.png")

    aligned = pd.read_csv(ROOT / "outputs/monthly_regression_analysis/aligned_analysis_dataset.csv")
    aligned["month"] = pd.to_datetime(aligned["month"])
    fig, ax1 = plt.subplots(figsize=(9.1, 3.8))
    ax1.plot(aligned.month, aligned.michigan_sentiment, color="#15374B", lw=1.55, label="Michigan sentiment")
    ax1.set_ylabel("Consumer sentiment", color="#15374B", fontsize=8.5)
    ax2 = ax1.twinx()
    ax2.plot(aligned.month, aligned.news_tone_index_deepseek, color="#0E766E", lw=1.15, alpha=.8, label="Media tone")
    ax2.set_ylabel("Media tone index", color="#0E766E", fontsize=8.5)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    chart_style(ax1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#AAB9C0")
    ax2.tick_params(colors="#0E766E", labelsize=8)
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    assets["overview"] = save_fig(fig, "series_overview.png")

    forecast = pd.read_csv(ROOT / "outputs/monthly_regression_analysis/forecast_results.csv")
    forecast = forecast[forecast["outcome"] == "michigan_sentiment"].copy()
    forecast["month"] = pd.to_datetime(forecast["month"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.1, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.45, 1]})
    ax1.fill_between(forecast.month, forecast.econ_prediction_interval_low, forecast.econ_prediction_interval_high, color="#D9E8E4", alpha=.75, label="95% interval")
    ax1.plot(forecast.month, forecast.observed, color="#15374B", lw=1.7, label="Observed")
    ax1.plot(forecast.month, forecast.ols_econ, color="#0E766E", lw=1.35, label="Economic baseline")
    ax1.plot(forecast.month, forecast.ols_media, color="#C77A28", lw=1.05, label="Baseline + media")
    ax1.set_ylabel("Sentiment", fontsize=8.5)
    ax1.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper right")
    chart_style(ax1)
    ax2.axhline(0, color="#8798A0", lw=.8)
    ax2.fill_between(forecast.month, 0, forecast.sentiment_gap, where=forecast.sentiment_gap < 0, color="#B94D45", alpha=.6)
    ax2.fill_between(forecast.month, 0, forecast.sentiment_gap, where=forecast.sentiment_gap >= 0, color="#0E766E", alpha=.6)
    ax2.plot(forecast.month, forecast.sentiment_gap, color="#15374B", lw=.8)
    ax2.set_ylabel("Gap", fontsize=8.5)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    chart_style(ax2)
    fig.tight_layout()
    assets["gap"] = save_fig(fig, "sentiment_gap.png")

    media = pd.read_csv(ROOT / "outputs/monthly_regression_analysis/media_significance_summary.csv")

    def forest(ids, labels, name, xlim=None):
        sub = media.set_index("model_id").loc[ids].reset_index()
        fig, ax = plt.subplots(figsize=(8.4, 3.3))
        y = np.arange(len(sub))[::-1]
        xerr = np.vstack([sub.estimate - sub.ci_95_low, sub.ci_95_high - sub.estimate])
        cols = ["#0E766E" if p < .05 else "#C77A28" for p in sub.raw_p_value]
        for i in range(len(sub)):
            ax.errorbar(sub.estimate.iloc[i], y[i], xerr=xerr[:, i:i+1], fmt="o", color=cols[i], ecolor=cols[i], capsize=3, ms=5)
        ax.axvline(0, color="#8798A0", lw=.9)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Media-tone coefficient (95% HAC CI)", fontsize=8.5)
        if xlim:
            ax.set_xlim(*xlim)
        chart_style(ax)
        fig.tight_layout()
        return save_fig(fig, name)

    assets["basic_forest"] = forest(
        ["H_M2_econ_media", "H_M3_controls", "H_T5_M2"],
        ["Economic controls + LLM tone", "Full controls + LLM tone", "Economic controls + T5 tone"],
        "basic_association.png",
    )
    assets["dynamic_forest"] = forest(
        ["H_DYN_M2_media", "H_DYN_M3_controls", "H_DIFF_M2"],
        ["Lagged sentiment + media", "Lagged sentiment + full controls", "First-difference model"],
        "dynamic_association.png",
        xlim=(-.01, .19),
    )

    metrics = pd.read_csv(ROOT / "outputs/monthly_regression_analysis/forecast_metrics.csv")
    metrics = metrics[
        (metrics["outcome"] == "michigan_sentiment")
        & metrics.model.isin(["naive", "ols_econ", "ols_media", "ols_full"])
    ].copy()
    display = {"naive": "Last value", "ols_econ": "Economic", "ols_media": "+ media", "ols_full": "Full controls"}
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    bars = ax.bar([display[x] for x in metrics.model], metrics.rmse, color=["#8EA1AA", "#15374B", "#C77A28", "#0E766E"])
    for bar, v in zip(bars, metrics.rmse):
        ax.text(bar.get_x() + bar.get_width()/2, v + .07, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Holdout RMSE (lower is better)", fontsize=8.5)
    chart_style(ax)
    fig.tight_layout()
    assets["forecast"] = save_fig(fig, "forecast_rmse.png")

    iv = pd.read_csv(ROOT / "outputs/threshold_iv_analysis/posthoc_expanded_grid/control_lag_iv_model_summary.csv")
    ivrow = iv.loc[iv.model_id == "QUADRATIC_TREND"].iloc[0]
    iv = iv[iv["status"] == "estimated"].copy()
    iv["label"] = iv.model_id.str.replace("_", " ").str.title()
    iv = iv.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.3, 4.15))
    y = np.arange(len(iv))
    for i, row in iv.iterrows():
        selected = row.model_id == "QUADRATIC_TREND"
        color = "#0E766E" if selected else "#8EA1AA"
        ax.errorbar(
            row.iv_media_estimate,
            y[i],
            xerr=[[row.iv_media_estimate - row.iv_ci_95_low], [row.iv_ci_95_high - row.iv_media_estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.5,
            ms=5.6 if selected else 4.0,
            lw=1.5 if selected else 1.0,
        )
    ax.axvline(0, color="#8798A0", lw=.9)
    labels = [label + ("  [selected]" if model == "QUADRATIC_TREND" else "") for label, model in zip(iv.label, iv.model_id)]
    ax.set_yticks(y, labels)
    ax.set_xlabel("2SLS media-tone coefficient with 95% HAC CI", fontsize=8.5)
    chart_style(ax)
    fig.tight_layout()
    assets["iv"] = save_fig(fig, "iv_specification_forest.png")

    fs = pd.read_csv(ROOT / "outputs/threshold_iv_analysis/posthoc_expanded_grid/control_lag_first_stage_results.csv")
    fs = fs[fs.model_id == "QUADRATIC_TREND"].copy()
    label_map = {"z_u3_signed_r24": "U-3 rare event", "z_pcegrid_up_r24": "PCE up event", "z_pcegrid_down_r24": "PCE down event"}
    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    y = np.arange(len(fs))[::-1]
    xerr = np.vstack([fs.estimate - fs.ci_95_low, fs.ci_95_high - fs.estimate])
    for i in range(len(fs)):
        color = ["#8EA1AA", "#C77A28", "#0E766E"][i]
        ax.errorbar(fs.estimate.iloc[i], y[i], xerr=xerr[:, i:i+1], fmt="o", color=color, ecolor=color, capsize=3, ms=5)
    ax.axvline(0, color="#8798A0", lw=.8)
    ax.set_yticks(y, [label_map.get(x, x) for x in fs.instrument])
    ax.set_xlabel("First-stage coefficient with 95% HAC CI", fontsize=8.5)
    chart_style(ax)
    fig.tight_layout()
    assets["first_stage"] = save_fig(fig, "first_stage.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.55), gridspec_kw={"wspace": .34})
    for ax, direction, color in [(ax1, "Upward crossing", "#B94D45"), (ax2, "Downward crossing", "#0E766E")]:
        ax.axvline(0, color="#15374B", lw=1.2, ls="--")
        if direction.startswith("Up"):
            ax.annotate("", xy=(.62, .76), xytext=(-.62, .30), arrowprops=dict(arrowstyle="->", color=color, lw=2.1))
            ax.scatter([-.62, .62], [.30, .76], color=color, s=28, zorder=3)
            ax.text(.03, .92, "Zup = 1", color=color, fontsize=8, fontweight="bold")
        else:
            ax.annotate("", xy=(-.62, .30), xytext=(.62, .76), arrowprops=dict(arrowstyle="->", color=color, lw=2.1))
            ax.scatter([.62, -.62], [.76, .30], color=color, s=28, zorder=3)
            ax.text(.03, .92, "Zdown = 1", color=color, fontsize=8, fontweight="bold")
        ax.text(0, .04, "integer threshold k", ha="center", va="bottom", fontsize=7.4, color="#15374B")
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1.05)
        ax.set_title(direction, fontsize=9, color="#15374B", fontweight="bold")
        ax.set_xlabel("Released value relative to k", fontsize=7.5)
        ax.set_ylabel("previous → current release", fontsize=7.3)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#AAB9C0")
        ax.tick_params(axis="x", labelsize=7, colors="#5D6B75")
    fig.tight_layout()
    assets["threshold_design"] = save_fig(fig, "threshold_crossing_design.png")

    assets["dashboard"] = ASSET_DIR / "dashboard.png"
    return assets


def on_page(canvas, doc):
    page = canvas.getPageNumber()
    canvas.saveState()
    if page > 1:
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(.5)
        canvas.line(.72 * inch, 10.31 * inch, 7.78 * inch, 10.31 * inch)
        canvas.setFont("Arial-Bold", 6.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(.72 * inch, 10.43 * inch, "ECONOMIC PERCEPTION MONITOR")
        canvas.drawRightString(7.78 * inch, 10.43 * inch, "AI-ASSISTED FINANCIAL TEXT ANALYSIS | OCTOBER 2026")
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(.45)
    canvas.line(.72 * inch, .56 * inch, 7.78 * inch, .56 * inch)
    canvas.setFont("Arial", 6.7)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(4.25 * inch, .34 * inch, str(page))
    canvas.restoreState()


def build_pdf(assets: dict[str, Path]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=.72 * inch,
        rightMargin=.72 * inch,
        topMargin=.72 * inch,
        bottomMargin=.72 * inch,
        title="Economic Narratives, Consumer Sentiment, and the 2026 Midterms",
        author="Kuanhong Cai",
        subject="NLP methods, media tone indices, association, causal evidence, and dashboard",
    )
    story = []

    # Page 1 — cover and abstract
    story += [
        P(
            "student_ID : kc6534<br/>"
            "github : <link href='https://github.com/kc6534-JerryCai/NLP_final_project' "
            "color='#0E766E'>https://github.com/kc6534-JerryCai/NLP_final_project</link>",
            STUDENT_INFO,
        ),
        Spacer(1, .04 * inch),
    ]
    story += [P("ECONOMIC NARRATIVES, CONSUMER SENTIMENT,<br/>AND THE 2026 MIDTERMS", TITLE)]
    story += [P("An AI-assisted measurement, association, forecasting, and causal-diagnostic framework", SUBTITLE)]
    story += [P("Kuanhong Cai<br/>Applications in Finance · Final Project<br/>October 2026", AUTHOR)]
    story += [Spacer(1, .27 * inch)]
    story += [
        callout(
            "ABSTRACT",
            "This report builds a monthly measure of economic-news tone from New York Times articles and asks whether media narratives help explain U.S. consumer sentiment beyond measured macroeconomic conditions. Four NLP routes are compared: direct LLM labeling, a FinBERT baseline, a customized multi-task EconMediaBERT model, and a fine-tuned FLAN-T5 generator. The resulting tone series is joined to University of Michigan sentiment, unemployment, inflation, GDP, income, and price-level measures. Static regressions show only weak evidence, while dynamic and first-difference specifications are stronger. A final post-hoc 2SLS model produces a conventionally significant coefficient, but its first stage is concentrated in one instrument and the result fails multiple-testing and weak-instrument-robust confirmation. The evidence therefore supports association and useful monitoring—not a definitive causal claim. A public dashboard communicates the index, sentiment gap, releases, and diagnostics.",
        )
    ]
    story += [Spacer(1, .16 * inch), P("KEY RESULTS", LABEL), Spacer(1, 3)]
    story += [
        bullet("The direct-LLM and T5 indices cover January 2010 through October 2026; October is partial."),
        bullet("Static media association: +0.203 sentiment points per tone-index point, p = 0.070."),
        bullet("Dynamic association: +0.122, p &lt; 0.001, after adding lagged sentiment."),
        bullet("Post-hoc 2SLS: +0.141, raw p = 0.015; BH q = 0.134 and Anderson–Rubin p = 0.126."),
    ]
    story += [Spacer(1, .1 * inch), P("Keywords: economic news, consumer sentiment, FinBERT, FLAN-T5, LLM annotation, 2SLS, forecasting, dashboard", CAPTION)]
    story += [PageBreak()]

    # Page 2 — question and design
    story += page_title("1", "Research Question and Empirical Design", "Motivation")
    story += [callout("PRIMARY RESEARCH QUESTION", "Do economic-news narratives explain consumer sentiment beyond observable economic conditions—and, under stronger assumptions, do plausibly exogenous shifts in media tone cause changes in sentiment?")]
    story += [P("The project separates three increasingly demanding claims. Measurement asks whether economic narratives can be transformed into a reproducible monthly index. Association asks whether that index covaries with sentiment after controls. Causality asks whether an instrument shifts media tone without directly shifting sentiment.")]
    story += [P("ANALYTIC LADDER", H2)]
    ladder = [
        ["Stage", "Question", "Primary output", "Claim strength"],
        ["1. Measurement", "What is the tone and topic of economic coverage?", "Monthly LLM and T5 tone indices", "Descriptive"],
        ["2. Gap", "How far is observed sentiment from an economic-only forecast?", "Observed minus model prediction", "Diagnostic"],
        ["3. Association", "Does tone add explanatory power?", "Static, dynamic, and difference regressions", "Conditional correlation"],
        ["4. Causality", "Can threshold-event instruments isolate exogenous tone?", "2SLS plus weak-IV diagnostics", "Suggestive only"],
    ]
    story += [make_table(ladder, [1.05*inch, 2.15*inch, 2.15*inch, 1.47*inch], font_size=6.9)]
    story += [P("DATA MAP", H2)]
    data_map = [
        ["Source", "Coverage / role", "Frequency"],
        ["New York Times article corpus", "Rule-screened economic candidates; headline + abstract", "Article → month"],
        ["Direct LLM and fine-tuned NLP outputs", "Relevance, topic, tone, frame, review flags", "Article"],
        ["University of Michigan", "Headline consumer sentiment; current conditions; expectations", "Monthly"],
        ["BLS / BEA macro data", "U-3, U-6, PCE inflation, GDP, real disposable income", "Monthly / quarterly"],
        ["Release archive proxies", "Threshold events for IV timing", "Release month"],
    ]
    story += [make_table(data_map, [1.65*inch, 3.8*inch, 1.37*inch], font_size=7.0)]
    story += [P("The joined analysis contains 202 monthly observations from 2010-01 through 2026-10. The primary association sample is 185 observations ending in 2026-06 because publication lags and feature availability constrain the complete-case window.", BODY_SMALL)]
    story += [PageBreak()]

    # Page 3 — direct LLM
    story += page_title("2", "NLP Method I: Direct LLM Annotation", "Measurement")
    story += [P("A direct-labeling model converts each candidate article into a structured record. GPT-6 Luna compact-v2 was used to build and audit a 10,000-item teacher set; a DeepSeek pipeline then labeled the screened corpus used for the long monthly series. The prompt enforces a compact schema so aggregation never depends on free-form prose.")]
    schema = [
        ["Field", "Allowed values / meaning"],
        ["relevance", "relevant · irrelevant · uncertain"],
        ["primary_topic", "macroeconomy · labor market · policy · household economy · cost of living/income"],
        ["tone", "negative (−1) · neutral (0) · positive (+1)"],
        ["frame", "hardship · resilience · unequal outcomes · future risk · macro-household disconnect"],
        ["quality flags", "needs_review, parse/schema checks, evidence consistency"],
    ]
    story += [make_table(schema, [1.35*inch, 5.47*inch], font_size=7.2)]
    story += [P("PRODUCTION SCALE", H2)]
    prod = [
        ["Pipeline checkpoint", "Total", "Relevant", "Irrelevant", "Uncertain"],
        ["GPT teacher/audit set", "10,000", "5,662", "4,102", "236"],
        ["Rule-screened corpus sent to LLM", "41,273", "—", "—", "—"],
        ["DeepSeek direct labels", "41,273", "21,602", "18,159", "1,512"],
    ]
    story += [make_table(prod, [2.5*inch, 1.0*inch, 1.1*inch, 1.1*inch, 1.12*inch], font_size=7.1, aligns={1:"RIGHT",2:"RIGHT",3:"RIGHT",4:"RIGHT"})]
    story += [P("VALIDATION AND INTERPRETATION", H2)]
    story += [
        bullet("On a 600-row GPT-referenced test, DeepSeek relevance agreement was 82.0%; conditional topic and tone accuracies were 89.2% and 85.8% when both systems marked the item relevant."),
        bullet("The comparison target is another model’s label, not human gold. Agreement is therefore a consistency check, not proof of semantic truth."),
        bullet("The direct-LLM route is the richest method and powers the primary monthly index, but it is also the most prompt- and provider-dependent."),
    ]
    story += [callout("QUALITY DISCLOSURE", "The GPT teacher labels are machine-generated. Review flags and schema checks reduce obvious failures, but no claim in this report treats the teacher labels as a human-validated ground truth.", bg=PALE_ORANGE, border=ORANGE)]
    story += [PageBreak()]

    # Page 4 — FinBERT/EconMediaBERT
    story += page_title("3", "NLP Methods II–III: FinBERT and EconMediaBERT", "Discriminative models")
    story += [P("FinBERT supplies a finance-domain language representation. The project uses it in two related but distinct ways: a lightweight baseline with a largely frozen encoder and a customized EconMediaBERT multi-task classifier whose encoder and new heads are jointly fine-tuned.")]
    compare = [
        ["Component", "FinBERT baseline", "EconMediaBERT"],
        ["Base model", "ProsusAI/finbert", "ProsusAI/finbert"],
        ["Training scope", "New heads; encoder held fixed", "Full encoder + topic/tone heads"],
        ["Outputs", "Topic and tone class probabilities", "Joint primary-topic and tone predictions"],
        ["Trainable parameters", "6,152", "109,488,392"],
        ["Selected schedule", "Head LR 2e−4; epoch 3", "Encoder LR 3e−5; head LR 1e−4; epoch 5"],
        ["Use in project", "Low-cost reference baseline", "Main discriminative fine-tuning experiment"],
    ]
    story += [make_table(compare, [1.55*inch, 2.6*inch, 2.67*inch], font_size=7.0)]
    story += [P("MULTI-TASK OBJECTIVE", H2)]
    story += [callout("JOINT LOSS", "L = Ltopic + Ltone. Each cross-entropy term is masked when its target is unavailable. Input is the article headline plus abstract, truncated to 192 tokens. Model selection uses the average validation macro-F1 across topic and tone.")]
    story += [P("TRAINING DESIGN", H2)]
    design = [
        ["Element", "Specification"],
        ["Compact-v2 split", "4,500 train / 500 validation / 600 test; all four methods share the same GPT-referenced test set"],
        ["Optimization", "Effective batch 32; five epochs; encoder LR grid 1e−5, 2e−5, 3e−5"],
        ["Selection", "Average validation topic and tone macro-F1"],
        ["Inference", "One topic and one tone per article; missing relevance-filtered targets remain masked"],
    ]
    story += [make_table(design, [1.45*inch, 5.37*inch], font_size=7.1)]
    story += [P("Important scope condition: the archived FinBERT/EconMediaBERT predictions cover the held-out evaluation set, not the entire 2010–2026 corpus. They validate the architecture but are not silently substituted into the monthly long-run index.", BODY_SMALL)]
    story += [PageBreak()]

    # Page 5 — T5 and four-method comparison
    story += page_title("4", "NLP Method IV and Four-Method Benchmark", "Generative model and final comparison")
    story += [P("FLAN-T5 treats classification as constrained text generation. Each article is mapped to a short target such as <b>topic=economic_policy; tone=negative</b>. The final comparison below puts direct LLM labeling, the FinBERT heads-only baseline, full EconMediaBERT, and fine-tuned FLAN-T5 on the same 600-row compact-v2 test set.")]
    t5_design = [
        ["Element", "Selected configuration"],
        ["Base model", "google/flan-t5-base; full encoder, decoder, and output layer trained"],
        ["Grid", "Learning rates 3e−5 and 1e−4; five epochs"],
        ["Selected completed run", "LR 3e−5, epoch 5; the 1e−4 compact-v2 trial stopped before epoch 1 completed"],
        ["Batch / precision", "Effective batch 16; BF16; padding targets masked with −100"],
        ["Validation", "Strict generated-output parsing; average topic/tone macro-F1; no invalid test outputs"],
    ]
    story += [make_table(t5_design, [1.52*inch, 5.3*inch], font_size=7.0)]
    story += [Spacer(1, 4), add_image(assets["model_perf"], 6.55*inch, 2.76*inch)]
    story += [P("Figure 1. Accuracy and macro-F1 for four methods, with Direct LLM shown both on all 600 test rows and on the 492-row relevant-only subset. Macro-F1 gives equal importance to each class and is therefore essential under class imbalance.", CAPTION)]
    perf = [
        ["Method", "Rel. acc.", "Topic acc.", "Topic F1", "Tone acc.", "Tone F1", "Valid output"],
        ["Direct LLM", "82.0%", "73.17%", "0.776", "70.33%", "0.745", "69.5% strict schema"],
        ["<b>Direct LLM<br/>(relevant-only)</b>", "<b>n=492</b>", "<b>89.23%</b>", "<b>0.859</b>", "<b>85.77%</b>", "<b>0.818</b>", "<b>100% on subset</b>"],
        ["FinBERT", "N/A", "49.33%", "0.223", "66.50%", "0.504", "100% prediction"],
        ["EconMediaBERT", "N/A", "86.50%", "0.836", "76.17%", "0.657", "100% prediction"],
        ["FLAN-T5", "N/A", "86.33%", "0.825", "76.17%", "0.710", "100% format"],
    ]
    story += [make_table(perf, [1.25*inch, .75*inch, .85*inch, .85*inch, .85*inch, .85*inch, 1.42*inch], font_size=6.25, aligns={1:"RIGHT",2:"RIGHT",3:"RIGHT",4:"RIGHT",5:"RIGHT"})]
    story += [P("Direct LLM relevant-only row: conditional on both systems marking the article relevant (n=492), topic accuracy is 89.23%, tone accuracy is 85.77%, and joint exact match is 76.83%. All scores still measure agreement with GPT-generated labels, not accuracy against independent human gold.", BODY_TINY)]
    story += [PageBreak()]

    # Page 6 — indices
    story += page_title("5", "Monthly News-Tone Indices", "Time-series output")
    story += [callout("INDEX DEFINITION", "ToneIndex<sub>t</sub> = 100 × mean(tone<sub>i,t</sub>) = 100 × (N<sub>positive,t</sub> − N<sub>negative,t</sub>) / N<sub>valid,t</sub>. Each valid article receives equal weight; there is no smoothing, standardization, or length weighting.")]
    story += [Spacer(1, 6), add_image(assets["tone_index"], 6.82*inch, 3.0*inch)]
    story += [P("Figure 2. Monthly direct-LLM and fine-tuned-T5 tone indices on their common valid sample, January 2010–October 2026. The shaded region marks the initial COVID-19 shock. Higher values denote more positive economic coverage.", CAPTION)]
    index_table = [
        ["Property", "Direct LLM / DeepSeek", "Fine-tuned T5"],
        ["Valid article labels", "21,602", "23,114"],
        ["Common-valid comparison", "21,602", "21,602"],
        ["Tone support", "−1, 0, +1", "−1, 0, +1"],
        ["Primary role", "Main media measure", "Alternative measurement / robustness"],
    ]
    story += [make_table(index_table, [1.95*inch, 2.44*inch, 2.43*inch], font_size=7.1)]
    story += [P("Two cautions matter. First, the final month is partial through 4 October 2026. Second, monthly indices summarize the tone of selected NYT economic coverage; they are not a population-weighted measure of what every voter read.", BODY_SMALL)]
    story += [PageBreak()]

    # Page 7 — gap
    story += page_title("6", "The Sentiment Gap", "Economic baseline")
    story += [P("The gap converts a forecasting residual into an interpretable monitoring signal. An expanding-window economic model is estimated using information available before each holdout month; the residual is the difference between observed Michigan sentiment and the economic-only prediction.")]
    story += [callout("GAP DEFINITION", "SentimentGap<sub>t</sub> = ObservedSentiment<sub>t</sub> − EconomicOnlyPrediction<sub>t</sub>. A negative value indicates sentiment below this model’s baseline—not irrationality, partisan bias, or a causal media effect.")]
    story += [Spacer(1, 6), add_image(assets["gap"], 6.82*inch, 4.12*inch)]
    story += [P("Figure 3. Expanding-window holdout forecasts from August 2023 through October 2026. The top panel compares observed sentiment with economic-only and economic-plus-media predictions; the bottom panel shows the sentiment gap.", CAPTION)]
    gap_data = pd.read_csv(ROOT / "outputs/monthly_regression_analysis/sentiment_gap.csv")
    last_gap = gap_data.iloc[-1]
    story += [make_table([
        ["Latest available reading", "Observed", "Economic baseline", "Gap", "Interval status"],
        [str(last_gap["month"])[:7], f"{last_gap.observed:.1f}", f"{last_gap.ols_econ:.1f}", f"{last_gap.sentiment_gap:+.1f}", "Inside 95% interval" if bool(last_gap.interval_covered) else "Outside interval"],
    ], [1.55*inch, 1.1*inch, 1.45*inch, .9*inch, 1.82*inch], font_size=7.1)]
    story += [PageBreak()]

    # Page 8 — basic association
    story += page_title("7", "Association: Basic Model", "Static specification")
    story += [P("The basic model asks whether media tone is related to sentiment after conditioning on unemployment, inflation, and real GDP growth. HAC(3) standard errors allow short-run serial correlation and heteroskedasticity.")]
    story += [callout("BASIC MODEL", "Sentiment<sub>t</sub> = α + β Tone<sub>t</sub> + γ<sub>1</sub>U3<sub>t</sub> + γ<sub>2</sub>PCEInflation<sub>t</sub> + γ<sub>3</sub>RealGDPGrowth<sub>t</sub> + ε<sub>t</sub>")]
    story += [Spacer(1, 6), add_image(assets["basic_forest"], 6.6*inch, 2.58*inch)]
    story += [P("Figure 4. Static media-tone estimates and 95% HAC confidence intervals. The main direct-LLM estimate is positive but narrowly misses the 5% threshold; adding broader controls attenuates the coefficient.", CAPTION)]
    basic = [
        ["Specification", "β media", "HAC SE", "raw p", "BH q", "95% CI", "1-SD", "R²"],
        ["Economic controls + LLM", "0.203", "0.111", "0.070", "N/A<br/>(primary)", "[−0.017, 0.423]", "+2.58", "0.517"],
        ["Full controls + LLM", "0.060", "0.079", "0.449", "0.499", "[−0.096, 0.216]", "+0.76", "0.808"],
        ["Economic controls + T5", "0.068", "0.0849", "0.422", "0.496", "[−0.099, 0.236]", "+0.69", "0.493"],
    ]
    story += [make_table(basic, [1.72*inch, .63*inch, .66*inch, .58*inch, .70*inch, 1.13*inch, .70*inch, .70*inch], font_size=6.15, aligns={1:"RIGHT",2:"RIGHT",3:"RIGHT",4:"RIGHT",6:"RIGHT",7:"RIGHT"})]
    story += [P("BH q is not applicable to the pre-specified primary row; the correction is reported for the exploratory comparison family.", BODY_TINY)]
    story += [P("Magnitude: a one-standard-deviation increase in direct-LLM tone (12.71 index points) corresponds to about +2.58 sentiment points in the basic model. Because the coefficient is not conventionally significant and the model is contemporaneous, this result is best described as weak positive association.")]
    story += [callout("INTERPRETATION", "The basic specification is the cleanest cross-sectional-in-time comparison, but it omits the strong persistence in consumer sentiment. Its p-value should not be upgraded into evidence of causality.", bg=PALE_ORANGE, border=ORANGE)]
    story += [PageBreak()]

    # Page 9 — dynamic
    story += page_title("8", "Association: Dynamic and Difference Models", "Persistence and change")
    story += [P("Consumer sentiment is highly persistent. The dynamic specification adds lagged sentiment, while the first-difference specification relates month-to-month changes in sentiment to changes in media tone and macro variables. Both reduce the chance that a common slow-moving trend alone drives the association.")]
    story += [add_image(assets["dynamic_forest"], 6.62*inch, 2.62*inch)]
    story += [P("Figure 5. Dynamic and first-difference estimates. All three direct-LLM specifications shown here remain positive and conventionally significant after HAC inference and Benjamini–Hochberg correction.", CAPTION)]
    dynamic = [
        ["Specification", "β media", "HAC SE", "raw p", "BH q", "95% CI", "1-SD", "R²"],
        ["Lagged sentiment + media", "0.122", "0.0248", "2.1e−6", "2.4e−5", "[0.0727, 0.1707]", "+1.55", "0.932"],
        ["Lagged sentiment + full controls", "0.098", "0.0280", "0.00056", "0.00224", "[0.0431, 0.1537]", "+1.25", "0.941"],
        ["First-difference model", "0.094", "0.0247", "0.00020", "0.00101", "[0.0449, 0.1422]", "+1.17", "0.181"],
    ]
    story += [make_table(dynamic, [1.72*inch, .63*inch, .66*inch, .58*inch, .62*inch, 1.18*inch, .70*inch, .73*inch], font_size=5.95, aligns={1:"RIGHT",2:"RIGHT",3:"RIGHT",4:"RIGHT",6:"RIGHT",7:"RIGHT"})]
    story += [P("First-difference sample: n = 195; adjusted R² = 0.164. Because the dependent variable is the monthly change in sentiment, its R² is not directly comparable with the R² from level specifications.", BODY_TINY)]
    story += [P("The dynamic findings are the strongest association results in the project. They show that more positive media tone is aligned with higher sentiment even after conditioning on the previous month’s sentiment. They still do not resolve reverse causality: journalists may respond to the same emerging information that shifts households, or sentiment itself may affect which stories receive attention.")]
    story += [callout("ASSOCIATION VERDICT", "There is a robust conditional relationship in dynamic specifications, but the result remains observational. It supports monitoring and forecasting analysis, not a standalone causal statement.")]
    story += [PageBreak()]

    # Page 10 — forecasting
    story += page_title("9", "Forecasting and Practical Value", "Out-of-sample test")
    story += [P("Predictive value is evaluated with expanding-window one-step-ahead forecasts. The holdout begins in August 2023 and contains 38 monthly forecasts through October 2026. All model choices are compared to the same economic-only baseline and a last-value benchmark.")]
    story += [add_image(assets["forecast"], 6.35*inch, 2.78*inch)]
    story += [P("Figure 6. Root mean squared error across the holdout. Lower values are better.", CAPTION)]
    forecast_tbl = [
        ["Model", "MAE", "RMSE", "RMSE change vs. econ"],
        ["Last-value benchmark", "3.42", "4.20", "14.72% better"],
        ["Economic-only OLS", "4.31", "4.92", "Baseline"],
        ["Economic + media OLS", "4.33", "4.91", "0.19% better"],
        ["Full-controls OLS", "3.77", "4.40", "10.60% better"],
    ]
    story += [make_table(forecast_tbl, [2.4*inch, 1.0*inch, 1.0*inch, 2.42*inch], font_size=7.15, aligns={1:"RIGHT",2:"RIGHT",3:"RIGHT"})]
    story += [P("The media index contributes almost no incremental out-of-sample RMSE improvement to the simple economic model. The full-control model improves more, yet the naive last-value benchmark remains best. The economic model’s conformal interval covers 97.4% of holdout observations, so the gap is typically a within-range diagnostic rather than a forecasting failure.")]
    story += [P("FINANCIAL INTERPRETATION", H2)]
    story += [
        bullet("Use the media index as a regime and narrative monitor, not as an isolated trading signal."),
        bullet("The gap is most useful when paired with release calendars and uncertainty bands."),
        bullet("Strong in-sample significance does not automatically translate into forecast gains."),
    ]
    story += [PageBreak()]

    # Page 11 — causal design
    story += page_title("10", "Causality: Threshold-Crossing Natural Experiment", "Quasi-random instrumental-variables design")
    story += [P("The identification strategy is a <b>natural experiment</b>: conditional on the released level and change, crossing a salient round-number threshold is treated as a quasi-random shock to the news value of the release. The threshold indicator instruments monthly media tone. This is not a literal randomized controlled trial; the design is credible only if near-threshold crossings affect sentiment through media tone rather than through a direct economic or policy channel.")]
    story += [callout("INSTRUMENT ASSIGNMENT", "For integer threshold k: Z<sub>up,t</sub> = 1(previous value &lt; k and current value ≥ k); Z<sub>down,t</sub> = 1(previous value ≥ k and current value &lt; k). Core thresholds are U-3 unemployment at 4%–10% and PCE inflation at 3%–7%, with a 24-month rarity rule.")]
    story += [add_image(assets["threshold_design"], 6.42*inch, 1.92*inch)]
    story += [P("Figure 7. Threshold-crossing assignment. A release that barely crosses a salient integer activates the instrument; releases that stay on the same side do not. Controls absorb the continuous economic level and change.", CAPTION)]
    iv_design = [
        ["Econometric stage", "Specification"],
        ["First stage", "Tone<sub>t</sub> = πZ<sub>t</sub> + δX<sub>t</sub> + u<sub>t</sub>"],
        ["Reduced form", "Sentiment<sub>t</sub> = ρZ<sub>t</sub> + θX<sub>t</sub> + v<sub>t</sub>"],
        ["2SLS", "Sentiment<sub>t</sub> = β Tone-hat<sub>t</sub> + γX<sub>t</sub> + e<sub>t</sub>"],
        ["Selected post-hoc extension", "Signed U-3 integer crossing plus separate PCE up/down crossings on a 0.5-point grid, including 2%; quadratic time trend"],
    ]
    story += [make_table(iv_design, [1.55*inch, 5.27*inch], font_size=6.8)]
    story += [P("SELECTED MODEL: FIRST STAGE", H2)]
    story += [add_image(assets["first_stage"], 6.15*inch, 1.72*inch)]
    story += [P("Figure 8. First-stage coefficients and 95% HAC confidence intervals. Joint F-type = 15.08 and partial R² = 0.074, but the aggregate strength is concentrated in the PCE-down instrument.", CAPTION)]
    story += [PageBreak()]

    # Page 12 — final IV result
    story += page_title("11", "Causal Result: Significant, but Not Confirmatory", "2SLS estimates and specification sensitivity")
    story += [P("The final significant result is the <b>QUADRATIC_TREND</b> specification. It estimates a positive effect of instrumented media tone on Michigan sentiment. The coefficient plot below places that result inside the complete fixed post-hoc control/timing menu rather than displaying it in isolation.")]
    story += [add_image(assets["iv"], 6.42*inch, 2.42*inch)]
    story += [P("Figure 9. 2SLS media-tone coefficients and 95% HAC confidence intervals across the twelve post-hoc specifications. The highlighted quadratic-trend estimate is the final raw-significant result; several intervals are wide and cross zero.", CAPTION)]
    iv_tbl = [
        ["Dependent variable: Michigan sentiment", "OLS", "2SLS"],
        ["<b>Panel A. Structural equation</b>", "", ""],
        ["Media-tone index", "0.119***", "0.141**"],
        ["HAC standard error", "(0.038)", "(0.057)"],
        ["95% confidence interval", "[0.043, 0.194]", "[0.028, 0.254]"],
        ["Observations", "175", "175"],
        ["Macro controls; linear + quadratic trend", "Yes", "Yes"],
        ["HAC bandwidth", "3", "3"],
        ["<b>Panel B. Identification diagnostics</b>", "", ""],
        ["Excluded threshold instruments", "N/A", "3"],
        ["First-stage F-type / partial R² / joint p", "N/A", "15.08 / 0.074 / 8.2e-10"],
        ["Anderson-Rubin p / BH q", "N/A", "0.1257 / 0.2516"],
        ["Over-identification J (p-value)", "N/A", "0.034 (0.9831)"],
        ["Conventional 2SLS p / BH q", "N/A", "0.0152 / 0.1336"],
    ]
    econ_tbl = make_table(iv_tbl, [3.72*inch, 1.45*inch, 1.65*inch], font_size=5.9, row_bgs=False, aligns={1:"CENTER",2:"CENTER"})
    econ_tbl.setStyle(TableStyle([
        ("SPAN", (0, 1), (-1, 1)),
        ("SPAN", (0, 8), (-1, 8)),
        ("BACKGROUND", (0, 1), (-1, 1), PALE_BLUE),
        ("BACKGROUND", (0, 8), (-1, 8), PALE_BLUE),
        ("TEXTCOLOR", (0, 1), (-1, 1), NAVY),
        ("TEXTCOLOR", (0, 8), (-1, 8), NAVY),
        ("TOPPADDING", (0, 0), (-1, -1), 2.3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3),
    ]))
    story += [econ_tbl]
    story += [P("Notes: HAC(3) standard errors. * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01 using unadjusted tests. Instruments are the signed rare U-3 crossing and separate PCE upward/downward crossings. The specification is post-hoc; BH q-values and the weak-IV-robust Anderson-Rubin test govern the confirmatory interpretation.", BODY_TINY)]
    story += [P("WHY THE FIRST STAGE HAS FLAWS", H2)]
    story += [
        bullet("Two of three instruments are individually weak; nearly all first-stage signal comes from PCE-down events."),
        bullet("Only seven U-3 and six PCE event months enter the 175-observation sample, making the design sensitive to sparse episodes."),
        bullet("The control/timing specification was examined after earlier results, so the raw p-value carries post-selection risk."),
        bullet("Inflation-threshold events may affect confidence directly through prices and policy expectations, threatening exclusion."),
    ]
    story += [callout("CAUSALITY VERDICT", "The result is suggestive and worth reporting, but it cannot 100% confirm causality. The defensible statement is: one post-hoc 2SLS specification is significant by a conventional raw test, while multiple-testing and weak-instrument-robust evidence do not confirm the effect.", bg=PALE_ORANGE, border=ORANGE)]
    story += [PageBreak()]

    # Page 13 — dashboard
    story += page_title("12", "Interactive Dashboard", "Communication layer")
    story += [P("The dashboard packages the research outputs for presentation and monitoring. It reports the latest sentiment, gap, media tone, causal-evidence status, release calendar, diagnostics, and data-health warnings without requiring the audience to read regression tables.")]
    story += [add_image(assets["dashboard"], 6.82*inch, 4.69*inch)]
    story += [P("Figure 10. US Economic Perception Monitor, captured 16 October 2026. The landing view displays the latest headline sentiment, sentiment gap, media index, explicit causal verdict, uncertainty interval, and monitoring cutoff.", CAPTION)]
    dash_features = [
        ["Dashboard section", "Purpose"],
        ["Overview", "Latest state, observed-versus-predicted chart, gap and media signal"],
        ["Sentiment gap", "Residual history and uncertainty-aware interpretation"],
        ["Media signals", "Direct-LLM and T5 tone indices with topic context"],
        ["Releases", "Macro-release timing and event context"],
        ["Diagnostics / data health", "Model caveats, causal tests, coverage and partial-month warnings"],
    ]
    story += [make_table(dash_features, [1.85*inch, 4.97*inch], font_size=7.0)]
    story += [P("Public URL: <link href='https://economic-perception-monitor-2026.alert-dream-4640.chatgpt.site' color='#0E766E'>economic-perception-monitor-2026.alert-dream-4640.chatgpt.site</link>", BODY_SMALL)]
    story += [PageBreak()]

    # Page 14 — limitations
    story += page_title("13", "Limitations, Reproducibility, and Responsible Use", "Research controls")
    limitations = [
        ["Risk", "Why it matters", "Mitigation / disclosure"],
        ["Machine-generated labels", "Teacher errors can propagate to fine-tuned models", "Schema validation, review flags, cross-model agreement; no human-gold claim"],
        ["Single-outlet corpus", "NYT coverage is not the full information environment", "Interpret index as outlet-specific; add outlets in future work"],
        ["Incomplete NLP coverage", "EconMediaBERT has test-set predictions only", "Long series uses direct LLM and T5; scope stated explicitly"],
        ["Revised macro data", "Pseudo-out-of-sample inputs may differ from real-time vintages", "Release proxies used in IV; dashboard flags availability"],
        ["Survey transition", "Changes in Michigan collection may affect comparability", "Treat as a data limitation; test breaks in extensions"],
        ["Multiple testing", "Exploratory specifications inflate false positives", "BH correction and Anderson–Rubin reported beside raw p-values"],
        ["Exclusion restriction", "Macro events may directly move confidence", "Causal result labeled suggestive, not confirmed"],
    ]
    story += [make_table(limitations, [1.45*inch, 2.42*inch, 2.95*inch], font_size=6.6)]
    story += [P("REPRODUCIBLE OUTPUT CHAIN", H2)]
    chain = [
        ["Layer", "Saved artifact"],
        ["Annotations", "Structured JSON/CSV labels, progress files, validation reports"],
        ["NLP training", "Experiment histories, selected checkpoints, label mappings, test metrics"],
        ["Indices", "Monthly direct-LLM/T5 CSVs, common-sample audit, topic-level series"],
        ["Econometrics", "Aligned dataset, coefficient tables, forecasts, gaps, IV diagnostics"],
        ["Communication", "Static web bundle, public dashboard, this final PDF"],
    ]
    story += [make_table(chain, [1.5*inch, 5.32*inch], font_size=7.0)]
    story += [callout("RESPONSIBLE-USE RULE", "The dashboard and report are research tools. They should not be presented as voter-level persuasion measurement, proof of media bias, or a validated trading strategy.", bg=PALE_ORANGE, border=ORANGE)]
    story += [PageBreak()]

    # Page 15 — conclusion and references
    story += page_title("14", "Conclusion and References", "Synthesis")
    story += [P("The project demonstrates a complete finance-oriented NLP workflow: structured LLM annotation, discriminative and generative fine-tuning, monthly index construction, macroeconomic alignment, association testing, causal diagnostics, forecasting, and dashboard delivery. The methods agree on the broad feasibility of measuring economic narratives, while the econometric evidence is more nuanced.")]
    conclusion = [
        ["Question", "Answer supported by the evidence"],
        ["Can economic news tone be measured?", "Yes. Direct LLM and T5 pipelines produce auditable monthly indices from 2010–2026."],
        ["Is media tone associated with sentiment?", "Yes in dynamic and difference models; static evidence is weaker."],
        ["Does media improve forecasting?", "Only marginally in the simple holdout comparison; a naive benchmark remains difficult to beat."],
        ["Is the relationship causal?", "Not confirmed. One final 2SLS estimate is raw-significant, but stronger diagnostics do not validate it."],
        ["What is the practical contribution?", "A transparent monitoring system that keeps uncertainty and identification limits visible."],
    ]
    story += [make_table(conclusion, [2.15*inch, 4.67*inch], font_size=7.1)]
    story += [P("REFERENCES AND DATA SOURCES", H2)]
    refs = [
        "Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.",
        "Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.",
        "University of Michigan, Surveys of Consumers: Index of Consumer Sentiment, Current Conditions, and Expectations.",
        "U.S. Bureau of Labor Statistics: U-3 unemployment and U-6 labor underutilization series and release archives.",
        "U.S. Bureau of Economic Analysis: PCE price index, real GDP, real disposable personal income, and release materials.",
        "New York Times article corpus used under the project’s local data-access workflow.",
        "Project outputs: monthly index methodology, fine-tuning report, monthly regression findings, and post-hoc IV sensitivity report (October 2026).",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(P(f"{i}. {ref}", BODY_SMALL))
    story += [Spacer(1, 5), callout("FINAL TAKEAWAY", "Economic narratives contain information about consumer sentiment, particularly in models that account for persistence. The evidence is useful for measurement and monitoring, but the causal channel remains an open research question.")]

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    assets = build_charts()
    missing = [str(path) for path in assets.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required assets: {missing}")
    build_pdf(assets)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
