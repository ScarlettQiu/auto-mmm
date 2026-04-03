"""Generate markdown + PowerPoint analysis report from MMM results.

Usage:
    python report_builder.py                  # from latest results
    python report_builder.py --round 3
    python report_builder.py --summary "Claude's interpretation text"
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from compare import (
    contribution_comparison,
    disagreements,
    model_fit_summary,
    roi_comparison,
    top_channels,
)


# ── Markdown report ──────────────────────────────────────────────────────────

def build_markdown(results: dict, summary_text: str = "") -> str:
    roi_df     = roi_comparison(results)
    contrib_df = contribution_comparison(results)
    fit_df     = model_fit_summary(results)
    top        = top_channels(results)
    disc       = disagreements(roi_df)

    run_at = results.get("run_at", "")[:10]
    lines = [
        f"# MMM Analysis Report",
        f"*Generated {run_at} · {results.get('train_periods', '?')} training periods · "
        f"{results.get('test_periods', '?')} holdout periods*",
        "",
    ]

    if summary_text:
        lines += ["## Executive Summary", "", summary_text, ""]

    lines += [
        "## Model Fit",
        "",
        fit_df.to_markdown() if hasattr(fit_df, "to_markdown") else fit_df.to_string(),
        "",
        "## ROI by Channel",
        "",
        "*Higher = more revenue generated per unit of spend. "
        "Agreement column reflects cross-model consistency.*",
        "",
        roi_df.to_markdown() if hasattr(roi_df, "to_markdown") else roi_df.to_string(),
        "",
        "## Channel Contribution (%)",
        "",
        "*Share of total KPI attributed to each channel.*",
        "",
        contrib_df.to_markdown() if hasattr(contrib_df, "to_markdown") else contrib_df.to_string(),
        "",
        "## Top Channels by Model",
        "",
    ]
    for model, chans in top.items():
        lines.append(f"- **{model}**: {', '.join(chans)}")
    lines.append("")

    if not disc.empty:
        lines += [
            "## High Disagreement Channels",
            "",
            "*These channels should be interpreted with caution — models differ significantly.*",
            "",
            disc.to_markdown() if hasattr(disc, "to_markdown") else disc.to_string(),
            "",
        ]
    else:
        lines += ["## Model Agreement", "", "All three models broadly agree on channel ROI rankings.", ""]

    lines += [
        "## Data Caveat",
        "",
        f"This analysis uses {results.get('data_periods', '?')} monthly periods. "
        "MMM typically benefits from 2+ years of weekly data. "
        "Interpret uncertainty ranges with this in mind.",
        "",
    ]

    return "\n".join(lines)


# ── PowerPoint report ─────────────────────────────────────────────────────────

def build_pptx(results: dict, summary_text: str = "") -> io.BytesIO:
    try:
        from pptx import Presentation
        from pptx.chart.data import ChartData
        from pptx.dml.color import RGBColor
        from pptx.enum.chart import XL_CHART_TYPE
        from pptx.util import Inches, Pt
    except ImportError:
        raise ImportError("python-pptx not installed. Run: pip install python-pptx")

    W = Inches(13.33)
    H = Inches(7.5)
    MARGIN = Inches(0.6)
    BLUE  = RGBColor(0x1F, 0x77, 0xB4)
    DARK  = RGBColor(0x1A, 0x1A, 0x2E)
    GRAY  = RGBColor(0xF0, 0xF4, 0xF8)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    MID   = RGBColor(0x8A, 0x9B, 0xB0)

    MODEL_COLORS = {
        "ridge":           RGBColor(0x1F, 0x77, 0xB4),
        "pymc":            RGBColor(0xFF, 0x7F, 0x0E),
        "lightweight_mmm": RGBColor(0x2C, 0xA0, 0x2C),
    }

    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H

    def blank():
        return prs.slides.add_slide(prs.slide_layouts[6])

    def rect(slide, l, t, w, h, color):
        s = slide.shapes.add_shape(1, l, t, w, h)
        s.fill.solid(); s.fill.fore_color.rgb = color
        s.line.fill.background()
        return s

    def text(slide, l, t, w, h, txt, size=12, bold=False, color=DARK, wrap=True, font=None):
        txb = slide.shapes.add_textbox(l, t, w, h)
        tf = txb.text_frame; tf.word_wrap = wrap
        p = tf.paragraphs[0]; p.text = txt
        run = p.runs[0]
        run.font.size = Pt(size); run.font.bold = bold
        run.font.color.rgb = color
        if font: run.font.name = font
        return txb

    def header(slide, title, dark=False):
        rect(slide, 0, 0, W, Inches(1.05), DARK if dark else BLUE)
        text(slide, MARGIN, Inches(0.18), W - MARGIN, Inches(0.7),
             title, size=22, bold=True, color=WHITE)

    run_at = results.get("run_at", "")[:10]
    n_models = sum(1 for r in results["models"].values() if not r.get("skipped"))

    # ── Slide 1: Title ────────────────────────────────────────────────────
    s = blank()
    rect(s, 0, 0, Inches(0.35), H, BLUE)
    text(s, Inches(0.85), Inches(2.0), Inches(11.5), Inches(1.4),
         "Marketing Mix Model Analysis", size=34, bold=True, color=DARK)
    text(s, Inches(0.85), Inches(3.7), Inches(11.5), Inches(0.6),
         f"{results.get('train_periods','?')} training periods  ·  "
         f"{n_models} models compared  ·  {run_at}",
         size=14, color=MID)

    # ── Slide 2: Model fit ────────────────────────────────────────────────
    s = blank()
    header(s, "Model Fit Comparison")
    fit_df = model_fit_summary(results)

    y = Inches(1.2)
    for model_name, row in fit_df.iterrows():
        color = MODEL_COLORS.get(model_name, BLUE)
        rect(s, MARGIN, y, Inches(0.08), Inches(0.7), color)
        label = f"{model_name}   R²={row['train_r2']}   Train MAPE={row['train_mape']}   Test MAPE={row['test_mape']}"
        text(s, MARGIN + Inches(0.2), y + Inches(0.1),
             W - MARGIN * 2, Inches(0.5), label, size=13, color=DARK)
        y += Inches(0.95)

    if summary_text:
        text(s, MARGIN, Inches(4.5), W - MARGIN * 2, Inches(2.7),
             summary_text[:600], size=10, color=DARK)

    # ── Slide 3: ROI comparison chart ─────────────────────────────────────
    roi_df = roi_comparison(results)
    if not roi_df.empty:
        s = blank()
        header(s, "ROI by Channel — All Models")
        model_cols = [c for c in roi_df.columns if c in MODEL_COLORS]
        channels = roi_df.index.tolist()

        chart_data = ChartData()
        chart_data.categories = channels
        for m in model_cols:
            chart_data.add_series(m, [round(float(roi_df.loc[ch, m]), 4) for ch in channels])

        cs = s.shapes.add_chart(
            XL_CHART_TYPE.COLUMN_CLUSTERED,
            MARGIN, Inches(1.15), W - MARGIN * 2, Inches(6.0), chart_data
        )
        chart = cs.chart
        chart.has_legend = True
        for i, m in enumerate(model_cols):
            chart.series[i].format.fill.solid()
            chart.series[i].format.fill.fore_color.rgb = MODEL_COLORS.get(m, BLUE)

    # ── Slide 4: Contribution % ───────────────────────────────────────────
    contrib_df = contribution_comparison(results)
    if not contrib_df.empty:
        s = blank()
        header(s, "Channel Contribution % — All Models")
        model_cols = [c for c in contrib_df.columns if c in MODEL_COLORS]
        channels = contrib_df.index.tolist()

        chart_data = ChartData()
        chart_data.categories = channels
        for m in model_cols:
            chart_data.add_series(m, [round(float(contrib_df.loc[ch, m]), 2) for ch in channels])

        cs = s.shapes.add_chart(
            XL_CHART_TYPE.BAR_CLUSTERED,
            MARGIN, Inches(1.15), W - MARGIN * 2, Inches(6.0), chart_data
        )
        chart = cs.chart
        chart.has_legend = True
        for i, m in enumerate(model_cols):
            chart.series[i].format.fill.solid()
            chart.series[i].format.fill.fore_color.rgb = MODEL_COLORS.get(m, BLUE)

    # ── Slide 5: Agreement / disagreement table ───────────────────────────
    s = blank()
    header(s, "Cross-Model Agreement Summary")

    if not roi_df.empty:
        agree_df = roi_df[["mean_roi", "cv_pct", "agreement"]].reset_index()
        n = len(agree_df) + 1
        table = s.shapes.add_table(
            n, 4, MARGIN, Inches(1.2),
            W - MARGIN * 2, Inches(min(5.8, n * 0.55))
        ).table
        table.columns[0].width = Inches(3.5)
        table.columns[1].width = Inches(2.5)
        table.columns[2].width = Inches(2.0)
        table.columns[3].width = Inches(3.0)

        for j, col in enumerate(["Channel", "Mean ROI", "CV %", "Agreement"]):
            cell = table.cell(0, j)
            cell.text = col
            cell.fill.solid(); cell.fill.fore_color.rgb = BLUE
            for para in cell.text_frame.paragraphs:
                for run in para.runs:
                    run.font.bold = True; run.font.size = Pt(10)
                    run.font.color.rgb = WHITE

        for i, (_, row) in enumerate(agree_df.iterrows()):
            bg = GRAY if i % 2 == 0 else WHITE
            vals = [row["channel"], f"{row['mean_roi']:.4f}", f"{row['cv_pct']:.1f}%", row["agreement"]]
            for j, val in enumerate(vals):
                cell = table.cell(i + 1, j)
                cell.text = str(val)
                cell.fill.solid(); cell.fill.fore_color.rgb = bg
                for para in cell.text_frame.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(9); run.font.color.rgb = DARK

    # ── Slide 6: Recommendations ──────────────────────────────────────────
    s = blank()
    header(s, "Key Takeaways & Recommendations")

    top = top_channels(results)
    bullet_y = Inches(1.3)

    text(s, MARGIN, bullet_y, W - MARGIN * 2, Inches(0.4),
         "Top Channels by ROI (model consensus):", size=13, bold=True, color=DARK)
    bullet_y += Inches(0.5)

    # Find channels that appear in top 3 for multiple models
    from collections import Counter
    all_top = [ch for chans in top.values() for ch in chans]
    consensus = [ch for ch, cnt in Counter(all_top).most_common() if cnt >= 2]
    text(s, MARGIN + Inches(0.2), bullet_y, W - MARGIN * 2, Inches(0.4),
         "  ·  ".join(consensus) if consensus else "See ROI slide for per-model rankings",
         size=12, color=DARK)
    bullet_y += Inches(0.7)

    text(s, MARGIN, bullet_y, W - MARGIN * 2, Inches(0.4),
         "Data Caveat:", size=13, bold=True, color=DARK)
    bullet_y += Inches(0.4)
    text(s, MARGIN + Inches(0.2), bullet_y, W - MARGIN * 2, Inches(0.8),
         f"{results.get('data_periods','?')} monthly periods used. "
         "Recommend collecting 2+ years of weekly data for production MMM.",
         size=11, color=MID)

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--summary", default="")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--no-pptx", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_dir = Path(cfg.get("results_dir", "./results"))
    out_dir.mkdir(exist_ok=True)

    results_path = (
        Path("rounds") / f"R{args.round:02d}_results.json"
        if args.round else out_dir / "latest.json"
    )
    results = json.loads(results_path.read_text())

    # Markdown
    md = build_markdown(results, args.summary)
    md_path = out_dir / "report.md"
    md_path.write_text(md)
    print(f"Markdown report → {md_path}")
    print(md)

    # PowerPoint
    if not args.no_pptx:
        try:
            buf = build_pptx(results, args.summary)
            pptx_path = out_dir / "report.pptx"
            pptx_path.write_bytes(buf.read())
            print(f"PowerPoint deck → {pptx_path}")
        except ImportError as e:
            print(f"Skipping pptx: {e}")


if __name__ == "__main__":
    main()
