"""
update_presentation.py — patches presentation.html with the latest round results.

Run after each round:
    python update_presentation.py

Reads:
    results/roi_comparison.csv
    results/contribution_comparison.csv
    results/model_fit.csv
    state.json

Writes:
    presentation.html  (JS data blocks + title + round labels)
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
PRESENTATION = ROOT / "presentation.html"


def load_results():
    roi = pd.read_csv(ROOT / "results" / "roi_comparison.csv")
    contrib = pd.read_csv(ROOT / "results" / "contribution_comparison.csv")
    fit = pd.read_csv(ROOT / "results" / "model_fit.csv")
    state = json.loads((ROOT / "state.json").read_text())
    return roi, contrib, fit, state


def build_roi_js(roi_df):
    """Build the ROI JS array from roi_comparison.csv."""
    # Use lightweight_mmm (NNLS) as the display ROI; fallback to mean_roi
    rows = []
    for _, r in roi_df.iterrows():
        ch = r["channel"]
        val = float(r.get("lightweight_mmm", r["mean_roi"]))
        # Color logic
        if val >= 10:
            color = "#E07070"   # warning — likely artefact
            lbl = f"{ch} ⚠️"
        elif val >= 3:
            color = "#A91E2C"
            lbl = ch
        elif val > 0:
            color = "#D63031"
            lbl = ch
        else:
            color = "#1a2240"   # navy-dark for zero channels
            lbl = ch
        top = val == roi_df["lightweight_mmm"].max() and val > 0
        rows.append(f"  {{ lbl:'{lbl}', roi:{val:.3f}, color:'{color}', top:{'true' if top else 'false'} }}")
    return "// AUTO-UPDATE: ROI_DATA_START\nconst ROI = [\n" + ",\n".join(rows) + "\n];\n// AUTO-UPDATE: ROI_DATA_END"


def build_contrib_js(contrib_df):
    """Build NNLS contribution JS array from contribution_comparison.csv."""
    colors = {
        "meta_facebook":   "#A91E2C",
        "meta_instagram":  "#D63031",
        "google_shopping": "#c06060",
        "google_search":   "#E07070",
    }
    default_color = "#d0d0da"

    # Filter to channels with NNLS > 0
    nnls_rows = contrib_df[contrib_df["lightweight_mmm"] > 0].sort_values("lightweight_mmm", ascending=False)

    ridge_js = 'const RIDGE = [\n  { lbl:\'Baseline\',  pct:100.0, color:\'#3a3a44\' },\n  { lbl:\'All Media (zeroed)\', pct:0.0, color:\'#A91E2C\' },\n];'

    nnls_entries = []
    for _, r in nnls_rows.iterrows():
        ch = r["channel"]
        pct = float(r["lightweight_mmm"])
        # Pretty label
        lbl_map = {
            "meta_facebook": "Meta Facebook", "meta_instagram": "Meta Instagram",
            "google_facebook": "Meta Facebook", "google_search": "Google Search",
            "google_shopping": "Google Shopping", "google_pmax": "Google PMax",
            "google_display": "Google Display", "google_video": "Google Video",
            "meta_other": "Meta Other",
        }
        lbl = lbl_map.get(ch, ch.replace("_", " ").title())
        color = colors.get(ch, default_color)
        nnls_entries.append(f"  {{ lbl:'{lbl}', pct:{pct:.2f}, color:'{color}' }}")

    nnls_js = "const NNLS = [\n" + ",\n".join(nnls_entries) + "\n];"

    return f"// AUTO-UPDATE: CONTRIB_DATA_START\n{ridge_js}\n{nnls_js}\n// AUTO-UPDATE: CONTRIB_DATA_END"


def get_best_mape(fit_df):
    """Return best (model_name, test_mape) across all models."""
    best_row = fit_df.loc[fit_df["test_mape"].str.rstrip("%").astype(float).idxmin()]
    return best_row["model"], best_row["test_mape"]


def patch_html(html, roi_js, contrib_js, state, fit_df):
    round_n = state["current_round"]

    # ── JS data blocks ─────────────────────────────────────────────────
    html = re.sub(
        r"// AUTO-UPDATE: ROI_DATA_START.*?// AUTO-UPDATE: ROI_DATA_END",
        roi_js, html, flags=re.DOTALL
    )
    html = re.sub(
        r"// AUTO-UPDATE: CONTRIB_DATA_START.*?// AUTO-UPDATE: CONTRIB_DATA_END",
        contrib_js, html, flags=re.DOTALL
    )

    # ── Title ──────────────────────────────────────────────────────────
    html = re.sub(
        r"<title>Auto-MMM — Round \d+ Results</title>",
        f"<title>Auto-MMM — Round {round_n} Results</title>",
        html
    )

    # ── Badge ──────────────────────────────────────────────────────────
    html = re.sub(
        r"Round \d+ · Critic Approved ✓",
        f"Round {round_n} · Critic Approved ✓",
        html
    )

    # ── PyMC MAPE ──────────────────────────────────────────────────────
    pymc_row = fit_df[fit_df["model"] == "pymc"].iloc[0]
    pymc_mape_raw = float(pymc_row["test_mape"].rstrip("%"))
    pymc_r2 = float(pymc_row["train_r2"])
    color = "#5CC45C" if pymc_mape_raw < 20 else "#E07070"
    html = re.sub(
        r'<div class="mape-n" style="color:#[0-9A-Fa-f]+;">[0-9.]+%</div>\s*<div class="mape-l">Test MAPE · R²=[0-9.]+</div>',
        f'<div class="mape-n" style="color:{color};">{pymc_mape_raw:.1f}%</div>\n      <div class="mape-l">Test MAPE · R²={pymc_r2:.2f}</div>',
        html
    )

    # ── S6 sub line ────────────────────────────────────────────────────
    from pandas import read_csv
    contrib = read_csv(ROOT / "results" / "contribution_comparison.csv")
    fb_pct = contrib.loc[contrib["channel"] == "meta_facebook", "lightweight_mmm"].values
    ig_pct = contrib.loc[contrib["channel"] == "meta_instagram", "lightweight_mmm"].values
    if len(fb_pct) and len(ig_pct):
        html = re.sub(
            r"Meta Facebook [0-9.]+% · Meta Instagram [0-9.]+% · NNLS only · Round \d+",
            f"Meta Facebook {fb_pct[0]:.1f}% · Meta Instagram {ig_pct[0]:.1f}% · NNLS only · Round {round_n}",
            html
        )

    # ── NNLS Contribution label ────────────────────────────────────────
    html = re.sub(
        r"NNLS Contribution \(Round \d+\)",
        f"NNLS Contribution (Round {round_n})",
        html
    )

    # ── ROI highlight label ────────────────────────────────────────────
    html = re.sub(
        r"(Meta Instagram ROI · Round) \d+",
        f"\\1 {round_n}",
        html
    )

    # ── Bottom line MAPE ───────────────────────────────────────────────
    html = re.sub(
        r"Best test accuracy reached [0-9.]+% \(Round \d+ PyMC\)",
        f"Best test accuracy reached {pymc_mape_raw:.1f}% (Round {round_n} PyMC)",
        html
    )

    return html


def main():
    if not PRESENTATION.exists():
        print("ERROR: presentation.html not found")
        sys.exit(1)

    print("Loading results...")
    roi, contrib, fit, state = load_results()

    print("Building JS data...")
    roi_js = build_roi_js(roi)
    contrib_js = build_contrib_js(contrib)

    print("Patching presentation.html...")
    html = PRESENTATION.read_text(encoding="utf-8")
    html = patch_html(html, roi_js, contrib_js, state, fit)
    PRESENTATION.write_text(html, encoding="utf-8")

    round_n = state["current_round"]
    best_mape = state.get("best_test_mape", "?")
    print(f"Done — Round {round_n} · best MAPE {best_mape}%")
    print(f"Updated: {PRESENTATION}")


if __name__ == "__main__":
    main()
