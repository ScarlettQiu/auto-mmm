"""Auto-MMM Dashboard — beautiful web report from results JSON.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Auto-MMM Report",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #f8f9fb; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #eaecf0; }

/* Hide Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

/* Tight top padding */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #eaecf0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; color: #111; }
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #667085; font-weight: 500; }
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* Section headers */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #111;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #534AB7;
    display: inline-block;
}

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.88rem;
    font-weight: 500;
    color: #667085;
    padding: 0.5rem 1rem;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #534AB7;
    border-bottom: 2px solid #534AB7;
    font-weight: 600;
}

/* Sidebar label */
.sidebar-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #667085;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}

/* Insight card */
.insight-card {
    background: #ffffff;
    border: 1px solid #eaecf0;
    border-left: 4px solid #534AB7;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    color: #111;
    line-height: 1.6;
}
.insight-card.warning { border-left-color: #F79009; }
.insight-card.success { border-left-color: #1D9E75; }
.insight-card.danger  { border-left-color: #E24B4A; }

/* Model pill badges */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 4px;
}
.pill-ridge   { background: #EEF4FF; color: #1F77B4; }
.pill-pymc    { background: #FFF4ED; color: #C05500; }
.pill-nnls    { background: #ECFDF3; color: #187818; }

/* Round progress bar */
.round-bar-wrap { background: #f2f2f2; border-radius: 99px; height: 8px; margin-top: 4px; }
.round-bar-fill { background: #534AB7; border-radius: 99px; height: 8px; }

/* Agreement badge */
.agree-high   { color: #1D9E75; font-weight: 600; }
.agree-medium { color: #F79009; font-weight: 600; }
.agree-low    { color: #E24B4A; font-weight: 600; }
</style>
"""

PURPLE = "#534AB7"
PURPLE_LIGHT = "#F5F4FF"
GREEN  = "#1D9E75"
ORANGE = "#F79009"
RED    = "#E24B4A"
GRAY   = "#667085"

MODEL_COLORS = {
    "ridge":           "#1F77B4",
    "pymc":            "#FF7F0E",
    "lightweight_mmm": "#2CA02C",
}

MODEL_LABELS = {
    "ridge":           "Ridge",
    "pymc":            "PyMC (Bayesian)",
    "lightweight_mmm": "LightweightMMM / NNLS",
}

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_results(path: str) -> dict:
    return json.loads(Path(path).read_text())


def load_round_history() -> list[dict]:
    rounds_dir = Path("rounds")
    history = []
    for f in sorted(rounds_dir.glob("R*_results.json")):
        data = json.loads(f.read_text())
        for model_name, res in data.get("models", {}).items():
            if not res.get("skipped"):
                history.append({
                    "round": data["round"],
                    "model": MODEL_LABELS.get(model_name, model_name),
                    "train_r2": res.get("train_r2"),
                    "train_mape": res.get("train_mape"),
                    "test_mape": res.get("test_mape"),
                })
    return history


def load_round_analyses() -> dict[int, str]:
    rounds_dir = Path("rounds")
    out = {}
    for f in sorted(rounds_dir.glob("R*_analysis.md")):
        n = int(f.stem.split("_")[0][1:])
        out[n] = f.read_text()
    return out


def load_tuning_log() -> dict[int, str]:
    rounds_dir = Path("rounds")
    out = {}
    for f in sorted(rounds_dir.glob("R*_tuning.md")):
        n = int(f.stem.split("_")[0][1:])
        out[n] = f.read_text()
    return out


# ── Chart helpers ─────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=12, color="#111"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def roi_chart(roi_df: pd.DataFrame) -> go.Figure:
    model_cols = [c for c in roi_df.columns if c in MODEL_COLORS]
    channels   = roi_df.index.tolist()
    fig = go.Figure()
    for m in model_cols:
        fig.add_trace(go.Bar(
            name=MODEL_LABELS.get(m, m),
            y=channels,
            x=[round(float(roi_df.loc[ch, m]), 4) for ch in channels],
            orientation="h",
            marker_color=MODEL_COLORS[m],
            marker_line_width=0,
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="group",
        title="ROI by Channel",
        height=max(320, len(channels) * 55),
        xaxis=dict(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ccc"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def contribution_chart(contrib_df: pd.DataFrame) -> go.Figure:
    model_cols = [c for c in contrib_df.columns if c in MODEL_COLORS]
    channels   = contrib_df.index.tolist()
    fig = go.Figure()
    for m in model_cols:
        fig.add_trace(go.Bar(
            name=MODEL_LABELS.get(m, m),
            x=channels,
            y=[round(float(contrib_df.loc[ch, m]), 2) for ch in channels],
            marker_color=MODEL_COLORS[m],
            marker_line_width=0,
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="group",
        title="Channel Contribution % of GMV",
        height=380,
        yaxis=dict(gridcolor="#f0f0f0", title="% of GMV"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def mape_history_chart(history: list[dict]) -> go.Figure:
    df = pd.DataFrame(history)
    fig = go.Figure()
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("round")
        key = [k for k, v in MODEL_LABELS.items() if v == model]
        color = MODEL_COLORS.get(key[0] if key else "", PURPLE)
        fig.add_trace(go.Scatter(
            x=sub["round"], y=sub["test_mape"],
            mode="lines+markers",
            name=model,
            line=dict(color=color, width=2.5),
            marker=dict(size=8),
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Test MAPE Across Rounds (lower = better)",
        height=320,
        xaxis=dict(title="Round", dtick=1, gridcolor="#f0f0f0"),
        yaxis=dict(title="Test MAPE (%)", gridcolor="#f0f0f0"),
    )
    return fig


def agreement_chart(roi_df: pd.DataFrame) -> go.Figure:
    if "cv_pct" not in roi_df.columns:
        return go.Figure()
    df = roi_df[["cv_pct", "mean_roi"]].reset_index()
    colors = [GREEN if v < 20 else (ORANGE if v < 50 else RED) for v in df["cv_pct"]]
    fig = go.Figure(go.Bar(
        x=df["channel"], y=df["cv_pct"],
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.0f}%" for v in df["cv_pct"]],
        textposition="outside",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Cross-Model Agreement (CV%) — lower = more agreement",
        height=320,
        yaxis=dict(title="CV %", gridcolor="#f0f0f0"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        shapes=[
            dict(type="line", x0=-0.5, x1=len(df)-0.5, y0=20, y1=20,
                 line=dict(color=GREEN, dash="dot", width=1.5)),
            dict(type="line", x0=-0.5, x1=len(df)-0.5, y0=50, y1=50,
                 line=dict(color=RED, dash="dot", width=1.5)),
        ],
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem">'
            f'<span style="font-size:1.4rem">📈</span>'
            f'<div><p style="margin:0;font-size:1rem;font-weight:700;color:#111">Auto-MMM</p>'
            f'<p style="margin:0;font-size:0.72rem;color:{GRAY}">Marketing Mix Model Report</p></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Round selector
        rounds_dir = Path("rounds")
        result_files = sorted(rounds_dir.glob("R*_results.json"), reverse=True)
        round_options = [int(f.stem.split("_")[0][1:]) for f in result_files]

        if not round_options:
            st.error("No results found. Run `python run_models.py` first.")
            return

        st.markdown('<p class="sidebar-label">Select Round</p>', unsafe_allow_html=True)
        selected_round = st.selectbox(
            "Round", round_options,
            format_func=lambda x: f"Round {x}" + (" (latest)" if x == round_options[0] else ""),
            label_visibility="collapsed",
        )

        results_path = rounds_dir / f"R{selected_round:02d}_results.json"
        results = load_results(str(results_path))

        st.divider()
        st.markdown('<p class="sidebar-label">Data</p>', unsafe_allow_html=True)
        st.markdown(f"**Periods:** {results.get('data_periods','?')} monthly")
        st.markdown(f"**Train:** {results.get('train_periods','?')}  ·  **Holdout:** {results.get('test_periods','?')}")
        st.markdown(f"**Run:** {results.get('run_at','')[:10]}")

        st.divider()
        st.markdown('<p class="sidebar-label">Models Run</p>', unsafe_allow_html=True)
        for model_name, res in results["models"].items():
            label = MODEL_LABELS.get(model_name, model_name)
            if res.get("skipped"):
                st.markdown(f"⏭️ ~~{label}~~")
            else:
                r2   = res.get("train_r2", "?")
                mape = res.get("test_mape", "?")
                st.markdown(f"✅ **{label}**")
                st.caption(f"R²={r2} · test MAPE={mape}%")

        st.divider()
        st.markdown('<p class="sidebar-label">Rounds Completed</p>', unsafe_allow_html=True)
        n_rounds = len(round_options)
        st.markdown(
            f'<div class="round-bar-wrap">'
            f'<div class="round-bar-fill" style="width:{min(100, n_rounds*25)}%"></div>'
            f'</div>'
            f'<p style="font-size:0.78rem;color:{GRAY};margin-top:4px">{n_rounds} round{"s" if n_rounds>1 else ""} completed</p>',
            unsafe_allow_html=True,
        )

    # ── Build comparison DataFrames ───────────────────────────────────────────
    from compare import roi_comparison, contribution_comparison, model_fit_summary

    roi_df     = roi_comparison(results)
    contrib_df = contribution_comparison(results)
    fit_df     = model_fit_summary(results)
    history    = load_round_history()
    analyses   = load_round_analyses()
    tuning     = load_tuning_log()

    # ── Header ────────────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(
            f'<h1 style="font-size:1.6rem;font-weight:700;color:#111;margin:0">'
            f'Marketing Mix Model Report</h1>'
            f'<p style="color:{GRAY};font-size:0.88rem;margin:4px 0 0 0">'
            f'Round {selected_round} · {results.get("data_periods","?")} months of data · '
            f'{results.get("run_at","")[:10]}</p>',
            unsafe_allow_html=True,
        )
    with col_h2:
        st.download_button(
            "📥 Download report (.md)",
            data=Path("results/report.md").read_text() if Path("results/report.md").exists() else "",
            file_name=f"mmm_report_round{selected_round}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("<hr style='margin:0.75rem 0 1.25rem 0;border:none;border-top:1px solid #eaecf0'>",
                unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 Channel Analysis",
        "⚖️ Model Comparison",
        "📈 Round History",
        "🗒️ Agent Logs",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        # Key metrics
        active_models = {k: v for k, v in results["models"].items() if not v.get("skipped")}
        best_model = min(active_models.items(), key=lambda x: x[1].get("test_mape", 999))
        best_name, best_res = best_model

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best test MAPE", f"{best_res.get('test_mape')}%",
                  delta=f"{MODEL_LABELS.get(best_name, best_name)}",
                  delta_color="off")
        c2.metric("Models run", str(len(active_models)))
        c3.metric("Training periods", str(results.get("train_periods", "?")))
        c4.metric("Rounds completed", str(len(round_options)))

        st.markdown("")

        # Top channel insight
        if not roi_df.empty and "mean_roi" in roi_df.columns:
            top_ch = roi_df["mean_roi"].idxmax()
            top_roi = roi_df.loc[top_ch, "mean_roi"]
            top_cv  = roi_df.loc[top_ch, "cv_pct"] if "cv_pct" in roi_df.columns else 100

            agree_label = "✅ High agreement" if top_cv < 20 else ("⚠️ Medium agreement" if top_cv < 50 else "⚠️ Low agreement — treat as directional")
            card_type   = "success" if top_cv < 20 else "warning"

            st.markdown(
                f'<div class="insight-card {card_type}">'
                f'<strong>Top channel: {top_ch}</strong><br>'
                f'Mean ROI = <strong>{top_roi:.3f}</strong> across all models · '
                f'Cross-model agreement: {agree_label} (CV = {top_cv:.0f}%)'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Data caveat
        st.markdown(
            f'<div class="insight-card warning">'
            f'<strong>Data limitation:</strong> This analysis uses {results.get("data_periods","?")} monthly periods. '
            f'MMM standard is 100+ weekly observations. Results are <strong>directional only</strong> — '
            f'do not make large budget shifts without collecting more data first.'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Analyst summary for this round
        if selected_round in analyses:
            st.markdown('<p class="section-header">Analyst Summary</p>', unsafe_allow_html=True)
            # Extract just the Preliminary Recommendation section
            text = analyses[selected_round]
            if "## Preliminary Recommendation" in text:
                rec = text.split("## Preliminary Recommendation")[1].strip()
                st.markdown(
                    f'<div class="insight-card">{rec}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(text[:800] + "…")

        # MAPE history sparkline
        if len(history) > 0:
            st.markdown('<p class="section-header">Test MAPE Trend</p>', unsafe_allow_html=True)
            st.plotly_chart(mape_history_chart(history), use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — CHANNEL ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<p class="section-header">ROI by Channel</p>', unsafe_allow_html=True)
        st.caption("Revenue generated per unit of spend. Channels at zero may reflect insufficient data, not zero effect.")
        if not roi_df.empty:
            st.plotly_chart(roi_chart(roi_df), use_container_width=True)

        st.markdown('<p class="section-header">Channel Contribution % of GMV</p>', unsafe_allow_html=True)
        st.caption("Share of total KPI attributed to each channel. Typical MMM range: 20–60% total media.")
        if not contrib_df.empty:
            st.plotly_chart(contribution_chart(contrib_df), use_container_width=True)

        # Detailed table
        st.markdown('<p class="section-header">Full Attribution Table</p>', unsafe_allow_html=True)
        if not roi_df.empty:
            display_cols = [c for c in roi_df.columns if c in MODEL_COLORS or c in ["mean_roi", "cv_pct", "agreement"]]
            display_df = roi_df[display_cols].copy()
            display_df.columns = [MODEL_LABELS.get(c, c) for c in display_df.columns]

            def color_agreement(val):
                if "High" in str(val): return f"color: {GREEN}; font-weight: 600"
                if "Medium" in str(val): return f"color: {ORANGE}; font-weight: 600"
                if "Low" in str(val): return f"color: {RED}; font-weight: 600"
                return ""

            styled = display_df.style\
                .format({c: "{:.4f}" for c in display_df.columns if display_df[c].dtype == float})\
                .applymap(color_agreement, subset=[c for c in display_df.columns if "agreement" in c.lower()] or display_df.columns[-1:])\
                .set_properties(**{"background-color": "#fff", "font-size": "0.88rem"})
            st.dataframe(styled, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — MODEL COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<p class="section-header">Model Fit Metrics</p>', unsafe_allow_html=True)

        cols = st.columns(len(active_models))
        for i, (model_name, res) in enumerate(active_models.items()):
            color = MODEL_COLORS.get(model_name, PURPLE)
            label = MODEL_LABELS.get(model_name, model_name)
            with cols[i]:
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #eaecf0;border-top:4px solid {color};'
                    f'border-radius:10px;padding:1rem 1.1rem;margin-bottom:0.5rem">'
                    f'<p style="margin:0 0 0.5rem 0;font-size:0.9rem;font-weight:700;color:#111">{label}</p>'
                    f'<p style="margin:0;font-size:0.78rem;color:{GRAY}">Train R²</p>'
                    f'<p style="margin:0 0 0.3rem 0;font-size:1.2rem;font-weight:700;color:#111">{res.get("train_r2","?")}</p>'
                    f'<p style="margin:0;font-size:0.78rem;color:{GRAY}">Test MAPE</p>'
                    f'<p style="margin:0 0 0.3rem 0;font-size:1.2rem;font-weight:700;color:#111">{res.get("test_mape","?")}%</p>'
                    f'<p style="margin:0;font-size:0.75rem;color:{GRAY}">{res.get("note","")[:60]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<p class="section-header">Cross-Model Agreement</p>', unsafe_allow_html=True)
        st.caption("CV% = coefficient of variation across models. Green < 20% · Orange 20–50% · Red > 50%")
        if not roi_df.empty:
            st.plotly_chart(agreement_chart(roi_df), use_container_width=True)

        st.markdown('<p class="section-header">Model Guide</p>', unsafe_allow_html=True)
        comparison_rows = [
            ("Speed",             "⚡ Seconds",         "🐢 10–60 min",       "🔄 Minutes (JAX)"),
            ("Uncertainty",       "Bootstrap CIs",      "Full posterior",      "None (NNLS) / Posterior (JAX)"),
            ("Saturation curve",  "❌ Linear only",     "✅ Hill built-in",    "✅ Hill built-in"),
            ("Negative ROI risk", "⚠️ Yes (shrinkage)", "Low (priors)",        "✅ No (NNLS positive)"),
            ("Min data needed",   "50+ weekly obs",     "80+ weekly obs",      "50+ weekly obs"),
            ("Best for",          "Quick baseline",     "Production decisions","Positive-constrained fast"),
        ]
        comp_df = pd.DataFrame(comparison_rows, columns=["Dimension", "Ridge", "PyMC (Bayesian)", "LightweightMMM"])
        st.dataframe(comp_df.set_index("Dimension"), use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — ROUND HISTORY
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<p class="section-header">Performance Across Rounds</p>', unsafe_allow_html=True)
        if history:
            st.plotly_chart(mape_history_chart(history), use_container_width=True)

            hist_df = pd.DataFrame(history).sort_values(["round", "model"])
            st.dataframe(
                hist_df.rename(columns={
                    "round": "Round", "model": "Model",
                    "train_r2": "Train R²", "train_mape": "Train MAPE (%)",
                    "test_mape": "Test MAPE (%)"
                }).set_index("Round"),
                use_container_width=True,
            )

        st.markdown('<p class="section-header">Tuning Log</p>', unsafe_allow_html=True)
        if tuning:
            for round_num in sorted(tuning.keys()):
                with st.expander(f"Round {round_num} — Tuner", expanded=(round_num == selected_round)):
                    st.markdown(tuning[round_num])
        else:
            st.caption("No tuning logs found.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — AGENT LOGS
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        rounds_dir = Path("rounds")

        # Analyst log
        st.markdown('<p class="section-header">Analyst Reports</p>', unsafe_allow_html=True)
        for round_num in sorted(analyses.keys(), reverse=True):
            with st.expander(f"Round {round_num} — Analyst", expanded=(round_num == selected_round)):
                st.markdown(analyses[round_num])

        # Critic reviews
        st.markdown('<p class="section-header">Critic Reviews</p>', unsafe_allow_html=True)
        for f in sorted(rounds_dir.glob("R*_review.md"), reverse=True):
            n = int(f.stem.split("_")[0][1:])
            with st.expander(f"Round {n} — Critic Review", expanded=(n == selected_round)):
                st.markdown(f.read_text())


if __name__ == "__main__":
    main()
