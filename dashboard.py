"""Auto-MMM Dashboard — data exploration + model results.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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

.stApp { background: #f8f9fb; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #eaecf0; }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #eaecf0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; color: #111; }
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #667085; font-weight: 500; }

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #111;
    margin: 2rem 0 0.5rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #A91E2C;
}

.section-header {
    font-size: 1.0rem;
    font-weight: 700;
    color: #111;
    margin: 1.25rem 0 0.5rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #eaecf0;
    display: inline-block;
}

.insight-card {
    background: #ffffff;
    border: 1px solid #eaecf0;
    border-left: 4px solid #A91E2C;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    color: #111;
    line-height: 1.6;
}
.insight-card.warning { border-left-color: #F79009; }
.insight-card.success { border-left-color: #1D9E75; }

.sidebar-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #667085;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
</style>
"""

BRAND   = "#A91E2C"
GREEN   = "#1D9E75"
ORANGE  = "#F79009"
RED     = "#E24B4A"
GRAY    = "#667085"
LGRAY   = "#f8f9fb"

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

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=12, color="#111"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_results(path: str) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data
def load_metadata() -> dict:
    p = Path("metadata.json")
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def load_raw_data(data_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_exploration_report() -> str:
    p = Path("rounds/R01_data_exploration.md")
    return p.read_text() if p.exists() else ""


def load_round_history() -> list[dict]:
    history = []
    for f in sorted(Path("rounds").glob("R*_results.json")):
        data = json.loads(f.read_text())
        for model_name, res in data.get("models", {}).items():
            if not res.get("skipped"):
                history.append({
                    "round": data["round"],
                    "model": MODEL_LABELS.get(model_name, model_name),
                    "model_key": model_name,
                    "train_r2": res.get("train_r2"),
                    "test_mape": res.get("test_mape"),
                })
    return history


# ── Chart helpers ─────────────────────────────────────────────────────────────

def kpi_timeseries(df: pd.DataFrame, date_col: str, kpi_col: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df[date_col], y=df[kpi_col],
        mode="lines+markers",
        line=dict(color=BRAND, width=2.5),
        marker=dict(size=6),
        name=kpi_col,
        fill="tozeroy",
        fillcolor=f"rgba(169,30,44,0.08)",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"KPI Over Time: {kpi_col}",
        height=300,
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig


def channel_spend_bar(df: pd.DataFrame, channels: list[str]) -> go.Figure:
    totals = {ch: df[ch].sum() for ch in channels if ch in df.columns}
    totals = dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))
    fig = go.Figure(go.Bar(
        x=list(totals.keys()),
        y=list(totals.values()),
        marker_color=BRAND,
        marker_line_width=0,
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Total Spend by Channel",
        height=320,
        yaxis=dict(title="Total Spend", gridcolor="#f0f0f0"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, channels: list[str], kpi_col: str) -> go.Figure:
    cols = [c for c in channels + [kpi_col] if c in df.columns]
    if len(cols) < 2:
        return go.Figure()
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Channel & KPI Correlation Matrix",
        height=max(320, len(cols) * 45),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def anomaly_chart(df: pd.DataFrame, date_col: str, kpi_col: str) -> go.Figure | None:
    if kpi_col not in df.columns or date_col not in df.columns:
        return None
    mean = df[kpi_col].mean()
    std  = df[kpi_col].std()
    if std == 0:
        return None
    z = (df[kpi_col] - mean) / std
    anomalies = df[z.abs() > 3]
    if anomalies.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[kpi_col],
        mode="lines", line=dict(color=GRAY, width=1.5), name=kpi_col,
    ))
    fig.add_trace(go.Scatter(
        x=anomalies[date_col], y=anomalies[kpi_col],
        mode="markers", marker=dict(color=RED, size=12, symbol="x"),
        name="Anomaly (z > 3σ)",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="KPI Anomalies (z-score > 3σ)",
        height=280,
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig


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
        height=360,
        yaxis=dict(gridcolor="#f0f0f0", title="% of GMV"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def mape_trend_chart(history: list[dict]) -> go.Figure:
    df = pd.DataFrame(history)
    fig = go.Figure()
    for model_key in df["model_key"].unique():
        sub = df[df["model_key"] == model_key].sort_values("round")
        fig.add_trace(go.Scatter(
            x=sub["round"], y=sub["test_mape"],
            mode="lines+markers",
            name=MODEL_LABELS.get(model_key, model_key),
            line=dict(color=MODEL_COLORS.get(model_key, BRAND), width=2.5),
            marker=dict(size=8),
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Test MAPE Across Rounds (lower = better)",
        height=300,
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
        height=300,
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

        result_files = sorted(Path("rounds").glob("R*_results.json"), reverse=True)
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

        results_path = Path("rounds") / f"R{selected_round:02d}_results.json"
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
                st.markdown(f"✅ **{label}**")
                st.caption(f"R²={res.get('train_r2','?')} · test MAPE={res.get('test_mape','?')}%")

        st.divider()
        if Path("results/report.md").exists():
            st.download_button(
                "📥 Download report (.md)",
                data=Path("results/report.md").read_text(),
                file_name=f"mmm_report_round{selected_round}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # ── Load supporting data ──────────────────────────────────────────────────
    meta     = load_metadata()
    cfg_path = Path("config.json")
    cfg      = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    data_path = (
        cfg.get("data_path") or
        meta.get("source_path") or
        meta.get("file_path") or ""
    )
    raw_df   = load_raw_data(data_path) if data_path else pd.DataFrame()
    channels = (
        cfg.get("channel_columns") or
        meta.get("channels") or
        [c.get("column") for c in meta.get("fields", []) if c.get("type") == "channel"]
    )
    channels = [c for c in (channels or []) if c]

    kpi_col  = cfg.get("kpi_column") or meta.get("kpi") or ""
    date_col = cfg.get("date_column") or meta.get("date_column") or ""

    # Detect from raw_df if still missing
    if raw_df is not None and not raw_df.empty:
        if not date_col:
            for c in raw_df.columns:
                if "date" in c.lower() or "week" in c.lower() or "period" in c.lower():
                    date_col = c; break
        if not kpi_col:
            for c in raw_df.columns:
                if c.lower() in ("sales", "revenue", "gmv", "conversions", "kpi"):
                    kpi_col = c; break

    exploration_text = load_exploration_report()

    from compare import roi_comparison, contribution_comparison
    roi_df     = roi_comparison(results)
    contrib_df = contribution_comparison(results)
    history    = load_round_history()

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<h1 style="font-size:1.6rem;font-weight:700;color:#111;margin:0">'
        f'Marketing Mix Model Report</h1>'
        f'<p style="color:{GRAY};font-size:0.88rem;margin:4px 0 1rem 0">'
        f'Round {selected_round} · {results.get("data_periods","?")} months · '
        f'{results.get("run_at","")[:10]}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:0 0 1.5rem 0;border:none;border-top:1px solid #eaecf0'>",
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — DATA EXPLORATION
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown('<p class="section-title">Data Exploration</p>', unsafe_allow_html=True)

    # Readiness score callout (parse from exploration report)
    if exploration_text:
        for line in exploration_text.splitlines():
            if "Readiness Score:" in line or line.startswith("**Score:"):
                st.markdown(
                    f'<div class="insight-card">{line.strip()}</div>',
                    unsafe_allow_html=True,
                )
                break

    if not raw_df.empty and date_col and kpi_col:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<p class="section-header">KPI Over Time</p>', unsafe_allow_html=True)
            st.plotly_chart(kpi_timeseries(raw_df, date_col, kpi_col),
                            use_container_width=True)

        with col_b:
            anom_fig = anomaly_chart(raw_df, date_col, kpi_col)
            if anom_fig:
                st.markdown('<p class="section-header">KPI Anomalies</p>', unsafe_allow_html=True)
                st.plotly_chart(anom_fig, use_container_width=True)
            else:
                st.markdown('<p class="section-header">KPI Distribution</p>',
                            unsafe_allow_html=True)
                fig_hist = px.histogram(raw_df, x=kpi_col, nbins=20,
                                        color_discrete_sequence=[BRAND])
                fig_hist.update_layout(**CHART_LAYOUT, title="KPI Distribution", height=280)
                st.plotly_chart(fig_hist, use_container_width=True)

        if channels:
            col_c, col_d = st.columns(2)
            with col_c:
                st.markdown('<p class="section-header">Channel Spend</p>',
                            unsafe_allow_html=True)
                st.plotly_chart(channel_spend_bar(raw_df, channels),
                                use_container_width=True)
            with col_d:
                st.markdown('<p class="section-header">Correlation Matrix</p>',
                            unsafe_allow_html=True)
                st.plotly_chart(correlation_heatmap(raw_df, channels, kpi_col),
                                use_container_width=True)

    elif exploration_text:
        # No raw data available — show exploration report text
        st.markdown(exploration_text)
    else:
        st.caption("Run Round 1 to generate the data exploration report.")

    # Collinearity / anomaly flags from metadata
    warnings = meta.get("collinearity_warnings", []) + meta.get("anomaly_warnings", [])
    if warnings:
        st.markdown('<p class="section-header">Data Quality Flags</p>',
                    unsafe_allow_html=True)
        for w in warnings:
            st.markdown(
                f'<div class="insight-card warning">{w}</div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — MODEL RESULTS
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown('<p class="section-title">Model Results</p>', unsafe_allow_html=True)

    active_models = {k: v for k, v in results["models"].items() if not v.get("skipped")}

    # Key metric cards
    m_cols = st.columns(len(active_models))
    for i, (model_name, res) in enumerate(active_models.items()):
        color = MODEL_COLORS.get(model_name, BRAND)
        label = MODEL_LABELS.get(model_name, model_name)
        with m_cols[i]:
            st.markdown(
                f'<div style="background:#fff;border:1px solid #eaecf0;border-top:4px solid {color};'
                f'border-radius:10px;padding:0.9rem 1rem;margin-bottom:0.75rem">'
                f'<p style="margin:0 0 0.4rem 0;font-size:0.88rem;font-weight:700;color:#111">{label}</p>'
                f'<p style="margin:0;font-size:0.72rem;color:{GRAY}">Train R²</p>'
                f'<p style="margin:0 0 0.25rem 0;font-size:1.15rem;font-weight:700;color:#111">'
                f'{res.get("train_r2","?")}</p>'
                f'<p style="margin:0;font-size:0.72rem;color:{GRAY}">Test MAPE</p>'
                f'<p style="margin:0;font-size:1.15rem;font-weight:700;color:#111">'
                f'{res.get("test_mape","?")}%</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ROI + Contribution side by side
    col_roi, col_contrib = st.columns(2)
    with col_roi:
        st.markdown('<p class="section-header">ROI by Channel</p>', unsafe_allow_html=True)
        st.caption("Revenue per unit of spend.")
        if not roi_df.empty:
            st.plotly_chart(roi_chart(roi_df), use_container_width=True)

    with col_contrib:
        st.markdown('<p class="section-header">Channel Contribution %</p>',
                    unsafe_allow_html=True)
        st.caption("Share of KPI attributed to each channel.")
        if not contrib_df.empty:
            st.plotly_chart(contribution_chart(contrib_df), use_container_width=True)

    # MAPE trend + Agreement side by side
    col_mape, col_agree = st.columns(2)
    with col_mape:
        st.markdown('<p class="section-header">Test MAPE Across Rounds</p>',
                    unsafe_allow_html=True)
        if history:
            st.plotly_chart(mape_trend_chart(history), use_container_width=True)
        else:
            st.caption("Only one round completed — trend not yet available.")

    with col_agree:
        st.markdown('<p class="section-header">Cross-Model Agreement</p>',
                    unsafe_allow_html=True)
        st.caption("CV% < 20% = high agreement (green).")
        if not roi_df.empty:
            st.plotly_chart(agreement_chart(roi_df), use_container_width=True)

    # Data caveat
    st.markdown(
        f'<div class="insight-card warning" style="margin-top:1rem">'
        f'<strong>Data caveat:</strong> This analysis uses {results.get("data_periods","?")} '
        f'monthly periods. MMM standard is 100+ weekly observations. '
        f'Results are <strong>directional only</strong>.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
