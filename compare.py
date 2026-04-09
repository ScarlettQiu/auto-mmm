"""Compare results across all three MMM models and surface agreements/disagreements.

Usage:
    python compare.py                     # compare latest results
    python compare.py --round 3           # compare a specific round
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(round_num: int | None = None, results_dir: str = "./results") -> dict:
    if round_num:
        path = Path("rounds") / f"R{round_num:02d}_results.json"
    else:
        path = Path(results_dir) / "latest.json"
    return json.loads(path.read_text())


def roi_comparison(results: dict) -> pd.DataFrame:
    """Build a channel × model ROI table."""
    rows = []
    for model_name, res in results["models"].items():
        if res.get("skipped"):
            continue
        for ch, roi in res.get("roi", {}).items():
            rows.append({"channel": ch, "model": model_name, "roi": roi})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).pivot(index="channel", columns="model", values="roi")
    # Flag channels where any model returned negative ROI (data quality signal)
    model_cols = [c for c in df.columns if c not in ("mean_roi", "std_roi", "cv_pct", "agreement")]
    df["has_negative"] = (df[model_cols] < 0).any(axis=1)
    df["mean_roi"]   = df[model_cols].mean(axis=1)
    df["std_roi"]    = df[model_cols].std(axis=1)
    df["cv_pct"]     = (df["std_roi"] / (df["mean_roi"].abs() + 1e-8) * 100).round(1)
    df["agreement"]  = df.apply(
        lambda row: "⚠️ Negative ROI" if row["has_negative"]
        else ("✅ High" if row["cv_pct"] < 20 else ("⚠️ Medium" if row["cv_pct"] < 50 else "❌ Low")),
        axis=1,
    )
    return df.round(4).sort_values("mean_roi", ascending=False)


def contribution_comparison(results: dict) -> pd.DataFrame:
    """Build a channel × model % contribution table."""
    rows = []
    for model_name, res in results["models"].items():
        if res.get("skipped"):
            continue
        for ch, pct in res.get("channel_contribution_pct", {}).items():
            rows.append({"channel": ch, "model": model_name, "contribution_pct": pct})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).pivot(index="channel", columns="model", values="contribution_pct")
    model_cols = [c for c in df.columns if c not in ("mean_pct", "spread")]
    df["mean_pct"] = df[model_cols].mean(axis=1)
    df["spread"]   = df[model_cols].max(axis=1) - df[model_cols].min(axis=1)
    # Warn if any model's contributions don't sum to ~100%
    for col in model_cols:
        total = df[col].sum()
        if not (80 <= total <= 120):  # allow some slack for baseline exclusion
            print(f"  [WARNING] {col} contributions sum to {total:.1f}% (expected ~100%)")
    return df.round(2).sort_values("mean_pct", ascending=False)


def model_fit_summary(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, res in results["models"].items():
        if res.get("skipped"):
            rows.append({
                "model": model_name,
                "train_r2": "N/A", "train_mape": "N/A", "test_mape": "N/A",
                "status": f"Skipped: {res.get('error', '')[:60]}"
            })
        else:
            rows.append({
                "model": model_name,
                "train_r2": res.get("train_r2"),
                "train_mape": f"{res.get('train_mape')}%",
                "test_mape": f"{res.get('test_mape')}%",
                "status": res.get("note", "OK"),
            })
    return pd.DataFrame(rows).set_index("model")


def top_channels(results: dict, top_n: int = 3) -> dict[str, list[str]]:
    """Return top N channels by mean ROI per model."""
    out = {}
    for model_name, res in results["models"].items():
        if res.get("skipped"):
            continue
        roi = res.get("roi", {})
        ranked = sorted(roi.items(), key=lambda x: x[1], reverse=True)
        out[model_name] = [ch for ch, _ in ranked[:top_n]]
    return out


def disagreements(roi_df: pd.DataFrame, threshold_cv: float = 30.0) -> pd.DataFrame:
    """Channels where models disagree most — CV > threshold."""
    if roi_df.empty or "cv_pct" not in roi_df.columns:
        return pd.DataFrame()
    return roi_df[roi_df["cv_pct"] > threshold_cv][["mean_roi", "cv_pct", "agreement"]]


def print_report(results: dict) -> str:
    lines = []

    def h(title): lines.append(f"\n{'='*60}\n{title}\n{'='*60}")
    def p(text):  lines.append(str(text))

    h("MODEL FIT COMPARISON")
    p(model_fit_summary(results).to_string())

    roi_df = roi_comparison(results)
    contrib_df = contribution_comparison(results)

    h("ROI BY CHANNEL (all models)")
    p(roi_df.to_string())

    h("CONTRIBUTION % BY CHANNEL (all models)")
    p(contrib_df.to_string())

    top = top_channels(results)
    h("TOP 3 CHANNELS BY MODEL")
    for model, chans in top.items():
        p(f"  {model:20s}: {', '.join(chans)}")

    disc = disagreements(roi_df)
    h("HIGH DISAGREEMENT CHANNELS (CV > 30%)")
    if disc.empty:
        p("  All models broadly agree on channel ROI rankings.")
    else:
        p(disc.to_string())

    report = "\n".join(lines)
    print(report)
    return report


def save_comparison(results: dict, cfg: dict) -> None:
    out_dir = Path(cfg.get("results_dir", "./results"))
    out_dir.mkdir(exist_ok=True)

    roi_df     = roi_comparison(results)
    contrib_df = contribution_comparison(results)
    fit_df     = model_fit_summary(results)

    roi_df.to_csv(out_dir / "roi_comparison.csv")
    contrib_df.to_csv(out_dir / "contribution_comparison.csv")
    fit_df.to_csv(out_dir / "model_fit.csv")
    print(f"\nCSVs saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    import json
    cfg = json.loads(Path(args.config).read_text())
    results = load_results(args.round, cfg.get("results_dir", "./results"))
    print_report(results)
    save_comparison(results, cfg)
