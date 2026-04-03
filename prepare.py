"""Data loading and preprocessing for auto-MMM.

Loads the DT Mart dataset, applies adstock and Hill saturation
transforms, and returns a model-ready DataFrame.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_config(path: str = "config.json") -> dict:
    return json.loads(Path(path).read_text())


def load_raw(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["data_path"]).expanduser()
    df = pd.read_csv(path)
    # Parse date
    df["date"] = pd.to_datetime(df[cfg["date_column"]], format=cfg["date_format"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def geometric_adstock(x: np.ndarray, decay: float, max_lag: int) -> np.ndarray:
    """Apply geometric adstock: carryover effect with exponential decay."""
    result = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        for lag in range(min(t + 1, max_lag + 1)):
            result[t] += x[t - lag] * (decay ** lag)
    return result


def hill_saturation(x: np.ndarray, slope: float, ec: float) -> np.ndarray:
    """Apply Hill saturation function: diminishing returns."""
    x_scaled = x / (x.max() + 1e-8)
    return x_scaled ** slope / (x_scaled ** slope + ec ** slope)


def preprocess(cfg: dict, adstock_decays: dict | None = None) -> pd.DataFrame:
    """Load, clean, apply transforms, return model-ready DataFrame."""
    df = load_raw(cfg)

    channels = cfg["media_channels"]
    kpi = cfg["kpi_column"]
    controls = cfg["control_variables"]
    max_lag = cfg["adstock_max_lag"]
    slope = cfg["hill_slope"]
    ec = cfg["hill_ec"]

    # Default adstock decay per channel (can be tuned)
    default_decays = {
        "TV": 0.7,
        "Digital": 0.3,
        "Sponsorship": 0.5,
        "Content.Marketing": 0.4,
        "Online.marketing": 0.3,
        "Affiliates": 0.2,
        "SEM": 0.2,
        "Radio": 0.5,
    }
    decays = {**default_decays, **(adstock_decays or {})}

    out = pd.DataFrame()
    out["date"] = df["date"]
    out["kpi"] = df[kpi].values.astype(float)

    # Apply adstock then Hill saturation per channel
    for ch in channels:
        raw = df[ch].fillna(0).values.astype(float)
        adstocked = geometric_adstock(raw, decays.get(ch, 0.4), max_lag)
        out[f"{ch}_adstock"] = adstocked
        out[f"{ch}_saturated"] = hill_saturation(adstocked, slope, ec)

    # Control variables (normalised)
    for ctrl in controls:
        if ctrl in df.columns:
            vals = df[ctrl].fillna(df[ctrl].median()).values.astype(float)
            out[ctrl] = (vals - vals.mean()) / (vals.std() + 1e-8)

    # Trend + seasonality
    out["trend"] = np.arange(len(out)) / len(out)
    out["month_num"] = df["date"].dt.month

    return out


def train_test_split(df: pd.DataFrame, holdout: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.iloc[:-holdout].copy(), df.iloc[-holdout:].copy()


def summary(df: pd.DataFrame, cfg: dict) -> None:
    channels = cfg["media_channels"]
    print(f"\n{'='*60}")
    print(f"Dataset: {len(df)} periods  |  KPI: {cfg['kpi_column']}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nKPI stats:")
    print(f"  Mean: {df['kpi'].mean():,.0f}  |  Std: {df['kpi'].std():,.0f}")
    print(f"\nMedia spend (adstocked):")
    for ch in channels:
        col = f"{ch}_adstock"
        if col in df.columns:
            print(f"  {ch:25s}: mean={df[col].mean():12,.0f}  max={df[col].max():12,.0f}")
    print("="*60)


if __name__ == "__main__":
    cfg = load_config()
    df = preprocess(cfg)
    summary(df, cfg)
    print("\nFirst 3 rows:")
    print(df[["date", "kpi"] + [f"{ch}_saturated" for ch in cfg["media_channels"]]].head(3).to_string())
