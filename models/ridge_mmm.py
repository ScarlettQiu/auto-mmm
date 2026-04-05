"""Ridge MMM — fast, robust baseline model.

Uses sklearn Ridge regression on adstock+saturated media inputs.
Bootstraps 200 samples for uncertainty estimates.
Always runnable — no heavy dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def _feature_cols(df: pd.DataFrame, channels: list[str], controls: list[str]) -> list[str]:
    cols = [f"{ch}_saturated" for ch in channels if f"{ch}_saturated" in df.columns]
    cols += [c for c in controls if c in df.columns]
    cols += [c for c in ["trend"] if c in df.columns]
    return cols


def _mape(y_true, y_pred) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def run(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
) -> dict:
    channels = cfg["media_channels"]
    controls = cfg["control_variables"]
    feat_cols = _feature_cols(train_df, channels, controls)

    X_train = train_df[feat_cols].values
    y_train = train_df["kpi"].values
    X_test  = test_df[feat_cols].values
    y_test  = test_df["kpi"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Cross-validated Ridge to find best alpha
    alphas = np.logspace(-2, 4, 50)
    model = RidgeCV(alphas=alphas, cv=min(5, len(train_df) - 1))
    model.fit(X_train_s, y_train)

    y_pred_train = model.predict(X_train_s)
    y_pred_test  = model.predict(X_test_s)

    train_r2   = r2_score(y_train, y_pred_train)
    train_mape = _mape(y_train, y_pred_train)
    test_mape  = _mape(y_test, y_pred_test)

    # Bootstrap for coefficient uncertainty
    n_boot = cfg.get("n_bootstrap", 200)
    boot_coefs = np.zeros((n_boot, len(feat_cols)))
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        idx = rng.integers(0, len(X_train_s), len(X_train_s))
        m = Ridge(alpha=model.alpha_)
        m.fit(X_train_s[idx], y_train[idx])
        boot_coefs[i] = m.coef_

    coef_mean = boot_coefs.mean(axis=0)
    coef_std  = boot_coefs.std(axis=0)

    # Channel contributions — decompose fitted values in ORIGINAL feature space.
    # StandardScaler mean-centres features, so coef * X_train_s sums to ~0 for
    # every channel regardless of true effect.  Convert to original scale first:
    #   coef_orig[j] = coef_std[j] / scaler.scale_[j]
    #   contrib(t)   = coef_orig[j] * X_train[t, j]
    coef_orig = coef_mean / scaler.scale_
    baseline  = model.intercept_
    channel_contribs = {}

    for ch in channels:
        col = f"{ch}_saturated"
        if col not in feat_cols:
            continue
        idx = feat_cols.index(col)
        contrib = coef_orig[idx] * X_train[:, idx]
        channel_contribs[ch] = float(contrib.sum())

    total_media_contrib = sum(channel_contribs.values())
    total_kpi = float(y_train.sum())

    # ROI = channel_contribution / total_spend_on_channel
    roi = {}
    for ch in channels:
        spend = train_df[f"{ch}_adstock"].sum() if f"{ch}_adstock" in train_df.columns else 0
        contrib = channel_contribs.get(ch, 0)
        roi[ch] = float(contrib / spend) if spend > 0 else 0.0

    # Coefficient table
    coef_df = pd.DataFrame({
        "feature": feat_cols,
        "coef_mean": coef_mean,
        "coef_std": coef_std,
        "coef_lower": coef_mean - 1.96 * coef_std,
        "coef_upper": coef_mean + 1.96 * coef_std,
    })

    return {
        "model": "ridge",
        "alpha": float(model.alpha_),
        "train_r2": round(train_r2, 4),
        "train_mape": round(train_mape, 2),
        "test_mape": round(test_mape, 2),
        "baseline_contribution": round(float(baseline * len(train_df)), 2),
        "total_media_contribution": round(total_media_contrib, 2),
        "channel_contributions": {k: round(v, 2) for k, v in channel_contribs.items()},
        "channel_contribution_pct": {
            k: round(v / total_kpi * 100, 2) if total_kpi else 0
            for k, v in channel_contribs.items()
        },
        "roi": {k: round(v, 4) for k, v in roi.items()},
        "coef_table": coef_df.to_dict(orient="records"),
        "y_pred_train": y_pred_train.tolist(),
        "y_actual_train": y_train.tolist(),
        "y_pred_test": y_pred_test.tolist(),
        "y_actual_test": y_test.tolist(),
    }
