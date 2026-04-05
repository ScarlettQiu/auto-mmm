"""LightweightMMM — Google's JAX-based MMM.

Uses geometric adstock + Hill saturation internally.
Requires: pip install lightweight_mmm

Falls back to a numpy-based MMM if JAX/lightweight_mmm unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mape(y_true, y_pred) -> float:
    mask = np.array(y_true) != 0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def run(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
) -> dict:
    try:
        from lightweight_mmm import lightweight_mmm, preprocessing
        import jax.numpy as jnp
        return _run_lightweight(train_df, test_df, cfg, lightweight_mmm, preprocessing, jnp)
    except Exception:
        # ImportError (not installed) or runtime error (JAX API incompatibility)
        return _run_numpy_fallback(train_df, test_df, cfg)


def _run_lightweight(train_df, test_df, cfg, lightweight_mmm, preprocessing, jnp) -> dict:
    channels = cfg["media_channels"]
    feat_cols = [f"{ch}_saturated" for ch in channels if f"{ch}_saturated" in train_df.columns]
    valid_channels = [ch for ch in channels if f"{ch}_saturated" in train_df.columns]

    media = train_df[feat_cols].values.astype(np.float32)
    target = train_df["kpi"].values.astype(np.float32)

    scaler_media = preprocessing.CustomScaler(divide_operation=jnp.mean)
    scaler_target = preprocessing.CustomScaler(divide_operation=jnp.mean)
    media_s = scaler_media.fit_transform(media)
    target_s = scaler_target.fit_transform(target)

    mmm = lightweight_mmm.LightweightMMM(model_name="hill_adstock")
    mmm.fit(
        media=media_s,
        media_prior=jnp.ones(len(feat_cols)),
        target=target_s,
        number_warmup=cfg.get("pymc_tune", 500),
        number_samples=cfg.get("pymc_samples", 1000),
        number_chains=2,
        seed=42,
    )

    media_contribution, baseline = mmm.get_contribution_decomposition(media_s)
    media_contribution_original = scaler_target.inverse_transform(media_contribution)

    total_kpi = float(target.sum())
    channel_contribs = {}
    for i, ch in enumerate(valid_channels):
        channel_contribs[ch] = float(media_contribution_original[:, i].sum())

    roi = {}
    for ch in valid_channels:
        spend = train_df[f"{ch}_adstock"].sum() if f"{ch}_adstock" in train_df.columns else 1
        roi[ch] = float(channel_contribs[ch] / spend) if spend > 0 else 0.0

    y_pred_train = scaler_target.inverse_transform(mmm.trace["mu"].mean(axis=0))
    train_r2 = 1 - np.var(target - y_pred_train) / np.var(target)
    train_mape = _mape(target, y_pred_train)

    # Test
    media_test = test_df[feat_cols].values.astype(np.float32)
    media_test_s = scaler_media.transform(media_test)
    y_pred_test = scaler_target.inverse_transform(
        mmm.predict(media=media_test_s, extra_features=None).mean(axis=0)
    )
    test_mape = _mape(test_df["kpi"].values, y_pred_test)

    return {
        "model": "lightweight_mmm",
        "train_r2": round(float(train_r2), 4),
        "train_mape": round(train_mape, 2),
        "test_mape": round(test_mape, 2),
        "channel_contributions": {k: round(v, 2) for k, v in channel_contribs.items()},
        "channel_contribution_pct": {
            k: round(v / total_kpi * 100, 2) if total_kpi else 0
            for k, v in channel_contribs.items()
        },
        "roi": {k: round(v, 4) for k, v in roi.items()},
        "y_pred_train": y_pred_train.tolist(),
        "y_actual_train": target.tolist(),
        "y_pred_test": y_pred_test.tolist(),
        "y_actual_test": test_df["kpi"].tolist(),
    }


def _run_numpy_fallback(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict) -> dict:
    """Numpy MMM with non-negative least squares — no JAX needed."""
    from scipy.optimize import nnls

    channels = cfg["media_channels"]
    feat_cols = [f"{ch}_saturated" for ch in channels if f"{ch}_saturated" in train_df.columns]
    valid_channels = [ch for ch in channels if f"{ch}_saturated" in train_df.columns]
    ctrl_cols = [c for c in cfg["control_variables"] if c in train_df.columns]

    all_cols = feat_cols + ctrl_cols + ["trend"]
    present_cols = [c for c in all_cols if c in train_df.columns]

    X_train = np.column_stack([train_df[c].values for c in present_cols]) if present_cols else np.ones((len(train_df), 1))
    X_test  = np.column_stack([test_df[c].values  for c in present_cols if c in test_df.columns]) if present_cols else np.ones((len(test_df), 1))
    y_train = train_df["kpi"].values.astype(float)
    y_test  = test_df["kpi"].values.astype(float)

    # Enforce a minimum organic baseline so NNLS cannot attribute 100% of revenue
    # to paid media.  baseline_fraction (default 40%) of mean weekly KPI is reserved
    # as organic/baseline before NNLS sees the data.
    baseline_fraction = float(cfg.get("nnls_baseline_fraction", 0.40))
    baseline_floor = baseline_fraction * float(y_train.mean())
    y_nnls = np.maximum(y_train - baseline_floor, 0.0)

    # Non-negative least squares on incremental revenue only
    coefs, _ = nnls(X_train, y_nnls)

    y_pred_train = X_train @ coefs + baseline_floor
    y_pred_test  = X_test  @ coefs + baseline_floor

    train_r2   = 1 - np.var(y_train - y_pred_train) / (np.var(y_train) + 1e-10)
    train_mape = _mape(y_train, y_pred_train)
    test_mape  = _mape(y_test, y_pred_test)

    total_kpi = float(y_train.sum())
    channel_contribs = {}

    for ch in valid_channels:
        col = f"{ch}_saturated"
        if col not in present_cols:
            channel_contribs[ch] = 0.0
            continue
        col_idx = present_cols.index(col)
        channel_contribs[ch] = float((coefs[col_idx] * X_train[:, col_idx]).sum())

    roi = {}
    for ch in valid_channels:
        spend = train_df[f"{ch}_adstock"].sum() if f"{ch}_adstock" in train_df.columns else 1
        roi[ch] = float(channel_contribs[ch] / spend) if spend > 0 else 0.0

    return {
        "model": "lightweight_mmm",
        "note": "lightweight_mmm/JAX not installed — used scipy NNLS fallback",
        "train_r2": round(float(train_r2), 4),
        "train_mape": round(train_mape, 2),
        "test_mape": round(test_mape, 2),
        "channel_contributions": {k: round(v, 2) for k, v in channel_contribs.items()},
        "channel_contribution_pct": {
            k: round(v / total_kpi * 100, 2) if total_kpi else 0
            for k, v in channel_contribs.items()
        },
        "roi": {k: round(v, 4) for k, v in roi.items()},
        "y_pred_train": y_pred_train.tolist(),
        "y_actual_train": y_train.tolist(),
        "y_pred_test": y_pred_test.tolist(),
        "y_actual_test": y_test.tolist(),
    }
