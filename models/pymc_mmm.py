"""PyMC-Marketing Bayesian MMM.

Uses DelayedSaturatedMMM for full posterior uncertainty.
Requires: pip install pymc-marketing
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
        import pymc as pm
        from pymc_marketing.mmm import DelayedSaturatedMMM
    except ImportError:
        return {
            "model": "pymc",
            "error": "pymc-marketing not installed. Run: pip install pymc-marketing",
            "skipped": True,
        }

    channels = cfg["media_channels"]
    controls = [c for c in cfg["control_variables"] if c in train_df.columns]

    date_col = "date"
    kpi_col  = "kpi"

    # PyMC-Marketing expects raw spend + date; it handles adstock/saturation internally
    # Re-load raw spend from adstock columns (pre-adstock = original spend)
    # We use the saturated columns as X since we've already transformed
    channel_cols = [f"{ch}_saturated" for ch in channels if f"{ch}_saturated" in train_df.columns]
    valid_channels = [ch for ch in channels if f"{ch}_saturated" in train_df.columns]

    X_cols = channel_cols + [c for c in controls if c in train_df.columns]

    try:
        mmm = DelayedSaturatedMMM(
            date_column=date_col,
            channel_columns=channel_cols,
            control_columns=[c for c in controls if c in train_df.columns],
            adstock_max_lag=1,  # already adstocked
        )

        mmm.fit(
            train_df.rename(columns={kpi_col: "y"}),
            target_col="y",
            chains=2,
            draws=cfg.get("pymc_samples", 500),
            tune=cfg.get("pymc_tune", 250),
            progressbar=False,
        )

        # Posterior predictive
        ppc = mmm.sample_posterior_predictive(train_df)
        y_pred_train = ppc["y"].mean(("chain", "draw")).values

        # Channel contributions from posterior
        contributions = mmm.get_channel_contributions_share_of_contribution_original_scale()
        total_kpi = float(train_df[kpi_col].sum())

        channel_contribs = {}
        for ch, col in zip(valid_channels, channel_cols):
            share = float(contributions.sel(channel=col).mean())
            channel_contribs[ch] = share * total_kpi

        roi = {}
        for ch in valid_channels:
            adstock_col = f"{ch}_adstock"
            spend = train_df[adstock_col].sum() if adstock_col in train_df.columns else 1
            roi[ch] = float(channel_contribs[ch] / spend) if spend > 0 else 0.0

        train_r2 = 1 - np.var(train_df[kpi_col].values - y_pred_train) / np.var(train_df[kpi_col].values)
        train_mape = _mape(train_df[kpi_col].values, y_pred_train)

        # Test prediction
        ppc_test = mmm.sample_posterior_predictive(test_df)
        y_pred_test = ppc_test["y"].mean(("chain", "draw")).values
        test_mape = _mape(test_df[kpi_col].values, y_pred_test)

    except Exception as e:
        # Fallback: simple Bayesian linear regression via PyMC directly
        return _fallback_pymc(train_df, test_df, cfg, str(e))

    return {
        "model": "pymc",
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
        "y_actual_train": train_df[kpi_col].tolist(),
        "y_pred_test": y_pred_test.tolist(),
        "y_actual_test": test_df[kpi_col].tolist(),
    }


def _fallback_pymc(train_df, test_df, cfg, original_error: str) -> dict:
    """Simple PyMC Bayesian linear regression as fallback."""
    try:
        import pymc as pm

        channels = cfg["media_channels"]
        feat_cols = [f"{ch}_saturated" for ch in channels if f"{ch}_saturated" in train_df.columns]
        valid_channels = [ch for ch in channels if f"{ch}_saturated" in train_df.columns]

        X = train_df[feat_cols].values
        y = train_df["kpi"].values.astype(float)

        # Normalise
        X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
        y_mean, y_std = y.mean(), y.std() + 1e-8
        Xs = (X - X_mean) / X_std
        ys = (y - y_mean) / y_std

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            betas = pm.HalfNormal("betas", sigma=1, shape=Xs.shape[1])
            sigma = pm.HalfNormal("sigma", sigma=1)
            mu = alpha + pm.math.dot(Xs, betas)
            pm.Normal("y", mu=mu, sigma=sigma, observed=ys)
            trace = pm.sample(
                draws=cfg.get("pymc_samples", 500),
                tune=cfg.get("pymc_tune", 250),
                chains=2,
                progressbar=False,
                return_inferencedata=True,
            )

        # Posterior means → predictions
        alpha_m = float(trace.posterior["alpha"].mean())
        betas_m = trace.posterior["betas"].mean(("chain", "draw")).values
        y_pred_s = alpha_m + Xs @ betas_m
        y_pred_train = y_pred_s * y_std + y_mean

        X_test = test_df[feat_cols].values
        Xs_test = (X_test - X_mean) / X_std
        y_pred_test = (alpha_m + Xs_test @ betas_m) * y_std + y_mean

        total_kpi = float(y.sum())
        channel_contribs = {}
        for i, ch in enumerate(valid_channels):
            contrib = float((betas_m[i] * Xs[:, i]).sum() * y_std)
            channel_contribs[ch] = contrib

        roi = {}
        for ch in valid_channels:
            spend = train_df[f"{ch}_adstock"].sum() if f"{ch}_adstock" in train_df.columns else 1
            roi[ch] = float(channel_contribs[ch] / spend) if spend > 0 else 0.0

        train_r2 = 1 - np.var(y - y_pred_train) / np.var(y)
        train_mape = _mape(y, y_pred_train)
        test_mape = _mape(test_df["kpi"].values, y_pred_test)

        return {
            "model": "pymc",
            "note": f"Used fallback Bayesian LM (DelayedSaturatedMMM error: {original_error[:80]})",
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
            "y_actual_train": y.tolist(),
            "y_pred_test": y_pred_test.tolist(),
            "y_actual_test": test_df["kpi"].tolist(),
        }

    except Exception as e2:
        return {
            "model": "pymc",
            "error": f"PyMC failed: {e2}. Original: {original_error}",
            "skipped": True,
        }
