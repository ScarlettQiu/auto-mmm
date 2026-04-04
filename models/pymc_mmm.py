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
        from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
    except ImportError:
        return {
            "model": "pymc",
            "error": "pymc-marketing not installed. Run: pip install pymc-marketing",
            "skipped": True,
        }

    channels = cfg["media_channels"]
    controls = [c for c in cfg["control_variables"] if c in train_df.columns]
    kpi_col  = "kpi"

    channel_cols = [f"{ch}_saturated" for ch in channels if f"{ch}_saturated" in train_df.columns]
    valid_channels = [ch for ch in channels if f"{ch}_saturated" in train_df.columns]

    try:
        import pytensor
        pytensor.config.cxx = ""  # suppress g++ warning

        mmm = MMM(
            date_column="date",
            channel_columns=channel_cols,
            control_columns=controls,
            adstock=GeometricAdstock(l_max=cfg.get("adstock_max_lag", 1)),
            saturation=HillSaturation(),
        )

        X_train = train_df.drop(columns=[kpi_col])
        y_train = train_df[kpi_col]
        X_test  = test_df.drop(columns=[kpi_col])

        mmm.fit(
            X=X_train, y=y_train,
            progressbar=False,
            draws=cfg.get("pymc_samples", 300),
            tune=cfg.get("pymc_tune", 200),
            chains=1,
        )

        import numpy as _np
        ppc_train = mmm.sample_posterior_predictive(X_train, combined=True, original_scale=True)
        y_pred_train = _np.array(ppc_train["y"]).mean(axis=1)  # shape (dates, samples) → (dates,)
        ppc_test = mmm.sample_posterior_predictive(X_test, combined=True, original_scale=True)
        y_pred_test = _np.array(ppc_test["y"]).mean(axis=1)

        total_kpi = float(y_train.sum())
        contributions = mmm.compute_channel_contribution_original_scale()
        # channel coord is the saturated column name e.g. "TV_saturated"
        channel_contribs = {}
        for ch, col in zip(valid_channels, channel_cols):
            channel_contribs[ch] = float(contributions.sel(channel=col).values.sum())

        roi = {}
        for ch in valid_channels:
            spend = train_df[f"{ch}_adstock"].sum() if f"{ch}_adstock" in train_df.columns else 1
            roi[ch] = float(channel_contribs[ch] / spend) if spend > 0 else 0.0

        train_r2   = 1 - np.var(y_train.values - y_pred_train) / np.var(y_train.values)
        train_mape = _mape(y_train.values, y_pred_train)
        test_mape  = _mape(test_df[kpi_col].values, y_pred_test)

    except Exception as e:
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

        with pm.Model():
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
