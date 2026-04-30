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


_PYMC_TIMEOUT_SEC = 300


def _subprocess_target(train_dict, test_dict, cfg, queue):
    """Runs in a child process so we can hard-kill it if PyTensor hangs."""
    try:
        import pandas as pd
        train_df = pd.DataFrame(train_dict)
        train_df["date"] = pd.to_datetime(train_df["date"])
        test_df = pd.DataFrame(test_dict)
        test_df["date"] = pd.to_datetime(test_df["date"])
        result = _run_pymc(train_df, test_df, cfg)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("err", str(e)))


def run(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
) -> dict:
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_subprocess_target,
        args=(train_df.to_dict("list"), test_df.to_dict("list"), cfg, queue),
        daemon=True,
    )
    p.start()
    p.join(timeout=_PYMC_TIMEOUT_SEC)
    if p.is_alive():
        p.kill()
        p.join()
        return _fallback_pymc(train_df, test_df, cfg, f"PyMC timed out after {_PYMC_TIMEOUT_SEC}s")
    status, payload = queue.get()
    if status == "ok":
        return payload
    return _fallback_pymc(train_df, test_df, cfg, payload)


def _run_pymc(
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
    controls = [c for c in cfg.get("control_variables", []) if c in train_df.columns]
    kpi_col  = "kpi"

    # PyMC-Marketing needs raw spend, not pre-transformed columns.
    # Rebuild a raw-spend DataFrame aligned with train/test dates.
    try:
        import pandas as _pd
        from pathlib import Path as _Path
        raw = _pd.read_csv(_Path(cfg["data_path"]).expanduser())
        date_col_raw = cfg.get("date_column", "date")
        raw[date_col_raw] = _pd.to_datetime(raw[date_col_raw])
        raw = raw.rename(columns={date_col_raw: "date", cfg["kpi_column"]: kpi_col})
        raw = raw[["date", kpi_col] + channels].copy()
        for ch in channels:
            # Small epsilon prevents 0^negative_alpha in Hill saturation during NUTS
            raw[ch] = raw[ch].fillna(0).clip(lower=1e-6)
    except Exception as e:
        return _fallback_pymc(train_df, test_df, cfg, f"raw load failed: {e}")

    valid_channels = [ch for ch in channels if ch in raw.columns]
    channel_cols = valid_channels

    # Align train/test dates
    train_dates = train_df["date"].values
    test_dates  = test_df["date"].values
    raw_train = raw[raw["date"].isin(train_dates)].sort_values("date").reset_index(drop=True)
    raw_test  = raw[raw["date"].isin(test_dates)].sort_values("date").reset_index(drop=True)

    try:
        import pytensor
        # Use clang if available (macOS without g++) — dramatically faster than pure Python
        import shutil
        if shutil.which("clang++") and not shutil.which("g++"):
            pytensor.config.cxx = "clang++"
        elif not shutil.which("g++"):
            pytensor.config.cxx = ""

        mmm_kwargs = dict(
            date_column="date",
            channel_columns=channel_cols,
            adstock=GeometricAdstock(l_max=cfg.get("adstock_max_lag", 1)),
            saturation=HillSaturation(),
        )
        if controls:
            mmm_kwargs["control_columns"] = controls
        mmm = MMM(**mmm_kwargs)

        X_train = raw_train.drop(columns=[kpi_col])
        y_train = raw_train[kpi_col].astype(float)
        X_test  = raw_test.drop(columns=[kpi_col])

        mmm.fit(
            X=X_train, y=y_train,
            progressbar=False,
            draws=cfg.get("pymc_samples", 200),
            tune=cfg.get("pymc_tune", 100),
            chains=1,
        )

        import numpy as _np

        def _ppc_mean(ppc_arr, n_periods):
            # With combined=True, ppc["y"] is (dates, samples) — average over axis=1
            arr = _np.array(ppc_arr)
            if arr.ndim == 2:
                return arr.mean(axis=1) if arr.shape[0] == n_periods else arr.mean(axis=0)
            return arr.mean(axis=tuple(i for i in range(arr.ndim - 1)))  # last dim = dates

        ppc_train = mmm.sample_posterior_predictive(X_train, combined=True, original_scale=True)
        y_pred_train = _ppc_mean(ppc_train["y"], len(y_train))
        ppc_test = mmm.sample_posterior_predictive(X_test, combined=True, original_scale=True)
        y_pred_test = _ppc_mean(ppc_test["y"], len(X_test))

        total_kpi = float(y_train.sum())
        contributions = mmm.compute_channel_contribution_original_scale()
        # dims: (chain, draw, date, channel) — mean over chain+draw, then sum over date
        channel_contribs = {}
        for ch in valid_channels:
            try:
                sel = contributions.sel(channel=ch)
                # squeeze out chain/draw dims, sum over date
                vals = sel.mean(dim=[d for d in sel.dims if d != "date"]).values
                channel_contribs[ch] = float(vals.sum())
            except Exception:
                channel_contribs[ch] = 0.0

        roi = {}
        for ch in valid_channels:
            spend = float(raw_train[ch].sum()) if ch in raw_train.columns else 1.0
            roi[ch] = float(channel_contribs[ch] / spend) if spend > 0 else 0.0

        train_r2   = 1 - np.var(y_train.values - y_pred_train) / np.var(y_train.values)
        train_mape = _mape(y_train.values, y_pred_train)
        test_mape  = _mape(raw_test[kpi_col].values.astype(float), y_pred_test)

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
        "y_actual_train": raw_train[kpi_col].tolist(),
        "y_pred_test": y_pred_test.tolist(),
        "y_actual_test": raw_test[kpi_col].tolist(),
    }


def _fallback_pymc(train_df, test_df, cfg, original_error: str) -> dict:
    """Simple PyMC Bayesian linear regression as fallback."""
    try:
        import pymc as pm
        import pytensor, shutil
        if shutil.which("clang++") and not shutil.which("g++"):
            pytensor.config.cxx = "clang++"
        elif not shutil.which("g++"):
            pytensor.config.cxx = ""

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
            # Normal (not HalfNormal) — allows negative coefficients for channels
            # with negative true ROI; HalfNormal forces positivity and collapses to zero
            betas = pm.Normal("betas", mu=0, sigma=1, shape=Xs.shape[1])
            sigma = pm.HalfNormal("sigma", sigma=1)
            mu = alpha + pm.math.dot(Xs, betas)
            pm.Normal("y", mu=mu, sigma=sigma, observed=ys)
            trace = pm.sample(
                draws=cfg.get("pymc_samples", 500),
                tune=cfg.get("pymc_tune", 250),
                chains=1,
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
