"""Run all MMM models and save results.

Usage:
    python run_models.py                          # run all 3 models
    python run_models.py --models ridge           # just Ridge
    python run_models.py --models ridge pymc      # Ridge + PyMC
    python run_models.py --round 3                # tag results as round 3
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from prepare import load_config, preprocess, train_test_split, summary


MODEL_REGISTRY = {
    "ridge":          "models.ridge_mmm",
    "pymc":           "models.pymc_mmm",
    "lightweight_mmm": "models.lightweight_mmm",
}


def run_all(cfg: dict, model_names: list[str], round_num: int) -> dict:
    df = preprocess(cfg)
    summary(df, cfg)

    holdout = cfg.get("holdout_periods", 2)
    train_df, test_df = train_test_split(df, holdout)
    print(f"\nTrain: {len(train_df)} periods  |  Test (holdout): {len(test_df)} periods\n")

    results = {
        "run_at": datetime.now().isoformat(),
        "round": round_num,
        "data_periods": len(df),
        "train_periods": len(train_df),
        "test_periods": len(test_df),
        "models": {},
    }

    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(f"  [SKIP] Unknown model: {name}")
            continue

        print(f"  Running {name}...", end=" ", flush=True)
        try:
            import importlib
            mod = importlib.import_module(MODEL_REGISTRY[name])
            res = mod.run(train_df, test_df, cfg)
            results["models"][name] = res
            if res.get("skipped"):
                print(f"SKIPPED ({res.get('error', '')})")
            else:
                r2   = res.get("train_r2", "n/a")
                mape = res.get("train_mape", "n/a")
                print(f"done  R²={r2}  MAPE={mape}%")
        except Exception as e:
            print(f"ERROR: {e}")
            results["models"][name] = {"model": name, "error": str(e), "skipped": True}

    return results


def save_results(results: dict, cfg: dict, round_num: int) -> Path:
    out_dir = Path(cfg.get("results_dir", "./results"))
    out_dir.mkdir(exist_ok=True)
    rounds_dir = Path(cfg.get("rounds_dir", "./rounds"))
    rounds_dir.mkdir(exist_ok=True)

    # Latest results
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(results, indent=2, default=str))

    # Round-specific
    round_file = rounds_dir / f"R{round_num:02d}_results.json"
    round_file.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nResults saved → {latest}")
    print(f"Round file   → {round_file}")
    return latest


def update_state(round_num: int, results: dict) -> None:
    state_path = Path("state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    state["current_round"] = round_num
    state["last_run"] = results["run_at"]

    # Track best model per metric
    best = state.get("best", {})
    for model_name, res in results["models"].items():
        if res.get("skipped"):
            continue
        r2 = res.get("train_r2", -999)
        prev_best = best.get("r2", {}).get("value", -999)
        if r2 > prev_best:
            best["r2"] = {"model": model_name, "value": r2, "round": round_num}

    state["best"] = best
    state_path.write_text(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["ridge", "pymc", "lightweight_mmm"])
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_all(cfg, args.models, args.round)
    save_results(results, cfg, args.round)
    update_state(args.round, results)


if __name__ == "__main__":
    main()
