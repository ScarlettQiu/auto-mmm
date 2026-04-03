# Auto-MMM Orchestrator

You are an autonomous Marketing Mix Modeling agent. Your job is to run three MMM models, compare results, and generate a final analysis report. Follow these steps precisely.

---

## Setup (run once)

```bash
cd /Users/qiuyu/auto-mmm
pip install -r requirements.txt
```

Verify data is accessible:
```bash
python prepare.py
```

Expected output: a summary table showing 12 periods, 8 channels, KPI stats.

---

## Round Loop

Each round = one full experiment cycle. Read `state.json` to find the current round number, then increment.

### Step 1 — Read current state
```bash
cat state.json
```
Note `current_round`. The next round = `current_round + 1`. Call it `N`.

### Step 2 — Run all three models
```bash
python run_models.py --round N
```

This runs Ridge, PyMC (Bayesian), and LightweightMMM (NNLS fallback if JAX unavailable).
Results saved to `results/latest.json` and `rounds/R{N:02d}_results.json`.

### Step 3 — Compare models
```bash
python compare.py
```

Read the output carefully:
- Which channels have highest ROI across all models?
- Where do models disagree (CV > 30%)?
- Which model fits best (R², MAPE)?

### Step 4 — Write your interpretation
Based on Step 3, write a 2–4 sentence summary covering:
1. Top 2–3 channels by consensus ROI
2. Any major disagreements and why they might exist
3. One recommendation for the marketing team
4. One note about data limitations

Save it as `rounds/R{N:02d}_summary.md`.

### Step 5 — Generate report
```bash
python report_builder.py --round N --summary "YOUR SUMMARY TEXT HERE"
```

This produces:
- `results/report.md` — full markdown report
- `results/report.pptx` — PowerPoint deck

### Step 6 — Update state
`state.json` is updated automatically by `run_models.py`. Confirm the round number incremented correctly.

---

## Iteration Guide

If `test_mape > 30%` for all models, consider:
- Adjusting `adstock_max_lag` in `config.json` (try 1 or 2 instead of 3)
- Changing `hill_slope` (try 1.5 or 3.0)
- Removing a control variable that may be collinear

If models strongly disagree on a channel (CV > 50%), note it explicitly in the summary — it usually means:
- High collinearity between that channel and another
- Insufficient variation in spend to identify the effect
- The channel operates on a different lag than assumed

---

## Output Files

| File | Contents |
|---|---|
| `results/latest.json` | Full model results (latest round) |
| `results/roi_comparison.csv` | Channel × model ROI table |
| `results/contribution_comparison.csv` | Channel × model contribution % |
| `results/model_fit.csv` | R², MAPE per model |
| `results/report.md` | Full analysis report |
| `results/report.pptx` | PowerPoint deck for stakeholders |
| `rounds/R{N}_results.json` | Results for each round |
| `rounds/R{N}_summary.md` | Agent's interpretation per round |

---

## Data Context

- Dataset: DT Mart (Indian e-commerce), Jan–Dec 2015/2016
- KPI: `total_gmv` (Gross Merchandise Value in INR)
- Media channels: TV, Digital, Sponsorship, Content Marketing, Online Marketing, Affiliates, SEM, Radio
- Controls: NPS (brand health), total discount
- **Note: Only 12 monthly data points. Use cautious language about confidence.**

---

## Start

Begin by running Step 1. If `current_round` is 0, this is a fresh start — run the full loop from Step 2.
