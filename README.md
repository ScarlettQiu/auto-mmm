# Auto-MMM

Autonomous Marketing Mix Modeling powered by Claude. Point an AI agent at `program.md`, and it runs three MMM models, critiques its own analysis, iterates on configuration, and produces a stakeholder PowerPoint — without human involvement.

---

## Architecture

```
ORCHESTRATOR (program.md)
        │
        ├── TUNER (agents/tuner.md)
        │       Reads prior round fit metrics, proposes one config change
        │       (adstock decay, Hill slope, PyMC samples). Edits config.json.
        │       Returns: CONFIG_UPDATED or NO_CHANGE
        │
        ├── [run_models.py — runs all 3 MMM models]
        │
        ├── ANALYST (agents/analyst.md)
        │       Interprets ROI rankings, contribution %, model agreement.
        │       Writes a business narrative with actual numbers.
        │       Returns: ANALYSIS_DONE
        │
        ├── CRITIC (agents/critic.md)
        │       Runs 6 checks: overfitting, sign correctness, contribution
        │       plausibility, consensus honesty, collinearity, sample size.
        │       Returns: APPROVED or REVISE: <reason>
        │         → if REVISE: Analyst fixes once, Critic re-reviews
        │         → max one revision cycle per round
        │
        └── REPORTER (agents/reporter.md)
                Rewrites analysis in plain English for CMO audience.
                Runs report_builder.py → report.md + report.pptx
                Returns: REPORT_DONE
```

**Flow each round:**
```
Round 1:  [skip Tuner] → Models → Analyst → Critic → Reporter
Round 2+: Tuner → Models → Analyst → Critic → Reporter
```

---

## Three MMM Models

| Model | Method | Uncertainty | Requires |
|---|---|---|---|
| **Ridge** | Regularised regression + bootstrap (200 samples) | Confidence intervals | sklearn only |
| **PyMC** | Full Bayesian with DelayedSaturatedMMM | Posterior distribution | `pip install pymc-marketing` |
| **LightweightMMM** | Google's JAX-based Hill + adstock | Posterior samples | `pip install lightweight_mmm` |

All three models fall back gracefully if optional dependencies are missing — Ridge always runs, LightweightMMM falls back to scipy NNLS.

---

## What Each Agent Does

### Tuner
- Reads previous round's R², MAPE, and ROI agreement scores
- Proposes exactly **one** config change per round (keeps experiments comparable)
- Decision rules: reduce adstock lag for zero-ROI channels, adjust Hill slope when models disagree, increase PyMC samples when Bayesian fit is poor

### Analyst
- Identifies top channels by consensus ROI (appears in top 3 for 2+ models)
- Flags negative ROI values, high CV% channels, and plausibility issues
- Writes `rounds/R{N}_analysis.md` in under 400 words with actual numbers

### Critic (most important)
Runs six checks before any report is published:

| Check | What it catches |
|---|---|
| Overfitting | R²=1.0 on small samples — Ridge ROI numbers become unreliable |
| Sign correctness | Negative ROI despite real spend — collinearity or data issue |
| Contribution plausibility | Media <5% or >80% of KPI — model failure |
| Consensus honesty | Analyst cherry-picked agreeing models, ignored disagreements |
| Collinearity | Channels that move together confusing the model |
| Sample size caveat | 12 monthly periods — limitation must be communicated |

### Reporter
- Rewrites the approved analysis in plain English — no statistical jargon
- Audience: marketing director or CMO
- Runs `report_builder.py` to generate `report.pptx` (6-slide deck)

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/ScarlettQiu/auto-mmm.git
cd auto-mmm
pip install -r requirements.txt
```

Optional (recommended for full Bayesian model):
```bash
pip install pymc-marketing
```

### 2. Download the dataset

```bash
pip install kagglehub
python -c "
import kagglehub
kagglehub.dataset_download('datatattle/dt-mart-market-mix-modeling')
"
```

The dataset path will be printed — it defaults to `~/.cache/kagglehub/...`. It's already set in `config.json`.

### 3. Verify setup

```bash
python prepare.py
```

Expected: summary table showing 12 periods, 8 channels, KPI stats.

### 4. Run the agent loop

Open a `claude` terminal session in this directory:

```
Read program.md and run the loop.
```

Claude will orchestrate all four agents autonomously. You'll see output like:

```
Auto-MMM starting. Reading state.json…
Round 1: skipping Tuner (no prior results)
Running models... ridge ✓  lightweight_mmm ✓  pymc ✓
Spawning Analyst...
  ANALYSIS_DONE: rounds/R01_analysis.md
Spawning Critic...
  REVISE: Overfitting not acknowledged — Ridge R²=1.0 on 10 data points
Analyst revising...
  ANALYSIS_DONE: rounds/R01_analysis.md (revised)
Critic re-reviewing...
  APPROVED
Spawning Reporter...
  REPORT_DONE: results/report.md + results/report.pptx
Round 1 complete.
```

### 5. Run more rounds

Each round the Tuner tries one config improvement. Typical progression:
- Round 1: baseline results
- Round 2: Tuner adjusts adstock lag, re-runs
- Round 3: Tuner adjusts Hill slope, re-runs
- ...until test MAPE stops improving

---

## Output Files

| File | Contents |
|---|---|
| `results/report.md` | Final stakeholder report (plain English) |
| `results/report.pptx` | 6-slide PowerPoint deck |
| `results/roi_comparison.csv` | Channel × model ROI table |
| `results/contribution_comparison.csv` | Channel × model contribution % |
| `results/model_fit.csv` | R², train MAPE, test MAPE per model |
| `results/latest.json` | Full raw results (latest round) |
| `rounds/R{N}_results.json` | Raw model output per round |
| `rounds/R{N}_tuning.md` | Tuner's config change log |
| `rounds/R{N}_analysis.md` | Analyst's interpretation |
| `rounds/R{N}_review.md` | Critic's six-check review |
| `state.json` | Current round, best scores, run history |

---

## Dataset

Uses the [DT Mart Market Mix Modeling](https://www.kaggle.com/datasets/datatattle/dt-mart-market-mix-modeling) dataset from Kaggle:

- **KPI**: `total_gmv` (Gross Merchandise Value, INR)
- **Channels**: TV, Digital, Sponsorship, Content Marketing, Online Marketing, Affiliates, SEM, Radio
- **Controls**: NPS (brand health), total discount
- **Period**: Jul 2015 – Jun 2016 (12 monthly observations)
- **Limitation**: 12 periods is below the MMM standard of 100+ weekly observations. Results are directionally useful but should be validated with more data.

To use your own data, update `config.json`:

```json
{
  "data_path": "./your_data.csv",
  "kpi_column": "revenue",
  "date_column": "week",
  "date_format": "%Y-%m-%d",
  "media_channels": ["tv", "paid_search", "paid_social", "email"],
  "control_variables": ["discount", "seasonality_index"]
}
```

---

## Project Structure

```
auto-mmm/
├── program.md              ← orchestrator (start here)
├── config.json             ← dataset + model parameters
├── prepare.py              ← data loading, adstock + Hill transforms
├── run_models.py           ← runs all 3 models, saves results
├── compare.py              ← ROI/contribution comparison, agreement scoring
├── report_builder.py       ← generates report.md + report.pptx
├── state.json              ← round counter, best scores
├── agents/
│   ├── analyst.md          ← interprets results, writes narrative
│   ├── critic.md           ← six-check quality gate
│   ├── tuner.md            ← iterates config between rounds
│   └── reporter.md         ← plain-English report for stakeholders
├── models/
│   ├── ridge_mmm.py        ← Ridge + bootstrap
│   ├── pymc_mmm.py         ← Bayesian MMM
│   └── lightweight_mmm.py  ← Google LightweightMMM / NNLS fallback
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- No GPU required — runs on Mac/Linux/Windows
- Claude Code CLI (`claude`) for the agent loop

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
tabulate>=0.9.0
python-pptx>=0.6.21
```

Optional:
```
pymc-marketing    # full Bayesian MMM
lightweight_mmm   # Google JAX-based MMM
jax jaxlib        # required for lightweight_mmm
```

