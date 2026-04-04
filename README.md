# Auto-MMM

Autonomous Marketing Mix Modeling powered by Claude. Point an AI agent at `program.md` and it runs **three independent MMM models**, critiques its own analysis through a four-agent loop, iterates on model configuration, and produces a stakeholder report — without human involvement.

The three-model approach is deliberate: **Ridge** (fast, regularised), **PyMC** (Bayesian), and **LightweightMMM** (positive-constrained) each make different assumptions. Where all three agree, you can act with confidence. Where they disagree, that's a diagnostic — thin data, collinearity, or a modelling assumption worth questioning. No single model can tell you this.

The four specialist agents keep roles separated so no single agent can both produce and approve its own output:

| Agent | Role | Output |
|---|---|---|
| **Tuner** | Proposes one config change per round based on prior fit metrics | Updated `config.json` |
| **Analyst** | Interprets ROI, contributions, and model agreement into a business narrative | `rounds/R{N}_analysis.md` |
| **Critic** | Runs 6 quality checks — overfitting, sign correctness, plausibility, consensus honesty, collinearity, sample size | `APPROVED` or `REVISE` |
| **Reporter** | Rewrites the approved analysis in plain English for a CMO audience, generates deck | `report.md` + `report.pptx` |

---

## Architecture

```
ORCHESTRATOR (program.md)
        │
        ├── TUNER (agents/tuner.md)
        │       Reads prior round fit metrics, proposes one config change
        │       (adstock decay, Hill slope, PyMC samples). Edits config.json.
        │       One change per round keeps experiments comparable.
        │       Returns: CONFIG_UPDATED or NO_CHANGE
        │
        ├── run_models.py
        │       Runs all 3 MMM models in sequence:
        │         • Ridge      — regularised regression + 200-sample bootstrap CI
        │         • PyMC       — full Bayesian with DelayedSaturatedMMM
        │         • LightMMM   — Google JAX-based Hill + adstock
        │       Fallback to scipy NNLS only if JAX / pymc-marketing not installed.
        │       Saves results/latest.json + rounds/R{N}_results.json
        │
        ├── ANALYST (agents/analyst.md)
        │       Reads model output, writes business narrative with actual numbers.
        │       Covers ROI rankings, model agreement/disagreement, contribution %,
        │       and what the data can and cannot support.
        │       Returns: ANALYSIS_DONE
        │
        ├── CRITIC (agents/critic.md)
        │       Six-point quality gate — challenges the Analyst before anything
        │       reaches the report:
        │         1. Overfitting (R²=1.0 on small samples)
        │         2. Sign correctness (negative ROI despite confirmed spend)
        │         3. Contribution plausibility (<5% or >80% attributed to media)
        │         4. Consensus honesty (did Analyst ignore model disagreements?)
        │         5. Collinearity (channels that co-moved, confusing attribution)
        │         6. Sample size caveat (limitation clearly communicated?)
        │       On REVISE: Analyst fixes once, Critic re-reviews. Max one cycle.
        │       Returns: APPROVED or REVISE: <reason>
        │
        └── REPORTER (agents/reporter.md)
                Only runs after APPROVED. Rewrites findings in plain English —
                no jargon, no model names in the headline, no unexplained CIs.
                Audience: marketing director or CMO.
                Runs report_builder.py → report.md + report.pptx
                Returns: REPORT_DONE
```

**Flow each round:**
```
Round 1:  [skip Tuner] → Models → Analyst → Critic → Reporter
Round 2+: Tuner → Models → Analyst → Critic → Reporter
```

---

## Why Three MMM Models?

No single MMM model is right in all situations. Each makes different assumptions about how media drives sales — and those assumptions matter a lot when you're deciding where to shift budget.

Running three models in parallel solves three real problems:

**1. No model is always correct**
Ridge is fast and transparent but assumes a linear spend-response relationship and can shrink small channel effects to zero. PyMC captures diminishing returns and uncertainty but is slow and sensitive to prior choices. LightweightMMM enforces positive-only channel effects but may over-attribute to correlated channels. Each has blind spots the others don't share.

**2. Agreement builds confidence, disagreement reveals risk**
When all three models rank the same channel as the top performer, you can act with confidence. When they disagree, that's a signal — either the data is too thin to isolate that channel's effect, or there's collinearity between channels that needs investigating. A single model can't tell you this.

**3. Different models suit different data situations**

| Situation | Best model |
|---|---|
| Quick first pass, any data size | Ridge — runs in seconds |
| Small dataset (<30 periods) | PyMC — priors compensate for thin data |
| Production budget decisions | PyMC — full credible intervals |
| Need positive-constrained estimates fast | LightweightMMM |
| Final validation | All three — consensus = trustworthy |

The auto-MMM loop runs all three, scores their agreement, and flags where they diverge — so you know exactly how much to trust each finding before it reaches the report.

---

## Three MMM Models

| Model | Method | Uncertainty | Requires |
|---|---|---|---|
| **Ridge** | Regularised regression + bootstrap (200 samples) | Confidence intervals | sklearn only |
| **PyMC** | Full Bayesian with DelayedSaturatedMMM | Posterior distribution | `pip install pymc-marketing` |
| **LightweightMMM** | Google's JAX-based Hill + adstock | Posterior samples | `pip install lightweight_mmm` |

Ridge always runs with no extra dependencies. LightweightMMM and PyMC fall back to scipy NNLS only if JAX / pymc-marketing are not installed — install them to get full model outputs.

---

## The Four Agent Roles

Auto-MMM uses four specialist agents coordinated by a single orchestrator (`program.md`). Separating the roles matters — an agent that both runs models and writes the report will unconsciously justify whatever the models produce. The Critic exists precisely to prevent this.

### Tuner
**Role:** Iterates model configuration between rounds to improve fit.

The Tuner reads the previous round's performance metrics (R², MAPE, ROI agreement scores) and proposes exactly **one** config change — adstock decay, Hill slope, or PyMC sampling depth. One change per round keeps experiments comparable so you can measure what actually helped.

Decision rules:
- 6 channels showing zero ROI → reduce `adstock_max_lag` (cross-channel bleed on small data)
- Models strongly disagree on top channel (CV > 50%) → reduce `hill_slope` to flatten saturation
- PyMC test MAPE > 40% → increase sampling iterations
- All models test MAPE < 15% → no change needed

### Analyst
**Role:** Interprets raw numbers and writes a business narrative.

The Analyst reads the model output files and translates them into findings: which channels have the highest consensus ROI, where models agree vs disagree, whether contribution percentages are plausible, and what the data can and cannot tell you. It writes `rounds/R{N}_analysis.md` in under 400 words, always referencing actual numbers.

The Analyst is deliberately kept separate from the Critic — it should state what it sees, not pre-emptively soften findings out of caution.

### Critic
**Role:** Quality gate — challenges the Analyst before anything reaches the report.

The Critic is the most important agent. It runs six checks on the Analyst's interpretation:

| Check | What it catches |
|---|---|
| **Overfitting** | R²=1.0 on small samples — Ridge ROI numbers become unreliable |
| **Sign correctness** | Negative ROI despite confirmed spend — collinearity or data error |
| **Contribution plausibility** | Media <5% or >80% of KPI — model failure or misattribution |
| **Consensus honesty** | Analyst cited only agreeing models and ignored disagreements |
| **Collinearity** | Channels that moved together in the same periods, confusing the model |
| **Sample size caveat** | Data limitation not communicated clearly enough for a non-technical reader |

If the Critic issues a `REVISE`, the Analyst fixes the specific issues and the Critic re-reviews once. After one revision cycle the Critic must approve — no infinite loops. This mirrors how a good data science team works: one round of review, then a decision.

### Reporter
**Role:** Translates the approved analysis into stakeholder language.

The Reporter only runs after the Critic has issued `APPROVED`. It rewrites the Analyst's findings in plain English — no statistical jargon, no model names in the headline, no confidence intervals without explanation. The audience is a marketing director or CMO who needs to know where to put budget, not how MCMC works.

It then runs `report_builder.py` to produce:
- `results/report.md` — full written report
- `results/report.pptx` — 15-slide PowerPoint deck with model overviews, comparison tables, ROI charts, and recommended actions

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/ScarlettQiu/auto-mmm.git
cd auto-mmm
pip install -r requirements.txt
```

Install JAX and the full model libraries (recommended):
```bash
pip install jax jaxlib lightweight_mmm pymc-marketing
```

Both install cleanly on Python 3.10 (CPU, Apple Silicon and x86). Without them, LightweightMMM and PyMC fall back to scipy NNLS.

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

Recommended (enables full LightweightMMM and PyMC models):
```
jax>=0.6.0
jaxlib>=0.6.0
lightweight_mmm>=0.1.9
pymc-marketing
```

