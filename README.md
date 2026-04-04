# Auto-MMM

Autonomous Marketing Mix Modeling powered by Claude. Point an AI agent at `program.md` and it runs **three independent MMM models**, critiques its own analysis through a five-agent loop, iterates on model configuration, and produces a stakeholder report — without human involvement.

The three-model approach is deliberate: **Ridge** (fast, regularised), **PyMC** (Bayesian), and **LightweightMMM** (positive-constrained) each make different assumptions. Where all three agree, you can act with confidence. Where they disagree, that's a diagnostic — thin data, collinearity, or a modelling assumption worth questioning. No single model can tell you this.

The five specialist agents keep roles separated so no single agent can both produce and approve its own output:

| Agent | Role | Output |
|---|---|---|
| **Data Explorer** | EDA before training — collinearity, anomalies, VIF, readiness score | `rounds/R01_data_exploration.md` |
| **Tuner** | Proposes one config change per round based on prior fit metrics | Updated `config.json` |
| **Analyst** | Interprets ROI, contributions, and model agreement into a business narrative | `rounds/R{N}_analysis.md` |
| **Critic** | Runs 6 quality checks — overfitting, sign correctness, plausibility, consensus honesty, collinearity, sample size | `APPROVED` or `REVISE` |
| **Reporter** | Rewrites the approved analysis in plain English for a CMO audience, generates deck | `report.md` + `report.pptx` |

---

## Architecture

```
discover.py  ←→  Notion knowledge layer (field definitions, business context, known issues)
     ↓
metadata.json + config.json
     ↓
ORCHESTRATOR (program.md)
        │
        ├── DATA EXPLORER (agents/data_explorer.md)  ← Round 1 only
        │       EDA on raw dataset: overview, KPI distribution, channel spend,
        │       collinearity (VIF), anomalies, multi-entity check, readiness score.
        │       Returns: EXPLORATION_DONE
        │
        ├── TUNER (agents/tuner.md)                  ← Round 2+ only
        │       Reads prior round fit metrics, proposes one config change
        │       (adstock decay, Hill slope, PyMC samples). Edits config.json.
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
        │       Reads model output + EDA report, writes business narrative.
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
Round 1:  Data Explorer → [skip Tuner] → Models → Analyst → Critic → Reporter
Round 2+: [skip Explorer] → Tuner → Models → Analyst → Critic → Reporter
```

---

## Knowledge Layer (Notion)

Auto-MMM uses a **Notion knowledge layer** as a live data dictionary. Business teams can update field descriptions, expected ROI ranges, and known data issues directly in Notion — no code changes needed. Every time `discover.py` runs it pulls the latest knowledge and merges it into `metadata.json`, which all five agents read.

Three Notion databases:

| Database | What it stores |
|---|---|
| **Field Definitions** | Column name, label, type (channel/kpi/control), unit, expected ROI min/max, description |
| **Business Context** | Brand, market, currency, seasonality notes, typical media share |
| **Known Issues** | Data quality problems with severity (high/medium/low) and recommended action |

### Set up Notion integration

1. Go to **https://www.notion.so/my-integrations** → New integration → copy the token
2. Open your Notion page → `...` → Connections → connect your integration
3. Run discovery with your token:

```bash
python discover.py --source csv --path ./data.csv \
  --notion-token $NOTION_TOKEN
```

Store the token as an environment variable — never hard-code it:
```bash
export NOTION_TOKEN=ntn_...
```

---

## Data Sources

`prepare.py` supports three data sources via `config.json`:

| Source | Config key | Install |
|---|---|---|
| CSV file | `"source": "csv"` | none |
| BigQuery | `"source": "bigquery"` | `pip install google-cloud-bigquery` |
| Google Sheets | `"source": "gsheet"` | `pip install gspread google-auth` |

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

### 2. Point at your data

**Option A — use discover.py (recommended for any new dataset):**
```bash
python discover.py --source csv --path ./your_data.csv \
  --notion-token $NOTION_TOKEN   # optional — enriches with Notion knowledge
```

This auto-detects columns, generates `config.json` and `metadata.json`, and pulls your knowledge layer from Notion.

**Option B — use the included Conjura eCommerce MMM dataset:**

`data.csv` in this repo is a ready-to-use sample: 132 weekly observations for an Apparel brand (2021–2024), with 8 Google + Meta spend channels and revenue as the KPI. Already profiled and ready to run.

Source: [Multi-Region Marketing Mix Modeling MMM Dataset for Several eCommerce Brands](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841) — Conjura via Figshare. Contains 93 brands across multiple regions and verticals.

```bash
# Already included — just run the loop:
Read program.md and run the loop.
```

**Option C — use the DT Mart dataset (original example):**
```bash
pip install kagglehub
python -c "
import kagglehub
kagglehub.dataset_download('datatattle/dt-mart-market-mix-modeling')
"
```

### 3. Run the agent loop

Open a `claude` terminal session in this directory:

```
Read program.md and run the loop.
```

Claude orchestrates all five agents. Round 1 runs the Data Explorer first:

```
Auto-MMM starting. Reading state.json…
Round 1: skipping Tuner (no prior results)
Spawning Data Explorer...
  EXPLORATION_DONE: rounds/R01_data_exploration.md
  Readiness score: 3/5 — 12 periods, August anomaly flagged
Running models... ridge ✓  pymc ✓  lightweight_mmm ✓
Spawning Analyst...
  ANALYSIS_DONE: rounds/R01_analysis.md
Spawning Critic...
  REVISE: 0% media contribution must be labelled as model failure, not neutral finding
Analyst revising...
  APPROVED
Spawning Reporter...
  REPORT_DONE: results/report.md + results/report.pptx
Round 1 complete.
```

### 4. Run more rounds

Each round the Tuner tries one config improvement:
- Round 1: baseline + EDA
- Round 2: Tuner adjusts adstock lag, re-runs
- Round 3: Tuner adjusts Hill slope, re-runs
- ...until test MAPE stops improving

### 5. New or updated dataset

```bash
# New rows appended — re-profile, keep round history
python discover.py --source csv --path ./data.csv --notion-token $NOTION_TOKEN
Read program.md and run the loop.

# BigQuery source
python discover.py --source bigquery \
  --query "SELECT * FROM project.dataset.mmm_weekly" \
  --project my-gcp-project \
  --notion-token $NOTION_TOKEN
```

---

## Why Three MMM Models?

No single MMM model is right in all situations. Each makes different assumptions about how media drives sales.

**1. No model is always correct**
Ridge is fast and transparent but can shrink correlated channels to zero. PyMC captures diminishing returns and uncertainty but is slow and sensitive to prior choices. LightweightMMM enforces positive-only ROI but may over-attribute to correlated channels. Each has blind spots the others don't share.

**2. Agreement builds confidence, disagreement reveals risk**
When all three rank the same channel as top performer, you can act. When they disagree, that's a diagnostic — thin data, collinearity, or a modelling assumption worth questioning. A single model can't tell you this.

**3. Different models suit different situations**

| Situation | Best model |
|---|---|
| Quick first pass, any data size | Ridge — runs in seconds |
| Small dataset (<30 periods) | PyMC — priors compensate for thin data |
| Production budget decisions | PyMC — full credible intervals |
| Need positive-constrained estimates fast | LightweightMMM |
| Final validation | All three — consensus = trustworthy |

---

## Three MMM Models

| Model | Method | Uncertainty | Requires |
|---|---|---|---|
| **Ridge** | Regularised regression + bootstrap (200 samples) | Confidence intervals | sklearn only |
| **PyMC** | Full Bayesian with DelayedSaturatedMMM | Posterior distribution | `pip install pymc-marketing` |
| **LightweightMMM** | Google's JAX-based Hill + adstock | Posterior samples | `pip install lightweight_mmm` |

Ridge always runs with no extra dependencies. LightweightMMM and PyMC fall back to scipy NNLS only if JAX / pymc-marketing are not installed.

---

## The Five Agent Roles

### Data Explorer
**Role:** EDA on the raw dataset before any model training — runs once per dataset (Round 1 only).

Produces a structured report covering: dataset overview, KPI distribution, channel spend analysis, pairwise collinearity (Pearson r + VIF), anomaly detection (z > 3σ), multi-entity check, and a 1–5 readiness score with specific recommended actions. The Analyst and Critic read this report every round to ground their interpretation in data quality facts.

### Tuner
**Role:** Iterates model configuration between rounds to improve fit.

Proposes exactly **one** config change per round — adstock decay, Hill slope, or PyMC sampling depth. One change per round keeps experiments comparable.

Decision rules:
- 6 channels zero ROI → reduce `adstock_max_lag`
- Models strongly disagree (CV > 50%) → reduce `hill_slope`
- PyMC test MAPE > 40% → increase sampling iterations
- All models test MAPE < 15% → no change needed

### Analyst
**Role:** Interprets raw model numbers into a business narrative.

Reads model output + EDA report. Covers ROI rankings, model agreement/disagreement, contribution plausibility, and what the data can and cannot support. Under 400 words, always cites actual numbers.

### Critic
**Role:** Quality gate — challenges the Analyst before anything reaches the report.

Runs six checks. Issues `REVISE` with a specific reason if any check fails. Analyst fixes once, Critic re-reviews. Max one revision cycle — no infinite loops.

| Check | What it catches |
|---|---|
| **Overfitting** | R²=1.0 on small samples |
| **Sign correctness** | Negative ROI despite confirmed spend |
| **Contribution plausibility** | Media <5% or >80% of KPI |
| **Consensus honesty** | Analyst ignored model disagreements |
| **Collinearity** | Channels that co-moved, confusing attribution |
| **Sample size caveat** | Limitation not clearly communicated |

### Reporter
**Role:** Translates the approved analysis into stakeholder language.

Only runs after `APPROVED`. Plain English, no jargon, no model names in the headline. Uses business-friendly channel labels from the Notion knowledge layer. Produces `report.md` + `report.pptx`.

---

## Output Files

| File | Contents |
|---|---|
| `metadata.json` | Dataset profile + Notion knowledge layer (channels, ROI ranges, known issues) |
| `rounds/R01_data_exploration.md` | EDA report: collinearity, anomalies, readiness score |
| `results/report.md` | Final stakeholder report (plain English) |
| `results/report.pptx` | PowerPoint deck with model overviews, ROI charts, recommendations |
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

## Project Structure

```
auto-mmm/
├── program.md              ← orchestrator (start here)
├── discover.py             ← auto-profiles dataset, fetches Notion knowledge layer
├── config.json             ← dataset + model parameters (auto-generated by discover.py)
├── metadata.json           ← dataset profile + Notion knowledge (read by all agents)
├── prepare.py              ← multi-source data loader (CSV / BigQuery / GSheet)
├── run_models.py           ← runs all 3 models, saves results
├── compare.py              ← ROI/contribution comparison, agreement scoring
├── report_builder.py       ← generates report.md + report.pptx
├── state.json              ← round counter, best scores
├── data_dictionary.csv     ← optional: your column descriptions (imported by discover.py)
├── agents/
│   ├── data_explorer.md    ← EDA agent (Round 1 only)
│   ├── analyst.md          ← interprets results, writes narrative
│   ├── critic.md           ← six-check quality gate
│   ├── tuner.md            ← iterates config between rounds
│   └── reporter.md         ← plain-English report for stakeholders
├── models/
│   ├── ridge_mmm.py        ← Ridge + bootstrap
│   ├── pymc_mmm.py         ← Bayesian MMM (DelayedSaturatedMMM)
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
statsmodels>=0.14.0
```

Recommended (enables full LightweightMMM and PyMC models):
```
jax>=0.6.0
jaxlib>=0.6.0
lightweight_mmm>=0.1.9
pymc-marketing
```

Optional (for non-CSV data sources):
```
google-cloud-bigquery   # BigQuery
gspread google-auth     # Google Sheets
```
