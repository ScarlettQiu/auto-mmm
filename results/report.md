# MMM Analysis Report
*Generated 2026-04-06 · 128 training periods · 4 holdout periods*

## Data Exploration Findings

# Data Exploration — Round 1

**Dataset:** data.csv
**Date:** 2026-04-03
**Readiness Score:** 4 / 5

---

## 1. Dataset Overview

- **Rows:** 132 | **Columns:** 13 | **Frequency:** Weekly
- **Date range:** 2021-11-22 to 2024-05-27
- **Missing values:** 0 across all columns
- **Duplicate rows:** 0
- **Columns:** date, 8 channel spend (float64), first_orders (int64), revenue (float64), all_orders (int64), all_revenue (float64)

---

## 2. KPI Analysis (revenue)

| Stat | Value |
|------|-------|
| Mean | $30,293,151 |
| Median | $24,250,275 |
| Std | $20,266,293 |
| Min | $10,620,990 |
| Max | $113,974,910 |
| p5 | $13,625,234 |
| p25 | $18,886,596 |
| p75 | $32,342,290 |
| p95 | $75,628,300 |

Revenue is right-skewed (mean >> median), with a coefficient of variation ~67%. Mean revenue is relatively stable across years (2022: $29.9M, 2023: $31.0M, 2024: $29.1M), suggesting no strong upward trend. Quarter-over-quarter growth is volatile (mean +18.8%, range −51.2% to +139.7%), indicating strong seasonal spikes. No zero or null KPI periods.

---

## 3. Channel Spend Analysis

| Channel | Total Spend | Mean/Week | Max Week | Zero% | r(KPI) | Flag |
|---------|-------------|-----------|----------|-------|--------|------|
| google_search | $18.0M | $136,691 | $595,248 (2023-12-18) | 0.0% | 0.49 | |
| google_shopping | $3.3M | $25,147 | $1,233,383 (2022-05-30) | 76.5% | 0.34 | SPARSE |
| google_pmax | $74.2M | $562,113 | $1,753,163 (2022-12-19) | 2.3% | 0.29 | |
| google_display | $6.6M | $50,029 | $123,577 (2023-02-13) | 0.8% | 0.02 | |
| google_video | $16.2M | $122,892 | $324,414 (2023-02-20) | 2.3% | −0.05 | |
| meta_facebook | $624.0M | $4,727,632 | $23,143,379 (2022-05-30) | 0.0% | 0.61 | |
| meta_instagram | $226.0M | $1,712,201 | $11,000,380 (2024-05-13) | 0.0% | 0.41 | |
| meta_other | $0.5M | $3,673 | $26,714 (2022-11-07) | 7.6% | 0.21 | |

**google_shopping** has 76.5% zero-spend weeks — sparse, may not be identifiable. Meta channels dominate spend (~88% of total). google_display (r=0.02) and google_video (r=−0.05) show near-zero correlation with revenue.

---

## 4. Collinearity Check

No channel pairs exceed |r| > 0.7. Closest pair: meta_instagram vs meta_other (r=0.674), approaching but below threshold.

VIF scores — all channels below 10 (no severe collinearity):

| Channel | VIF |
|---------|-----|
| meta_facebook | 6.57 |
| google_search | 5.23 |
| google_display | 4.92 |
| google_pmax | 4.34 |
| google_video | 4.11 |
| meta_instagram | 3.81 |
| meta_other | 2.97 |
| google_shopping | 1.61 |

---

## 5. Anomalies

**KPI outliers (z > 3σ):**

| Date | Revenue | Z-score | Action |
|------|---------|---------|--------|
| 2022-05-30 | $113,974,910 | 4.13 | Investigate — coincides with spend spikes |
| 2023-05-29 | $113,025,700 | 4.08 | Investigate — possible promotion |
| 2023-12-11 | $104,083,700 | 3.64 | Investigate — holiday season spike |
| 2023-05-22 | $94,653,890 | 3.18 | Keep with flag |

**Channel spend outliers (z > 4σ):**

| Date | Channel | Z-score | Action |
|------|---------|---------|--------|
| 2022-05-30 | google_shopping | 10.19 | Investigate — extreme single-week spike |
| 2022-05-30 | meta_facebook | 5.97 | Investigate — correlated with KPI spike |
| 2023-12-18 | google_search | 5.66 | Keep with flag — holiday uplift |
| 2022-11-07 | meta_other | 4.87 | Keep with flag |
| 2024-05-13 | meta_instagram | 4.11 | Keep with flag |

The 2022-05-30 week shows synchronized spikes in google_shopping (z=10.2), meta_facebook (z=6.0), and revenue (z=4.1) — likely a major campaign or promotional event.

---

## 6. Multi-Entity Check

No entity columns detected. Single-entity dataset — no filtering required.

---

## 7. Readiness Verdict

**Score: 4/5**

**Drivers:**
- 132 weekly periods — sufficient for MMM
- No missing values or duplicate rows
- No severe collinearity (max VIF=6.57, no pairs |r|>0.7)
- google_shopping is sparse (76.5% zeros) — identifiability risk
- 4 KPI outliers + 5 channel spend outliers require review
- google_display and google_video show near-zero/negative KPI correlation — may need careful priors

### Recommended actions before modelling:
1. **Exclude or handle google_shopping carefully** — 76.5% zero-spend weeks will produce unreliable adstock and response curve estimates; consider aggregating into google_pmax or setting a strong zero-effect prior.
2. **Flag 2022-05-30 as a potential outlier week** — simultaneous google_shopping (z=10.2) and meta_facebook (z=6.0) spend spikes alongside a KPI spike (z=4.1) may distort channel attribution; consider adding a binary event indicator.
3. **Apply informative priors for google_display and google_video** — near-zero correlations with revenue suggest weak or delayed effects; without priors, the model may assign noise-driven coefficients.


## The Three Models

### Ridge Regression MMM
*Fast, transparent baseline — always runs*

Fits a regularised linear regression on adstock-transformed spend. Bootstraps 200 samples to estimate coefficient uncertainty. Alpha (regularisation strength) selected via 5-fold cross-validation.

**Pros:** Runs in seconds — no heavy dependencies · Coefficients are fully interpretable · Bootstrap gives confidence intervals per channel · Good baseline to sanity-check other models against

**Cons:** Assumes linear spend-response (no saturation curve) · Prone to overfitting on small samples (R²=1.0 is a warning sign) · Shrinks small coefficients to zero — may under-attribute niche channels · No uncertainty propagation — point estimates only

**Best for:** Quick first-pass analysis when data is limited · Sanity-checking Bayesian results · Teams without Python/stats expertise (simple to explain) · Datasets with 50+ weekly observations and low collinearity

### Bayesian MMM (PyMC-Marketing)
*Full posterior uncertainty — the gold standard*

Uses DelayedSaturatedMMM from PyMC-Marketing. Fits a full Bayesian model with prior distributions on adstock decay, Hill saturation, and channel coefficients. MCMC sampling (NUTS) produces a posterior distribution over all parameters — every ROI estimate comes with a credible interval.

**Pros:** Full uncertainty quantification — know how confident you are · Encodes domain knowledge via priors (e.g. TV decays slower than Digital) · Handles small samples better by regularising through priors · Posterior predictive checks catch model misspecification · Industry standard for rigorous MMM (used by Meta, Google)

**Cons:** Slow — 10–60 minutes depending on data size and chain count · Requires pymc-marketing installation and familiarity with Bayesian stats · Prior choices matter — wrong priors can bias results · Results harder to explain to non-technical stakeholders

**Best for:** Production MMM where budget decisions involve significant spend · Datasets with 80+ weekly observations · When you need credible intervals on ROI for board-level decisions · Teams with data science capability to interpret posteriors

### LightweightMMM / NNLS
*Google's JAX-based MMM with NNLS fallback*

Google's LightweightMMM uses JAX for fast GPU/CPU Bayesian inference with built-in Hill adstock. When JAX is unavailable, falls back to Non-Negative Least Squares (NNLS via scipy) — a constrained regression that enforces positive channel coefficients, preventing the negative-ROI problem common in Ridge.

**Pros:** NNLS fallback enforces positive ROI — no negative coefficients · LightweightMMM is significantly faster than PyMC on GPU · Built-in Hill + adstock transforms — no manual feature engineering · Open source from Google — well tested at scale · Good middle ground between Ridge speed and PyMC rigour

**Cons:** NNLS fallback loses uncertainty quantification · JAX installation can be complex (especially on Mac M-series) · LightweightMMM is less actively maintained than PyMC-Marketing · NNLS can over-attribute to channels that happen to correlate with KPI

**Best for:** Teams who want faster Bayesian inference than PyMC · Google Cloud / GCP environments where JAX runs natively · As a cross-check against PyMC results · When you need positive-constrained estimates without full Bayes

## Model Comparison

| Dimension | Ridge | PyMC | LightweightMMM |
|---|---|---|---|
| Speed | ⚡ Seconds | 🐢 10–60 min | 🔄 Minutes (JAX) |
| Uncertainty | Bootstrap CIs | Full posterior | NNLS: none / JAX: posterior |
| Saturation curve | ❌ Linear only | ✅ Hill (built-in) | ✅ Hill (built-in) |
| Adstock | Manual (prepare.py) | ✅ Built-in | ✅ Built-in |
| Negative ROI risk | ⚠️ Yes (shrinkage) | Low (priors) | ✅ No (NNLS positive) |
| Min data needed | 50+ weekly obs | 80+ weekly obs | 50+ weekly obs |
| Install complexity | ✅ pip sklearn | ⚠️ pymc-marketing | ⚠️ JAX required |
| Explainability | ✅ High | Medium | Medium |
| Industry adoption | Baseline standard | ⭐ Gold standard | Google internal |

## When to Use Which Model

| Scenario | Data | Recommended | Reason |
|---|---|---|---|
| Quick first-pass analysis | Any size | **Ridge** | Results in seconds, easy to interpret, good sanity check |
| Small dataset (<30 periods) | Monthly, <30 obs | **PyMC** | Priors regularise better than Ridge shrinkage on thin data |
| Production budget decisions | Weekly, 80+ obs | **PyMC** | Full credible intervals needed for significant spend decisions |
| Fast iteration / experiment loop | Weekly, 50+ obs | **Ridge → LightweightMMM** | Ridge for speed, LightweightMMM for positive-constrained check |
| GCP / GPU environment | Any size | **LightweightMMM** | JAX runs natively on GCP — fastest Bayesian option |
| Board-level reporting | Weekly, 100+ obs | **PyMC + Ridge as sanity check** | Posterior credible intervals defensible; Ridge validates direction |
| High collinearity between channels | Any | **PyMC** | Priors on each channel help disentangle correlated spend patterns |

## Model Fit (This Run)

| model           |   train_r2 | train_mape   | test_mape   | status                                                                            |
|:----------------|-----------:|:-------------|:------------|:----------------------------------------------------------------------------------|
| ridge           |     0.3538 | 39.06%       | 23.17%      | OK                                                                                |
| pymc            |     0.4422 | 40.37%       | 13.12%      | Used fallback Bayesian LM (DelayedSaturatedMMM error: PyMC MMM sampling timed out |
|                 |            |              |             | Apply node that caused the error: Composite{...}(Spe)                             |
| lightweight_mmm |     0.3528 | 43.86%       | 21.53%      | lightweight_mmm/JAX not installed — used scipy NNLS fallback                      |

## ROI by Channel

*Higher = more revenue per unit of spend. Agreement = cross-model consistency.*

| channel         |   lightweight_mmm |   pymc |    ridge |   mean_roi |   std_roi |   cv_pct | agreement   |
|:----------------|------------------:|-------:|---------:|-----------:|----------:|---------:|:------------|
| meta_other      |            0      |     -0 | 264.217  |    88.0725 |  124.553  |    141.4 | ❌ Low       |
| google_shopping |           41.1114 |     -0 |  16.7001 |    19.2705 |   16.8818 |     87.6 | ❌ Low       |
| google_search   |            0      |     -0 |  40.0696 |    13.3565 |   18.889  |    141.4 | ❌ Low       |
| meta_instagram  |            2.6735 |      0 |   1.2977 |     1.3237 |    1.0916 |     82.5 | ❌ Low       |
| meta_facebook   |            1.7309 |     -0 |   1.4767 |     1.0692 |    0.7631 |     71.4 | ❌ Low       |
| google_pmax     |            0      |      0 |   0.4417 |     0.1472 |    0.2082 |    141.4 | ❌ Low       |
| google_video    |            0      |      0 |  -4.0132 |    -1.3377 |    1.8918 |    141.4 | ❌ Low       |
| google_display  |            0      |      0 | -51.03   |   -17.01   |   24.0558 |    141.4 | ❌ Low       |

## Channel Contribution (%)

| channel         |   lightweight_mmm |   pymc |   ridge |   mean_pct |   spread |
|:----------------|------------------:|-------:|--------:|-----------:|---------:|
| meta_facebook   |             41.97 |     -0 |   35.81 |      25.93 |    41.97 |
| meta_instagram  |             19.96 |      0 |    9.69 |       9.88 |    19.96 |
| google_search   |              0    |     -0 |   21.82 |       7.27 |    21.82 |
| google_shopping |              4.68 |     -0 |    1.9  |       2.19 |     4.68 |
| meta_other      |              0    |     -0 |    4.59 |       1.53 |     4.59 |
| google_pmax     |              0    |      0 |    1.04 |       0.35 |     1.04 |
| google_video    |              0    |      0 |   -2.86 |      -0.95 |     2.86 |
| google_display  |              0    |      0 |  -13.51 |      -4.5  |    13.51 |

## Top Channels by Model

- **ridge**: meta_other, google_search, google_shopping
- **pymc**: google_search, google_shopping, google_pmax
- **lightweight_mmm**: google_shopping, meta_instagram, meta_facebook

## High Disagreement Channels

*Interpret with caution — models give significantly different answers.*

| channel         |   mean_roi |   cv_pct | agreement   |
|:----------------|-----------:|---------:|:------------|
| meta_other      |    88.0725 |    141.4 | ❌ Low       |
| google_shopping |    19.2705 |     87.6 | ❌ Low       |
| google_search   |    13.3565 |    141.4 | ❌ Low       |
| meta_instagram  |     1.3237 |     82.5 | ❌ Low       |
| meta_facebook   |     1.0692 |     71.4 | ❌ Low       |
| google_pmax     |     0.1472 |    141.4 | ❌ Low       |
| google_video    |    -1.3377 |    141.4 | ❌ Low       |
| google_display  |   -17.01   |    141.4 | ❌ Low       |

## Data Caveat

This analysis uses 132 monthly periods. MMM typically benefits from 2+ years of weekly data. Interpret uncertainty ranges with this in mind.
