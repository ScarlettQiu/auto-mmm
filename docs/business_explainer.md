# PrismMMM — Plain English Guide
### What We Did, How It Works, and What We Found

*For business readers with no statistics or math background.*

---

## What is Marketing Mix Modeling?

Imagine you're running a shop and spending money on several different types of advertising — Facebook ads, Google ads, Instagram ads. At the end of each week, you count your sales. The question is: **which advertising actually caused those sales?**

That's harder than it sounds. Sales go up in November because of holiday shopping. Sales go up when you run a promotion. Sales go up when Facebook *and* Google *and* Instagram all happen to spend more the same week. How do you separate what *advertising* caused from what *seasons and promotions* caused?

**Marketing Mix Modeling (MMM)** is a technique that looks at 2+ years of weekly data — every week's revenue alongside every week's advertising spend — and tries to mathematically separate those causes. It answers: *"If we spent nothing on Meta Facebook this year, how much revenue would we have lost?"*

---

## What is "Regression"?

Think of it like this. You have 132 weeks of data. Each week you know how much you spent on each advertising channel and how much revenue you made.

Regression is a way of finding the best explanation for why revenue goes up and down. It asks: *"For every extra dollar spent on Meta Facebook, how many dollars of extra revenue do we typically see?"* That number — dollars of revenue per dollar of ad spend — is called **ROI (Return on Investment)**.

We didn't use just one method to calculate this. We used **three different mathematical approaches** to cross-check each other. If all three agree, you can trust the answer. If they disagree, something is making the picture unclear — maybe the channels were all spending more at the same time, or there isn't enough data.

---

## Part 1: Checking the Data Before We Start

Before running any models, the **Data Explorer agent** examined the raw data — like a doctor reviewing test results before making a diagnosis.

### What the data looks like

- **132 weeks** of data, November 2021 to May 2024
- **8 advertising channels**: Meta Facebook, Meta Instagram, Google Search, Google Shopping, Google PMax, Google Display, Google Video, Meta Other
- Weekly revenue ranged from **$10.6M to $114M**
- Average weekly revenue: **$30.3M**
- The data is clean — no missing weeks, no duplicate entries

### Who's spending what

| Channel | Total Spend (2.5 years) | Notable |
|---|---|---|
| Meta Facebook | $624M | By far the biggest — 74% of Meta budget |
| Meta Instagram | $226M | Second — 26% of Meta budget |
| Google PMax | $74M | Largest Google channel |
| Google Search | $18M | |
| Google Video | $16M | |
| Google Display | $6.6M | |
| Google Shopping | $3.3M | Almost never used — 76% of weeks had $0 spend |
| Meta Other | $0.5M | Tiny |

**Meta is spending ~88% of the total advertising budget.** Google channels make up the remaining 12%.

### What the data explorer flagged as concerning

**1. Google Shopping is nearly inactive.**
76% of weeks had zero dollars spent on Google Shopping. This is a problem for modeling because you can't measure the effect of something that barely ran. The model will struggle to say anything reliable about Google Shopping.

**2. Unusual weeks that need explaining.**
On **30 May 2022**, something extraordinary happened: revenue hit $114M (the highest week in the dataset), Google Shopping spend spiked to 10× its normal level, and Meta Facebook also spiked. This happened all at once. It's almost certainly a major sales event or promotion. These "outlier" weeks can distort the model's understanding if not handled carefully — like trying to understand a person's typical diet by including one week where they ate at 10 restaurants for a special occasion.

Similar spikes happened around **holiday seasons** (November–December) and **May 2023**.

**3. Google Display and Google Video show almost no relationship with revenue.**
When a channel's spend goes up and revenue doesn't move with it, that's a signal the channel either doesn't work, or its effect is too delayed or too mixed with other things for the model to see it clearly.

**4. Meta Facebook shows the strongest raw relationship with revenue** — when Facebook spend goes up, revenue tends to go up with it. Google Search is second. Instagram is third.

**Readiness score: 4 out of 5** — good data, with a few known issues to watch.

---

## Part 2: How the Models Work

We used three different modeling approaches. Here's what each one does in plain English:

### Ridge — "The Disciplined Estimator"
Looks at all channels simultaneously and tries to find how much each one contributes to revenue. It's "disciplined" because it automatically reduces extreme answers — if a channel seems to explain 500% of revenue, it pulls that number back toward something more realistic. Fast and transparent.

### PyMC Bayesian — "The Careful Statistician"
Does the same thing but starts with some prior beliefs ("we expect ad spend to have a positive effect") and then updates those beliefs with the data. Like a scientist who says "I don't think this channel is magic, but let me check the data and adjust." This approach also gives you a range of uncertainty, not just a single number. It's slower but gives richer outputs.

### NNLS — "The No-Negatives Rule"
Forces all channel effects to be positive — it refuses to say "this channel caused us to lose money." The idea is that companies don't usually run advertising that actively destroys sales; if a model returns a negative number, it's probably a data confusion problem, not reality. This model also enforces a "baseline" — the revenue that would happen even without any advertising (organic sales, word of mouth, repeat customers).

---

### Two concepts the models use

**Adstock (carryover effect)**
When you see a Facebook ad today, you might not buy until next week. Or you might see the ad, forget about it, and then remember the brand when you're ready to shop two weeks later. "Adstock" captures this — it means today's ad spend has a lingering effect on future weeks' revenue.

Different channels carry over for different amounts of time:

| Channel | How long the effect lingers | Why |
|---|---|---|
| Google Search | Days | People search when they're ready to buy — effect is immediate |
| Meta Facebook / Instagram | 2–3 weeks | Social ads build awareness that lingers |
| Google Video | 3–4 weeks | Brand-building video has longer recall |
| Google Display | 2–3 weeks | Awareness channel, medium carryover |

In Round 4, we gave the model these real-world benchmarks from a knowledge layer instead of using one average for all channels. This made a significant difference — more on that below.

**Hill Saturation (diminishing returns)**
If you spend $1M on Facebook and get 1,000 customers, spending $2M won't get you 2,000 customers. At some point, you've reached most of the people who would buy, and extra spend produces less and less return. This is called "diminishing returns," and the Hill curve captures it mathematically. We tuned this in Round 3 and it significantly improved the model's accuracy.

---

## Part 3: Four Rounds of Improvement

The agent ran four rounds, each time trying one improvement and measuring whether it helped.

### How we measure accuracy

We held back the last 4 weeks of data — the model never saw these weeks during training. After fitting the model on the other 128 weeks, we asked it to predict those 4 held-back weeks. The gap between its predictions and reality is the **MAPE (Mean Absolute Percentage Error)**. Lower is better.

Think of it like: if the model predicts $30M revenue in a week and actual revenue was $26M, the error is 13.3%. A MAPE of 13% means the model is on average 13% off on weeks it has never seen before.

### Round 1 — Starting point
Default settings. Best accuracy: **23.2% error**. A reasonable starting point but too uncertain for confident budget decisions.

### Round 2 — Fixed the timing assumption
The model was originally assuming ad effects linger for 2 months. The agent noticed that for digital channels, this was too long — digital advertising effects are faster. It shortened this to 1 month. Result: **error dropped to 20.4%**.

### Round 3 — Fixed the saturation curve
The model was assuming channels needed to spend a lot before seeing meaningful returns. The agent adjusted the saturation threshold to be more sensitive to lower spending levels. This was the biggest improvement: **error dropped to 13.1%**. At this level, the model is predicting within 13% of actual revenue on weeks it had never seen — directionally trustworthy.

### Round 4 — Added domain knowledge about channel behavior
Instead of treating all channels the same way, we told the model what we know about how different types of advertising actually work in the real world. Search ads fade in days. Video ads linger for weeks. Social ads are in between.

The effect was striking. **Meta Facebook went from "high disagreement between models" to "high agreement" for the first time** — the two models that compute channel-level results went from disagreeing by 71% on Meta Facebook's ROI to disagreeing by only 7.9%. Domain knowledge the model could not discover from data alone made the biggest difference in reliability.

---

## Part 4: What We Found

### The most reliable finding: Meta Facebook works

Both independent models agreed that Meta Facebook generates approximately **$1.60 in revenue for every $1 spent**. That's a 60% return on ad spend. More importantly, the two models only disagreed by **7.9%** on this number — the closest agreement achieved across all channels across all four rounds. This is not noise. This is a consistent signal.

Meta Facebook received **$624M in total spend** over the period and appears to account for roughly **36–42% of all media-driven revenue**.

### Meta Instagram appears even more efficient per dollar

Instagram ROI is approximately **$1.99–$2.67 per dollar spent** (depending on the model). That's higher than Facebook. The two models disagree more (34.6% gap), so this is a directional signal, not a confirmed fact.

What it suggests: **Instagram may be getting relatively less investment than its effectiveness warrants.** Facebook received 74% of the Meta budget; Instagram got 26%. If Instagram truly delivers higher return per dollar, the current allocation may not be optimal.

### The Google channels are unclear — do not act on them yet

Google Search, Display, Video, and PMax all show contradictory results across models. One model says Google Search delivers $40 of revenue per dollar; the other says $0. That's not a real finding — it's a sign the data is too tangled to give a clean answer.

Possible reasons:
- Google and Meta spending often move together. When one goes up, so does the other. The model finds it hard to separate which one caused the sales.
- Some Google channels ran inconsistently (like Shopping, which was off 76% of weeks)
- The Bayesian model (PyMC) wasn't able to run its full version this round, limiting the three-way cross-check

**The right call is to not make budget decisions on Google channels from this analysis alone.**

### The organic baseline

Not all revenue comes from advertising. People who already know the brand buy again. Word of mouth brings new customers. Seasonal demand rises regardless of ads. The model estimates this "would have happened anyway" revenue is approximately **33% of total revenue**. The remaining ~67% is attributed to media channels.

---

## Part 5: What This Means for Budget Decisions

### What you can act on today
- **Meta Facebook** is confirmed by two independent models as delivering positive ROI (~1.6×). Maintaining current investment is supported by the data.
- **Meta Instagram** looks more efficient per dollar than Facebook. Consider testing a modest reallocation — even shifting 5–10% of Facebook budget to Instagram is low-risk given the directional signal.

### What you should not act on yet
Do not cut or grow Google channels based on this analysis. The models disagree too much. The right path for Google channels is an **incrementality experiment** — running ads in some regions and not others to directly measure the effect, without relying on modeling assumptions.

### What would make this more reliable
- A fifth round with the full Bayesian model enabled would give three-model confirmation on Meta channels
- Adding a "was this week a promotion?" flag to the data would help the model separate promotional spikes from normal advertising effects
- Geo-experiments for Google channels would give direct causal evidence

---

## The Key Limitation

This analysis used **132 weeks of historical data from a public ecommerce dataset** — not a real company's data. The results are directionally illustrative. They show what this kind of analysis can find and how the agent system works, but they are not a basis for a specific company's budget decisions.

For a real budget decision, you would need:
- Your own company's data
- At least 2–3 years of weekly revenue and spend data
- Business context (promotions, price changes, competitor activity) to explain unusual weeks
- Ideally some incrementality experiments to calibrate the model

The finding most likely to transfer to real businesses: **not all advertising channels are equally efficient, models consistently disagree more on Google channels than Meta channels, and Instagram tends to appear more efficient per dollar than Facebook despite receiving less budget.** These are patterns worth investigating with your own data.

---

## Summary Table

| Question | Answer |
|---|---|
| What did we analyse? | 132 weeks of ecommerce ad spend and revenue, 8 channels |
| How many models? | 3 independent models cross-checking each other |
| How many rounds of tuning? | 4 rounds, one improvement per round |
| Best model accuracy? | 13.1% error on held-out weeks |
| Most reliable channel finding? | Meta Facebook — confirmed by 2 models, 7.9% disagreement |
| Most efficient channel per dollar? | Meta Instagram (directional, not confirmed) |
| Organic baseline (no-ads revenue)? | ~33% of total revenue |
| Google channels? | Too noisy — do not act on results yet |
| What's next? | Run Round 5 with full Bayesian model + geo-experiments for Google |

---

*Dataset: [Multi-Region MMM Dataset for eCommerce Brands](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841/3?file=46779652), Figshare, CC BY 4.0. Results are illustrative and not from a real brand.*

*Generated by PrismMMM — [github.com/ScarlettQiu/prismmmm](https://github.com/ScarlettQiu/prismmmm)*
