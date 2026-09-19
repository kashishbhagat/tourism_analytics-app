# Tourism Experience Analytics — Final Report

**Domain:** Tourism · **Tasks:** Regression, Classification, Recommendation

---

## 1. Purpose

Tourism platforms hold visit histories they mostly use for reporting. This
project turns that history into three decisions:

1. **Will this visitor enjoy this attraction?** Catch a poor match before the
   trip rather than reading about it in a review afterwards.
2. **How is this visitor travelling?** Target packages and plan on-site
   resources by travel party.
3. **What should they see next?** Rank attractions to lift engagement and
   retention.

---

## 2. Data and preparation

Nine linked tables: one fact table of visits, eight dimensions covering visitor
geography (continent → region → country → city) and attraction attributes
(type, city, address).

### Cleaning

Every transformation is logged to `reports/cleaning_log.md` by the code itself,
so the audit trail cannot drift from what actually ran. The order is deliberate:

1. Normalise column names and dtypes.
2. Standardise categorical text — **before** deduplication, so that `' FAMILY '`
   and `Family` collapse to one value first. Deduplicating first would leave
   both variants in place.
3. Drop exact and primary-key duplicates.
4. Repair invalid values: ratings outside 1–5 and months outside 1–12 are set to
   null rather than clipped, because a rating of 9 is a data error, not a strong
   opinion to be squashed to 5.
5. Enforce referential integrity — drop visits whose visitor or attraction has
   no matching dimension record.
6. Impute what remains.
7. Join into one master table.

**Targets are dropped, not imputed.** Rows missing a rating or a visit mode are
removed rather than filled. Imputing a target teaches the model the imputation
rule instead of visitor behaviour. Feature columns (month, year) are imputed
with the modal value, which preserves seasonality better than a constant and
keeps the row usable.

**Joins preserve the fact grain.** All joins are LEFT from the visit table, and
the row count is asserted afterwards. A duplicate key in any dimension would
silently multiply rows and inflate every downstream count; the assertion catches
it instead. Unmatched lookups are labelled `Unknown` rather than dropped, so a
missing city does not cost an otherwise complete visit record.

On the sample data this removes about 3.3% of rows across 16 logged steps.

---

## 3. Exploratory analysis

Seven figures, each paired with a computed finding — the narrative is generated
from the data, so it stays true when the dataset is swapped. Full report:
`reports/eda_report.md`.

The findings that shaped the modelling:

**Ratings are strongly left-skewed** — around 82% of visits score 4 or 5, mean
4.21. Satisfaction data is an optimistic, censored signal. This is why every
model is reported against a baseline: an R² that looks modest against a 1–5
range is substantial against the actual variance available.

**Visit mode genuinely drives satisfaction.** Couples rate highest (4.37),
Business lowest (3.88) — a 0.49-point gap. Not noise.

**But geography barely determines visit mode.** Chi-square is significant, yet
Cramér's V is about 0.05. This sets a realistic ceiling on the classifier, and
is the reason its modest score is a property of the problem rather than a
failure of tuning.

**Demand is concentrated.** The top three continents account for 74% of visits;
the top 20% of attractions capture a disproportionate share of traffic. A
recommender that only surfaces popular items reinforces this concentration,
which is why the hybrid blends in a content signal to keep the tail
discoverable.

**The user-item matrix is over 99% sparse.** This is the core constraint on
collaborative filtering and the reason a content-based cold-start path is
required rather than optional.

---

## 4. Modelling

### 4.1 Regression — predicting ratings

Six algorithms compared on an identical split. **Ridge Regression** won with
test R² 0.384 and RMSE 0.630, against a mean-prediction baseline of R² −0.000.
Five-fold cross-validation gives 0.393 ± 0.005, so the result is stable.

Top features by permutation importance on held-out data: the attraction's
historical mean rating, then the visitor's own mean rating, then season.
Attraction identity dominates — *what* you visit matters more than *who* you are.

The leakage control described in the README is what makes these numbers
trustworthy. An earlier version reported CV 0.57 against test 0.28; the gap was
the tell, and fixing it raised the honest score.

### 4.2 Classification — predicting visit mode

Five algorithms compared. **Random Forest** won on macro F1 (0.245) against a
majority-class baseline of 0.091 — a 2.7× lift, but low in absolute terms.

Selection is on macro F1, not accuracy, because the classes are imbalanced and
accuracy rewards ignoring the rare, commercially valuable segments. Models that
support it use balanced class weights.

Per-class results show the model is most reliable on Business travellers
(recall ≈ 0.53) and weakest on Solo (≈ 0.10) — the smallest class.

**How to use it:** rank campaign audiences by predicted probability, not decide
about individuals. The app surfaces the full probability distribution and warns
when the top two classes are within 10 percentage points.

### 4.3 Recommendation

Five methods compared under a temporal leave-last-out protocol:

| Method | Hit rate@10 | MAP@10 | NDCG@10 |
|---|---|---|---|
| **Hybrid** | 0.186 | 0.083 | **0.107** |
| Content-based | 0.192 | 0.056 | 0.087 |
| Popularity | 0.118 | 0.052 | 0.067 |
| Collaborative | 0.103 | 0.045 | 0.058 |
| SVD | 0.085 | 0.030 | 0.043 |

The hybrid wins on ranking quality by 59% over popularity. Content-based has a
marginally higher hit rate but ranks its hits worse — it finds the right
attraction but buries it further down the list, which is what NDCG and MAP
capture and raw hit rate does not.

---

## 5. Business recommendations

**Prioritise the fix-first quadrant.** Attraction types with above-median
traffic but below-average satisfaction are where service investment returns
most, because the audience already exists. The `attraction_type_quadrant` SQL
query classifies every type into Star / Fix first / Promote / Deprioritise.

**Promote the under-served quadrant.** Types that score above average on
satisfaction but draw below-median traffic are the cheapest growth available —
the experience already delivers, only the visibility is missing.

**Act on the four visitor segments.** The `user_segments` query splits visitors
by engagement and satisfaction:

| Segment | Action |
|---|---|
| Loyal advocates | Referral and review programmes — lowest-cost growth |
| Frequent but unsatisfied | Highest-value intervention; frequent visitors having a poor time |
| Occasional advocates | Re-engagement; they enjoy the product but visit rarely |
| At risk | Diagnose before spending on retention |

**Use rating prediction as a pre-emptive filter.** Where the predicted rating
falls materially below average, either reset expectations in the listing or
route the visitor to a better-fitting attraction. This converts a future
one-star review into a redirect.

**Plan staffing against the seasonal curve, not the annual average.** Monthly
demand swings substantially peak-to-trough.

**Bundle by cross-type affinity.** The `cross_type_affinity` query finds
attraction types visited by the same people, which drives "visitors who enjoyed
X also enjoyed Y" packaging without needing a model.

---

## 6. Limitations

**The classifier is weak in absolute terms.** Macro F1 of 0.25 supports audience
ranking, not individual decisions. The available features genuinely do not
determine travel party; booking channel, party size or lead time would likely
help more than further tuning.

**No true cold-start evaluation for attractions.** New attractions with no
ratings fall back to content similarity, which is sensible but untested here —
the held-out protocol only covers items already in the catalogue.

**Ratings are self-selected.** People who visit an attraction are already
predisposed toward it, and people who bother to rate are not a random sample of
visitors. Predicted ratings describe likely reviewers, not all visitors.

**Popularity feedback loops are not modelled.** Deploying a recommender changes
the behaviour it was trained on. The content blend mitigates concentration but
does not eliminate it; this needs monitoring after launch, not just before.

**All figures in this report come from the sample dataset**, since the source
data is distributed via a Google Drive link. The pipeline is unchanged for the
real files — re-run it to produce real figures.

---

## 7. Deliverables

| Deliverable | Location |
|---|---|
| Cleaned dataset | `data/processed/master_dataset.parquet` / `.csv` |
| Cleaning documentation | `reports/cleaning_log.md` (auto-generated, 16 steps) |
| EDA report and figures | `reports/eda_report.md`, `reports/figures/` |
| Source code | `src/`, `scripts/`, `run_pipeline.py` |
| SQL analysis | `sql/analysis.sql` (10 queries), `data/processed/tourism.db` |
| Trained models | `models/*.joblib` |
| All metrics | `models/metrics.json` |
| Application | `app/streamlit_app.py` |
| This report | `reports/final_report.md` |
