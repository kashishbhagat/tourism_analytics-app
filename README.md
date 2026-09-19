# Tourism Experience Analytics

Classification, prediction and recommendation on tourism visit data.

Three models over one shared pipeline:

| Task | Question | Output |
|---|---|---|
| Regression | How much will this visitor enjoy this attraction? | Predicted rating, 1–5 |
| Classification | How is this visitor travelling? | Business / Couples / Family / Friends / Solo |
| Recommendation | What should this visitor see next? | Ranked list of attractions |

All three are served through a Streamlit application, alongside the exploratory
analysis and a SQL layer.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Get data into data/raw/
python scripts/generate_sample_data.py     # sample data, or drop the real files in

# 2. Clean, analyse, train
python run_pipeline.py

# 3. Launch the app
streamlit run app/streamlit_app.py
```

The pipeline takes around four minutes on the sample dataset.

### Using the real dataset

Put the nine source files (or one multi-sheet workbook) in `data/raw/` and run
`python run_pipeline.py`. Nothing else changes.

The loader resolves tables by fuzzy-matching filenames and sheet names against
the aliases in `src/config.py`, so `Transaction.xlsx`, `transaction data.csv`
and a `Tourism.xlsx` with nine sheets all work. If a table cannot be matched by
name, it falls back to matching on column signature; if it still cannot be
resolved, the error names exactly which table is missing and what files were
found, rather than failing later with a `KeyError`.

---

## Results

Measured on a held-out 20% of the sample dataset.

| Task | Best model | Score | Baseline |
|---|---|---|---|
| Rating prediction | Ridge Regression | R² 0.384, RMSE 0.630 | R² −0.000 (predict the mean) |
| Visit mode | Random Forest | macro F1 0.245 | 0.091 (majority class) |
| Recommendation | Hybrid | NDCG@10 0.107 | 0.067 (popularity) |

Six algorithms are compared for regression and five for classification; the
full comparison tables are written to `models/metrics.json` and shown in the app.

Three things worth knowing before reading these numbers:

**Ridge beating the gradient boosters is not a bug.** Ratings are close to
additive in the aggregate features (attraction quality + visitor generosity +
trip context), which is exactly what a linear model fits. The tree ensembles
overfit the high-cardinality ordinal encodings of `Attraction` and
`AttractionCity`. On the real dataset a tree model may well win — the
comparison runs either way and picks by score, not by assumption.

**The visit-mode classifier is weak in absolute terms.** It beats the
majority-class baseline by 2.7×, but macro F1 of 0.25 is not a model to make
individual decisions with. The EDA explains why: the association between
visitor origin and visit mode is statistically significant but weak
(Cramér's V ≈ 0.05). Geography does not determine who you travel with. The app
flags this on the scorecard rather than presenting the number bare, and warns
when the top two classes are within 10 percentage points.

**Accuracy is the wrong metric for the classifier**, so it is not the selection
metric. Visit modes are imbalanced, and a model that never predicts Business or
Solo can score well on accuracy while being useless for exactly the segments
worth targeting. The winner is selected on macro F1, which weights every class
equally. Accuracy is reported alongside for reference.

---

## How it fits together

```
data/raw/                9 source tables
      │
      ▼
src/data_loader.py       resolve files → 9 DataFrames
      │
      ▼
src/cleaning.py          dedupe, repair, standardise, integrate
      │                  → data/processed/master_dataset.parquet
      │                  → reports/cleaning_log.md
      ├──────────────────────────────┬────────────────────────┐
      ▼                              ▼                        ▼
src/eda.py                    src/database.py          src/features.py
7 figures + findings          SQLite + 10 queries      leak-free features
      │                              │                        │
      │                              │      ┌─────────────────┼─────────────────┐
      │                              │      ▼                 ▼                 ▼
      │                              │  regression.py  classification.py  recommender.py
      │                              │      │                 │                 │
      └──────────────────────────────┴──────┴────────┬────────┴─────────────────┘
                                                     ▼
                                          app/streamlit_app.py
```

### Layout

```
tourism_analytics/
├── run_pipeline.py              orchestrator; --stages to run part of it
├── requirements.txt
├── src/
│   ├── config.py                paths, schema, feature lists, constants
│   ├── data_loader.py           tolerant raw-file resolution
│   ├── cleaning.py              cleaning + integration, with an audit log
│   ├── features.py              feature engineering and leakage control
│   ├── eda.py                   figures paired with computed findings
│   ├── database.py              SQLite builder and named-query runner
│   └── models/
│       ├── common.py            shared preprocessing and metric helpers
│       ├── regression.py        rating prediction
│       ├── classification.py    visit mode prediction
│       └── recommender.py       collaborative + content + hybrid
├── app/streamlit_app.py         5-page application
├── scripts/generate_sample_data.py
├── sql/analysis.sql             10 documented analytical queries
├── data/{raw,processed}/
├── models/                      trained models + metrics.json
└── reports/                     cleaning log, EDA report, figures
```

---

## Preventing target leakage

This is the part of the project most worth reading, because getting it wrong
inflates the reported scores while making the model worse.

Features such as `AttractionAvgRating` and `UserAvgRating` are by far the
strongest predictors — an attraction's historical mean correlates around 0.6
with the rating an individual visit receives. Computing them over the whole
dataset leaks held-out answers into the training features.

`TargetAggregates` in `src/features.py` is therefore a fit/transform estimator:

- **Fitted on the training fold only.** The split happens before any
  target-derived feature is computed.
- **Smoothed toward the global mean** in proportion to how little data supports
  each group, so an attraction with one 5-star rating does not get a feature
  value of 5.0.
- **Leave-one-out corrected on the training fold**, removing each row's own
  rating from its own user, attraction and affinity aggregates.
- **Falls back to global priors** for users and attractions unseen at predict
  time, which is also what makes cold-start prediction work in the app.

The leave-one-out correction on `UserAttractionTypeAffinity` mattered most,
because those groups are small enough for a single rating to dominate its own
feature. Before the fix, cross-validated R² was 0.57 against a test R² of 0.28 —
the gap was the symptom. After it, CV (0.393) and test (0.384) agree, and the
honest score is *higher* than the leaky one, because the training features now
match the distribution the model sees at test time.

---

## Recommendation approach

Three methods, compared on the same held-out data:

- **Collaborative filtering** — item-item cosine similarity over the
  mean-centred user-item matrix, plus a truncated-SVD latent factor model.
  Centring removes each visitor's personal generosity so similarity reflects
  agreement on relative preference, not shared optimism. Scores are shrunk by
  how much similarity actually supports them; without this, thinly-rated items
  dominate the top-N.
- **Content-based filtering** — cosine similarity over attraction attributes.
  Handles cold-start users, where collaborative filtering has nothing to work with.
- **Hybrid** — a weighted blend of both plus a popularity prior. This is the
  default served by the app.

**Evaluation uses a temporal split**: each visitor's most recent visit is held
out, and the model ranks everything they had not already seen. A random split
would let the model use a visitor's future visits to predict their past ones,
which flatters the score and is not how the system runs in production.

Reported metrics are Hit Rate@10, Precision@10, Recall@10, MAP@10 and NDCG@10,
plus RMSE for the SVD model's rating reconstruction. For a top-N system the
ranking metrics matter far more than RMSE — the system is judged on what it puts
in front of the visitor, not on how well it reconstructs ratings for items the
visitor will never be shown.

---

## A note on the sample data

`scripts/generate_sample_data.py` exists because the dataset is distributed via
a Google Drive link. It produces all nine tables with the documented column
names and dtypes, and deliberately injects the defects a real export has:
duplicate rows, ratings outside 1–5, impossible months, orphan foreign keys,
inconsistent city casing and whitespace, and missing values. The cleaning stage
removes about 3.3% of rows and logs every action.

Ratings are generated with real structure — latent attraction quality, visitor
generosity bias, a visit-mode effect and per-visitor type preferences — so the
models have genuine signal to find rather than noise to overfit.

One detail worth flagging: the generator makes 55% of visits preference-driven
rather than drawing every visit from a global popularity distribution. In an
earlier version, item choice was independent of the visitor, which meant there
was no personal signal in *which* attraction someone visited, and a popularity
ranking was by construction the optimal recommender. That is not how real
tourism data behaves, and it made the personalised methods look worthless. With
taste-driven selection, the hybrid model beats popularity by 59% on NDCG@10.

**Numbers produced from the sample data are not claims about the real
dataset.** Re-run the pipeline on the real files to get real figures.

---

## Reproducibility

Every random operation is seeded from `RANDOM_STATE` in `src/config.py`.
Re-running the pipeline on the same input reproduces the same models and metrics.

The SQL layer includes a `data_quality_check` query whose counts should all be
zero; the pipeline runs it after training and reports PASS or FAIL, so a
regression in the cleaning stage surfaces immediately.
