# Exploratory Data Analysis

Generated automatically by `src/eda.py`. Every figure below is paired with the finding it supports.

## Dataset overview

| Metric | Value |
|--------|-------|
| Transactions (cleaned) | 38,854 |
| Distinct users | 2,996 |
| Distinct attractions | 350 |
| Countries represented | 66 |
| Attraction types | 15 |
| Mean rating | 4.208 |
| Period covered | 2020-2024 |

## Key findings

1. Ratings are strongly left-skewed (skew = -0.73): 81.7% of all visits are rated 4 or 5, and the mean is 4.21. Satisfaction data is a censored, optimistic signal — models should be judged against a mean-prediction baseline, not against a naive 0-to-5 range.

2. **Couples** travellers are the most satisfied (mean 4.37) and **Business** the least (mean 3.88), a gap of 0.49 points. Visit mode is a real driver of satisfaction, not noise.

3. User demand is geographically concentrated: the top three continents (Asia, Europe, Africa) account for 74.0% of all visits, and **Norway** alone contributes 3.3%. Marketing spend and language support should follow this concentration rather than being spread evenly.

4. **Fix-first segment:** Ballets, Neighborhoods, Nature & Wildlife Areas draw above-median traffic but below-average satisfaction. These are the categories where service improvements return the most, because the audience already exists.

5. **Under-promoted segment:** Flea & Street Markets, Religious Sites, Ancient Ruins score above average on satisfaction but attract below-median traffic — strong candidates for promotion, since the experience already delivers.

6. The association between continent and visit mode is statistically significant but weak in magnitude (chi-square p = 1.26e-97, Cramer's V = 0.058). Geography alone does not determine how people travel, which sets a realistic ceiling on the visit-mode classifier and is why its accuracy should be read against the majority-class baseline.

7. Visit modes are imbalanced (Family is 3.0x more common than Solo), so the classifier is selected on macro F1 rather than accuracy — accuracy would reward ignoring the rare, commercially valuable segments.

8. Demand peaks in month 9 and bottoms out in month 5, a 1.12x swing. Staffing and dynamic pricing should be planned against this curve; the `Season` feature is included in both models for this reason.

9. Demand follows a long tail: the top 20% of attractions capture 53.4% of all visits. A recommender that only surfaces popular items will reinforce this concentration, which is why the hybrid model blends a content signal to keep the tail discoverable.

10. The user-item matrix is 97.2331% sparse with a median of 8 visits per user. This sparsity is the core constraint on collaborative filtering and the reason a content-based fallback is needed for cold-start users.

11. An attraction's historical mean rating correlates 0.55 with the rating an individual visit receives — by far the strongest single predictor, and confirmed as the top feature by permutation importance. Note this is computed here for exploration only; in the models it is recomputed on the training fold alone to avoid leaking the target.

## Figures

### Rating distribution and visit mode

![Rating distribution and visit mode](figures/01_rating_distribution.png)

### Geographic distribution of demand

![Geographic distribution of demand](figures/02_geographic_distribution.png)

### Attraction types: demand vs satisfaction

![Attraction types: demand vs satisfaction](figures/03_attraction_analysis.png)

### Visit mode composition

![Visit mode composition](figures/04_visit_mode_analysis.png)

### Seasonality and trend

![Seasonality and trend](figures/05_temporal_analysis.png)

### User and attraction behaviour

![User and attraction behaviour](figures/06_user_behaviour.png)

### Feature correlations

![Feature correlations](figures/07_correlations.png)
