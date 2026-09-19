"""
Feature engineering.

The features split into two families, and the distinction is the single most
important correctness point in this project:

  1. **Deterministic features** (`add_basic_features`) depend only on the row
     itself -- season from month, recency from year. These can be computed on
     the full dataset safely.

  2. **Target-derived aggregates** (`TargetAggregates`) summarise the rating
     column: average rating per user, per attraction, per country, and so on.
     These are enormously predictive, and computing them over the full dataset
     leaks the test-set answer into the training features. A model built that
     way reports an inflated R2 that collapses in production.

     `TargetAggregates` is therefore a fit/transform estimator: it is fitted on
     the training fold only, then applied to train and test alike. Unseen keys
     fall back to the global mean, and every group mean is shrunk toward the
     global mean in proportion to how little data supports it
     (empirical-Bayes smoothing), so an attraction with a single 5-star rating
     does not get a feature value of 5.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

# Smoothing strength: a group needs ~this many observations before its own
# mean outweighs the global prior.
SMOOTHING = 10.0


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-local features. No target information is used, so this is leak-free."""
    df = df.copy()

    df["Season"] = df["VisitMonth"].map(SEASON_BY_MONTH).fillna("Unknown")

    # Months elapsed since the first visit in the dataset -- a recency proxy
    # that lets the model pick up drift in ratings over time.
    base_year = int(df["VisitYear"].min())
    df["MonthsSinceFirstVisit"] = (df["VisitYear"] - base_year) * 12 + df["VisitMonth"]

    return df


class TargetAggregates:
    """
    Fit/transform estimator for smoothed, target-derived group statistics.

    Usage:
        agg = TargetAggregates().fit(train_df)
        train_df = agg.transform(train_df)
        test_df  = agg.transform(test_df)     # uses train-fold statistics only
    """

    # (feature_prefix, grouping key)
    GROUPS = [
        ("User", "UserId"),
        ("Attraction", "AttractionId"),
        ("Country", "Country"),
        ("AttractionType", "AttractionType"),
    ]

    def __init__(self, smoothing: float = SMOOTHING, target: str = "Rating"):
        self.smoothing = smoothing
        self.target = target
        self.global_mean_: float | None = None
        self.global_std_: float | None = None
        self.stats_: dict[str, pd.DataFrame] = {}
        self.affinity_: pd.Series | None = None
        self.fitted_ = False

    def fit(self, df: pd.DataFrame) -> "TargetAggregates":
        y = df[self.target].astype(float)
        self.global_mean_ = float(y.mean())
        self.global_std_ = float(y.std(ddof=0))

        for prefix, key in self.GROUPS:
            g = df.groupby(key, observed=True)[self.target].agg(["count", "mean", "std"])
            # Empirical-Bayes shrinkage toward the global mean.
            w = g["count"] / (g["count"] + self.smoothing)
            g[f"{prefix}AvgRating"] = w * g["mean"] + (1 - w) * self.global_mean_
            g[f"{prefix}VisitCount"] = g["count"]
            g[f"{prefix}RatingStd"] = g["std"].fillna(self.global_std_)
            self.stats_[prefix] = g[
                [f"{prefix}AvgRating", f"{prefix}VisitCount", f"{prefix}RatingStd"]
            ]

        # How much this user likes this attraction type, relative to the global
        # mean. Captures "beach person" vs "museum person". Counts and raw means
        # are retained so the training fold can be leave-one-out corrected --
        # these groups are small, so a row's own rating dominates its own
        # affinity value if left uncorrected.
        ut = df.groupby(["UserId", "AttractionType"], observed=True)[self.target].agg(["count", "mean"])
        ut.columns = ["AffinityCount", "AffinityMean"]
        self.affinity_ = ut

        self.fitted_ = True
        return self

    def _shrink(self, count, mean):
        w = count / (count + self.smoothing)
        return w * mean + (1 - w) * self.global_mean_ - self.global_mean_

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("TargetAggregates.transform called before fit().")
        out = df.copy()

        for prefix, key in self.GROUPS:
            stats = self.stats_[prefix]
            joined = out[[key]].merge(
                stats, left_on=key, right_index=True, how="left"
            )
            for col in stats.columns:
                values = joined[col].to_numpy()
                # Unseen key at predict time -> neutral prior.
                if col.endswith("AvgRating"):
                    fill = self.global_mean_
                elif col.endswith("RatingStd"):
                    fill = self.global_std_
                else:
                    fill = 0.0
                out[col] = np.where(pd.isna(values), fill, values)

        aff = out[["UserId", "AttractionType"]].merge(
            self.affinity_, left_on=["UserId", "AttractionType"],
            right_index=True, how="left",
        )
        count = aff["AffinityCount"].fillna(0.0).to_numpy()
        mean = aff["AffinityMean"].fillna(self.global_mean_).to_numpy()
        out["UserAttractionTypeAffinity"] = self._shrink(count, mean)
        out["_AffinityCount"] = count
        out["_AffinityMean"] = mean

        return out

    def transform_loo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the training fold with leave-one-out correction.

        Without it, `UserAvgRating` for a training row contains that row's own
        rating, which the model can partially invert -- especially for users
        with few visits. This removes the row's own contribution from the
        user- and attraction-level means, and from the user/attraction-type
        affinity, whose groups are small enough that a single row's own rating
        would otherwise dominate its own feature value.
        """
        out = self.transform(df)
        y = df[self.target].astype(float).to_numpy()

        # --- affinity: small groups, so this is the most severe leak ---------
        a_count = out["_AffinityCount"].to_numpy().astype(float)
        a_mean = out["_AffinityMean"].to_numpy().astype(float)
        a_n_loo = np.maximum(a_count - 1, 0)
        a_mean_loo = np.where(
            a_n_loo > 0, (a_mean * a_count - y) / np.maximum(a_n_loo, 1e-9), self.global_mean_
        )
        out["UserAttractionTypeAffinity"] = self._shrink(a_n_loo, a_mean_loo)

        for prefix, key in (("User", "UserId"), ("Attraction", "AttractionId")):
            counts = out[f"{prefix}VisitCount"].to_numpy().astype(float)
            means = out[f"{prefix}AvgRating"].to_numpy().astype(float)
            # Undo shrinkage -> raw sum -> remove self -> re-shrink.
            w = counts / (counts + self.smoothing)
            raw_mean = np.where(w > 0, (means - (1 - w) * self.global_mean_) / np.maximum(w, 1e-9),
                                self.global_mean_)
            raw_sum = raw_mean * counts
            n_loo = np.maximum(counts - 1, 0)
            loo_mean = np.where(n_loo > 0, (raw_sum - y) / np.maximum(n_loo, 1e-9),
                                self.global_mean_)
            w_loo = n_loo / (n_loo + self.smoothing)
            out[f"{prefix}AvgRating"] = w_loo * loo_mean + (1 - w_loo) * self.global_mean_
            out[f"{prefix}VisitCount"] = n_loo

        return out


def feature_columns(include_rating: bool = False) -> tuple[list[str], list[str]]:
    """Return (categorical, numeric) feature name lists for a model pipeline."""
    num = list(NUMERIC_FEATURES)
    if include_rating:
        num = num + ["Rating"]
    return list(CATEGORICAL_FEATURES), num


def prepare_model_frame(df: pd.DataFrame, include_rating: bool = False) -> pd.DataFrame:
    """Select and order the model feature columns, tolerating absent ones."""
    cat, num = feature_columns(include_rating)
    cols = [c for c in cat + num if c in df.columns]
    missing = set(cat + num) - set(cols)
    if missing:
        raise KeyError(
            f"Feature columns missing from frame: {sorted(missing)}. "
            "Did you run add_basic_features() and TargetAggregates.transform()?"
        )
    return df[cols]
