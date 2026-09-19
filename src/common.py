"""
Shared plumbing for the supervised models.

Two preprocessing strategies are provided because linear and tree models want
different encodings of the same columns:

  * `one_hot`  -- for linear/SVM-style models, which need orthogonal dummies
                  and scaled numerics.
  * `ordinal`  -- for tree ensembles, which split on ordinal codes perfectly
                  well and would otherwise blow up to thousands of sparse
                  columns on high-cardinality fields like `Attraction`.

Both set `handle_unknown` so that a category seen only at predict time (a new
attraction, a new country) degrades gracefully instead of raising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_preprocessor(
    categorical: list[str],
    numeric: list[str],
    kind: str = "ordinal",
    max_onehot_cardinality: int = 60,
    frame: pd.DataFrame | None = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for the given feature lists.

    When `kind="one_hot"` and a `frame` is supplied, categorical columns with
    more distinct values than `max_onehot_cardinality` are ordinal-encoded
    instead, to keep the design matrix tractable.
    """
    if kind not in ("one_hot", "ordinal"):
        raise ValueError(f"kind must be 'one_hot' or 'ordinal', got {kind!r}")

    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    if kind == "ordinal":
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        return ColumnTransformer(
            [("cat", cat_pipe, categorical), ("num", num_pipe, numeric)],
            remainder="drop",
        )

    # one-hot, splitting out high-cardinality columns
    low, high = list(categorical), []
    if frame is not None:
        low, high = [], []
        for c in categorical:
            (high if frame[c].nunique(dropna=True) > max_onehot_cardinality else low).append(c)

    onehot_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)),
        ]
    )
    transformers = [("num", num_pipe, numeric)]
    if low:
        transformers.append(("cat_low", onehot_pipe, low))
    if high:
        high_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("cat_high", high_pipe, high))

    return ColumnTransformer(transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Best-effort readable feature names for importance plots."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(preprocessor.transform_output_shape_[1])]


def permutation_importance_df(
    model, X: pd.DataFrame, y, scoring: str, n_repeats: int = 5, random_state: int = 42, top: int = 20
) -> pd.DataFrame:
    """
    Permutation importance on the *original* columns.

    Computed on the held-out set and on raw column names, so the result is
    interpretable to a stakeholder ("attraction identity matters most")
    rather than referring to encoded dummy indices.
    """
    from sklearn.inspection import permutation_importance

    r = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=1,
    )
    return (
        pd.DataFrame(
            {"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std}
        )
        .sort_values("importance", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def safe_float(x) -> float:
    """Convert numpy scalars to plain floats so metrics serialise to JSON."""
    return float(np.asarray(x).item()) if np.ndim(x) == 0 else float(np.mean(x))
