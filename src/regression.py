"""
Task 1 -- Regression: predict the rating a user will give an attraction.

Several algorithms are trained and compared on an identical split so the
comparison is fair; the best by test R2 is persisted. A mean-prediction
baseline is included because R2 alone is easy to misread -- it establishes
what "no model at all" scores on this data.

Leakage control: the target-derived aggregates are fitted on the training
fold only, and the training fold uses leave-one-out versions of the user and
attraction means (see src/features.py).
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from ..config import (
    RANDOM_STATE,
    REG_MODEL_PATH,
    REG_TARGET,
    TEST_SIZE,
    RATING_MAX,
    RATING_MIN,
)
from ..features import TargetAggregates, feature_columns, prepare_model_frame
from .common import build_preprocessor, permutation_importance_df, safe_float

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False


def candidate_models(frame, categorical, numeric) -> dict[str, tuple[Pipeline, str]]:
    """Return {name: (pipeline, encoding_kind)} for every algorithm to compare."""
    models: dict[str, tuple] = {
        "Baseline (mean)": (DummyRegressor(strategy="mean"), "ordinal"),
        "Ridge Regression": (Ridge(alpha=1.0, random_state=RANDOM_STATE), "one_hot"),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=200, max_depth=14, min_samples_leaf=5,
                n_jobs=-1, random_state=RANDOM_STATE,
            ),
            "ordinal",
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=250, learning_rate=0.08, max_depth=4, random_state=RANDOM_STATE
            ),
            "ordinal",
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBRegressor(
                n_estimators=500, learning_rate=0.06, max_depth=6,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
                random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
            ),
            "ordinal",
        )
    if HAS_LGBM:
        models["LightGBM"] = (
            LGBMRegressor(
                n_estimators=600, learning_rate=0.05, num_leaves=48,
                subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1,
            ),
            "ordinal",
        )

    built = {}
    for name, (est, kind) in models.items():
        pre = build_preprocessor(categorical, numeric, kind=kind, frame=frame)
        built[name] = (Pipeline([("prep", pre), ("model", est)]), kind)
    return built


def train(master: pd.DataFrame, verbose: bool = True) -> dict:
    """Train, compare, evaluate and persist the rating regressor."""
    categorical, numeric = feature_columns(include_rating=False)

    # --- split BEFORE any target-derived feature is computed ---------------
    train_df, test_df = train_test_split(
        master, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    agg = TargetAggregates().fit(train_df)
    train_feat = agg.transform_loo(train_df)   # leave-one-out on the train fold
    test_feat = agg.transform(test_df)         # train-fold statistics only

    X_train = prepare_model_frame(train_feat, include_rating=False)
    y_train = train_feat[REG_TARGET].astype(float)
    X_test = prepare_model_frame(test_feat, include_rating=False)
    y_test = test_feat[REG_TARGET].astype(float)

    results = []
    fitted: dict[str, Pipeline] = {}

    for name, (pipe, _kind) in candidate_models(X_train, categorical, numeric).items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        pred = np.clip(pipe.predict(X_test), RATING_MIN, RATING_MAX)

        results.append(
            {
                "model": name,
                "r2": safe_float(r2_score(y_test, pred)),
                "rmse": safe_float(np.sqrt(mean_squared_error(y_test, pred))),
                "mae": safe_float(mean_absolute_error(y_test, pred)),
                "train_seconds": round(time.time() - t0, 2),
            }
        )
        fitted[name] = pipe
        if verbose:
            r = results[-1]
            print(f"    {name:22s} R2={r['r2']:+.4f}  RMSE={r['rmse']:.4f}  "
                  f"MAE={r['mae']:.4f}  ({r['train_seconds']}s)")

    comparison = pd.DataFrame(results).sort_values("r2", ascending=False).reset_index(drop=True)

    # Never let the trivial baseline "win" a tie -- pick the best real model.
    real = comparison[comparison["model"] != "Baseline (mean)"]
    best_name = real.iloc[0]["model"]
    best_model = fitted[best_name]

    # --- cross-validated stability check on the winner ---------------------
    cv = cross_val_score(
        best_model, X_train, y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2", n_jobs=1,
    )

    importance = permutation_importance_df(
        best_model, X_test, y_test, scoring="r2", n_repeats=3, random_state=RANDOM_STATE
    )

    artefact = {
        "pipeline": best_model,
        "aggregates": agg,
        "feature_columns": list(X_train.columns),
        "categorical": categorical,
        "numeric": numeric,
        "best_model_name": best_name,
    }
    joblib.dump(artefact, REG_MODEL_PATH, compress=3)

    best_row = comparison[comparison["model"] == best_name].iloc[0]
    summary = {
        "task": "regression",
        "target": REG_TARGET,
        "best_model": best_name,
        "test_r2": best_row["r2"],
        "test_rmse": best_row["rmse"],
        "test_mae": best_row["mae"],
        "cv_r2_mean": safe_float(cv.mean()),
        "cv_r2_std": safe_float(cv.std()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "comparison": comparison.to_dict(orient="records"),
        "top_features": importance.to_dict(orient="records"),
        "model_path": str(REG_MODEL_PATH),
    }

    if verbose:
        print(f"    -> best: {best_name} (test R2={best_row['r2']:.4f}, "
              f"5-fold CV R2={cv.mean():.4f} +/- {cv.std():.4f})")

    return summary


def load_model(path=REG_MODEL_PATH) -> dict:
    return joblib.load(path)


def predict(artefact: dict, rows: pd.DataFrame) -> np.ndarray:
    """Score raw master-schema rows with a persisted regression artefact."""
    from ..features import add_basic_features

    df = rows.copy()
    if "Season" not in df.columns:
        df = add_basic_features(df)
    df = artefact["aggregates"].transform(df)
    X = df[artefact["feature_columns"]]
    return np.clip(artefact["pipeline"].predict(X), RATING_MIN, RATING_MAX)
