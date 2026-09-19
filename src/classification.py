"""
Task 2 -- Classification: predict a user's mode of visit
(Business / Couples / Family / Friends / Solo).

Two points of care:

*Metric choice.* Visit modes are imbalanced -- Family and Couples dominate,
Business and Solo are rare. Plain accuracy rewards a model that never predicts
the rare classes, which is exactly the opposite of what the business use case
needs (targeting a Business traveller is the high-value case). The winner is
therefore selected on **macro F1**, which weights every class equally, and
accuracy is reported alongside for reference.

*Class weighting.* Models that support `class_weight="balanced"` use it, so the
rare classes contribute proportionally to the loss.

`Rating` is available as a feature here: visit mode is inferred retrospectively
for segmentation, at which point the rating is already known.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from ..config import CLF_MODEL_PATH, CLF_TARGET, RANDOM_STATE, TEST_SIZE
from ..features import TargetAggregates, feature_columns, prepare_model_frame
from .common import build_preprocessor, permutation_importance_df, safe_float

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False


def candidate_models(frame, categorical, numeric, n_classes: int) -> dict[str, Pipeline]:
    models: dict[str, tuple] = {
        "Baseline (most frequent)": (
            DummyClassifier(strategy="most_frequent"), "ordinal"
        ),
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "one_hot",
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300, max_depth=16, min_samples_leaf=20,
                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
            ),
            "ordinal",
        ),
    }
    if HAS_LGBM:
        models["LightGBM"] = (
            LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=48,
                subsample=0.85, colsample_bytree=0.85, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            ),
            "ordinal",
        )
    if HAS_XGB:
        models["XGBoost"] = (
            XGBClassifier(
                n_estimators=400, learning_rate=0.06, max_depth=6,
                subsample=0.85, colsample_bytree=0.85,
                objective="multi:softprob", num_class=n_classes,
                random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
                eval_metric="mlogloss",
            ),
            "ordinal",
        )

    built = {}
    for name, (est, kind) in models.items():
        pre = build_preprocessor(categorical, numeric, kind=kind, frame=frame)
        built[name] = Pipeline([("prep", pre), ("model", est)])
    return built


def train(master: pd.DataFrame, verbose: bool = True) -> dict:
    categorical, numeric = feature_columns(include_rating=True)

    # Stratified split preserves the class mix in both folds.
    train_df, test_df = train_test_split(
        master, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=master[CLF_TARGET],
    )

    agg = TargetAggregates().fit(train_df)
    train_feat = agg.transform_loo(train_df)
    test_feat = agg.transform(test_df)

    X_train = prepare_model_frame(train_feat, include_rating=True)
    X_test = prepare_model_frame(test_feat, include_rating=True)

    le = LabelEncoder().fit(master[CLF_TARGET].astype(str))
    y_train = le.transform(train_feat[CLF_TARGET].astype(str))
    y_test = le.transform(test_feat[CLF_TARGET].astype(str))

    results, fitted = [], {}
    for name, pipe in candidate_models(X_train, categorical, numeric, len(le.classes_)).items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append(
            {
                "model": name,
                "accuracy": safe_float(accuracy_score(y_test, pred)),
                "f1_macro": safe_float(f1_score(y_test, pred, average="macro", zero_division=0)),
                "f1_weighted": safe_float(f1_score(y_test, pred, average="weighted", zero_division=0)),
                "precision_macro": safe_float(
                    precision_score(y_test, pred, average="macro", zero_division=0)
                ),
                "recall_macro": safe_float(
                    recall_score(y_test, pred, average="macro", zero_division=0)
                ),
                "train_seconds": round(time.time() - t0, 2),
            }
        )
        fitted[name] = pipe
        if verbose:
            r = results[-1]
            print(f"    {name:26s} acc={r['accuracy']:.4f}  F1(macro)={r['f1_macro']:.4f}  "
                  f"({r['train_seconds']}s)")

    comparison = (
        pd.DataFrame(results).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    )
    real = comparison[~comparison["model"].str.startswith("Baseline")]
    best_name = real.iloc[0]["model"]
    best_model = fitted[best_name]

    pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(
        y_test, pred, target_names=le.classes_, output_dict=True, zero_division=0
    )

    cv = cross_val_score(
        best_model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro", n_jobs=1,
    )

    importance = permutation_importance_df(
        best_model, X_test, y_test, scoring="f1_macro", n_repeats=3, random_state=RANDOM_STATE
    )

    joblib.dump(
        {
            "pipeline": best_model,
            "aggregates": agg,
            "label_encoder": le,
            "feature_columns": list(X_train.columns),
            "categorical": categorical,
            "numeric": numeric,
            "best_model_name": best_name,
        },
        CLF_MODEL_PATH,
        compress=3,
    )

    best_row = comparison[comparison["model"] == best_name].iloc[0]
    if verbose:
        print(f"    -> best: {best_name} (macro F1={best_row['f1_macro']:.4f}, "
              f"CV={cv.mean():.4f} +/- {cv.std():.4f})")

    return {
        "task": "classification",
        "target": CLF_TARGET,
        "best_model": best_name,
        "classes": le.classes_.tolist(),
        "test_accuracy": best_row["accuracy"],
        "test_f1_macro": best_row["f1_macro"],
        "test_f1_weighted": best_row["f1_weighted"],
        "cv_f1_macro_mean": safe_float(cv.mean()),
        "cv_f1_macro_std": safe_float(cv.std()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_distribution": (
            master[CLF_TARGET].value_counts(normalize=True).round(4).to_dict()
        ),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            c: {
                "precision": round(report[c]["precision"], 4),
                "recall": round(report[c]["recall"], 4),
                "f1": round(report[c]["f1-score"], 4),
                "support": int(report[c]["support"]),
            }
            for c in le.classes_
        },
        "comparison": comparison.to_dict(orient="records"),
        "top_features": importance.to_dict(orient="records"),
        "model_path": str(CLF_MODEL_PATH),
    }


def load_model(path=CLF_MODEL_PATH) -> dict:
    return joblib.load(path)


def predict(artefact: dict, rows: pd.DataFrame, return_proba: bool = False):
    """Predict visit mode labels (and optionally the full probability table)."""
    from ..features import add_basic_features

    df = rows.copy()
    if "Season" not in df.columns:
        df = add_basic_features(df)
    df = artefact["aggregates"].transform(df)
    X = df[artefact["feature_columns"]]

    codes = artefact["pipeline"].predict(X)
    labels = artefact["label_encoder"].inverse_transform(codes)
    if not return_proba:
        return labels

    proba = artefact["pipeline"].predict_proba(X)
    return labels, pd.DataFrame(proba, columns=artefact["label_encoder"].classes_)
