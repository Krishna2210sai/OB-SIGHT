"""Classical baselines for delivery-mode prediction.

Two feature sets are compared:
  * maternal clinical variables only
  * clinical variables + hand-crafted CTG descriptors

Models: logistic regression and histogram gradient boosting.  The baselines
are reference points for the deep-learning comparison, not clinical tools.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_DIR, PROCESSED_DIR
from src.preprocessing.ctg_features import FEATURE_NAMES

CLINICAL_RAW = [
    "age",
    "gravidity",
    "parity",
    "gest_weeks",
    "diabetes",
    "hypertension",
    "preeclampsia",
    "liquor_praecox",
    "pyrexia",
    "meconium",
    "presentation",
    "induced",
]
CATEGORICAL = ["presentation"]


def load_baseline_frame() -> pd.DataFrame:
    dataset = pd.read_csv(PROCESSED_DIR / "dataset.csv")
    ctg = pd.read_csv(PROCESSED_DIR / "ctg_features.csv")
    frame = dataset.merge(ctg, on="record_id", how="left", validate="one_to_one")
    missing = set(FEATURE_NAMES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing CTG features: {sorted(missing)}")
    return frame


def _numeric_pipeline() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def _preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    transformers = [("num", _numeric_pipeline(), numeric_cols)]
    if CATEGORICAL[0] in numeric_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)
        )
    return ColumnTransformer(transformers, remainder="drop")


def build_baselines() -> Tuple[Dict[str, Pipeline], Dict[str, List[str]]]:
    clinical_numeric = [c for c in CLINICAL_RAW if c not in CATEGORICAL]
    all_numeric = clinical_numeric + FEATURE_NAMES

    clinical_features = clinical_numeric + CATEGORICAL
    ctg_features = list(FEATURE_NAMES)
    combined_features = all_numeric + CATEGORICAL

    models = {
        "clinical_lr": Pipeline(
            [
                ("prep", _preprocessor(clinical_numeric)),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5, random_state=42)),
            ]
        ),
        "ctg_lr": Pipeline(
            [
                ("prep", _preprocessor(FEATURE_NAMES)),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5, random_state=42)),
            ]
        ),
        "combined_lr": Pipeline(
            [
                ("prep", _preprocessor(all_numeric)),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5, random_state=42)),
            ]
        ),
        "combined_gb": Pipeline(
            [
                ("prep", _preprocessor(all_numeric)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.05,
                        max_depth=3,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    feature_sets = {
        "clinical_lr": clinical_features,
        "ctg_lr": ctg_features,
        "combined_lr": combined_features,
        "combined_gb": combined_features,
    }
    return models, feature_sets


def train_baselines() -> Dict[str, dict]:
    frame = load_baseline_frame()
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    models, feature_sets = build_baselines()
    results: Dict[str, dict] = {}
    for name, model in models.items():
        cols = feature_sets[name]
        model.fit(train[cols], train["label"])
        joblib.dump({"model": model, "columns": cols}, MODEL_DIR / f"baseline_{name}.joblib")
        prob = model.predict_proba(test[cols])[:, 1]
        results[name] = {
            "columns": cols,
            "test_probabilities": prob.tolist(),
            "test_labels": test["label"].tolist(),
        }
    return results


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = train_baselines()
    summary = {name: {"n_columns": len(value["columns"]), "columns": value["columns"]} for name, value in results.items()}
    with open(MODEL_DIR / "baselines_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v["n_columns"] for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
