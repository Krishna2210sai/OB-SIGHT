"""Nested classical-model experiments for Phase B.

Logistic regression and histogram gradient boosting are evaluated on the
Phase A 40-feature tabular matrix.  Resampling (random oversampling or SMOTE)
and scaling are fitted inside each training partition only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.phase_b.config import (
    B0_DIR,
    CLASSICAL_GB_GRID,
    CLASSICAL_LR_C,
    DEFAULT_THRESHOLD,
    SEED,
    classical_result_path,
)
from src.phase_b.data import DevelopmentBundle, make_inner_folds, outer_train_validation
from src.phase_b.imbalance import (
    class_counts,
    compute_class_weights,
    random_oversample,
    smote_oversample,
)
from src.phase_b.metrics import (
    aggregate_folds,
    best_threshold,
    bootstrap_confidence_intervals,
    evaluate_predictions,
)


@dataclass
class FittedClassical:
    imputer: SimpleImputer
    scaler: StandardScaler
    classifier: object

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xt = self.scaler.transform(self.imputer.transform(X))
        return self.classifier.predict_proba(Xt)[:, 1]


def _candidate_grid(model: str) -> List[Dict]:
    if model == "logistic":
        return [{"C": float(value)} for value in CLASSICAL_LR_C]
    if model == "gradient_boosting":
        return [dict(item) for item in CLASSICAL_GB_GRID]
    raise ValueError(f"Unknown classical model: {model}")


def _make_classifier(
    model: str,
    params: Dict,
    strategy: str,
    seed: int,
):
    class_weight = "balanced" if strategy == "class_weight" else None
    if model == "logistic":
        return LogisticRegression(
            C=float(params["C"]),
            class_weight=class_weight,
            max_iter=5000,
            random_state=seed,
        )
    if model == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            l2_regularization=1.0,
            class_weight=class_weight,
            random_state=seed,
        )
    raise ValueError(f"Unknown classical model: {model}")


def fit_classical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: str,
    strategy: str,
    params: Dict,
    seed: int = SEED,
) -> FittedClassical:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xt = scaler.fit_transform(imputer.fit_transform(X_train))
    yt = y_train.copy()
    if strategy == "random_oversample":
        Xt, yt = random_oversample(Xt, yt, seed=seed)
    elif strategy == "smote":
        Xt, yt = smote_oversample(Xt, yt, k_neighbors=5, seed=seed)
    elif strategy not in ("none", "class_weight"):
        raise ValueError(f"Unknown classical strategy: {strategy}")
    classifier = _make_classifier(model, params, strategy, seed)
    classifier.fit(Xt, yt)
    return FittedClassical(imputer=imputer, scaler=scaler, classifier=classifier)


def _inner_select(
    X: np.ndarray,
    y: np.ndarray,
    outer_train_idx: np.ndarray,
    model: str,
    strategy: str,
    outer_fold: int,
    seed: int,
) -> Dict:
    inner_ids = make_inner_folds(y, outer_train_idx, seed=seed + outer_fold)
    candidates = []
    for params in _candidate_grid(model):
        rows = []
        oof_prob = np.full(len(outer_train_idx), np.nan, dtype=float)
        for inner in sorted(np.unique(inner_ids)):
            train_local = np.flatnonzero(inner_ids != inner)
            val_local = np.flatnonzero(inner_ids == inner)
            fitted = fit_classical(
                X[outer_train_idx[train_local]],
                y[outer_train_idx[train_local]],
                model,
                strategy,
                params,
                seed=seed + 100 * outer_fold + inner,
            )
            prob = fitted.predict_proba(X[outer_train_idx[val_local]])
            oof_prob[val_local] = prob
            metrics = evaluate_predictions(y[outer_train_idx[val_local]], prob)
            rows.append(
                {
                    "inner_fold": int(inner),
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "sensitivity": metrics["recall_sensitivity"],
                }
            )
        candidate = {
            "params": params,
            "mean_auroc": float(np.mean([row["auroc"] for row in rows])),
            "mean_auprc": float(np.mean([row["auprc"] for row in rows])),
            "threshold": float(best_threshold(y[outer_train_idx], oof_prob)),
            "folds": rows,
        }
        candidates.append(candidate)
    candidates.sort(key=lambda item: (item["mean_auroc"], item["mean_auprc"]), reverse=True)
    return {
        "outer_fold": int(outer_fold),
        "selected": candidates[0],
        "candidates": candidates,
    }


def run_classical_strategy(
    bundle: DevelopmentBundle,
    model: str,
    strategy: str,
    seed: int = SEED,
    force: bool = False,
    directory: Optional[Path] = None,
) -> Dict:
    directory = directory or classical_result_path(model, strategy)
    result_path = directory / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    X = bundle.frame[bundle.feature_names].to_numpy(dtype=float)
    y = bundle.y

    all_probs = np.full(len(bundle.frame), np.nan, dtype=float)
    fold_rows = []
    selections = {}

    for outer_fold in range(1, 6):
        outer_train_idx, outer_val_idx = outer_train_validation(bundle.folds, outer_fold)
        selection = _inner_select(
            X, y, outer_train_idx, model, strategy, outer_fold, seed
        )
        selections[str(outer_fold)] = selection
        params = selection["selected"]["params"]
        threshold = float(selection["selected"]["threshold"])
        fitted = fit_classical(
            X[outer_train_idx],
            y[outer_train_idx],
            model,
            strategy,
            params,
            seed=seed + 1000 * outer_fold,
        )
        prob = fitted.predict_proba(X[outer_val_idx])
        all_probs[outer_val_idx] = prob
        metrics_05 = evaluate_predictions(y[outer_val_idx], prob, threshold=DEFAULT_THRESHOLD)
        metrics_inner = evaluate_predictions(y[outer_val_idx], prob, threshold=threshold)
        fold_row = {
            "fold": int(outer_fold),
            "n_train": int(len(outer_train_idx)),
            "n_val": int(len(outer_val_idx)),
            "n_cesarean_train": int(y[outer_train_idx].sum()),
            "n_cesarean_val": int(y[outer_val_idx].sum()),
            "params": params,
            "inner_threshold": threshold,
            **metrics_05,
            "sensitivity_inner_threshold": metrics_inner["recall_sensitivity"],
            "specificity_inner_threshold": metrics_inner["specificity"],
        }
        fold_rows.append(fold_row)

    oof = pd.DataFrame(
        {
            "record_id": bundle.record_ids,
            "y_true": y,
            "prob": all_probs,
            "fold": bundle.folds["fold"].to_numpy(int),
        }
    )
    if oof["prob"].isna().any():
        raise RuntimeError("Missing out-of-fold classical predictions")
    pooled = evaluate_predictions(y, all_probs, threshold=DEFAULT_THRESHOLD)
    ci = bootstrap_confidence_intervals(y, all_probs, threshold=DEFAULT_THRESHOLD)
    pooled_with_ci = dict(pooled)
    for key, (low, high) in ci.items():
        pooled_with_ci[f"{key}_ci_low"] = low
        pooled_with_ci[f"{key}_ci_high"] = high
    aggregate = aggregate_folds(fold_rows)
    result = {
        "model": model,
        "strategy": strategy,
        "feature_names": bundle.feature_names,
        "n_records": int(len(oof)),
        "n_cesarean": int(y.sum()),
        "metrics": {
            "pooled": pooled_with_ci,
            "fold_mean": {key: value["mean"] for key, value in aggregate.items()},
            "fold_std": {key: value["std"] for key, value in aggregate.items()},
            "folds": fold_rows,
        },
        "selections": selections,
        "config": {"seed": seed, "inner_folds": 3, "outer_folds": 5},
        "class_weights": compute_class_weights(y),
        "class_counts": dict(zip(("negative", "positive"), class_counts(y))),
    }
    result_path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    oof.to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(directory / "fold_metrics.csv", index=False)
    return result


def run_b0_classical(
    bundle: DevelopmentBundle,
    model: str,
    seed: int = SEED,
    force: bool = False,
) -> Dict:
    """Reproduce the production class-balanced baselines on the Phase A folds."""
    directory = B0_DIR / model
    result_path = directory / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    X = bundle.frame[bundle.feature_names].to_numpy(dtype=float)
    y = bundle.y
    params = {"C": 0.5} if model == "logistic" else {"max_depth": 3, "learning_rate": 0.05}
    all_probs = np.full(len(bundle.frame), np.nan, dtype=float)
    fold_rows = []
    for outer_fold in range(1, 6):
        outer_train_idx, outer_val_idx = outer_train_validation(bundle.folds, outer_fold)
        fitted = fit_classical(
            X[outer_train_idx],
            y[outer_train_idx],
            model,
            "class_weight",
            params,
            seed=seed + 1000 * outer_fold,
        )
        prob = fitted.predict_proba(X[outer_val_idx])
        all_probs[outer_val_idx] = prob
        metrics = evaluate_predictions(y[outer_val_idx], prob, threshold=DEFAULT_THRESHOLD)
        fold_rows.append(
            {
                "fold": int(outer_fold),
                "n_train": int(len(outer_train_idx)),
                "n_val": int(len(outer_val_idx)),
                "n_cesarean_train": int(y[outer_train_idx].sum()),
                "n_cesarean_val": int(y[outer_val_idx].sum()),
                "params": params,
                **metrics,
            }
        )
    oof = pd.DataFrame(
        {
            "record_id": bundle.record_ids,
            "y_true": y,
            "prob": all_probs,
            "fold": bundle.folds["fold"].to_numpy(int),
        }
    )
    pooled = evaluate_predictions(y, all_probs, threshold=DEFAULT_THRESHOLD)
    ci = bootstrap_confidence_intervals(y, all_probs, threshold=DEFAULT_THRESHOLD)
    pooled_with_ci = dict(pooled)
    for key, (low, high) in ci.items():
        pooled_with_ci[f"{key}_ci_low"] = low
        pooled_with_ci[f"{key}_ci_high"] = high
    aggregate = aggregate_folds(fold_rows)
    result = {
        "model": model,
        "strategy": "production_class_weight",
        "feature_names": bundle.feature_names,
        "n_records": int(len(oof)),
        "n_cesarean": int(y.sum()),
        "metrics": {
            "pooled": pooled_with_ci,
            "fold_mean": {key: value["mean"] for key, value in aggregate.items()},
            "fold_std": {key: value["std"] for key, value in aggregate.items()},
            "folds": fold_rows,
        },
        "config": {"seed": seed, "params": params, "protocol": "outer folds only"},
    }
    result_path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    oof.to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(directory / "fold_metrics.csv", index=False)
    return result
