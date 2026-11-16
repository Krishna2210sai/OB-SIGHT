"""Evaluation metrics and uncertainty for Phase B."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "auroc",
    "auprc",
    "mcc",
    "brier",
)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # Include the right edge in the final bin.
    idx = np.clip(np.digitize(y_prob, bins[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(total)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if not np.isfinite(y_prob).all():
        raise ValueError("Predictions contain non-finite values")
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y_true)) > 1
    metrics = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if has_both else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if has_both else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if has_both else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metrics


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximising balanced accuracy on the supplied predictions.

    Callers must pass inner-fold predictions only, never outer validation data.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    candidates = np.unique(np.clip(y_prob, 0.0, 1.0))
    if len(candidates) == 1:
        return float(candidates[0])
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        score = balanced_accuracy_score(y_true, (y_prob >= threshold).astype(int))
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = []
    for cls in (0, 1):
        cls_idx = np.flatnonzero(y == cls)
        if len(cls_idx):
            indices.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
    return np.concatenate(indices)


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    keys = ("auroc", "auprc", "recall_sensitivity", "specificity", "mcc", "brier")
    values = {key: [] for key in keys}
    for _ in range(n_boot):
        idx = _stratified_indices(y_true, rng)
        metrics = evaluate_predictions(y_true[idx], y_prob[idx], threshold=threshold)
        for key in keys:
            value = metrics[key]
            if np.isfinite(value):
                values[key].append(value)
    out = {}
    for key in keys:
        if values[key]:
            out[key] = (
                float(np.percentile(values[key], 100 * alpha / 2)),
                float(np.percentile(values[key], 100 * (1 - alpha / 2))),
            )
        else:
            out[key] = (float("nan"), float("nan"))
    return out


def paired_bootstrap_difference(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Paired stratified bootstrap for strategy A minus strategy B.

    Both prediction vectors must be aligned to the same record order.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)
    rng = np.random.default_rng(seed)
    keys = ("auroc", "auprc", "recall_sensitivity", "specificity", "mcc")
    observed_a = evaluate_predictions(y_true, prob_a, threshold)
    observed_b = evaluate_predictions(y_true, prob_b, threshold)
    draws = {key: [] for key in keys}
    for _ in range(n_boot):
        idx = _stratified_indices(y_true, rng)
        ma = evaluate_predictions(y_true[idx], prob_a[idx], threshold)
        mb = evaluate_predictions(y_true[idx], prob_b[idx], threshold)
        for key in keys:
            if np.isfinite(ma[key]) and np.isfinite(mb[key]):
                draws[key].append(ma[key] - mb[key])
    out = {}
    for key in keys:
        diff = observed_a[key] - observed_b[key]
        if draws[key]:
            low = float(np.percentile(draws[key], 100 * alpha / 2))
            high = float(np.percentile(draws[key], 100 * (1 - alpha / 2)))
            arr = np.asarray(draws[key])
            p_value = float(2 * min((arr <= 0).mean(), (arr >= 0).mean()))
            p_value = min(1.0, p_value)
        else:
            low = high = p_value = float("nan")
        out[key] = {
            "difference": float(diff),
            "ci_low": low,
            "ci_high": high,
            "p_value": p_value,
        }
    return out


def aggregate_folds(fold_metrics: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    fold_metrics = list(fold_metrics)
    out = {}
    for key in METRIC_KEYS:
        values = [item[key] for item in fold_metrics if np.isfinite(item.get(key, np.nan))]
        out[key] = {
            "mean": float(np.mean(values)) if values else float("nan"),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        }
    return out
