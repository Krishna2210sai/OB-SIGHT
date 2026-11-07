"""Evaluation metrics shared by the baseline, deep learning and API reports."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y_true)) > 1
    metrics: Dict[str, float] = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(0.5 * ((tp / (tp + fn)) if (tp + fn) else 0.0) + 0.5 * (tn / (tn + fp)) if (tn + fp) else float("nan")),
        "auroc": float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if has_both else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    # One-vs-rest per-class summaries (useful for reporting).
    for cls, name in [(1, "cesarean"), (0, "vaginal")]:
        yt = (y_true == cls).astype(int)
        yp = (y_pred == cls).astype(int)
        metrics[f"{name}_precision"] = float(precision_score(yt, yp, zero_division=0))
        metrics[f"{name}_recall"] = float(recall_score(yt, yp, zero_division=0))
        metrics[f"{name}_f1"] = float(f1_score(yt, yp, zero_division=0))
    return metrics


def _stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    pos_sample = rng.choice(pos, size=len(pos), replace=True)
    neg_sample = rng.choice(neg, size=len(neg), replace=True)
    return np.concatenate([pos_sample, neg_sample])


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(n_boot):
        idx = _stratified_bootstrap_indices(y_true, rng)
        try:
            value = metric(y_true[idx], y_prob[idx])
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 100 * alpha / 2)), float(np.percentile(values, 100 * (1 - alpha / 2)))


def bootstrap_auroc(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    return bootstrap_ci(y_true, y_prob, roc_auc_score, n_boot=n_boot, seed=seed)


def bootstrap_auprc(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    return bootstrap_ci(y_true, y_prob, average_precision_score, n_boot=n_boot, seed=seed)


def paired_bootstrap_auc_difference(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired stratified bootstrap for AUC(a) - AUC(b).

    The same resampled records are used for both models, which preserves the
    correlation between their predictions.  With only seven test positives the
    interval is wide; it is reported as an uncertainty statement, not as proof
    of superiority.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob_a = np.asarray(y_prob_a, dtype=float)
    y_prob_b = np.asarray(y_prob_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    for _ in range(n_boot):
        idx = _stratified_bootstrap_indices(y_true, rng)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            diff = roc_auc_score(y_true[idx], y_prob_a[idx]) - roc_auc_score(
                y_true[idx], y_prob_b[idx]
            )
        except ValueError:
            continue
        if np.isfinite(diff):
            diffs.append(float(diff))
    if not diffs:
        return {"difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    observed = float(roc_auc_score(y_true, y_prob_a) - roc_auc_score(y_true, y_prob_b))
    return {
        "difference": observed,
        "ci_low": float(np.percentile(diffs, 100 * alpha / 2)),
        "ci_high": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        "n_boot": int(len(diffs)),
    }


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-frequency bins; returns (mean_predicted, observed_fraction, count)."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(y_prob)
    chunks = np.array_split(order, n_bins)
    means, observed, counts = [], [], []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        means.append(float(y_prob[chunk].mean()))
        observed.append(float(y_true[chunk].mean()))
        counts.append(int(len(chunk)))
    return np.array(means), np.array(observed), np.array(counts)


def find_threshold(y_true: np.ndarray, y_prob: np.ndarray, criterion: str = "youden") -> float:
    """Select a decision threshold on validation data only."""
    thresholds = np.unique(np.round(np.linspace(0.05, 0.95, 91), 3))
    best_t, best_score = 0.5, -np.inf
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        if criterion == "youden":
            score = m["recall_sensitivity"] + m["specificity"] - 1
        elif criterion == "f1":
            score = m["f1"]
        elif criterion == "sensitivity":
            score = m["recall_sensitivity"] - 0.01 * (1 - m["specificity"])
        else:
            raise ValueError(criterion)
        if score > best_score + 1e-12:
            best_t, best_score = float(t), score
    return best_t
