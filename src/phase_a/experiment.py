"""Cross-validated experiment runners for Phase A.

The held-out test split is never used here.  All Phase A comparisons use the
same fixed, record-level 5-fold partition of the development set
(``split in {train, val}``), so differences reflect preprocessing rather than
different data splits.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED
from src.evaluation.metrics import bootstrap_ci, compute_metrics
from src.models.deep_learning_model import forward_model
from src.phase_a.config import N_SPLITS
from src.phase_a.features import CLINICAL_RAW, candidate_feature_names
from src.phase_a.selection import BaseSelector, make_selector
from src.phase_a.signals import SignalScaler, stack_windows
from src.preprocessing.preprocess import ClinicalPreprocessor
from src.training.train import CTGDataset, TrainConfig, train_one_fold


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def phase_a_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    y_pred = (y_prob >= threshold).astype(int)
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    metrics["balanced_accuracy"] = (
        float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) else float("nan")
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics


def _threshold_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(matthews_corrcoef(y_true, (y_prob >= 0.5).astype(int)))


def _metric_summary(y_true: np.ndarray, y_prob: np.ndarray, fold_ids: np.ndarray) -> Dict:
    pooled = phase_a_metrics(y_true, y_prob)
    folds = []
    for fold in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold
        if len(np.unique(y_true[mask])) < 2:
            continue
        folds.append({"fold": int(fold), **phase_a_metrics(y_true[mask], y_prob[mask])})
    summary = {
        "pooled": pooled,
        "fold_mean": {},
        "fold_std": {},
        "folds": folds,
    }
    for key in ("auroc", "auprc", "accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "specificity", "f1", "mcc"):
        values = [item[key] for item in folds if np.isfinite(item.get(key, np.nan))]
        summary["fold_mean"][key] = float(np.mean(values)) if values else float("nan")
        summary["fold_std"][key] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    # Stratified bootstrap CIs on the pooled out-of-fold predictions.
    for key, fn in (
        ("auroc", roc_auc_score),
        ("auprc", average_precision_score),
        ("mcc", _threshold_mcc),
    ):
        try:
            low, high = bootstrap_ci(y_true, y_prob, fn, n_boot=1000, seed=42)
        except Exception:
            low, high = float("nan"), float("nan")
        summary["pooled"][f"{key}_ci_low"] = low
        summary["pooled"][f"{key}_ci_high"] = high
    return summary


# ---------------------------------------------------------------------------
# Folds and CV
# ---------------------------------------------------------------------------
def make_folds(metadata: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Fixed stratified folds on the development set (test excluded)."""
    y = metadata["label"].to_numpy(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(metadata), -1, dtype=int)
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        fold[val_idx] = k
    out = pd.DataFrame(
        {
            "record_id": metadata["record_id"].astype(str),
            "label": y,
            "split": metadata["split"].astype(str),
            "fold": fold,
        }
    )
    return out.sort_values(["fold", "record_id"]).reset_index(drop=True)


def _attach_folds(frame: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(folds[["record_id", "fold"]], on="record_id", how="left", validate="one_to_one")
    if merged["fold"].isna().any():
        raise ValueError("Some Phase A records have no fold assignment")
    return merged


def _build_estimator(model_name: str, selector: BaseSelector, seed: int):
    if model_name == "logistic":
        classifier = LogisticRegression(
            class_weight="balanced", C=0.5, max_iter=3000, random_state=seed
        )
    elif model_name == "gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_depth=3,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", selector),
            ("clf", classifier),
        ]
    )


def cv_baseline(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    method: str,
    model_name: str,
    seed: int = RANDOM_SEED,
) -> Dict:
    """Cross-validate one feature-selection / classifier combination."""
    data = _attach_folds(frame, folds)
    feature_names = candidate_feature_names(data)
    X = data[feature_names].to_numpy(dtype=float)
    y = data["label"].to_numpy(int)
    fold_ids = data["fold"].to_numpy(int)
    oof = np.full(len(data), np.nan, dtype=float)
    selected_by_fold: Dict[int, List[str]] = {}
    importances_by_fold: Dict[int, Dict[str, float]] = {}
    histories: List[Dict] = []

    for fold in sorted(np.unique(fold_ids)):
        train_mask = fold_ids != fold
        val_mask = fold_ids == fold
        selector = make_selector(method, feature_names, seed=seed)
        estimator = _build_estimator(model_name, selector, seed=seed)
        estimator.fit(X[train_mask], y[train_mask])
        prob = estimator.predict_proba(X[val_mask])[:, 1]
        oof[val_mask] = prob
        fitted_selector: BaseSelector = estimator.named_steps["select"]
        selected_by_fold[int(fold)] = fitted_selector.selected_features()
        if fitted_selector.importances_ is not None:
            importances_by_fold[int(fold)] = {
                name: float(value)
                for name, value in zip(feature_names, fitted_selector.importances_)
            }
        fold_metrics = phase_a_metrics(y[val_mask], prob)
        histories.append({"fold": int(fold), **fold_metrics})

    summary = _metric_summary(y, oof, fold_ids)
    result = {
        "method": method,
        "model": model_name,
        "n_records": int(len(data)),
        "n_features_input": len(feature_names),
        "n_features_mean_selected": float(np.mean([len(v) for v in selected_by_fold.values()])),
        "selected_by_fold": {str(k): v for k, v in selected_by_fold.items()},
        "metrics": summary,
        "fold_metrics": histories,
        "oof": {
            "record_id": data["record_id"].astype(str).tolist(),
            "y_true": y.tolist(),
            "prob": oof.tolist(),
            "fold": fold_ids.tolist(),
        },
    }
    return {
        "result": result,
        "oof_frame": pd.DataFrame(
            {"record_id": data["record_id"], "y_true": y, "prob": oof, "fold": fold_ids}
        ),
        "selected_by_fold": selected_by_fold,
        "importances_by_fold": importances_by_fold,
    }


def cv_deep(
    frame: pd.DataFrame,
    windows: Dict[str, Dict],
    folds: pd.DataFrame,
    variant: str = "multimodal",
    seed: int = RANDOM_SEED,
    epochs: int = 35,
    patience: int = 8,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict:
    """Cross-validate an existing deep architecture on a Phase A window."""
    data = _attach_folds(frame, folds).reset_index(drop=True)
    y = data["label"].to_numpy(int)
    fold_ids = data["fold"].to_numpy(int)
    oof = np.full(len(data), np.nan, dtype=float)
    histories: List[pd.DataFrame] = []
    best_epochs: Dict[int, int] = {}

    config = TrainConfig(
        variant=variant,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        scheduler_patience=max(2, patience // 2),
        seed=seed,
    )

    for fold in sorted(np.unique(fold_ids)):
        train_idx = np.flatnonzero(fold_ids != fold)
        val_idx = np.flatnonzero(fold_ids == fold)
        train_rids = data.loc[train_idx, "record_id"].astype(str).tolist()
        val_rids = data.loc[val_idx, "record_id"].astype(str).tolist()

        scaler = SignalScaler().fit(windows[rid] for rid in train_rids)
        Xs_train = stack_windows([windows[rid] for rid in train_rids], scaler)
        Xs_val = stack_windows([windows[rid] for rid in val_rids], scaler)

        clinical_pre = ClinicalPreprocessor().fit(
            data.loc[train_idx, CLINICAL_RAW].reset_index(drop=True)
        )
        Xc_train = clinical_pre.transform(data.loc[train_idx, CLINICAL_RAW]).astype(np.float32)
        Xc_val = clinical_pre.transform(data.loc[val_idx, CLINICAL_RAW]).astype(np.float32)

        model, history, info = train_one_fold(
            Xs_train,
            Xc_train,
            y[train_idx],
            Xs_val,
            Xc_val,
            y[val_idx],
            config,
            checkpoint_path=None,
            verbose=False,
        )
        histories.append(history.assign(fold=fold))
        best_epochs[int(fold)] = int(info["best_epoch"])

        model.eval()
        loader = torch.utils.data.DataLoader(
            CTGDataset(Xs_val, Xc_val, y[val_idx]),
            batch_size=batch_size,
            shuffle=False,
        )
        logits: List[np.ndarray] = []
        with torch.no_grad():
            for signal, clinical, _ in loader:
                out = forward_model(model, signal, clinical)
                logits.append(out.cpu().numpy())
        prob = 1.0 / (1.0 + np.exp(-np.concatenate(logits)))
        oof[val_idx] = prob

    summary = _metric_summary(y, oof, fold_ids)
    result = {
        "variant": variant,
        "n_records": int(len(data)),
        "epochs": epochs,
        "patience": patience,
        "best_epochs": {str(k): v for k, v in best_epochs.items()},
        "metrics": summary,
        "oof": {
            "record_id": data["record_id"].astype(str).tolist(),
            "y_true": y.tolist(),
            "prob": oof.tolist(),
            "fold": fold_ids.tolist(),
        },
    }
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    return {
        "result": result,
        "oof_frame": pd.DataFrame(
            {"record_id": data["record_id"], "y_true": y, "prob": oof, "fold": fold_ids}
        ),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_cv_result(
    directory: Path,
    name: str,
    result: Dict,
    oof: pd.DataFrame,
    extra: Optional[Dict] = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, **_json_safe(result)}
    if extra:
        payload.update(_json_safe(extra))
    (directory / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    oof.to_csv(directory / f"{name}_oof.csv", index=False)
