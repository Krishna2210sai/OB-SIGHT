"""Nested deep-learning experiments for Phase B.

The outer five folds are the fixed Phase A folds.  Inside each outer training
partition, three inner folds select a learning rate and an epoch budget.  The
outer model is then trained on the full outer training partition for the
selected number of epochs and evaluated once on the untouched outer validation
fold.  The outer validation fold is never used for early stopping.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.deep_learning_model import build_model, count_parameters, forward_model
from src.phase_a.signals import SignalScaler, stack_windows
from src.phase_b.config import (
    B0_DIR,
    DEEP_BATCH_SIZE,
    DEEP_INNER_EPOCHS,
    DEEP_LR_GRID,
    DEEP_MIN_EPOCHS,
    DEEP_OUTER_EPOCHS,
    DEEP_PATIENCE,
    DEEP_WEIGHT_DECAY,
    DEFAULT_TORCH_THREADS,
    SEED,
    augment_result_path,
    b0_result_path,
    deep_result_path,
    selection_path,
)
from src.phase_b.data import DevelopmentBundle, make_inner_folds, outer_train_validation
from src.phase_b.imbalance import (
    augment_batch,
    build_loss,
    class_counts,
    compute_class_weights,
    positive_weight,
    sampler_weights,
    make_weighted_sampler,
)
from src.phase_b.metrics import (
    aggregate_folds,
    best_threshold,
    bootstrap_confidence_intervals,
    evaluate_predictions,
    paired_bootstrap_difference,
)
from src.preprocessing.preprocess import ClinicalPreprocessor
from src.training.train import CTGDataset, set_seed


@dataclass
class DeepRunConfig:
    variant: str
    strategy: str = "weighted_bce"
    learning_rate: float = 1e-3
    epochs: int = DEEP_OUTER_EPOCHS
    batch_size: int = DEEP_BATCH_SIZE
    weight_decay: float = DEEP_WEIGHT_DECAY
    patience: int = DEEP_PATIENCE
    augment: str = "none"
    seed: int = SEED


_SELECTION_CACHE: Dict[str, Dict] = {}


def _configure_threads() -> None:
    # Fixed default thread count: CPU float reductions can differ between
    # thread counts, which would break bitwise reproducibility of deep runs.
    threads = int(os.environ.get("PHASE_B_THREADS", str(DEFAULT_TORCH_THREADS)))
    if threads > 0:
        torch.set_num_threads(threads)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def prepare_fold_arrays(
    bundle: DevelopmentBundle,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Fit signal and clinical preprocessing on the training partition only."""
    train_ids = bundle.frame.iloc[train_idx]["record_id"].astype(str).tolist()
    val_ids = bundle.frame.iloc[val_idx]["record_id"].astype(str).tolist()
    scaler = SignalScaler().fit(bundle.windows[rid] for rid in train_ids)
    Xs_train = stack_windows([bundle.windows[rid] for rid in train_ids], scaler)
    Xs_val = stack_windows([bundle.windows[rid] for rid in val_ids], scaler)

    clinical = ClinicalPreprocessor()
    clinical.fit(bundle.frame.iloc[train_idx][bundle.clinical_raw].reset_index(drop=True))
    Xc_train = clinical.transform(
        bundle.frame.iloc[train_idx][bundle.clinical_raw].reset_index(drop=True)
    ).astype(np.float32)
    Xc_val = clinical.transform(
        bundle.frame.iloc[val_idx][bundle.clinical_raw].reset_index(drop=True)
    ).astype(np.float32)
    metadata = {
        "signal_scaler": scaler.to_dict(),
        "clinical_means": clinical.means,
        "clinical_stds": clinical.stds,
        "clinical_medians": clinical.medians,
        "clinical_output_names": clinical.output_names,
    }
    return (
        Xs_train,
        Xc_train,
        bundle.y[train_idx],
        Xs_val,
        Xc_val,
        bundle.y[val_idx],
        metadata,
    )


def predict_array(
    model: nn.Module,
    signals: np.ndarray,
    clinical: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        CTGDataset(signals, clinical, np.zeros(len(signals), dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for signal, clin, _ in loader:
            logits = forward_model(model, signal, clin)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def fit_deep(
    signals_train: np.ndarray,
    clinical_train: np.ndarray,
    y_train: np.ndarray,
    signals_val: np.ndarray | None,
    clinical_val: np.ndarray | None,
    y_val: np.ndarray | None,
    config: DeepRunConfig,
) -> Tuple[nn.Module, pd.DataFrame, Dict]:
    """Train one deep model with one imbalance strategy.

    If validation arrays are supplied, early stopping is used and the best
    validation-loss state is restored.  Without validation arrays the model is
    trained for the fixed epoch budget and the final state is returned.
    """
    set_seed(int(config.seed))
    _configure_threads()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CTGDataset(signals_train, clinical_train, y_train, augment=False)
    generator = torch.Generator().manual_seed(int(config.seed))
    sampler = (
        make_weighted_sampler(y_train, seed=int(config.seed))
        if config.strategy == "weighted_sampling"
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator if sampler is None else None,
        num_workers=0,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            CTGDataset(signals_val, clinical_val, y_val, augment=False),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if signals_val is not None
        else None
    )

    model = build_model(config.variant, clinical_features=clinical_train.shape[1]).to(device)
    criterion = build_loss(config.strategy, y_train).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(2, config.patience // 2),
            min_lr=1e-6,
        )
        if val_loader is not None
        else None
    )

    history: List[Dict] = []
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0
    augment_generator = torch.Generator().manual_seed(config.seed + 17)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for signal, clinical, label in train_loader:
            signal, clinical, label = signal.to(device), clinical.to(device), label.to(device)
            if config.augment != "none":
                signal = augment_batch(signal, config.augment, augment_generator)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_model(model, signal, clinical)
            loss = criterion(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += float(loss.item()) * len(label)
            seen += len(label)
        train_loss = running / max(seen, 1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if val_loader is not None:
            val_probs = predict_array(model, signals_val, clinical_val, config.batch_size)
            val_loss = float(
                nn.functional.binary_cross_entropy(
                    torch.from_numpy(val_probs.astype(np.float32)).clamp(1e-6, 1 - 1e-6),
                    torch.from_numpy(y_val.astype(np.float32)),
                ).item()
            )
            val_metrics = evaluate_predictions(y_val, val_probs)
            row.update(
                {
                    "val_loss": val_loss,
                    "val_auroc": val_metrics["auroc"],
                    "val_auprc": val_metrics["auprc"],
                    "val_sensitivity": val_metrics["recall_sensitivity"],
                    "val_specificity": val_metrics["specificity"],
                }
            )
            scheduler.step(val_loss)
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        history.append(row)
        if val_loader is not None and epochs_without_improvement >= config.patience:
            break

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    info = {
        "best_epoch": int(best_epoch if val_loader is not None else config.epochs),
        "best_val_loss": float(best_loss if val_loader is not None else np.nan),
        "epochs_run": int(len(history)),
        "n_parameters": int(count_parameters(model)),
    }
    return model, history_df, info


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    config: DeepRunConfig,
    info: Dict,
    y_train: np.ndarray,
    fold: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": config.variant,
            "strategy": config.strategy,
            "fold": int(fold),
            "config": asdict(config),
            "class_counts": dict(zip(("negative", "positive"), class_counts(y_train))),
            "class_weights": compute_class_weights(y_train),
            "positive_weight": positive_weight(y_train),
            "info": info,
        },
        path,
    )


def _inner_selection(
    bundle: DevelopmentBundle,
    variant: str,
    outer_fold: int,
    outer_train_idx: np.ndarray,
    seed: int,
    inner_epochs: int,
    outer_epochs: int,
    patience: int,
) -> Dict:
    """Select learning rate and epoch budget on inner folds only."""
    labels = bundle.y[outer_train_idx]
    inner_fold_ids = make_inner_folds(bundle.y, outer_train_idx, seed=seed + outer_fold)
    candidates = []
    for lr in DEEP_LR_GRID:
        fold_rows = []
        oof_prob = np.full(len(outer_train_idx), np.nan, dtype=float)
        for inner in sorted(np.unique(inner_fold_ids)):
            inner_train_local = np.flatnonzero(inner_fold_ids != inner)
            inner_val_local = np.flatnonzero(inner_fold_ids == inner)
            inner_train_idx = outer_train_idx[inner_train_local]
            inner_val_idx = outer_train_idx[inner_val_local]
            (
                Xs_tr,
                Xc_tr,
                y_tr,
                Xs_va,
                Xc_va,
                y_va,
                _,
            ) = prepare_fold_arrays(bundle, inner_train_idx, inner_val_idx)
            cfg = DeepRunConfig(
                variant=variant,
                strategy="weighted_bce",
                learning_rate=lr,
                epochs=inner_epochs,
                patience=patience,
                seed=int(seed) + 100 * outer_fold + int(inner),
            )
            model, history, info = fit_deep(
                Xs_tr, Xc_tr, y_tr, Xs_va, Xc_va, y_va, cfg
            )
            prob = predict_array(model, Xs_va, Xc_va)
            oof_prob[inner_val_local] = prob
            metrics = evaluate_predictions(y_va, prob)
            fold_rows.append(
                {
                    "inner_fold": int(inner),
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "sensitivity": metrics["recall_sensitivity"],
                    "best_epoch": info["best_epoch"],
                    "epochs_run": info["epochs_run"],
                    "n_train": int(len(inner_train_idx)),
                    "n_val": int(len(inner_val_idx)),
                }
            )
        candidate = {
            "learning_rate": float(lr),
            "mean_auroc": float(np.mean([row["auroc"] for row in fold_rows])),
            "mean_auprc": float(np.mean([row["auprc"] for row in fold_rows])),
            "mean_best_epoch": float(np.mean([row["best_epoch"] for row in fold_rows])),
            "folds": fold_rows,
        }
        candidate["threshold"] = float(best_threshold(labels, oof_prob))
        candidates.append(candidate)

    candidates.sort(key=lambda item: (item["mean_auroc"], item["mean_auprc"]), reverse=True)
    selected = candidates[0]
    selected_epochs = int(
        np.clip(
            round(selected["mean_best_epoch"]),
            DEEP_MIN_EPOCHS,
            outer_epochs,
        )
    )
    return {
        "outer_fold": int(outer_fold),
        "selected_learning_rate": selected["learning_rate"],
        "selected_epochs": selected_epochs,
        "selected_inner_auroc": selected["mean_auroc"],
        "selected_inner_auprc": selected["mean_auprc"],
        "threshold_from_inner": selected["threshold"],
        "candidates": candidates,
        "n_inner_train": int(len(outer_train_idx)),
        "selection_loss": "weighted_bce",
    }


def load_or_compute_selection(
    bundle: DevelopmentBundle,
    variant: str,
    seed: int,
    inner_epochs: int,
    outer_epochs: int,
    patience: int,
    force: bool = False,
) -> Dict:
    path = selection_path(variant)
    # Selection is deterministic for a given variant.  Cache it in-process so a
    # forced run recomputes it once, not once per imbalance strategy.
    if variant in _SELECTION_CACHE:
        return _SELECTION_CACHE[variant]
    if path.exists() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _SELECTION_CACHE[variant] = payload
        return payload
    selections = {}
    for outer_fold in range(1, 6):
        outer_train_idx, _ = outer_train_validation(bundle.folds, outer_fold)
        selections[str(outer_fold)] = _inner_selection(
            bundle,
            variant,
            outer_fold,
            outer_train_idx,
            seed,
            inner_epochs,
            outer_epochs,
            patience,
        )
    payload = {
        "variant": variant,
        "inner_folds": 3,
        "selection_metric": "mean inner-fold AUROC (AUPRC tie-break)",
        "selection_loss": "weighted_bce",
        "outer_folds": selections,
    }
    _write_json(path, payload)
    _SELECTION_CACHE[variant] = payload
    return payload


def run_deep_strategy(
    bundle: DevelopmentBundle,
    variant: str,
    strategy: str,
    seed: int = SEED,
    outer_epochs: int = DEEP_OUTER_EPOCHS,
    patience: int = DEEP_PATIENCE,
    augment: str = "none",
    fixed_learning_rate: Optional[float] = None,
    use_inner_selection: bool = True,
    force: bool = False,
    directory: Optional[Path] = None,
) -> Dict:
    directory = directory or deep_result_path(variant, strategy)
    result_path = directory / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)

    if use_inner_selection:
        selection = load_or_compute_selection(
            bundle,
            variant,
            seed=seed,
            inner_epochs=min(15, outer_epochs),
            outer_epochs=outer_epochs,
            patience=patience,
            force=force,
        )
    else:
        selection = {}

    all_probs = np.full(len(bundle.frame), np.nan, dtype=float)
    fold_rows = []
    histories = []
    fold_details = {}

    for outer_fold in range(1, 6):
        outer_train_idx, outer_val_idx = outer_train_validation(bundle.folds, outer_fold)
        if use_inner_selection:
            chosen = selection["outer_folds"][str(outer_fold)]
            lr = float(chosen["selected_learning_rate"])
            epochs = int(chosen["selected_epochs"])
            inner_threshold = float(chosen["threshold_from_inner"])
        else:
            lr = float(fixed_learning_rate or 1e-3)
            epochs = int(outer_epochs)
            inner_threshold = 0.5

        (
            Xs_tr,
            Xc_tr,
            y_tr,
            Xs_va,
            Xc_va,
            y_va,
            prep_meta,
        ) = prepare_fold_arrays(bundle, outer_train_idx, outer_val_idx)
        cfg = DeepRunConfig(
            variant=variant,
            strategy=strategy,
            learning_rate=lr,
            epochs=epochs,
            patience=patience,
            augment=augment,
            seed=seed + 1000 * outer_fold,
        )
        model, history, info = fit_deep(Xs_tr, Xc_tr, y_tr, None, None, None, cfg)
        prob = predict_array(model, Xs_va, Xc_va)
        all_probs[outer_val_idx] = prob
        metrics_05 = evaluate_predictions(y_va, prob, threshold=0.5)
        metrics_inner = evaluate_predictions(y_va, prob, threshold=inner_threshold)
        fold_row = {
            "fold": int(outer_fold),
            "n_train": int(len(outer_train_idx)),
            "n_val": int(len(outer_val_idx)),
            "n_cesarean_train": int(y_tr.sum()),
            "n_cesarean_val": int(y_va.sum()),
            "learning_rate": lr,
            "epochs": epochs,
            "inner_threshold": inner_threshold,
            **metrics_05,
            "sensitivity_inner_threshold": metrics_inner["recall_sensitivity"],
            "specificity_inner_threshold": metrics_inner["specificity"],
        }
        fold_rows.append(fold_row)
        histories.append(history.assign(fold=outer_fold))
        _save_checkpoint(
            directory / f"fold{outer_fold}_checkpoint.pt",
            model,
            cfg,
            info,
            y_tr,
            outer_fold,
        )
        _write_json(
            directory / f"fold{outer_fold}.json",
            {
                "fold": fold_row,
                "training": {
                    "config": asdict(cfg),
                    "info": info,
                    "class_counts": dict(zip(("negative", "positive"), class_counts(y_tr))),
                    "class_weights": compute_class_weights(y_tr),
                    "positive_weight": positive_weight(y_tr),
                    "sampler_weights": (
                        sampler_weights(y_tr).tolist()
                        if strategy == "weighted_sampling"
                        else None
                    ),
                    "preprocessing": prep_meta,
                },
            },
        )
        fold_details[str(outer_fold)] = fold_row

    oof = pd.DataFrame(
        {
            "record_id": bundle.record_ids,
            "y_true": bundle.y,
            "prob": all_probs,
            "fold": bundle.folds["fold"].to_numpy(int),
        }
    )
    padded = oof.copy()
    for fold_value in sorted(oof["fold"].unique()):
        mask = oof["fold"] == fold_value
        if oof.loc[mask, "prob"].isna().any():
            raise RuntimeError(f"Missing out-of-fold predictions in fold {fold_value}")
    pooled = evaluate_predictions(oof["y_true"], oof["prob"], threshold=0.5)
    ci = bootstrap_confidence_intervals(
        oof["y_true"].to_numpy(int), oof["prob"].to_numpy(float), threshold=0.5
    )
    pooled_with_ci = dict(pooled)
    for key, (low, high) in ci.items():
        pooled_with_ci[f"{key}_ci_low"] = low
        pooled_with_ci[f"{key}_ci_high"] = high
    result = {
        "variant": variant,
        "strategy": strategy,
        "augment": augment,
        "n_records": int(len(oof)),
        "n_cesarean": int(oof["y_true"].sum()),
        "metrics": {
            "pooled": pooled_with_ci,
            "fold_mean": {k: v["mean"] for k, v in aggregate_folds(fold_rows).items()},
            "fold_std": {k: v["std"] for k, v in aggregate_folds(fold_rows).items()},
            "folds": fold_rows,
        },
        "inner_selection": selection,
        "fold_details": fold_details,
        "config": {
            "seed": seed,
            "outer_epochs": outer_epochs,
            "patience": patience,
            "use_inner_selection": use_inner_selection,
            "fixed_learning_rate": fixed_learning_rate,
        },
    }
    _write_json(result_path, result)
    oof.to_csv(directory / "predictions.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        directory / "training_history.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(directory / "fold_metrics.csv", index=False)
    return result


def run_b0_deep(
    bundle: DevelopmentBundle,
    variant: str,
    seed: int = SEED,
    outer_epochs: int = DEEP_OUTER_EPOCHS,
    patience: int = DEEP_PATIENCE,
    force: bool = False,
) -> Dict:
    """Baseline reproduction with the production weighted-BCE loss and 1e-3 LR."""
    directory = b0_result_path(variant)
    result_path = directory / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    return run_deep_strategy(
        bundle,
        variant,
        strategy="weighted_bce",
        seed=seed,
        outer_epochs=outer_epochs,
        patience=patience,
        fixed_learning_rate=1e-3,
        use_inner_selection=False,
        force=force,
        directory=directory,
    )


def run_augmentation(
    bundle: DevelopmentBundle,
    mode: str,
    variant: str = "multimodal",
    seed: int = SEED,
    outer_epochs: int = DEEP_OUTER_EPOCHS,
    force: bool = False,
) -> Dict:
    directory = augment_result_path(mode)
    result_path = directory / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    return run_deep_strategy(
        bundle,
        variant,
        strategy="weighted_bce",
        seed=seed,
        outer_epochs=outer_epochs,
        patience=DEEP_PATIENCE,
        augment=mode,
        fixed_learning_rate=1e-3,
        use_inner_selection=False,
        force=force,
        directory=directory,
    )
