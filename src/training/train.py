"""Training entry point for the deep learning models.

Examples
--------
Final model on the fixed train/val/test split:
    python -m src.training.train --variant multimodal
    python -m src.training.train --variant signal
    python -m src.training.train --variant bilstm
    python -m src.training.train --variant clinical

Five-fold cross-validation (used for a more stable performance estimate):
    python -m src.training.train --variant multimodal --cv
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from src.config import MODEL_DIR, PROCESSED_DIR, RANDOM_SEED
from src.models.deep_learning_model import build_model, count_parameters, forward_model

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


@dataclass
class TrainConfig:
    variant: str = "multimodal"
    epochs: int = 120
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    scheduler_patience: int = 8
    dropout_noise: float = 0.03
    seed: int = RANDOM_SEED
    num_workers: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


class CTGDataset(Dataset):
    def __init__(self, signals: np.ndarray, clinical: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.signals = torch.from_numpy(signals.astype(np.float32))
        self.clinical = torch.from_numpy(clinical.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        signal = self.signals[index].clone()
        if self.augment:
            # Small Gaussian noise on FHR/UC channels only; masks stay 0/1.
            noise = torch.randn(2, signal.shape[-1]) * 0.03
            signal[:2] = signal[:2] + noise
        return signal, self.clinical[index], self.labels[index]


def _predict(model: nn.Module, loader: DataLoader, device: torch.device | None = None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    logits, labels = [], []
    with torch.no_grad():
        for signal, clinical, label in loader:
            signal, clinical = signal.to(device), clinical.to(device)
            out = forward_model(model, signal, clinical)
            logits.append(out.cpu().numpy())
            labels.append(label.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def _metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "loss": float(nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(logits.astype(np.float32)), torch.from_numpy(labels.astype(np.float32))
        ).item()),
        "accuracy": float(accuracy_score(labels, pred)),
        "auroc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "auprc": float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
    }
    return metrics


def train_one_fold(
    signals_train: np.ndarray,
    clinical_train: np.ndarray,
    y_train: np.ndarray,
    signals_val: np.ndarray,
    clinical_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    checkpoint_path: Path | None = None,
    verbose: bool = True,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, float]]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CTGDataset(signals_train, clinical_train, y_train, augment=True)
    val_ds = CTGDataset(signals_val, clinical_val, y_val, augment=False)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, generator=generator,
                              num_workers=config.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers)

    model = build_model(config.variant, clinical_features=clinical_train.shape[1]).to(device)
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.scheduler_patience, min_lr=1e-6
    )

    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for signal, clinical, label in train_loader:
            signal, clinical, label = signal.to(device), clinical.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_model(model, signal, clinical)
            loss = criterion(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * len(label)
            n_seen += len(label)
        train_loss = running_loss / max(n_seen, 1)
        val_logits, val_labels = _predict(model, val_loader, device)
        val_metrics = _metrics_from_logits(val_logits, val_labels)
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(
                f"  epoch {epoch:3d} | train {train_loss:.4f} | val {val_metrics['loss']:.4f} "
                f"| AUC {val_metrics['auroc']:.3f} | AP {val_metrics['auprc']:.3f}"
            )

        if val_metrics["loss"] < best_loss - 1e-5:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "variant": config.variant,
                "clinical_features": int(clinical_train.shape[1]),
                "signal_channels": int(signals_train.shape[1]),
                "window_samples": int(signals_train.shape[2]),
                "config": asdict(config),
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "n_parameters": count_parameters(model),
            },
            checkpoint_path,
        )
    return model, history_df, {"best_epoch": best_epoch, "best_val_loss": best_loss}


def load_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sig = np.load(PROCESSED_DIR / "signals.npz", allow_pickle=True)
    clin = np.load(PROCESSED_DIR / "clinical.npz", allow_pickle=True)
    assert np.array_equal(sig["record_ids"], clin["record_ids"])
    return (
        sig["X"].astype(np.float32),
        clin["X"].astype(np.float32),
        sig["y"].astype(np.int64),
        sig["split"].astype(str),
    )


def train_final(config: TrainConfig) -> Dict:
    Xs, Xc, y, split = load_arrays()
    train = split == "train"
    val = split == "val"
    model, history, info = train_one_fold(
        Xs[train], Xc[train], y[train],
        Xs[val], Xc[val], y[val],
        config,
        checkpoint_path=MODEL_DIR / f"{config.variant}.pt",
    )
    history.to_csv(REPORT_DIR / f"training_history_{config.variant}.csv", index=False)
    result = {
        "variant": config.variant,
        "train_size": int(train.sum()),
        "val_size": int(val.sum()),
        "train_positives": int(y[train].sum()),
        "val_positives": int(y[val].sum()),
        "n_parameters": count_parameters(model),
        **info,
        "config": asdict(config),
    }
    with open(MODEL_DIR / f"train_summary_{config.variant}.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return result


def cross_validate(config: TrainConfig, n_splits: int = 5) -> Dict:
    Xs, Xc, y, _ = load_arrays()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(Xs, y), start=1):
        print(f"Fold {fold}/{n_splits} ({len(train_idx)} train / {len(val_idx)} val)")
        model, history, info = train_one_fold(
            Xs[train_idx], Xc[train_idx], y[train_idx],
            Xs[val_idx], Xc[val_idx], y[val_idx],
            config,
            checkpoint_path=MODEL_DIR / "cv" / f"{config.variant}_fold{fold}.pt",
        )
        logits, labels = _predict(model, DataLoader(CTGDataset(Xs[val_idx], Xc[val_idx], y[val_idx]),
                                                    batch_size=config.batch_size), torch.device("cpu"))
        probs = 1.0 / (1.0 + np.exp(-logits))
        oof[val_idx] = probs
        metrics = _metrics_from_logits(logits, labels)
        metrics.update({"fold": fold, "best_epoch": info["best_epoch"], **{k: int(v) for k, v in {
            "n_train": len(train_idx), "n_val": len(val_idx),
            "pos_train": int(y[train_idx].sum()), "pos_val": int(y[val_idx].sum()),
        }.items()}})
        fold_rows.append(metrics)
        print(f"  fold {fold}: AUC={metrics['auroc']:.3f} AP={metrics['auprc']:.3f}")

    oof_df = pd.DataFrame({"record_id": np.load(PROCESSED_DIR / "signals.npz", allow_pickle=True)["record_ids"],
                           "y_true": y, "prob": oof})
    oof_df.to_csv(REPORT_DIR / f"cv_oof_{config.variant}.csv", index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(REPORT_DIR / f"cv_folds_{config.variant}.csv", index=False)
    result = {
        "variant": config.variant,
        "n_splits": n_splits,
        "pooled_auroc": float(roc_auc_score(y, oof)),
        "pooled_auprc": float(average_precision_score(y, oof)),
        "mean_auroc": float(fold_df["auroc"].mean()),
        "std_auroc": float(fold_df["auroc"].std(ddof=1)),
        "mean_auprc": float(fold_df["auprc"].mean()),
        "std_auprc": float(fold_df["auprc"].std(ddof=1)),
        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "std_accuracy": float(fold_df["accuracy"].std(ddof=1)),
        "config": asdict(config),
        "folds": fold_rows,
    }
    with open(REPORT_DIR / f"cv_results_{config.variant}.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in {"folds", "config"}}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CTG delivery-mode models")
    parser.add_argument("--variant", choices=["multimodal", "signal", "bilstm", "clinical"], default="multimodal")
    parser.add_argument("--cv", action="store_true", help="Run stratified 5-fold cross-validation")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
    )
    start = time.time()
    if args.cv:
        cross_validate(config)
    else:
        train_final(config)
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
