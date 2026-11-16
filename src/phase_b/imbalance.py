"""Imbalance utilities: class weights, focal loss, sampling, SMOTE, augmentation.

Every resampling operation in this module is a pure function of the training
arrays it receives.  Nothing here can see a validation fold or the held-out
test split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import WeightedRandomSampler


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def class_counts(y: np.ndarray) -> Tuple[int, int]:
    y = np.asarray(y).astype(int).reshape(-1)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must be binary 0/1")
    counts = np.bincount(y, minlength=2)
    return int(counts[0]), int(counts[1])


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Balanced class weights from the supplied (training) labels only."""
    n_negative, n_positive = class_counts(y)
    n = n_negative + n_positive
    if n == 0:
        raise ValueError("Cannot compute class weights for an empty array")
    weights = {
        0: float(n / (2.0 * n_negative)) if n_negative else 1.0,
        1: float(n / (2.0 * n_positive)) if n_positive else 1.0,
    }
    return weights


def positive_weight(y: np.ndarray) -> float:
    n_negative, n_positive = class_counts(y)
    return float(n_negative / max(n_positive, 1))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Binary focal loss with class-frequency alpha.

    ``alpha`` is derived from the labels supplied at construction time.  With
    ``gamma=0`` and training-fold alpha this equals a class-weighted BCE up to
    the usual mean/sum reduction convention.
    """

    def __init__(
        self,
        y_train: np.ndarray | None = None,
        gamma: float = 1.0,
        alpha: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean, sum or none")
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is None:
            if y_train is None:
                raise ValueError("Either y_train or alpha must be provided")
            n_negative, n_positive = class_counts(y_train)
            n = n_negative + n_positive
            alpha = float(n_negative / n) if n else 0.5
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(strategy: str, y_train: np.ndarray) -> nn.Module:
    """Map a Phase B strategy name to its training loss."""
    if strategy in ("plain_bce", "weighted_sampling"):
        return nn.BCEWithLogitsLoss()
    if strategy == "weighted_bce":
        weight = positive_weight(y_train)
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([weight], dtype=torch.float32)
        )
    if strategy == "focal_gamma1":
        return FocalLoss(y_train=y_train, gamma=1.0)
    if strategy == "focal_gamma2":
        return FocalLoss(y_train=y_train, gamma=2.0)
    raise ValueError(f"Unknown deep imbalance strategy: {strategy}")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sampler_weights(y_train: np.ndarray) -> np.ndarray:
    """Per-sample weights whose inverse-frequency expectation balances classes."""
    y_train = np.asarray(y_train).astype(int)
    weights = compute_class_weights(y_train)
    return np.array([weights[int(label)] for label in y_train], dtype=np.float64)


def make_weighted_sampler(
    y_train: np.ndarray,
    seed: int,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Build a sampler over *training* indices only.

    The sampler returns indices; it never adds samples to the dataset, so a
    validation DataLoader built from the original dataset remains untouched.
    """
    y_train = np.asarray(y_train).astype(int)
    weights = torch.as_tensor(sampler_weights(y_train), dtype=torch.double)
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=int(num_samples or len(y_train)),
        replacement=True,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Tabular resampling (classical experiments only)
# ---------------------------------------------------------------------------
def random_oversample(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random minority oversampling with replacement on tabular data."""
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    n_negative, n_positive = class_counts(y)
    if n_negative == 0 or n_positive == 0 or n_negative == n_positive:
        return X.copy(), y.copy()
    minority = 1 if n_positive < n_negative else 0
    target = max(n_negative, n_positive)
    minority_idx = np.flatnonzero(y == minority)
    extra = target - len(minority_idx)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(minority_idx, size=extra, replace=True)
    idx = np.concatenate([np.arange(len(y)), sampled])
    rng.shuffle(idx)
    return X[idx].astype(np.float32), y[idx].astype(int)


def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal tabular SMOTE implementation, applied to training rows only.

    The minority class is oversampled to parity.  Each synthetic point is an
    interpolation between a real minority point and one of its minority-class
    nearest neighbours.  Raw CTG tensors must never be passed to this function.
    Real rows keep their original order and synthetic rows are appended, unless
    ``shuffle`` is requested.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if X.ndim != 2:
        raise ValueError("SMOTE expects a 2-D tabular matrix")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    n_negative, n_positive = class_counts(y)
    if n_negative == 0 or n_positive == 0 or n_negative == n_positive:
        return X.astype(np.float32), y.copy()
    minority = 1 if n_positive < n_negative else 0
    minority_idx = np.flatnonzero(y == minority)
    n_synthetic = abs(n_negative - n_positive)
    k = min(int(k_neighbors), len(minority_idx) - 1)
    if k < 1:
        return X.astype(np.float32), y.copy()

    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(X[minority_idx])
    _, nearest = neighbors.kneighbors(X[minority_idx])
    # Column 0 is the point itself.
    nearest = nearest[:, 1:]
    rng = np.random.default_rng(seed)
    base_choice = rng.integers(0, len(minority_idx), size=n_synthetic)
    neighbor_choice = rng.integers(0, k, size=n_synthetic)
    gap = rng.random(size=(n_synthetic, 1))
    base = X[minority_idx[base_choice]]
    neighbor = X[minority_idx[nearest[base_choice, neighbor_choice]]]
    synthetic = base + gap * (neighbor - base)
    X_out = np.vstack([X, synthetic]).astype(np.float32)
    y_out = np.concatenate([y, np.full(n_synthetic, minority, dtype=int)])
    if not shuffle:
        return X_out, y_out
    order = rng.permutation(len(y_out))
    return X_out[order], y_out[order]


# ---------------------------------------------------------------------------
# Controlled CTG augmentation (exploratory only)
# ---------------------------------------------------------------------------
def augment_batch(signal: torch.Tensor, mode: str, generator: torch.Generator) -> torch.Tensor:
    """Apply a controlled augmentation to a batch of ``(B, 4, T)`` tensors.

    Only the two physiological channels are perturbed; masks remain 0/1.  The
    time shift is small (<= 30 seconds at 1 Hz) and zero-fills the vacated
    samples, so it does not invent signal beyond the window.
    """
    if mode == "none":
        return signal
    if mode not in ("gaussian", "mix"):
        raise ValueError(f"Unknown augmentation mode: {mode}")
    out = signal.clone()
    noise = torch.randn(
        (out.shape[0], 2, out.shape[2]), generator=generator, dtype=out.dtype, device=out.device
    ) * 0.03
    out[:, :2, :] = out[:, :2, :] + noise
    if mode == "mix":
        scale = 1.0 + (torch.rand(out.shape[0], 1, 1, generator=generator) - 0.5) * 0.10
        out[:, :2, :] = out[:, :2, :] * scale
        shift = int(torch.randint(-30, 31, (1,), generator=generator).item())
        if shift != 0:
            shifted = torch.zeros_like(out)
            if shift > 0:
                shifted[:, :, shift:] = out[:, :, :-shift]
            else:
                shifted[:, :, :shift] = out[:, :, -shift:]
            out = shifted
    return out


@dataclass(frozen=True)
class ImbalanceSpec:
    strategy: str
    loss: str
    sampler: str
    notes: str


SPECS = {
    "plain_bce": ImbalanceSpec("plain_bce", "BCEWithLogitsLoss", "shuffle", "Neutral reference"),
    "weighted_bce": ImbalanceSpec(
        "weighted_bce", "BCEWithLogitsLoss(pos_weight=n_neg/n_pos)", "shuffle", "Production loss"
    ),
    "focal_gamma1": ImbalanceSpec(
        "focal_gamma1", "FocalLoss(gamma=1, alpha=n_neg/n)", "shuffle", "Focal, moderate"
    ),
    "focal_gamma2": ImbalanceSpec(
        "focal_gamma2", "FocalLoss(gamma=2, alpha=n_neg/n)", "shuffle", "Focal, aggressive"
    ),
    "weighted_sampling": ImbalanceSpec(
        "weighted_sampling",
        "BCEWithLogitsLoss",
        "WeightedRandomSampler(inverse class frequency)",
        "Sampling, no loss reweighting",
    ),
}
