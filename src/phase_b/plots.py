"""Figures for the Phase B imbalance study."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import confusion_matrix


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_roc_pr(
    curves: Dict[str, tuple],
    directory: Path,
    filename: str,
    title: str,
) -> None:
    _style()
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, (y_true, prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        precision, recall, _ = precision_recall_curve(y_true, prob)
        axes[0].plot(fpr, tpr, linewidth=1.6, label=label)
        axes[1].plot(recall, precision, linewidth=1.6, label=label)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC")
    axes[1].set_xlabel("Recall (cesarean sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–recall")
    for ax in axes:
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(directory / filename, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(
    curves: Dict[str, tuple],
    directory: Path,
    filename: str,
    title: str,
    n_bins: int = 8,
) -> None:
    _style()
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    for label, (y_true, prob) in curves.items():
        try:
            fraction, mean_pred = calibration_curve(
                y_true, prob, n_bins=n_bins, strategy="quantile"
            )
        except ValueError:
            continue
        ax.plot(mean_pred, fraction, marker="o", linewidth=1.4, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed cesarean fraction")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(directory / filename, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    matrix: Dict[str, int],
    directory: Path,
    filename: str,
    title: str,
) -> None:
    _style()
    directory.mkdir(parents=True, exist_ok=True)
    values = np.array(
        [[matrix.get("tn", 0), matrix.get("fp", 0)], [matrix.get("fn", 0), matrix.get("tp", 0)]],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(values, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], ["Vaginal", "Cesarean"])
    ax.set_yticks([0, 1], ["Vaginal", "Cesarean"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(directory / filename, bbox_inches="tight")
    plt.close(fig)


def plot_fold_metrics(
    fold_table: pd.DataFrame,
    directory: Path,
    filename: str,
    title: str,
) -> None:
    _style()
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(fold_table))
    for ax, metric, label in (
        (axes[0], "auprc", "PR-AUC"),
        (axes[1], "recall_sensitivity", "Cesarean sensitivity"),
    ):
        ax.bar(x, fold_table[metric], color="#2a6f97", alpha=0.85)
        ax.set_xticks(x, [f"F{int(v)}" for v in fold_table["fold"]])
        ax.set_ylim(0, 1)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by outer fold")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(directory / filename, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_comparison(
    table: pd.DataFrame,
    directory: Path,
    filename: str,
    metric: str,
    title: str,
) -> None:
    _style()
    directory.mkdir(parents=True, exist_ok=True)
    if table.empty:
        return
    pivot = table.pivot(index="variant", columns="strategy", values=metric)
    fig, ax = plt.subplots(figsize=(10, max(3.2, 0.55 * len(pivot) + 2)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(directory / filename, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(
    table: pd.DataFrame,
    directory: Path,
    filename: str,
    metrics: Sequence[str],
    title: str,
) -> None:
    _style()
    directory.mkdir(parents=True, exist_ok=True)
    if table.empty:
        return
    labels = table["label"].tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels) + 3), 4.5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, table[metric], width=width, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2, labels, rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / filename, bbox_inches="tight")
    plt.close(fig)
