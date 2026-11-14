"""Figures for the Phase A report."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

COLORS = ["#0f766e", "#b45309", "#1d4ed8", "#be123c", "#4d7c0f", "#7c3aed", "#0891b2"]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_quality(report: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for label, color, name in ((0, "#0f766e", "Vaginal"), (1, "#b45309", "Cesarean")):
        subset = report[report["label"] == label]
        axes[0].hist(subset["trimmed_duration_min"], bins=24, alpha=0.65, color=color, label=name)
    axes[0].set_xlabel("Trimmed recording duration (min)")
    axes[0].set_ylabel("Records")
    axes[0].set_title("Recording duration by outcome")
    axes[0].legend()

    data = [
        report.loc[report["label"] == 0, "fhr_missing_fraction_30m"],
        report.loc[report["label"] == 1, "fhr_missing_fraction_30m"],
        report.loc[report["label"] == 0, "uc_missing_fraction_30m"],
        report.loc[report["label"] == 1, "uc_missing_fraction_30m"],
    ]
    axes[1].boxplot(data, labels=["FHR\nvaginal", "FHR\ncesarean", "UC\nvaginal", "UC\ncesarean"])
    axes[1].set_ylabel("Missing fraction in 30-min window")
    axes[1].set_title("Missingness by outcome")
    axes[1].tick_params(axis="x", labelsize=8)

    counts = report.groupby(["quality_category", "label"]).size().unstack(fill_value=0)
    counts = counts.reindex(["good", "acceptable", "poor"]).fillna(0)
    bottom = np.zeros(len(counts))
    for label, color, name in ((0, "#0f766e", "Vaginal"), (1, "#b45309", "Cesarean")):
        values = counts[label].to_numpy() if label in counts.columns else np.zeros(len(counts))
        axes[2].bar(counts.index, values, bottom=bottom, color=color, label=name)
        bottom += values
    axes[2].set_ylabel("Records")
    axes[2].set_title("Descriptive quality categories")
    axes[2].legend()
    fig.suptitle("Phase A — CTG data-quality audit (all 552 records)", y=1.02)
    _save(fig, directory / "fig_quality_overview.png")


def plot_comparison(
    table: pd.DataFrame,
    directory: Path,
    filename: str,
    title: str,
    label_col: str = "configuration",
    metric: str = "auc_mean",
    error: str = "auc_std",
) -> None:
    if table.empty:
        return
    directory.mkdir(parents=True, exist_ok=True)
    table = table.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(table)), 4.4))
    values = table[metric].to_numpy(float)
    errors = table[error].fillna(0).to_numpy(float) if error in table else np.zeros(len(table))
    ax.barh(table[label_col].astype(str), values, xerr=errors, color=COLORS[0], alpha=0.85)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Pooled out-of-fold ROC-AUC (fold mean ± SD shown as error)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, directory / filename)


def plot_roc_pr(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    directory: Path,
    filename: str,
    title: str,
) -> None:
    if not curves:
        return
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, (name, (y_true, prob)) in enumerate(curves.items()):
        color = COLORS[i % len(COLORS)]
        fpr, tpr, _ = roc_curve(y_true, prob)
        precision, recall, _ = precision_recall_curve(y_true, prob)
        axes[0].plot(fpr, tpr, color=color, lw=1.6, label=name)
        axes[1].plot(recall, precision, color=color, lw=1.6, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("Pooled out-of-fold ROC")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Pooled out-of-fold precision-recall")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle(title)
    _save(fig, directory / filename)


def plot_confusion(matrix: Dict[str, int], directory: Path, filename: str, title: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    values = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(values, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Pred vaginal", "Pred cesarean"])
    ax.set_yticks([0, 1], ["True vaginal", "True cesarean"])
    ax.set_title(title)
    _save(fig, directory / filename)


def plot_selection_stability(stability: pd.DataFrame, directory: Path, filename: str, method: str) -> None:
    subset = stability[stability["method"] == method]
    if subset.empty:
        return
    directory.mkdir(parents=True, exist_ok=True)
    subset = subset.sort_values("selection_frequency", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(subset["feature"], subset["selection_frequency"], color=COLORS[0], alpha=0.85)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Selection frequency across 5 folds")
    ax.set_title(f"Feature-selection stability — {method}")
    ax.invert_yaxis()
    _save(fig, directory / filename)
