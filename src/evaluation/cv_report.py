"""Aggregate five-fold cross-validation results for all model families.

Deep-learning folds are produced by:
    python -m src.training.train --variant <v> --cv

This script adds cross-validated baselines, pools out-of-fold predictions, and
writes reports/cv_summary.json / reports/cv_summary.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import FIGURE_DIR, MODEL_DIR, PROCESSED_DIR, REPORT_DIR, RANDOM_SEED
from src.evaluation.metrics import compute_metrics
from src.models.baseline import build_baselines, load_baseline_frame

DL_VARIANTS = ["multimodal", "signal", "bilstm", "clinical"]
DL_LABELS = {
    "multimodal": "Multimodal CNN + clinical MLP",
    "signal": "CTG-only 1D CNN",
    "bilstm": "CTG-only CNN-BiLSTM",
    "clinical": "Clinical-only MLP",
}
BASELINE_ORDER = ["clinical_lr", "ctg_lr", "combined_lr", "combined_gb"]
BASELINE_LABELS = {
    "clinical_lr": "Logistic regression (clinical)",
    "ctg_lr": "Logistic regression (CTG descriptors)",
    "combined_lr": "Logistic regression (clinical + CTG)",
    "combined_gb": "Gradient boosting (clinical + CTG)",
}


def cv_baselines() -> dict:
    frame = load_baseline_frame()
    models, feature_sets = build_baselines()
    y = frame["label"].to_numpy(int)
    rows = []
    oof_map = {}
    for name in BASELINE_ORDER:
        cols = feature_sets[name]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        prob = cross_val_predict(models[name], frame[cols], y, cv=cv, method="predict_proba")[:, 1]
        oof_map[name] = prob
        fold_metrics = []
        for _, val_idx in cv.split(frame[cols], y):
            m = compute_metrics(y[val_idx], prob[val_idx])
            fold_metrics.append(m)
        rows.append(
            {
                "model": name,
                "pooled_auroc": compute_metrics(y, prob)["auroc"],
                "pooled_auprc": compute_metrics(y, prob)["auprc"],
                "mean_auroc": float(np.mean([m["auroc"] for m in fold_metrics])),
                "std_auroc": float(np.std([m["auroc"] for m in fold_metrics], ddof=1)),
                "mean_auprc": float(np.mean([m["auprc"] for m in fold_metrics])),
                "std_auprc": float(np.std([m["auprc"] for m in fold_metrics], ddof=1)),
            }
        )
        pd.DataFrame({"record_id": frame["record_id"], "y_true": y, "prob": prob}).to_csv(
            REPORT_DIR / f"cv_oof_baseline_{name}.csv", index=False
        )
    return {"folds": rows, "oof": oof_map}


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"deep_learning": {}, "baselines": {}}
    oof_curves = {}

    for variant in DL_VARIANTS:
        path = REPORT_DIR / f"cv_results_{variant}.json"
        if not path.exists():
            print(f"Missing {path}; run training with --cv")
            continue
        result = json.loads(path.read_text())
        oof_path = REPORT_DIR / f"cv_oof_{variant}.csv"
        oof = pd.read_csv(oof_path)
        summary["deep_learning"][variant] = {
            "pooled_auroc": result["pooled_auroc"],
            "pooled_auprc": result["pooled_auprc"],
            "mean_auroc": result["mean_auroc"],
            "std_auroc": result["std_auroc"],
            "mean_auprc": result["mean_auprc"],
            "std_auprc": result["std_auprc"],
            "mean_accuracy": result["mean_accuracy"],
            "std_accuracy": result["std_accuracy"],
            "folds": result["folds"],
        }
        oof_curves[DL_LABELS[variant]] = (oof["y_true"].to_numpy(), oof["prob"].to_numpy())

    baselines = cv_baselines()
    for row in baselines["folds"]:
        summary["baselines"][row["model"]] = {k: v for k, v in row.items() if k != "model"}
    for name, prob in baselines["oof"].items():
        frame = load_baseline_frame()
        oof_curves[name] = (frame["label"].to_numpy(), prob)

    with open(REPORT_DIR / "cv_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, (y, prob) in oof_curves.items():
        fpr, tpr, _ = roc_curve(y, prob)
        axes[0].plot(fpr, tpr, lw=1.6, label=name)
        precision, recall, _ = precision_recall_curve(y, prob)
        axes[1].plot(recall, precision, lw=1.6, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("5-fold cross-validation — pooled out-of-fold ROC")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("5-fold cross-validation — pooled out-of-fold PR")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_cv_roc_pr.png", dpi=150)
    plt.close(fig)

    lines = [
        "# Five-Fold Cross-Validation Summary",
        "",
        "Stratified 5-fold cross-validation on all 552 records. Pooled metrics use",
        "out-of-fold predictions. Fold mean ± SD shows how much the result depends on",
        "the particular split. These are exploratory estimates; the fixed held-out",
        "test set in `model_evaluation_report.md` is the primary untouched test.",
        "",
        "| Model | Pooled ROC-AUC | Fold mean AUC ± SD | Pooled PR-AUC | Fold mean PR-AUC ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, r in summary["deep_learning"].items():
        lines.append(
            f"| {DL_LABELS[variant]} | {r['pooled_auroc']:.3f} | {r['mean_auroc']:.3f} ± {r['std_auroc']:.3f} | "
            f"{r['pooled_auprc']:.3f} | {r['mean_auprc']:.3f} ± {r['std_auprc']:.3f} |"
        )
    for name, r in summary["baselines"].items():
        lines.append(
            f"| {BASELINE_LABELS.get(name, name)} | {r['pooled_auroc']:.3f} | {r['mean_auroc']:.3f} ± {r['std_auroc']:.3f} | "
            f"{r['pooled_auprc']:.3f} | {r['mean_auprc']:.3f} ± {r['std_auprc']:.3f} |"
        )
    lines += [
        "",
        "See `reports/figures/fig_cv_roc_pr.png`.",
        "",
        "Cross-validation reuses every record once as validation data, so it is not a",
        "replacement for external validation. The confidence intervals in the held-out",
        "test report remain wide because only 46 cesarean deliveries exist.",
    ]
    (REPORT_DIR / "cv_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in summary.items()}, indent=2)[:2000])


if __name__ == "__main__":
    main()
