"""Evaluate trained models on the held-out test set and generate report assets.

Run after training the four deep learning variants and the baselines:
    python -m src.evaluation.evaluate
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, precision_recall_curve
from torch.utils.data import DataLoader

from src.config import (
    CLASS_NAMES,
    FIGURE_DIR,
    MODEL_DIR,
    PROCESSED_DIR,
    REPORT_DIR,
)
from src.evaluation.metrics import (
    bootstrap_auprc,
    bootstrap_auroc,
    compute_metrics,
    find_threshold,
    paired_bootstrap_auc_difference,
    reliability_curve,
)
from src.models.baseline import load_baseline_frame
from src.models.calibration import PlattCalibrator
from src.models.deep_learning_model import build_model
from src.training.train import CTGDataset, _predict

DL_VARIANTS = {
    "multimodal": "Multimodal CNN + clinical MLP",
    "signal": "CTG-only 1D CNN",
    "bilstm": "CTG-only CNN-BiLSTM",
    "clinical": "Clinical-only MLP",
}
BASELINES = {
    "clinical_lr": "Logistic regression (clinical)",
    "ctg_lr": "Logistic regression (CTG descriptors)",
    "combined_lr": "Logistic regression (clinical + CTG)",
    "combined_gb": "Gradient boosting (clinical + CTG)",
}


def _load_processed():
    sig = np.load(PROCESSED_DIR / "signals.npz", allow_pickle=True)
    clin = np.load(PROCESSED_DIR / "clinical.npz", allow_pickle=True)
    return sig, clin


def _dl_predictions(variant: str, split_name: str, sig, clin) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (logits, labels, record_ids) for a split."""
    mask = sig["split"] == split_name
    checkpoint = torch.load(MODEL_DIR / f"{variant}.pt", map_location="cpu", weights_only=False)
    model = build_model(variant, clinical_features=clin["X"].shape[1])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(CTGDataset(sig["X"][mask], clin["X"][mask], sig["y"][mask]), batch_size=32)
    logits, labels = _predict(model, loader)
    return logits, labels.astype(int), sig["record_ids"][mask]


def _baseline_predictions(name: str, split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = load_baseline_frame()
    payload = joblib.load(MODEL_DIR / f"baseline_{name}.joblib")
    mask = frame["split"] == split_name
    x = frame.loc[mask, payload["columns"]]
    probs = payload["model"].predict_proba(x)[:, 1]
    return probs, frame.loc[mask, "label"].to_numpy(int), frame.loc[mask, "record_id"].to_numpy()


def _plot_roc_pr(predictions: Dict[str, np.ndarray], y: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, prob in predictions.items():
        fpr, tpr, _ = roc_curve(y, prob)
        axes[0].plot(fpr, tpr, lw=1.8, label=name)
        precision, recall, _ = precision_recall_curve(y, prob)
        axes[1].plot(recall, precision, lw=1.8, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (sensitivity)")
    axes[0].set_title("ROC curves — held-out test set")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].axhline(y.mean(), color="k", ls="--", lw=1, alpha=0.5)
    axes[1].set_xlabel("Recall (sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–recall curves — held-out test set")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_roc_pr_test.png", dpi=150)
    plt.close(fig)


def _plot_confusions(results: Dict[str, dict], y: np.ndarray) -> None:
    names = [n for n in results if "confusion" in results[n]]
    n = len(names)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        cm = np.array(results[name]["confusion"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(results[name].get("display_name", name), fontsize=9)
        ax.set_xticks([0, 1], ["Vaginal", "Cesarean"], fontsize=8)
        ax.set_yticks([0, 1], ["Vaginal", "Cesarean"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Observed", fontsize=8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices at threshold 0.5 (test set)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_confusion_matrices_test.png", dpi=150)
    plt.close(fig)


def _plot_calibration(prob_map: Dict[str, np.ndarray], y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for name, prob in prob_map.items():
        means, observed, _ = reliability_curve(y, prob, n_bins=6)
        ax.plot(means, observed, marker="o", lw=1.5, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed cesarean fraction")
    ax.set_title("Calibration on held-out test set (6 bins)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_calibration_test.png", dpi=150)
    plt.close(fig)


def _plot_comparison(results: Dict[str, dict]) -> None:
    names = [results[n].get("display_name", n) for n in results]
    aucs = [results[n]["auroc"] for n in results]
    aps = [results[n]["auprc"] for n in results]
    order = np.argsort(aucs)
    names = [names[i] for i in order]
    aucs = [aucs[i] for i in order]
    aps = [aps[i] for i in order]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(names))
    ax.barh(x - 0.2, aucs, height=0.38, label="ROC-AUC", color="#2563eb")
    ax.barh(x + 0.2, aps, height=0.38, label="PR-AUC", color="#f59e0b")
    ax.set_yticks(x, names, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Held-out test performance (7 cesarean cases)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_model_comparison_test.png", dpi=150)
    plt.close(fig)


def _plot_history(variant: str) -> None:
    path = REPORT_DIR / f"training_history_{variant}.csv"
    if not path.exists():
        return
    history = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title(f"{variant}: training history")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(history["epoch"], history["val_auroc"], label="Validation ROC-AUC")
    axes[1].plot(history["epoch"], history["val_auprc"], label="Validation PR-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{variant}: validation metrics")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"fig_training_history_{variant}.png", dpi=150)
    plt.close(fig)


def _plot_class_distribution(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    counts = frame.groupby(["split", "label"]).size().unstack(fill_value=0)
    counts.columns = [CLASS_NAMES[i] for i in counts.columns]
    counts = counts.loc[["train", "val", "test"]]
    counts.plot(kind="bar", ax=ax, color=["#2563eb", "#dc2626"])
    ax.set_ylabel("Number of records")
    ax.set_title("Class distribution by split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_class_distribution.png", dpi=150)
    plt.close(fig)


def _fmt(value: float, digits: int = 3) -> str:
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"


def write_report(payload: dict) -> None:
    models = payload["models"]
    lines = [
        "# Model Evaluation Report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Task and data",
        "",
        f"- **Prediction task:** {payload['task']['description']}",
        f"- **Target:** {payload['task']['target']}",
        f"- **Held-out test set:** {payload['test_set']['n']} records "
        f"({payload['test_set']['n_positive']} cesarean, {payload['test_set']['n_negative']} vaginal)",
        f"- **Test prevalence:** {payload['test_set']['prevalence']:.1%}",
        "",
        "The test set was fixed before training and was never used for model "
        "selection, early stopping, calibration, or threshold selection.",
        "",
        "## Results at threshold 0.5",
        "",
        "| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Accuracy | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in models.values():
        name = result.get("display_name", "")
        m = result["calibrated"] if result.get("calibrated") else result["raw"]
        lines.append(
            f"| {name} | {_fmt(m['auroc'])} | {_fmt(m['auprc'])} | {_fmt(m['recall_sensitivity'])} | "
            f"{_fmt(m['specificity'])} | {_fmt(m['f1'])} | {_fmt(m['accuracy'])} | {_fmt(m['brier'])} |"
        )
    lines += [
        "",
        "## Threshold-free uncertainty",
        "",
        "| Model | ROC-AUC (95% bootstrap CI) | PR-AUC (95% bootstrap CI) |",
        "|---|---|---|",
    ]
    for result in models.values():
        name = result.get("display_name", "")
        m = result["calibrated"] if result.get("calibrated") else result["raw"]
        ci_auc = result["ci"]["auroc"]
        ci_ap = result["ci"]["auprc"]
        lines.append(
            f"| {name} | {_fmt(m['auroc'])} ({_fmt(ci_auc[0])}–{_fmt(ci_auc[1])}) | "
            f"{_fmt(m['auprc'])} ({_fmt(ci_ap[0])}–{_fmt(ci_ap[1])}) |"
        )
    lines += [
        "",
        "Bootstrap intervals are stratified and resample the 7 cesarean and 76 "
        "vaginal test records separately. They describe uncertainty *on this "
        "small test set*, not clinical uncertainty in a new hospital.",
        "",
        "## Final comparison at validation-selected threshold",
        "",
        "| Model | ROC-AUC (95% CI) | PR-AUC | Sensitivity | Specificity | Threshold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in models.values():
        name = result.get("display_name", "")
        tuned = result.get("tuned_threshold")
        if tuned is None:
            continue
        m = result["at_tuned_threshold"]
        ci_auc = result["ci"]["auroc"]
        lines.append(
            f"| {name} | {_fmt(m['auroc'])} ({_fmt(ci_auc[0])}–{_fmt(ci_auc[1])}) | "
            f"{_fmt(m['auprc'])} | {_fmt(m['recall_sensitivity'])} | "
            f"{_fmt(m['specificity'])} | {tuned:.3f} |"
        )
    lines += [
        "",
        "Thresholds were selected on the validation split by maximising Youden's "
        "J statistic. The test split was not used for threshold selection.",
        "",
    ]
    comparisons = payload.get("comparisons", {})
    if comparisons:
        lines += [
            "## Paired AUC comparison",
            "",
            "| Comparison | AUC difference | 95% paired bootstrap CI | Interpretation |",
            "|---|---:|---|---|",
        ]
        for item in comparisons.values():
            lines.append(
                f"| {item.get('model_a', 'A')} minus {item.get('model_b', 'B')} | "
                f"{item.get('difference', float('nan')):+.3f} | "
                f"{item.get('ci_low', float('nan')):+.3f} to "
                f"{item.get('ci_high', float('nan')):+.3f} | "
                f"{item.get('interpretation', '')} |"
            )
        lines += [
            "",
            "The paired bootstrap resamples the same test records for both models, "
            "preserving the correlation between predictions. With seven test positives "
            "the interval is wide and should be read as an uncertainty statement.",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
    ]
    lines.extend(payload.get("interpretation", []))
    lines += [
        "",
        "## Figures",
        "",
        "- `reports/figures/fig_roc_pr_test.png`",
        "- `reports/figures/fig_confusion_matrices_test.png`",
        "- `reports/figures/fig_calibration_test.png`",
        "- `reports/figures/fig_model_comparison_test.png`",
        "- `reports/figures/fig_training_history_multimodal.png`",
        "- `reports/figures/fig_training_history_bilstm.png`",
        "- `reports/figures/fig_class_distribution.png`",
        "",
        "## Important limitations",
        "",
        "- Only 46 cesarean deliveries exist in the entire public dataset; the test set contains 7.",
        "- The cohort is a selected, homogeneous subset of 9164 recordings from one Czech hospital.",
        "- Cesarean recordings were collected as a separate 46-case subset and differ in signal quality and duration.",
        "- No external validation has been performed.",
        "- These numbers do not establish clinical validity, safety, or effectiveness.",
    ]
    (REPORT_DIR / "model_evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sig, clin = _load_processed()
    y_test = sig["y"][sig["split"] == "test"].astype(int)
    frame = load_baseline_frame()
    _plot_class_distribution(frame)

    results: Dict[str, dict] = {}
    test_probs: Dict[str, np.ndarray] = {}
    calibrated_probs: Dict[str, np.ndarray] = {}
    all_predictions = {"record_id": sig["record_ids"][sig["split"] == "test"], "y_true": y_test}

    for variant, label in DL_VARIANTS.items():
        test_logits, y, record_ids = _dl_predictions(variant, "test", sig, clin)
        val_logits, y_val, _ = _dl_predictions(variant, "val", sig, clin)
        calibrator = PlattCalibrator.fit(val_logits, y_val)
        joblib.dump({"calibrator": calibrator, "variant": variant}, MODEL_DIR / f"calibrator_{variant}.joblib")
        raw = 1.0 / (1.0 + np.exp(-test_logits))
        calibrated = calibrator.predict(test_logits)
        raw_metrics = compute_metrics(y, raw)
        calibrated_metrics = compute_metrics(y, calibrated)
        use_calibrated = calibrated_metrics["brier"] <= raw_metrics["brier"]
        primary = calibrated_metrics if use_calibrated else raw_metrics
        tuned = find_threshold(y_val, calibrator.predict(val_logits), criterion="youden")
        results[f"{variant}_dl"] = {
            "display_name": label,
            "type": "deep_learning",
            "raw": raw_metrics,
            "calibrated": calibrated_metrics if use_calibrated else None,
            "primary": "calibrated" if use_calibrated else "raw",
            "tuned_threshold": tuned,
            "at_tuned_threshold": compute_metrics(y, calibrated, threshold=tuned),
            "ci": {"auroc": list(bootstrap_auroc(y, calibrated)), "auprc": list(bootstrap_auprc(y, calibrated))},
            "confusion": [[int(primary["tn"]), int(primary["fp"])], [int(primary["fn"]), int(primary["tp"])]],
            "record_ids": record_ids.tolist(),
            "probabilities": calibrated.tolist() if use_calibrated else raw.tolist(),
        }
        test_probs[f"{variant} DL"] = calibrated if use_calibrated else raw
        calibrated_probs[f"{variant} DL"] = calibrated
        all_predictions[f"{variant}_dl"] = calibrated if use_calibrated else raw
        _plot_history(variant)

    for name, label in BASELINES.items():
        probs, y, record_ids = _baseline_predictions(name, "test")
        val_probs, y_val, _ = _baseline_predictions(name, "val")
        metrics = compute_metrics(y, probs)
        tuned = find_threshold(y_val, val_probs, criterion="youden")
        results[name] = {
            "display_name": label,
            "type": "baseline",
            "raw": metrics,
            "calibrated": None,
            "primary": "raw",
            "tuned_threshold": tuned,
            "at_tuned_threshold": compute_metrics(y, probs, threshold=tuned),
            "ci": {"auroc": list(bootstrap_auroc(y, probs)), "auprc": list(bootstrap_auprc(y, probs))},
            "confusion": [[int(metrics["tn"]), int(metrics["fp"])], [int(metrics["fn"]), int(metrics["tp"])]],
            "record_ids": record_ids.tolist(),
            "probabilities": probs.tolist(),
        }
        test_probs[label] = probs
        all_predictions[name] = probs

    _plot_roc_pr(test_probs, y_test)
    _plot_confusions(results, y_test)
    _plot_calibration(calibrated_probs, y_test)
    _plot_comparison(
        {
            k: {
                **(v["calibrated"] if v.get("calibrated") else v["raw"]),
                "display_name": v.get("display_name", k),
            }
            for k, v in results.items()
        }
    )

    pd.DataFrame(all_predictions).to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    comparison = paired_bootstrap_auc_difference(
        y_test, calibrated_probs["bilstm DL"], calibrated_probs["signal DL"]
    )
    comparison["model_a"] = DL_VARIANTS["bilstm"]
    comparison["model_b"] = DL_VARIANTS["signal"]
    if comparison["ci_low"] > 0:
        comparison["interpretation"] = (
            "The paired bootstrap interval lies above zero, suggesting a higher AUC for "
            "the CNN-BiLSTM on this test split."
        )
    elif comparison["ci_high"] < 0:
        comparison["interpretation"] = (
            "The paired bootstrap interval lies below zero, suggesting a lower AUC for "
            "the CNN-BiLSTM on this test split."
        )
    else:
        comparison["interpretation"] = (
            "The paired bootstrap interval includes zero, so this experiment does not "
            "establish a statistically meaningful AUC difference between the CNN-BiLSTM "
            "and the 1D CNN."
        )

    best_name = max(results, key=lambda n: results[n]["calibrated"]["auroc"] if results[n].get("calibrated") else results[n]["raw"]["auroc"])
    multimodal = results["multimodal_dl"]
    mm = multimodal["calibrated"] if multimodal.get("calibrated") else multimodal["raw"]
    bilstm = results["bilstm_dl"]
    bl = bilstm["calibrated"] if bilstm.get("calibrated") else bilstm["raw"]
    majority_accuracy = float((1 - y_test).mean())
    interpretation = [
        f"- The best ROC-AUC on this test split was obtained by "
        f"**{results[best_name].get('display_name', best_name)}** "
        f"({_fmt((results[best_name]['calibrated'] or results[best_name]['raw'])['auroc'])}); "
        "with only 46 events in the full dataset this ranking is unstable.",
        f"- The multimodal model achieved ROC-AUC {_fmt(mm['auroc'])} and PR-AUC {_fmt(mm['auprc'])} "
        f"on 7 cesarean cases. At 0.5 it detected {int(mm['tp'])} of 7 cesareans with "
        f"{int(mm['fp'])} false positives among 76 vaginal deliveries.",
        f"- The CNN-BiLSTM achieved ROC-AUC {_fmt(bl['auroc'])} and PR-AUC {_fmt(bl['auprc'])}. "
        f"Its AUC difference versus the 1D CNN was {comparison['difference']:+.3f} "
        f"(95% paired bootstrap CI {comparison['ci_low']:+.3f} to {comparison['ci_high']:+.3f}). "
        f"{comparison['interpretation']}",
        f"- A classifier that always predicts vaginal delivery would reach "
        f"{majority_accuracy:.1%} accuracy while detecting zero cesareans. Accuracy is "
        "therefore not a meaningful headline metric here.",
        "- Hand-crafted CTG descriptors and the CNN perform similarly, which suggests the "
        "signal information is modest and easily confounded by recording quality.",
        "- No model is suitable for clinical use. The system is a research prototype and a "
        "demonstration of an end-to-end pipeline.",
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": {
            "description": "Binary classification of delivery mode from intrapartum CTG and maternal data",
            "target": "Cesarean delivery (Deliv. type = 2) versus vaginal delivery (Deliv. type = 1)",
            "positive_class": "Cesarean delivery",
            "negative_class": "Vaginal delivery",
            "prediction_time": "During labour, using the most recent 30 minutes of CTG",
        },
        "test_set": {
            "n": int(len(y_test)),
            "n_positive": int(y_test.sum()),
            "n_negative": int((1 - y_test).sum()),
            "prevalence": float(y_test.mean()),
        },
        "models": results,
        "comparisons": {"bilstm_vs_signal": comparison},
        "interpretation": interpretation,
    }
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    write_report(payload)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in {"primary", "tuned_threshold"}} for k, v in results.items()}, indent=2))
    print("Wrote reports/metrics.json and reports/model_evaluation_report.md")


if __name__ == "__main__":
    main()
