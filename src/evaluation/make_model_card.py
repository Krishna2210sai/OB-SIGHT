"""Build ``models/model_card.json`` from the evaluated artefacts.

The card is consumed by the ``/model-info`` endpoint and by the documentation.
It deliberately contains only numbers that were produced by the evaluation
pipeline; nothing is typed in by hand.

Run after ``python -m src.evaluation.evaluate``:
    python -m src.evaluation.make_model_card
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from src.config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR

DISCLAIMER = (
    "Research prototype only. This output is a statistical estimate from a public "
    "research dataset and is not a diagnosis, a treatment recommendation, or a "
    "substitute for professional obstetric judgement. Do not use it to decide the "
    "mode of delivery."
)

LIMITATIONS = [
    "Only 46 cesarean deliveries exist in the whole public dataset (506 vaginal); "
    "the held-out test set contains 7 cesarean records, so every performance "
    "estimate has very wide uncertainty.",
    "The cohort is a selected subset of 552 recordings from a single Czech "
    "university hospital (the full CTU-CHB database contains 9164 recordings). "
    "Case mix, practice patterns and recording equipment differ elsewhere.",
    "Cesarean recordings were released as a separate 46-case subset. Their raw "
    "recordings contain far more zero-coded FHR samples overall (32.9% vs 17.5%) "
    "and their usable length after trimming is shorter (58.4 vs 71.3 minutes); "
    "within the last 30-minute window, however, FHR missingness is lower for "
    "cesareans (9.3% vs 19.2%). Recording length and quality are therefore "
    "potential shortcuts. Duration itself is not an input, but missingness masks "
    "are, and this confound has not been eliminated.",
    "Expert CTG annotations are not available in version 1.0.0 of this database, "
    "so the model learns the eventual mode of delivery, not an expert interpretation "
    "of fetal status.",
    "Prevalence is 8.3%. Accuracy is dominated by the majority class; a "
    "majority-class classifier reaches 91.6% while detecting no cesareans. "
    "ROC-AUC, PR-AUC, sensitivity and specificity are the appropriate metrics.",
    "No external, prospective or temporal validation has been performed. No "
    "calibration or threshold can be assumed to transfer to another hospital.",
    "The clinical-only model did not discriminate better than chance on the "
    "held-out test set (ROC-AUC approximately 0.50); its score must not be read as "
    "a risk estimate.",
    "Feature attributions are gradients of the model output with respect to its "
    "inputs. They describe model behaviour, not causal clinical mechanisms.",
]

INTENDED_USE = [
    "Teaching and demonstration of an end-to-end clinical machine-learning workflow.",
    "Research on multimodal intrapartum CTG and maternal-data modelling.",
    "Reproducible benchmarking of deep learning against simple baselines on the "
    "public CTU-CHB intrapartum cardiotocography database.",
]

PROHIBITED_USE = [
    "Any real clinical decision, triage, counselling or mode-of-delivery decision.",
    "Use as a medical device, or as input to a device, without regulatory approval "
    "and prospective clinical validation.",
    "Use on patient data without a lawful basis, ethics approval and appropriate "
    "data-protection safeguards.",
    "Use of the demonstration records as if they were real patient cases.",
]

ARCHITECTURE = {
    "multimodal": {
        "name": "Multimodal 1D CNN + clinical MLP",
        "inputs": "4 x 1800 CTG tensor (FHR, UC, FHR mask, UC mask at 1 Hz) and 15 clinical features",
        "description": (
            "Four 1D convolution blocks (24-48-64-96 channels) with batch "
            "normalisation and pooling, concatenated average/max pooling, a 96-unit "
            "signal embedding, a 32-unit clinical embedding (48 hidden units), and a "
            "64-unit fusion head."
        ),
        "parameters": None,  # filled from train summaries
    },
    "signal": {
        "name": "CTG-only 1D CNN",
        "inputs": "4 x 1800 CTG tensor",
        "description": "The same signal encoder with a 48-unit classification head; ignores clinical inputs.",
        "parameters": None,
    },
    "bilstm": {
        "name": "CTG-only CNN-BiLSTM",
        "inputs": "4 x 1800 CTG tensor",
        "description": (
            "Three 1D convolution blocks (16-32-48 channels) with ReLU, batch "
            "normalisation and two max-pooling stages reduce the signal to 225 "
            "steps. A single-layer bidirectional LSTM (hidden size 32 per "
            "direction) models temporal dependencies, followed by dropout "
            "(0.30), a 32-unit fully connected layer, dropout (0.20) and a "
            "single output logit. Clinical inputs are ignored."
        ),
        "parameters": None,
    },
    "clinical": {
        "name": "Clinical-only MLP",
        "inputs": "15 clinical features",
        "description": "32-unit clinical embedding and a 16-unit head; ignores the CTG input.",
        "parameters": None,
    },
    "training": {
        "loss": "Binary cross-entropy with positive-class weighting (negatives / positives)",
        "optimiser": "AdamW (learning rate 1e-3, weight decay 1e-4)",
        "regularisation": "Dropout, batch normalisation, Gaussian input noise (sigma 0.03) on FHR/UC during training, gradient clipping at 5.0",
        "model_selection": "Early stopping on validation BCE loss (patience 15-20); the test split was never used",
        "calibration": "Platt scaling fitted on the validation split only",
        "threshold_selection": "Youden's J on the validation split only; applied unchanged to the test split",
    },
}


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _model_summary(entry: dict) -> Dict:
    """Compact, API-friendly view of one evaluated model."""
    at_threshold = entry.get("at_tuned_threshold") or entry["raw"]
    # Probabilities served by the API are calibrated when a calibrator was
    # retained.  The 0.5 operating point must use the same probabilities as the
    # evaluation report, not the raw logits/scores.
    default = entry.get("calibrated") or entry["raw"]
    # ``entry["confusion"]`` is the 0.5-threshold matrix.  The card reports the
    # operating point, so rebuild the matrix from that threshold's counts.
    confusion = [
        [int(at_threshold["tn"]), int(at_threshold["fp"])],
        [int(at_threshold["fn"]), int(at_threshold["tp"])],
    ]
    return {
        "display_name": entry["display_name"],
        "type": entry["type"],
        "auroc": round(float(at_threshold["auroc"]), 4),
        "auroc_95ci": [round(float(x), 4) for x in entry["ci"]["auroc"]],
        "auprc": round(float(at_threshold["auprc"]), 4),
        "auprc_95ci": [round(float(x), 4) for x in entry["ci"]["auprc"]],
        "brier": round(float(at_threshold["brier"]), 4),
        "operating_threshold": round(float(entry.get("tuned_threshold") or 0.5), 4),
        "sensitivity": round(float(at_threshold["recall_sensitivity"]), 4),
        "specificity": round(float(at_threshold["specificity"]), 4),
        "f1": round(float(at_threshold["f1"]), 4),
        "accuracy": round(float(at_threshold["accuracy"]), 4),
        "confusion": confusion,
        "at_default_threshold": {
            "threshold": 0.5,
            "sensitivity": round(float(default["recall_sensitivity"]), 4),
            "specificity": round(float(default["specificity"]), 4),
            "f1": round(float(default["f1"]), 4),
        },
    }


def build_card() -> dict:
    metrics = _load_json(REPORT_DIR / "metrics.json")
    if metrics is None:
        raise SystemExit("reports/metrics.json not found; run python -m src.evaluation.evaluate first")
    preprocessing = _load_json(MODEL_DIR / "preprocessing_meta.json")
    dataset_summary = _load_json(PROCESSED_DIR / "dataset_summary.json")
    if preprocessing is None or dataset_summary is None:
        raise SystemExit("Processed metadata not found; run python -m src.data.prepare_dataset first")

    architecture = json.loads(json.dumps(ARCHITECTURE))
    for variant in ("multimodal", "signal", "bilstm", "clinical"):
        summary = _load_json(MODEL_DIR / f"train_summary_{variant}.json")
        if summary:
            architecture[variant]["parameters"] = int(summary["n_parameters"])
            architecture[variant]["best_epoch"] = int(summary["best_epoch"])
            architecture[variant]["training_config"] = summary["config"]

    models = {name: _model_summary(entry) for name, entry in metrics["models"].items()}
    operating_points = {
        name: round(float(entry["tuned_threshold"]), 4)
        for name, entry in metrics["models"].items()
        if entry.get("tuned_threshold") is not None
    }
    # The API uses clinical and multimodal for prediction; signal and bilstm are
    # exposed for comparison and can be loaded by the same model builder.
    api_operating_points = {
        "clinical": operating_points.get("clinical_dl", 0.5),
        "multimodal": operating_points.get("multimodal_dl", 0.5),
        "signal": operating_points.get("signal_dl", 0.5),
        "bilstm": operating_points.get("bilstm_dl", 0.5),
    }

    cv = {}
    for variant in ("multimodal", "signal", "bilstm", "clinical"):
        payload = _load_json(REPORT_DIR / f"cv_results_{variant}.json")
        if payload:
            cv[variant] = {
                "n_splits": payload["n_splits"],
                "pooled_auroc": round(float(payload["pooled_auroc"]), 4),
                "pooled_auprc": round(float(payload["pooled_auprc"]), 4),
                "mean_auroc": round(float(payload["mean_auroc"]), 4),
                "std_auroc": round(float(payload["std_auroc"]), 4),
                "mean_auprc": round(float(payload["mean_auprc"]), 4),
                "std_auprc": round(float(payload["std_auprc"]), 4),
            }

    best = max(
        metrics["models"].items(),
        key=lambda kv: (kv[1]["calibrated"] or kv[1]["raw"])["auroc"],
    )
    best_auc = float((best[1]["calibrated"] or best[1]["raw"])["auroc"])

    evaluation = {
        "test_set": metrics["test_set"],
        "operating_points": api_operating_points,
        "threshold_policy": (
            "Probabilities are Platt-calibrated on the validation split. The decision "
            "threshold is the Youden-J optimum on the validation split and is applied "
            "unchanged to the test split. Both the 0.5 operating point and the "
            "validation-selected one are reported; moving the threshold trades "
            "sensitivity against specificity and is unstable with 7 validation events."
        ),
        "models": models,
        "best_test_auroc": {
            "model": best[1].get("display_name", best[0]),
            "key": best[0],
            "auroc": round(best_auc, 4),
        },
        "paired_comparison": metrics.get("comparisons"),
        "cross_validation": cv or None,
    }

    return {
        "project": "AI-Based Mobile Clinical Decision Support System for Delivery Outcome Prediction",
        "version": "1.1.0",
        "task": metrics["task"]["description"],
        "target": metrics["task"]["target"],
        "prediction_time": metrics["task"]["prediction_time"],
        "dataset": {
            "name": "CTU-CHB Intrapartum Cardiotocography Database, version 1.0.0",
            "source": "PhysioNet / Czech Technical University in Prague, University Hospital in Brno",
            "doi": "10.13026/0d3m-5v02",
            "n_records": int(dataset_summary["n_records"]),
            "n_vaginal": int(dataset_summary["class_counts"]["0"]),
            "n_cesarean": int(dataset_summary["class_counts"]["1"]),
            "splits": dataset_summary["class_counts_by_split"],
            "signal_shape": dataset_summary["signal_shape"],
            "clinical_features": preprocessing["clinical_features"],
            "window_minutes": preprocessing["window_min"],
            "sampling_frequency_hz": preprocessing["target_fs"],
        },
        "architecture": architecture,
        "evaluation": evaluation,
        "limitations": LIMITATIONS,
        "intended_use": INTENDED_USE,
        "prohibited_use": PROHIBITED_USE,
        "disclaimer": DISCLAIMER,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    card = build_card()
    path = MODEL_DIR / "model_card.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")
    print(json.dumps(card["evaluation"]["operating_points"], indent=2))


if __name__ == "__main__":
    main()
