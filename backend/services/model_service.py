"""Inference service: loads the trained artefacts and produces predictions.

The service never writes uploaded files to disk and never logs clinical values.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import (
    CLASS_NAMES,
    MODEL_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    REPORT_DIR,
)
from src.models.deep_learning_model import build_model, forward_model
from src.preprocessing.preprocess import extract_window

DISCLAIMER = (
    "Research prototype only. This output is a statistical estimate from a public "
    "research dataset and is not a diagnosis, a treatment recommendation, or a "
    "substitute for professional obstetric judgement. Do not use it to decide the "
    "mode of delivery."
)

FEATURE_DISPLAY = {
    "age": "Maternal age",
    "gravidity": "Gravidity",
    "parity": "Parity",
    "gest_weeks": "Gestational age",
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "preeclampsia": "Preeclampsia",
    "liquor_praecox": "Liquor praecox / PROM",
    "pyrexia": "Maternal pyrexia",
    "meconium": "Meconium-stained fluid",
    "induced": "Induced labour",
    "presentation": "Fetal presentation",
}

CLINICAL_GROUPS = [
    ("presentation", ["presentation_1", "presentation_2", "presentation_3", "presentation_missing"]),
]


class ModelUnavailableError(RuntimeError):
    pass


# Number of teaching cases returned per outcome class. The held-out test split
# contains 7 cesareans, so 7 per class exposes every positive case plus a
# matched number of vaginal cases.
DEMO_CASES_PER_CLASS = 7


class ModelService:
    def __init__(self, model_dir: Path = MODEL_DIR, processed_dir: Path = PROCESSED_DIR):
        self.model_dir = model_dir
        self.processed_dir = processed_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessing = joblib.load(model_dir / "preprocessing.joblib")
        self.signal_pre = self.preprocessing["signal"]
        self.clinical_pre = self.preprocessing["clinical"]
        self._models: Dict[str, torch.nn.Module] = {}
        self._calibrators: Dict[str, object] = {}
        self._metrics: Optional[dict] = None
        self._card: Optional[dict] = None
        self._demo_frame: Optional[pd.DataFrame] = None
        self._demo_signals: Optional[np.ndarray] = None
        self._demo_clinical: Optional[np.ndarray] = None
        self.version = "1.1.0"
        # Operating points were selected on the validation split only and are
        # stored in the model card.  Fall back to 0.5 if the card is absent.
        self._thresholds: Dict[str, float] = {
            "clinical": 0.5,
            "multimodal": 0.5,
            "signal": 0.5,
            "bilstm": 0.5,
        }
        try:
            card = self.load_card()
            for key, value in card.get("evaluation", {}).get("operating_points", {}).items():
                if key in self._thresholds:
                    self._thresholds[key] = float(value)
        except ModelUnavailableError:
            pass

    # ------------------------------------------------------------------
    # Artefacts
    # ------------------------------------------------------------------
    def _load_model(self, variant: str) -> torch.nn.Module:
        if variant not in self._models:
            path = self.model_dir / f"{variant}.pt"
            if not path.exists():
                raise ModelUnavailableError(
                    f"Model '{variant}' is not available. Train it with "
                    f"'python -m src.training.train --variant {variant}'."
                )
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model = build_model(variant, clinical_features=len(self.clinical_pre.output_names))
            model.load_state_dict(checkpoint["state_dict"])
            model.to(self.device)
            model.eval()
            self._models[variant] = model
        return self._models[variant]

    def _calibrator(self, variant: str):
        if variant not in self._calibrators:
            path = self.model_dir / f"calibrator_{variant}.joblib"
            if path.exists():
                self._calibrators[variant] = joblib.load(path)["calibrator"]
            else:
                self._calibrators[variant] = None
        return self._calibrators[variant]

    def load_metrics(self) -> dict:
        if self._metrics is None:
            path = REPORT_DIR / "metrics.json"
            if not path.exists():
                raise ModelUnavailableError("Evaluation metrics are not available; run src.evaluation.evaluate")
            self._metrics = json.loads(path.read_text(encoding="utf-8"))
        return self._metrics

    def load_card(self) -> dict:
        if self._card is None:
            path = self.model_dir / "model_card.json"
            if not path.exists():
                raise ModelUnavailableError("Model card not available")
            self._card = json.loads(path.read_text(encoding="utf-8"))
            self.version = str(self._card.get("version", self.version))
        return self._card

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def _logit(self, variant: str, signal: np.ndarray, clinical: np.ndarray) -> Tuple[float, torch.Tensor, torch.Tensor]:
        model = self._load_model(variant)
        signal_t = torch.from_numpy(signal[None, ...].astype(np.float32)).to(self.device).requires_grad_(True)
        clinical_t = torch.from_numpy(clinical[None, ...].astype(np.float32)).to(self.device).requires_grad_(True)
        logit = forward_model(model, signal_t, clinical_t)
        return float(logit.detach().cpu().item()), signal_t, clinical_t

    def _probability(self, variant: str, logit: float) -> Tuple[float, bool]:
        calibrator = self._calibrator(variant)
        if calibrator is not None:
            prob = float(calibrator.predict(np.array([logit]))[0])
            return prob, True
        return float(1.0 / (1.0 + np.exp(-logit))), False

    def _clinical_attributions(
        self, variant: str, signal: np.ndarray, clinical: np.ndarray, raw: Dict[str, float]
    ) -> List[dict]:
        # CTG-only models never consume the clinical vector, so there is no
        # clinical gradient to attribute.
        if variant in {"signal", "bilstm"}:
            return []
        model = self._load_model(variant)
        signal_t = torch.from_numpy(signal[None, ...].astype(np.float32)).to(self.device)
        clinical_t = torch.from_numpy(clinical[None, ...].astype(np.float32)).to(self.device).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logit = forward_model(model, signal_t, clinical_t)
        logit.backward()
        grad = clinical_t.grad.detach().cpu().numpy()[0]
        contrib = grad * clinical
        names = list(self.clinical_pre.output_names)
        grouped: Dict[str, float] = {k: 0.0 for k in FEATURE_DISPLAY}
        for name, value in zip(names, contrib):
            base = name
            for group, members in CLINICAL_GROUPS:
                if name in members:
                    base = group
                    break
            grouped[base] = grouped.get(base, 0.0) + float(value)
        rows = []
        for key, value in grouped.items():
            display = FEATURE_DISPLAY.get(key, key)
            rows.append(
                {
                    "feature": key,
                    "display_name": display,
                    "input_value": float(raw.get(key, np.nan)) if raw.get(key) is not None else None,
                    "contribution": float(value),
                    "direction": "toward_cesarean" if value > 0 else "toward_vaginal",
                }
            )
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return rows[:6]

    def _signal_attributions(
        self, variant: str, signal: np.ndarray, clinical: np.ndarray
    ) -> List[dict]:
        if variant == "clinical":
            return []
        model = self._load_model(variant)
        signal_t = torch.from_numpy(signal[None, ...].astype(np.float32)).to(self.device).requires_grad_(True)
        clinical_t = torch.from_numpy(clinical[None, ...].astype(np.float32)).to(self.device)
        model.zero_grad(set_to_none=True)
        logit = forward_model(model, signal_t, clinical_t)
        logit.backward()
        grad = signal_t.grad.detach().cpu().numpy()[0]
        saliency = np.abs(grad[:2]).mean(axis=0)  # FHR + UC only
        bins = np.array_split(saliency, 6)
        values = np.array([float(b.mean()) for b in bins])
        total = values.sum()
        values = values / total if total > 0 else np.zeros_like(values)
        # The window is the most recent 30 minutes; label bins from oldest to newest.
        labels = [
            "25–30 min before end",
            "20–25 min before end",
            "15–20 min before end",
            "10–15 min before end",
            "5–10 min before end",
            "Most recent 5 min",
        ]
        return [
            {
                "window": label,
                "importance": float(value),
                "note": "Gradient-based attribution; not a clinical interpretation.",
            }
            for label, value in zip(labels, values)
        ]

    def _response(
        self,
        variant: str,
        logit: float,
        clinical_vector: np.ndarray,
        signal_vector: np.ndarray,
        raw: Dict[str, float],
        warnings: Optional[List[str]] = None,
        explain: bool = True,
    ) -> dict:
        prob, calibrated = self._probability(variant, logit)
        prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
        threshold = float(self._thresholds.get(variant, 0.5))
        predicted = 1 if prob >= threshold else 0
        warnings = list(warnings or [])
        if abs(prob - threshold) < 0.02:
            warnings.append(
                "The model score lies close to the decision threshold; the "
                "classification should be treated as unstable."
            )
        warnings.append("Research prototype — not for clinical decision-making.")
        return {
            "task": "Prediction of delivery mode (cesarean vs vaginal) from intrapartum CTG and maternal data",
            "prediction": CLASS_NAMES[predicted],
            "predicted_class": predicted,
            "cesarean_probability": prob,
            "vaginal_probability": 1.0 - prob,
            "decision_threshold": threshold,
            "model_variant": variant,
            "model_version": self.version,
            "calibrated": calibrated,
            "clinical_contributions": (
                self._clinical_attributions(variant, signal_vector, clinical_vector, raw) if explain else []
            ),
            "signal_contributions": (
                self._signal_attributions(variant, signal_vector, clinical_vector) if explain else []
            ),
            "warnings": warnings,
            "disclaimer": DISCLAIMER,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_clinical(self, clinical: Dict[str, float]) -> dict:
        vector = self.clinical_pre.transform_one(clinical)
        # The clinical-only model ignores the signal; a zero tensor keeps the
        # interface uniform and avoids fabricating a CTG.
        signal = np.zeros((4, self.signal_pre.window_min * 60 * self.signal_pre.target_fs), dtype=np.float32)
        logit, _, _ = self._logit("clinical", signal, vector)
        return self._response("clinical", logit, vector, signal, clinical)

    def predict_with_window(
        self, clinical: Dict[str, float], window: dict, variant: str = "multimodal"
    ) -> dict:
        vector = self.clinical_pre.transform_one(clinical)
        signal = self.signal_pre.transform_window(window)
        logit, _, _ = self._logit(variant, signal, vector)
        warnings = []
        if window.get("padded"):
            warnings.append("The uploaded recording is shorter than 30 minutes; the start was padded.")
        if window.get("fhr_missing_fraction", 0) > 0.5:
            warnings.append("More than half of the FHR window is missing or unusable.")
        if window.get("uc_missing_fraction", 1.0) >= 0.999:
            warnings.append("The uterine contraction channel is absent or entirely zero.")
        return self._response(variant, logit, vector, signal, clinical, warnings=warnings)

    def predict_ctg(
        self,
        clinical: Dict[str, float],
        fhr: np.ndarray,
        uc: np.ndarray,
        fs: float,
        variant: str = "multimodal",
    ) -> Tuple[dict, dict]:
        window = extract_window(fhr, uc, fs=fs)
        result = self.predict_with_window(clinical, window, variant=variant)
        summary = {
            "sampling_frequency_hz": float(fs),
            "duration_minutes": float(window["duration_min_available"]),
            "samples_used": int(len(window["fhr"])),
            "mean_fhr_bpm": float(np.mean(window["fhr"][window["fhr_mask"] > 0])) if (window["fhr_mask"] > 0).any() else None,
            "fhr_missing_fraction": float(window["fhr_missing_fraction"]),
            "uc_missing_fraction": float(window["uc_missing_fraction"]),
            "window_padded": bool(window["padded"]),
        }
        return result, summary

    # ------------------------------------------------------------------
    # Demonstration cases from the public dataset
    # ------------------------------------------------------------------
    def _load_demo(self) -> None:
        if self._demo_frame is not None:
            return
        frame = pd.read_csv(self.processed_dir / "dataset.csv")
        signals = np.load(self.processed_dir / "signals.npz", allow_pickle=True)
        clinical = np.load(self.processed_dir / "clinical.npz", allow_pickle=True)
        frame = frame[frame["split"] == "test"].copy().reset_index(drop=True)
        self._demo_frame = frame
        self._demo_signals = signals["X"][signals["split"] == "test"]
        self._demo_clinical = clinical["X"][clinical["split"] == "test"]

    def demo_cases(self) -> List[dict]:
        self._load_demo()
        assert self._demo_frame is not None
        rng = np.random.default_rng(7)
        frame = self._demo_frame
        chosen = []
        for label in [0, 1]:
            idx = frame.index[frame["label"] == label].to_list()
            rng.shuffle(idx)
            chosen.extend(idx[:DEMO_CASES_PER_CLASS])
        chosen = sorted(chosen, key=lambda i: (frame.loc[i, "label"], frame.loc[i, "record_id"]))
        cases = []
        for i in chosen:
            row = frame.loc[i]
            clinical = {
                "age": float(row["age"]),
                "gravidity": int(row["gravidity"]),
                "parity": int(row["parity"]),
                "gest_weeks": float(row["gest_weeks"]),
                "diabetes": bool(row["diabetes"]),
                "hypertension": bool(row["hypertension"]),
                "preeclampsia": bool(row["preeclampsia"]),
                "liquor_praecox": bool(row["liquor_praecox"]),
                "pyrexia": bool(row["pyrexia"]),
                "meconium": bool(row["meconium"]),
                "induced": bool(row["induced"]),
                "presentation": int(row["presentation"]),
            }
            cases.append(
                {
                    "record_id": str(row["record_id"]),
                    "label": CLASS_NAMES[int(row["label"])],
                    "split": "test",
                    "clinical": clinical,
                    "note": "De-identified public research record from the CTU-CHB database (held-out test split).",
                }
            )
        return cases

    def predict_demo(self, record_id: str, variant: str = "multimodal") -> dict:
        self._load_demo()
        assert self._demo_frame is not None
        frame = self._demo_frame
        matches = frame.index[frame["record_id"].astype(str) == str(record_id)]
        if not len(matches):
            raise KeyError(record_id)
        i = int(matches[0])
        row = frame.loc[i]
        clinical = {
            "age": float(row["age"]),
            "gravidity": int(row["gravidity"]),
            "parity": int(row["parity"]),
            "gest_weeks": float(row["gest_weeks"]),
            "diabetes": bool(row["diabetes"]),
            "hypertension": bool(row["hypertension"]),
            "preeclampsia": bool(row["preeclampsia"]),
            "liquor_praecox": bool(row["liquor_praecox"]),
            "pyrexia": bool(row["pyrexia"]),
            "meconium": bool(row["meconium"]),
            "induced": bool(row["induced"]),
            "presentation": int(row["presentation"]),
        }
        signal = self._demo_signals[i]
        vector = self._demo_clinical[i]
        logit, _, _ = self._logit(variant, signal, vector)
        result = self._response(
            variant, logit, vector, signal, clinical,
            warnings=["Demonstration record from the public research dataset; the true outcome is shown for teaching only."],
        )
        result["demo"] = True
        result["true_label"] = CLASS_NAMES[int(row["label"])]
        result["true_label_note"] = (
            "True outcome is shown only for teaching/demonstration of this public research "
            "dataset and is never available to the model at inference time."
        )
        # Reconstruct the summary from the stored signal masks.
        result["ctg_summary"] = {
            "sampling_frequency_hz": 1.0,
            "duration_minutes": 30.0,
            "samples_used": int(signal.shape[-1]),
            "mean_fhr_bpm": None,
            "fhr_missing_fraction": float(1 - signal[2].mean()),
            "uc_missing_fraction": float(1 - signal[3].mean()),
            "window_padded": False,
        }
        return result
