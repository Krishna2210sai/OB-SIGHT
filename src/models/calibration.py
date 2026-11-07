"""Probability calibration utilities.

``PlattCalibrator`` lives in a real module (not in the evaluation script) so that
joblib can unpickle it from the FastAPI backend and from the tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibrator:
    """Logistic (Platt) calibration of a model's log-odds."""

    model: LogisticRegression

    @classmethod
    def fit(cls, logits: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(np.asarray(logits).reshape(-1, 1), np.asarray(labels).astype(int))
        return cls(model)

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(logits).reshape(-1, 1))[:, 1]
