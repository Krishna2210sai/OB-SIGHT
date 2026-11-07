"""Shared pytest fixtures for the OB-SIGHT research prototype."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_DIR, PROCESSED_DIR, PROJECT_ROOT


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def synthetic_clinical() -> dict:
    return {
        "age": 31.0,
        "gravidity": 2,
        "parity": 1,
        "gest_weeks": 39.5,
        "diabetes": False,
        "hypertension": False,
        "preeclampsia": False,
        "liquor_praecox": False,
        "pyrexia": False,
        "meconium": False,
        "induced": False,
        "presentation": 1,
    }


@pytest.fixture(scope="session")
def dataset_frame() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "dataset.csv")


@pytest.fixture(scope="session")
def signals():
    return np.load(PROCESSED_DIR / "signals.npz", allow_pickle=True)


@pytest.fixture(scope="session")
def clinical():
    return np.load(PROCESSED_DIR / "clinical.npz", allow_pickle=True)


@pytest.fixture(scope="session")
def preprocessing():
    return joblib.load(MODEL_DIR / "preprocessing.joblib")


@pytest.fixture(scope="session")
def metrics_payload() -> dict:
    return json.loads((PROJECT_ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def model_card() -> dict:
    return json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client
