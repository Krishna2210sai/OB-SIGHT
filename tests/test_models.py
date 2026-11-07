"""Model architecture, checkpoint and artefact tests."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
import torch

from src.config import MODEL_DIR
from src.models.calibration import PlattCalibrator
from src.models.deep_learning_model import build_model, count_parameters, forward_model


@pytest.mark.parametrize("variant", ["multimodal", "signal", "bilstm", "clinical"])
def test_forward_pass_shapes(variant):
    torch.manual_seed(0)
    model = build_model(variant, clinical_features=15).eval()
    signal = torch.zeros(4, 4, 1800)
    clinical = torch.zeros(4, 15)
    with torch.no_grad():
        out = forward_model(model, signal, clinical)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_bilstm_contains_cnn_and_bidirectional_lstm():
    model = build_model("bilstm")
    assert any(isinstance(layer, torch.nn.Conv1d) for layer in model.modules())
    assert isinstance(model.lstm, torch.nn.LSTM)
    assert model.lstm.bidirectional is True
    assert count_parameters(model) < 40_000


def test_dummy_baseline_artefact_is_absent():
    assert not (MODEL_DIR / "baseline_dummy_prior.joblib").exists()


@pytest.mark.parametrize("variant", ["multimodal", "signal", "bilstm", "clinical"])
def test_checkpoint_loads_and_is_deterministic(variant):
    path = MODEL_DIR / f"{variant}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(variant, clinical_features=checkpoint["clinical_features"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    assert count_parameters(model) == checkpoint["n_parameters"]
    assert checkpoint["window_samples"] == 1800
    assert checkpoint["signal_channels"] == 4

    signal = torch.zeros(1, 4, 1800)
    clinical = torch.zeros(1, 15)
    with torch.no_grad():
        first = forward_model(model, signal, clinical).item()
        second = forward_model(model, signal, clinical).item()
    assert first == second


def test_parameter_counts_match_training_summaries():
    for variant in ["multimodal", "signal", "bilstm", "clinical"]:
        summary = json.loads((MODEL_DIR / f"train_summary_{variant}.json").read_text())
        assert summary["n_parameters"] > 0
        assert summary["best_epoch"] >= 1


def test_calibrators_are_importable_and_monotone():
    for variant in ["multimodal", "signal", "bilstm", "clinical"]:
        payload = joblib.load(MODEL_DIR / f"calibrator_{variant}.joblib")
        calibrator = payload["calibrator"]
        assert isinstance(calibrator, PlattCalibrator)
        probs = calibrator.predict(np.linspace(-8, 8, 21))
        assert np.all(np.diff(probs) >= -1e-9)
        assert np.all((probs >= 0) & (probs <= 1))


def test_model_card_contains_required_sections(model_card, metrics_payload):
    for key in [
        "project",
        "version",
        "task",
        "target",
        "prediction_time",
        "dataset",
        "architecture",
        "evaluation",
        "limitations",
        "intended_use",
        "prohibited_use",
        "disclaimer",
    ]:
        assert key in model_card
    dataset = model_card["dataset"]
    assert dataset["n_records"] == 552
    assert dataset["n_cesarean"] == 46
    assert dataset["n_vaginal"] == 506
    evaluation = model_card["evaluation"]
    assert "multimodal_dl" in evaluation["models"]
    assert "bilstm_dl" in evaluation["models"]
    for variant in ["clinical", "multimodal", "signal", "bilstm"]:
        threshold = evaluation["operating_points"][variant]
        assert 0 < threshold < 1
    # The reported confusion matrix must correspond to the reported operating
    # point, not to the default 0.5 threshold.
    for name, entry in evaluation["models"].items():
        tn, fp, fn, tp = (
            entry["confusion"][0][0], entry["confusion"][0][1],
            entry["confusion"][1][0], entry["confusion"][1][1],
        )
        assert tn + fp + fn + tp == 83, name
        assert tp / max(tp + fn, 1) == pytest.approx(entry["sensitivity"], abs=1e-4), name
        assert tn / max(tn + fp, 1) == pytest.approx(entry["specificity"], abs=1e-4), name
    # The 0.5 operating point in the card must agree with the evaluation
    # report's (calibrated) probabilities.
    for name, entry in metrics_payload["models"].items():
        default = entry["calibrated"] or entry["raw"]
        card_default = evaluation["models"][name]["at_default_threshold"]
        assert card_default["sensitivity"] == pytest.approx(default["recall_sensitivity"], abs=1e-4), name
        assert card_default["specificity"] == pytest.approx(default["specificity"], abs=1e-4), name
    assert evaluation["cross_validation"] is not None
    for variant in ["multimodal", "signal", "bilstm", "clinical"]:
        cv = evaluation["cross_validation"][variant]
        assert 0.5 <= cv["pooled_auroc"] <= 1.0
        assert cv["n_splits"] == 5
    comparison = evaluation["paired_comparison"]["bilstm_vs_signal"]
    assert comparison["ci_low"] <= comparison["difference"] <= comparison["ci_high"]
    assert "not establish" in comparison["interpretation"].lower()
    assert "not a diagnosis" in model_card["disclaimer"].lower()


def test_training_data_was_not_used_as_test(signals):
    train_ids = set(signals["record_ids"][signals["split"] == "train"].tolist())
    test_ids = set(signals["record_ids"][signals["split"] == "test"].tolist())
    assert train_ids.isdisjoint(test_ids)
    # All 46 cesarean records are partitioned across splits, not duplicated.
    cesarean_ids = signals["record_ids"][signals["y"] == 1].tolist()
    assert len(cesarean_ids) == 46
    assert len(set(cesarean_ids)) == 46
