"""Tests for metrics, calibration and thresholds."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    bootstrap_auprc,
    bootstrap_auroc,
    compute_metrics,
    find_threshold,
    reliability_curve,
)
from src.models.calibration import PlattCalibrator


def test_compute_metrics_matches_hand_calculation():
    y = np.array([0, 0, 1, 1, 0, 0])
    p = np.array([0.1, 0.4, 0.8, 0.6, 0.2, 0.9])
    m = compute_metrics(y, p, threshold=0.5)
    assert m["tp"] == 2 and m["fp"] == 1 and m["tn"] == 3 and m["fn"] == 0
    assert m["recall_sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(0.8)
    # 6 of 8 positive/negative pairs are correctly ordered (the 0.9 negative
    # outranks both positives), so AUROC is 0.75.
    assert m["auroc"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert m["n"] == 6 and m["n_positive"] == 2


def test_threshold_changes_sensitivity_and_specificity():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.05, 0.2, 0.4, 0.35, 0.8])
    low = compute_metrics(y, p, threshold=0.3)
    high = compute_metrics(y, p, threshold=0.5)
    assert low["recall_sensitivity"] >= high["recall_sensitivity"]
    assert low["specificity"] <= high["specificity"]
    assert high["tp"] == 1


def test_find_threshold_uses_youden_j():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.75, 0.8, 0.9])
    threshold = find_threshold(y, p, criterion="youden")
    m = compute_metrics(y, p, threshold=threshold)
    assert m["recall_sensitivity"] == 1.0
    assert m["specificity"] == 1.0


def test_bootstrap_intervals_are_ordered():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 8)
    p = np.clip(rng.normal(0.3 + 0.25 * y, 0.15, size=len(y)), 0, 1)
    low, high = bootstrap_auroc(y, p, n_boot=100)
    assert 0 <= low <= high <= 1
    low_ap, high_ap = bootstrap_auprc(y, p, n_boot=100)
    assert 0 <= low_ap <= high_ap <= 1


def test_reliability_curve_is_monotone_in_score_bins():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    p = np.array([0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.95])
    means, observed, counts = reliability_curve(y, p, n_bins=4)
    assert len(means) == len(observed) == len(counts) == 4
    assert np.all(np.diff(means) >= 0)
    assert counts.sum() == len(y)


def test_platt_calibrator_is_monotone_and_bounded():
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 2, size=200)
    labels = (rng.uniform(size=200) < 1 / (1 + np.exp(-logits))).astype(int)
    calibrator = PlattCalibrator.fit(logits, labels)
    grid = np.linspace(-6, 6, 50)
    probs = calibrator.predict(grid)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert np.all(np.diff(probs) >= -1e-12)
    assert isinstance(calibrator.predict(logits), np.ndarray)


def test_metrics_payload_is_internally_consistent(metrics_payload):
    test_set = metrics_payload["test_set"]
    assert test_set == {"n": 83, "n_positive": 7, "n_negative": 76, "prevalence": pytest.approx(7 / 83)}
    for name, entry in metrics_payload["models"].items():
        for key in ("raw",):
            assert entry[key]["n"] == 83
        primary = entry["calibrated"] or entry["raw"]
        assert 0 <= primary["auroc"] <= 1
        assert 0 <= primary["auprc"] <= 1
        assert len(entry["record_ids"]) == 83
        assert len(set(map(str, entry["record_ids"]))) == 83
        assert len(entry["probabilities"]) == 83
        cm = np.array(entry["confusion"])
        assert cm.shape == (2, 2)
        assert cm.sum() == 83
        assert cm[1, 0] + cm[1, 1] == 7
        assert entry["tuned_threshold"] is not None
        if entry["at_tuned_threshold"]:
            assert entry["at_tuned_threshold"]["threshold"] == pytest.approx(entry["tuned_threshold"])
