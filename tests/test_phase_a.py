"""Tests for the additive Phase A data-handling modules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.phase_a.config import QualityThresholds
from src.phase_a.experiment import cv_baseline, make_folds, phase_a_metrics
from src.phase_a.features import (
    CLINICAL_BINARY,
    CLINICAL_NUMERIC,
    PRESENTATION_FEATURES,
    candidate_feature_names,
)
from src.phase_a.quality import audit_record, longest_run_min, trim_zero_padding
from src.phase_a.selection import (
    CorrelationFilter,
    L1Selector,
    NoSelection,
    make_selector,
)
from src.phase_a.signals import extract_recent_window
from src.preprocessing.ctg_features import FEATURE_NAMES, extract_ctg_features


def _synthetic_signal(minutes: float = 40.0, fs: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    n = int(minutes * 60 * fs)
    t = np.arange(n) / fs
    fhr = 135.0 + 8.0 * np.sin(2 * np.pi * 0.03 * t)
    uc = 15.0 + 8.0 * np.sin(2 * np.pi * 0.01 * t)
    return fhr.astype(np.float32), uc.astype(np.float32)


def test_trim_zero_padding_removes_only_edges() -> None:
    fhr = np.array([0, 0, 120, 130, 140, 0, 0], dtype=float)
    uc = np.array([0, 0, 10, 12, 11, 0, 0], dtype=float)
    fhr_t, uc_t = trim_zero_padding(fhr, uc)
    assert np.allclose(fhr_t, [120, 130, 140])
    assert np.allclose(uc_t, [10, 12, 11])


def test_longest_run_min_counts_contiguous_missing() -> None:
    mask = np.zeros(4 * 60 * 10, dtype=bool)
    mask[4 * 60 * 2 : 4 * 60 * 7] = True
    assert longest_run_min(mask, fs=4.0) == 5.0


def test_audit_record_reports_missingness_and_support() -> None:
    fhr, uc = _synthetic_signal(minutes=40)
    fhr[15 * 60 * 4 : 20 * 60 * 4] = 0.0  # 5 missing minutes inside the last 30
    uc[16 * 60 * 4 : 17 * 60 * 4] = 0.0
    row = audit_record("synthetic", fhr, uc, label=0, fs=4.0, thresholds=QualityThresholds())
    assert row["supports_30m"] == 1
    assert row["supports_45m"] == 0
    assert row["window_supported"] == 1
    assert 0.15 < row["fhr_missing_fraction_30m"] < 0.18
    assert row["longest_fhr_zero_run_min_30m"] >= 4.9
    assert row["quality_category"] in {"good", "acceptable", "poor"}
    assert np.isfinite(row["implausible_fhr_fraction_30m"])
    assert row["fhr_min_positive"] > 0


def test_window_extraction_excludes_or_pads_explicitly() -> None:
    fhr, uc = _synthetic_signal(minutes=20)
    assert extract_recent_window(fhr, uc, fs=4.0, window_min=30, target_fs=1.0) is None

    window = extract_recent_window(
        fhr, uc, fs=4.0, window_min=30, target_fs=1.0, policy="pad_masked"
    )
    assert window is not None
    assert window["padded"] is True
    assert len(window["fhr"]) == 30 * 60
    assert len(window["uc"]) == 30 * 60
    # The padded 10 minutes must be flagged as missing, never as observed data.
    assert window["fhr_mask"][: 10 * 60 - 2].max() == 0
    assert window["fhr_mask"][10 * 60 + 2 :].min() == 1


def test_antialiasing_preserves_low_frequency_and_suppresses_high_frequency() -> None:
    fs = 4.0
    minutes = 30
    n = int(minutes * 60 * fs)
    t = np.arange(n) / fs
    low = 6.0 * np.sin(2 * np.pi * 0.05 * t)
    high = 6.0 * np.sin(2 * np.pi * 1.7 * t)
    fhr = (135.0 + low + high).astype(np.float32)
    uc = (15.0 + 5.0 * np.sin(2 * np.pi * 0.02 * t)).astype(np.float32)

    window = extract_recent_window(fhr, uc, fs=fs, window_min=minutes, target_fs=1.0)
    out = window["fhr"]
    assert len(out) == minutes * 60
    t_out = np.arange(len(out)) / 1.0
    reference = 6.0 * np.sin(2 * np.pi * 0.05 * t_out)
    # The 1.7 Hz component is above the 1 Hz Nyquist frequency and must be
    # removed by the anti-aliasing filter rather than folded into the output.
    residual = out - out.mean() - reference
    assert np.std(residual) < 1.0
    assert np.corrcoef(out, reference)[0, 1] > 0.98


def test_long_gap_stays_missing_after_resampling() -> None:
    fs = 4.0
    minutes = 30
    fhr, uc = _synthetic_signal(minutes=minutes)
    fhr[: 20 * 60 * 4] = 0.0
    window = extract_recent_window(fhr, uc, fs=fs, window_min=minutes, target_fs=1.0)
    assert window["fhr_mask"][: 19 * 60].mean() == 0.0
    assert window["fhr_mask"][21 * 60 :].mean() > 0.99
    # Missing samples are never presented as measured values.
    assert np.all(window["fhr"][window["fhr_mask"] == 0] == 0.0)


def test_ctg_features_are_finite_at_every_sampling_rate() -> None:
    for fs in (4.0, 2.0, 1.0, 0.5):
        n = int(30 * 60 * fs)
        t = np.arange(n) / fs
        window = {
            "fhr": (135 + 5 * np.sin(2 * np.pi * 0.05 * t)).astype(np.float32),
            "uc": (10 + 5 * np.sin(2 * np.pi * 0.02 * t)).astype(np.float32),
            "fhr_mask": np.ones(n, dtype=np.float32),
            "uc_mask": np.ones(n, dtype=np.float32),
        }
        features = extract_ctg_features(window, fs=fs)
        assert set(features) == set(FEATURE_NAMES)
        assert all(np.isfinite(value) for value in features.values())


def _phase_a_frame(n_records: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "record_id": [f"{1000 + i}" for i in range(n_records)],
            "label": ([0, 1] * (n_records // 2))[:n_records],
            "split": ["train"] * n_records,
        }
    )
    for name in CLINICAL_NUMERIC:
        frame[name] = rng.normal(0, 1, n_records)
    for name in CLINICAL_BINARY:
        frame[name] = rng.integers(0, 2, n_records)
    for name in PRESENTATION_FEATURES:
        frame[name] = rng.integers(0, 2, n_records)
    for name in FEATURE_NAMES:
        frame[name] = rng.normal(0, 1, n_records)
    return frame


def test_selectors_keep_protected_features_and_are_fold_specific() -> None:
    frame = _phase_a_frame()
    names = candidate_feature_names(frame)
    X = frame[names].to_numpy(float)
    y = frame["label"].to_numpy(int)

    no_selection = NoSelection(names).fit(X, y)
    assert no_selection.get_support().all()

    correlation = CorrelationFilter(names).fit(X, y)
    assert set(correlation.selected_features()) >= {
        "age",
        "gest_weeks",
        "parity",
        "meconium",
        "presentation_2",
        "presentation_3",
    }
    assert correlation.get_support().sum() <= len(names)

    l1 = L1Selector(names).fit(X, y)
    assert l1.get_support().sum() >= 5
    for name in ("age", "gest_weeks", "parity"):
        assert name in l1.selected_features()


def test_cv_baseline_uses_each_fold_training_data_only() -> None:
    frame = _phase_a_frame(n_records=80, seed=1)
    folds = make_folds(frame, n_splits=5, seed=42)
    output = cv_baseline(frame, folds, method="correlation", model_name="logistic", seed=42)
    result = output["result"]

    assert result["n_records"] == len(frame)
    assert set(result["selected_by_fold"]) == {"1", "2", "3", "4", "5"}
    assert len(output["oof_frame"]) == len(frame)
    assert output["oof_frame"]["prob"].between(0, 1).all()

    # Every fold is fitted independently: selections are stored per fold and
    # are not a single subset computed on the full dataset.
    assert isinstance(output["selected_by_fold"][1], list)
    assert result["n_features_input"] == len(candidate_feature_names(frame))


def test_make_folds_is_patient_level_and_stratified() -> None:
    frame = _phase_a_frame(n_records=100, seed=2)
    folds = make_folds(frame, n_splits=5, seed=42)
    assert len(folds) == len(frame)
    assert folds["record_id"].is_unique
    assert set(folds["fold"]) == {1, 2, 3, 4, 5}
    assert (folds.groupby("fold")["label"].sum() > 0).all()


def test_phase_a_metrics_contains_required_reporting_fields() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.7, 0.4, 0.9, 0.3, 0.8, 0.2, 0.5])
    metrics = phase_a_metrics(y_true, y_prob, threshold=0.5)
    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "auroc",
        "auprc",
        "mcc",
        "npv",
        "confusion_matrix",
    ):
        assert key in metrics
    cm = metrics["confusion_matrix"]
    assert cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"] == len(y_true)


def test_selector_registry_rejects_unknown_method() -> None:
    try:
        make_selector("does-not-exist", ["a", "b"])
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("Unknown selector should raise ValueError")
