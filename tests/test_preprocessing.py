"""Preprocessing tests: windowing, masking, interpolation and train-only fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FHR_VALID_MAX, FHR_VALID_MIN
from src.preprocessing.preprocess import (
    ClinicalPreprocessor,
    SignalPreprocessor,
    extract_window,
    trim_recording,
)


def _recording(minutes: int, fhr: float = 140.0, uc: float = 20.0, fs: int = 4):
    n = minutes * 60 * fs
    return np.full(n, fhr, dtype=np.float32), np.full(n, uc, dtype=np.float32)


def test_trim_removes_leading_and_trailing_padding():
    fhr = np.array([0, 0, 140, 141, 0, 0, 0], dtype=np.float32)
    uc = np.array([0, 0, 20, 21, 0, 0, 0], dtype=np.float32)
    fhr_t, uc_t = trim_recording(fhr, uc)
    assert np.array_equal(fhr_t, np.array([140, 141], dtype=np.float32))
    assert np.array_equal(uc_t, np.array([20, 21], dtype=np.float32))


def test_window_is_last_thirty_minutes_at_one_hz():
    fhr, uc = _recording(45)
    # Mark the first 15 minutes with an implausible value that must be dropped.
    fhr[: 15 * 60 * 4] = 999
    window = extract_window(fhr, uc, fs=4)
    assert window["fhr"].shape == (1800,)
    assert window["uc"].shape == (1800,)
    assert window["padded"] is False
    assert window["duration_min_available"] == pytest.approx(45.0, abs=0.01)
    assert np.allclose(window["fhr"], 140.0)
    assert np.allclose(window["fhr_mask"], 1.0)
    assert np.allclose(window["uc_mask"], 1.0)
    assert window["fhr_missing_fraction"] == 0.0


def test_short_recording_is_left_padded_and_masked():
    fhr, uc = _recording(10)
    window = extract_window(fhr, uc, fs=4)
    assert window["padded"] is True
    assert window["fhr"].shape == (1800,)
    # 10 minutes = 600 samples; the first 1200 samples are padding.
    assert np.all(window["fhr_mask"][:1200] == 0)
    assert np.all(window["fhr"][:1200] == 0)
    assert np.all(window["fhr_mask"][1200:] == 1)
    assert np.allclose(window["fhr"][1200:], 140.0)
    assert window["fhr_missing_fraction"] == pytest.approx(1200 / 1800, abs=1e-6)


def test_short_gaps_are_interpolated_and_long_gaps_remain_missing():
    fhr, uc = _recording(30)
    fs = 4
    # 2-second gap (8 samples) should be interpolated.
    fhr[1000:1008] = 0
    # 60-second gap (240 samples) must stay missing.
    fhr[2000:2240] = 0
    window = extract_window(fhr, uc, fs=fs)
    # The 4 Hz recording is block-averaged to 1 Hz, so sample indices are /4.
    assert window["fhr_mask"][250:252].mean() == pytest.approx(1.0)
    assert window["fhr_mask"][500:560].size == 60
    assert window["fhr_mask"][500:560].mean() == pytest.approx(0.0)
    assert window["fhr_missing_fraction"] > 0


def test_fhr_outside_physiological_range_is_treated_as_missing():
    fhr, uc = _recording(30)
    fhr[100:200] = 10.0       # below FHR_VALID_MIN
    fhr[300:400] = 300.0      # above FHR_VALID_MAX
    window = extract_window(fhr, uc, fs=4)
    # The blocks containing the invalid samples are at least partly masked.
    assert window["fhr_mask"][25:50].min() == 0
    assert window["fhr_mask"][75:100].min() == 0
    assert (window["fhr"][25:50] == 0).any()
    assert (window["fhr"][75:100] == 0).any()
    assert FHR_VALID_MIN < 140 < FHR_VALID_MAX


def test_signal_preprocessor_fits_observed_values_only():
    windows = []
    for offset in (0, 4):
        fhr, uc = _recording(30, fhr=138.0 + offset, uc=20.0)
        fhr[:200] = 0  # missing values must not enter the statistics
        windows.append(extract_window(fhr, uc, fs=4))
    pre = SignalPreprocessor().fit(windows)
    assert pre.fhr_mean == pytest.approx(140.0, abs=0.05)
    assert pre.fhr_std == pytest.approx(2.0, abs=0.05)
    assert pre.uc_mean == pytest.approx(20.0, abs=0.05)

    fhr, uc = _recording(30, fhr=140.0)
    transformed = pre.transform(fhr, uc)[0]
    assert transformed.shape == (4, 1800)
    assert np.isfinite(transformed).all()
    assert transformed[0].mean() == pytest.approx(0.0, abs=0.01)
    assert transformed[2].min() == 1.0


def test_clinical_preprocessor_output_layout_and_missingness():
    frame = pd.DataFrame(
        [
            {"age": 30, "gravidity": 2, "parity": 1, "gest_weeks": 39, "diabetes": 0,
             "hypertension": 0, "preeclampsia": 0, "liquor_praecox": 0, "pyrexia": 0,
             "meconium": 0, "induced": 1, "presentation": 1},
            {"age": 34, "gravidity": 3, "parity": 2, "gest_weeks": 40, "diabetes": 1,
             "hypertension": 0, "preeclampsia": 0, "liquor_praecox": 1, "pyrexia": 0,
             "meconium": 0, "induced": 0, "presentation": 2},
            {"age": 29, "gravidity": 1, "parity": 0, "gest_weeks": 38, "diabetes": 0,
             "hypertension": 1, "preeclampsia": 0, "liquor_praecox": 0, "pyrexia": 0,
             "meconium": 1, "induced": 0, "presentation": 3},
        ]
    )
    pre = ClinicalPreprocessor().fit(frame)
    assert len(pre.output_names) == 15
    assert pre.output_names[:4] == ["age", "gravidity", "parity", "gest_weeks"]
    assert pre.output_names[-4:] == [
        "presentation_1", "presentation_2", "presentation_3", "presentation_missing"
    ]

    vector = pre.transform_one(
        {
            "age": 31, "gravidity": 2, "parity": 1, "gest_weeks": 39.5,
            "diabetes": 0, "hypertension": 0, "preeclampsia": 0, "liquor_praecox": 0,
            "pyrexia": 0, "meconium": 0, "induced": 0, "presentation": np.nan,
        }
    )
    assert vector.shape == (15,)
    assert vector[-1] == 1.0  # explicit missing-presentation indicator
    # The category itself is imputed with the training mode (presentation=1).
    assert vector[-4] == 1.0 and vector[-3] == 0.0 and vector[-2] == 0.0


def test_fitted_clinical_preprocessor_matches_train_only_statistics(preprocessing, dataset_frame):
    pre = preprocessing["clinical"]
    train = dataset_frame[dataset_frame["split"] == "train"]
    for column in ["age", "gravidity", "parity", "gest_weeks"]:
        values = pd.to_numeric(train[column], errors="coerce")
        filled = values.fillna(pre.medians[column])
        assert pre.medians[column] == pytest.approx(float(values.median()))
        assert pre.means[column] == pytest.approx(float(filled.mean()), rel=1e-6, abs=1e-6)
        assert pre.stds[column] == pytest.approx(float(filled.std(ddof=0)), rel=1e-6, abs=1e-6)
    # A scaler fitted on all records would differ from the train-only values.
    assert abs(pre.means["age"] - float(dataset_frame["age"].mean())) < 5.0


def test_transform_is_deterministic(preprocessing):
    clinical = {
        "age": 30, "gravidity": 2, "parity": 1, "gest_weeks": 39,
        "diabetes": 0, "hypertension": 0, "preeclampsia": 0, "liquor_praecox": 0,
        "pyrexia": 0, "meconium": 0, "induced": 0, "presentation": 1,
    }
    first = preprocessing["clinical"].transform_one(clinical)
    second = preprocessing["clinical"].transform_one(clinical)
    assert np.array_equal(first, second)
