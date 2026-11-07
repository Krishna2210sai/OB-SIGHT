"""Dataset integrity and split-integrity tests.

These tests read the read-only source dataset but never write to it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATASET_DIR, LABEL_MAP, NUMERIC_FEATURES
from src.data.load_dataset import iter_record_ids, parse_header, read_signals


def test_source_dataset_is_present_and_complete():
    headers = sorted(DATASET_DIR.glob("*.hea"))
    data = sorted(DATASET_DIR.glob("*.dat"))
    assert len(headers) == 552
    assert len(data) == 552
    assert {p.stem for p in headers} == {p.stem for p in data}
    assert (DATASET_DIR / "RECORDS").exists()
    assert (DATASET_DIR / "SHA256SUMS.txt").exists()


def test_record_id_ranges_match_documented_delivery_types():
    ids = iter_record_ids()
    assert len(ids) == 552
    assert ids[0] == "1001"
    assert ids[-1] == "2046"
    assert sum(1 for r in ids if r.startswith("2")) == 46
    assert sum(1 for r in ids if r.startswith("1")) == 506


def test_header_metadata_and_label_mapping():
    vaginal = parse_header(DATASET_DIR / "1001.hea")
    cesarean = parse_header(DATASET_DIR / "2001.hea")

    assert vaginal.meta["deliv_type"] == 1
    assert vaginal.label == LABEL_MAP[1] == 0
    assert cesarean.meta["deliv_type"] == 2
    assert cesarean.label == LABEL_MAP[2] == 1
    for header in (vaginal, cesarean):
        assert header.n_signals == 2
        assert header.fs == 4.0
        assert header.n_samples > 0
        assert header.meta["age"] > 0


def test_raw_signal_layout_matches_header():
    header = parse_header(DATASET_DIR / "1001.hea")
    signals = read_signals("1001")
    assert signals.shape == (header.n_samples, 2)
    assert signals.dtype == np.float32
    # Values are physical units; zero denotes missing, not a negative FHR.
    assert np.nanmin(signals) >= 0
    assert np.nanmax(signals[:, 0]) <= 300
    # UC is an uncalibrated tocodynamometer channel; values above 120 are
    # physiologically plausible artefacts of the device scaling.
    assert np.nanmax(signals[:, 1]) <= 255


def test_processed_arrays_are_aligned_and_finite(signals, clinical):
    assert signals["X"].shape == (552, 4, 1800)
    assert clinical["X"].shape == (552, 15)
    assert signals["y"].shape == clinical["y"].shape == (552,)
    assert signals["X"].dtype == np.float32
    assert clinical["X"].dtype == np.float32
    assert np.isfinite(signals["X"]).all()
    assert np.isfinite(clinical["X"]).all()
    assert np.array_equal(signals["record_ids"], clinical["record_ids"])
    assert np.array_equal(signals["split"], clinical["split"])
    assert np.array_equal(signals["y"], clinical["y"])


def test_split_sizes_and_stratification(signals):
    split = signals["split"]
    y = signals["y"]
    assert set(np.unique(split)) == {"train", "val", "test"}
    assert dict(zip(*np.unique(split, return_counts=True))) == {"test": 83, "train": 386, "val": 83}
    for name in ["train", "val", "test"]:
        mask = split == name
        assert y[mask].sum() > 0, f"{name} split contains no cesarean cases"
    assert y[split == "train"].sum() == 32
    assert y[split == "val"].sum() == 7
    assert y[split == "test"].sum() == 7


def test_splits_are_disjoint_and_unique(dataset_frame):
    assert dataset_frame["record_id"].is_unique
    assert set(dataset_frame["split"]) == {"train", "val", "test"}
    per_split = [set(dataset_frame.loc[dataset_frame["split"] == s, "record_id"]) for s in ["train", "val", "test"]]
    assert not (per_split[0] & per_split[1])
    assert not (per_split[0] & per_split[2])
    assert not (per_split[1] & per_split[2])
    assert sum(len(s) for s in per_split) == 552


def test_clinical_columns_exclude_post_delivery_leakage(dataset_frame):
    leakage = {
        "ph",
        "bdecf",
        "pco2",
        "be",
        "apgar1",
        "apgar5",
        "nicu_days",
        "seizures",
        "hie",
        "intubation",
        "main_diag",
        "other_diag",
        "weight_g",
        "sex",
        "ii_stage",
        "noprogress",
        "ck_kp",
        "i_stage",
        "pos_ii_st",
        "rec_type",
        "sig2birth",
        "deliv_type",
        "label",
    }
    used = set(NUMERIC_FEATURES) | {
        "diabetes",
        "hypertension",
        "preeclampsia",
        "liquor_praecox",
        "pyrexia",
        "meconium",
        "induced",
        "presentation",
    }
    assert used <= set(dataset_frame.columns)
    assert used.isdisjoint(leakage)
    # The target must be the delivery type, not a fetal-status class.
    assert dataset_frame["label"].isin([0, 1]).all()
    assert dataset_frame["label"].sum() == 46
