"""Development-only data loading for Phase B.

The loader reads the Phase A 30-minute / 1 Hz cache.  It fails loudly if a
record from the held-out test split appears anywhere in the bundle.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RANDOM_SEED
from src.phase_a.features import (
    CLINICAL_RAW,
    build_feature_dataset,
    candidate_feature_names,
    load_development_metadata,
    load_raw_signals,
)
from src.phase_b.config import (
    INNER_FOLDS,
    OUTER_FOLDS,
    PHASE_A_CACHE,
    PHASE_A_FOLDS,
    PHASE_B_DIR,
    SEED,
    TARGET_FS,
    WINDOW_MIN,
)


@dataclass
class DevelopmentBundle:
    frame: pd.DataFrame
    windows: Dict[str, Dict]
    folds: pd.DataFrame
    feature_names: List[str]
    clinical_raw: List[str]

    def __post_init__(self) -> None:
        if self.frame["record_id"].duplicated().any():
            raise ValueError("Duplicate development record IDs")
        if self.folds["record_id"].duplicated().any():
            raise ValueError("Duplicate fold record IDs")
        if set(self.frame["record_id"]) != set(self.folds["record_id"]):
            raise ValueError("Feature frame and folds contain different records")
        if "test" in set(self.frame["split"].astype(str)):
            raise ValueError("Held-out test records leaked into Phase B")
        if not set(self.folds["fold"]).issubset(set(range(1, OUTER_FOLDS + 1))):
            raise ValueError("Unexpected outer fold labels")

    @property
    def y(self) -> np.ndarray:
        return self.frame["label"].to_numpy(dtype=int)

    @property
    def record_ids(self) -> List[str]:
        return self.frame["record_id"].astype(str).tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_cache() -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    metadata = load_development_metadata()
    raw = load_raw_signals(metadata["record_id"].astype(str))
    frame, windows, insufficient = build_feature_dataset(
        metadata, raw, window_min=WINDOW_MIN, target_fs=TARGET_FS
    )
    if insufficient:
        raise RuntimeError(
            f"Phase A cache expected all 469 development records, {len(insufficient)} were excluded"
        )
    PHASE_A_CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PHASE_A_CACHE / "features.csv", index=False)
    ids = list(windows)
    np.savez_compressed(
        PHASE_A_CACHE / "windows.npz",
        record_id=np.array(ids),
        fhr=np.stack([windows[rid]["fhr"] for rid in ids]),
        uc=np.stack([windows[rid]["uc"] for rid in ids]),
        fhr_mask=np.stack([windows[rid]["fhr_mask"] for rid in ids]),
        uc_mask=np.stack([windows[rid]["uc_mask"] for rid in ids]),
    )
    return frame, windows


def _load_windows(path: Path) -> Dict[str, Dict]:
    with np.load(path) as data:
        record_ids = [str(value) for value in data["record_id"]]
        # Materialise each array once; indexing NpzFile re-decompresses.
        fhr = data["fhr"]
        uc = data["uc"]
        fhr_mask = data["fhr_mask"]
        uc_mask = data["uc_mask"]
        return {
            rid: {
                "record_id": rid,
                "fhr": fhr[i],
                "uc": uc[i],
                "fhr_mask": fhr_mask[i],
                "uc_mask": uc_mask[i],
                "window_min": WINDOW_MIN,
                "target_fs": TARGET_FS,
            }
            for i, rid in enumerate(record_ids)
        }


def load_development_bundle() -> DevelopmentBundle:
    features_path = PHASE_A_CACHE / "features.csv"
    windows_path = PHASE_A_CACHE / "windows.npz"
    if not features_path.exists() or not windows_path.exists():
        frame, windows = _build_cache()
    else:
        frame = pd.read_csv(features_path)
        frame["record_id"] = frame["record_id"].astype(str)
        windows = _load_windows(windows_path)

    if not PHASE_A_FOLDS.exists():
        raise FileNotFoundError(
            "experiments/phase_a/folds.csv is missing. Run the Phase A fold stage first."
        )
    folds = pd.read_csv(PHASE_A_FOLDS)
    folds["record_id"] = folds["record_id"].astype(str)
    folds = folds.sort_values(["fold", "record_id"]).reset_index(drop=True)

    frame = frame.sort_values("record_id").reset_index(drop=True)
    folds = folds.set_index("record_id").loc[frame["record_id"]].reset_index()
    if not (folds["label"].to_numpy(int) == frame["label"].to_numpy(int)).all():
        raise ValueError("Fold labels do not match the feature frame")

    bundle = DevelopmentBundle(
        frame=frame,
        windows=windows,
        folds=folds,
        feature_names=candidate_feature_names(frame),
        clinical_raw=list(CLINICAL_RAW),
    )
    if len(bundle.frame) != 469:
        raise ValueError(f"Expected 469 development records, found {len(bundle.frame)}")
    return bundle


def make_inner_folds(
    labels: np.ndarray,
    indices: np.ndarray,
    n_splits: int = INNER_FOLDS,
    seed: int = SEED,
) -> np.ndarray:
    """Deterministic stratified inner folds inside one outer training set.

    ``labels`` is the full development label vector and ``indices`` are the
    outer-training row positions.  The returned array has length
    ``len(indices)`` and values ``1..n_splits``.  Only records in ``indices``
    are used, so an inner fold can never contain an outer validation record.
    """
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels).astype(int)
    indices = np.asarray(indices).astype(int)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("indices must be a non-empty 1-D array")
    subset = labels[indices]
    if len(np.unique(subset)) < 2:
        raise ValueError("Inner folds need both classes")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner = np.full(len(indices), -1, dtype=int)
    for fold, (_, val_local) in enumerate(skf.split(np.zeros(len(subset)), subset), start=1):
        inner[val_local] = fold
    if (inner < 0).any():
        raise RuntimeError("Inner fold assignment is incomplete")
    return inner


def outer_train_validation(folds: pd.DataFrame, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    labels = folds["fold"].to_numpy(int)
    train_idx = np.flatnonzero(labels != fold)
    val_idx = np.flatnonzero(labels == fold)
    if set(labels[val_idx].tolist()) != {fold}:
        raise RuntimeError("Outer fold isolation check failed")
    if np.intersect1d(train_idx, val_idx).size:
        raise RuntimeError("Outer train and validation overlap")
    return train_idx, val_idx


def write_manifest(bundle: DevelopmentBundle, path: Path) -> Dict:
    dataset_path = PROCESSED_DIR / "dataset.csv"
    payload = {
        "dataset": "CTU-CHB Intrapartum Cardiotocography Database v1.0.0",
        "window_minutes": WINDOW_MIN,
        "target_fs": TARGET_FS,
        "n_records": int(len(bundle.frame)),
        "n_cesarean": int(bundle.y.sum()),
        "split_counts": {
            str(k): int(v) for k, v in bundle.frame["split"].value_counts().items()
        },
        "record_ids": bundle.record_ids,
        "outer_folds": int(bundle.folds["fold"].nunique()),
        "outer_fold_sizes": {
            str(k): int(v) for k, v in bundle.folds["fold"].value_counts().sort_index().items()
        },
        "outer_fold_cesareans": {
            str(k): int(v)
            for k, v in bundle.folds.groupby("fold")["label"].sum().sort_index().items()
        },
        "processed_table_sha256": _sha256(dataset_path),
        "phase_a_folds": str(PHASE_A_FOLDS),
        "seed": SEED,
        "test_split_used": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def align_arrays(bundle: DevelopmentBundle) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Feature matrix, labels and record IDs in the bundle order."""
    X = bundle.frame[bundle.feature_names].to_numpy(dtype=float)
    y = bundle.y
    return X, y, bundle.folds["fold"].to_numpy(int), bundle.record_ids


def save_phase_b_folds(bundle: DevelopmentBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle.folds[["record_id", "label", "split", "fold"]].to_csv(path, index=False)
