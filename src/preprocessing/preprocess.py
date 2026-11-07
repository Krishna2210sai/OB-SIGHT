"""Deterministic preprocessing for CTG signals and maternal clinical data.

The same functions are used by the training pipeline, the evaluation scripts,
the explainability scripts and the FastAPI backend.  Fitted transformations are
stored with joblib so that inference cannot silently diverge from training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FHR_INTERP_MAX_SEC,
    FHR_VALID_MAX,
    FHR_VALID_MIN,
    FS_NATIVE,
    NUMERIC_FEATURES,
    SIGNAL_CHANNELS,
    TARGET_FS,
    UC_INTERP_MAX_SEC,
    WINDOW_MIN,
    WINDOW_SAMPLES,
)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def trim_recording(fhr: np.ndarray, uc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove leading/trailing samples where *both* channels are missing.

    The WFDB files are zero-padded at the end to a whole number of minutes.
    Without trimming, a "last 30 minutes" window could consist mostly of pad.
    """
    fhr = np.asarray(fhr, dtype=np.float32)
    uc = np.asarray(uc, dtype=np.float32)
    valid = (fhr > 0) | (uc > 0)
    if not valid.any():
        return fhr[:0], uc[:0]
    first = int(np.argmax(valid))
    last = len(valid) - int(np.argmax(valid[::-1]))
    return fhr[first:last], uc[first:last]


def _interpolate_short_gaps(
    values: np.ndarray, valid: np.ndarray, fs: float, max_gap_sec: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of short gaps; long gaps remain missing.

    Returns ``(filled, still_missing_mask)``.  Missing samples are encoded as
    0.0 in ``filled`` and 1.0 in ``still_missing_mask``.
    """
    values = np.asarray(values, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=bool)
    filled = values.copy()
    filled[~valid] = 0.0
    if valid.sum() < 2 or valid.all():
        return filled, (~valid).astype(np.float32)

    idx = np.flatnonzero(valid)
    interp = np.interp(np.arange(len(values)), idx, values[valid]).astype(np.float32)

    # Identify contiguous invalid runs and only accept runs up to max_gap_sec.
    changes = np.diff(np.r_[0, (~valid).astype(np.int8), 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    max_len = max(1, int(round(max_gap_sec * fs)))
    still_missing = (~valid).astype(np.float32)
    for start, end in zip(starts, ends):
        if (end - start) <= max_len:
            filled[start:end] = interp[start:end]
            still_missing[start:end] = 0.0
    return filled, still_missing


def resample_to_native(fhr: np.ndarray, uc: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample an arbitrary input sampling rate to 4 Hz by linear interpolation."""
    if abs(fs - FS_NATIVE) < 1e-9:
        return fhr.astype(np.float32), uc.astype(np.float32)
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    n_out = int(round(len(fhr) * FS_NATIVE / fs))
    t_in = np.arange(len(fhr), dtype=np.float64) / fs
    t_out = np.arange(n_out, dtype=np.float64) / FS_NATIVE
    fhr_out = np.interp(t_out, t_in, fhr.astype(np.float64)).astype(np.float32)
    uc_out = np.interp(t_out, t_in, uc.astype(np.float64)).astype(np.float32)
    return fhr_out, uc_out


def extract_window(
    fhr: np.ndarray,
    uc: np.ndarray,
    fs: float = FS_NATIVE,
    window_min: int = WINDOW_MIN,
    target_fs: int = TARGET_FS,
) -> dict:
    """Extract the most recent ``window_min`` minutes and return 1 Hz channels.

    Returns a dict with ``fhr``, ``uc`` (filled), ``fhr_mask``, ``uc_mask``
    (fraction of valid native samples in each target sample), and diagnostics.
    No fitted scaling is applied here.
    """
    fhr = np.asarray(fhr, dtype=np.float32).reshape(-1)
    uc = np.asarray(uc, dtype=np.float32).reshape(-1)
    if fhr.shape != uc.shape:
        raise ValueError("FHR and UC must have the same length")
    if len(fhr) == 0:
        raise ValueError("Empty signal")

    fhr, uc = trim_recording(fhr, uc)
    if len(fhr) == 0:
        raise ValueError("Signal contains no valid samples")
    fhr, uc = resample_to_native(fhr, uc, fs)
    duration_min_available = float(len(fhr) / FS_NATIVE / 60.0)

    n_window = int(window_min * 60 * FS_NATIVE)
    if len(fhr) > n_window:
        fhr = fhr[-n_window:]
        uc = uc[-n_window:]
    padded = len(fhr) < n_window
    if padded:
        pad = n_window - len(fhr)
        fhr = np.pad(fhr, (pad, 0))
        uc = np.pad(uc, (pad, 0))

    fhr_valid = (fhr > 0) & (fhr >= FHR_VALID_MIN) & (fhr <= FHR_VALID_MAX)
    uc_valid = uc > 0

    fhr_filled, fhr_still_missing = _interpolate_short_gaps(
        fhr, fhr_valid, FS_NATIVE, FHR_INTERP_MAX_SEC
    )
    uc_filled, uc_still_missing = _interpolate_short_gaps(
        uc, uc_valid, FS_NATIVE, UC_INTERP_MAX_SEC
    )
    # Never carry values into regions that were already pad or long gaps.
    # Short gaps that were successfully interpolated count as observed for the
    # mask channel so that the interpolated value is not discarded later.
    fhr_mask_native = (1.0 - fhr_still_missing).astype(np.float32)
    uc_mask_native = (1.0 - uc_still_missing).astype(np.float32)

    factor = FS_NATIVE // target_fs
    if FS_NATIVE % target_fs != 0:
        raise ValueError("target_fs must divide the native sampling frequency")

    def _block_mean(x: np.ndarray) -> np.ndarray:
        return x.reshape(-1, factor).mean(axis=1)

    def _masked_block_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        xr = x.reshape(-1, factor)
        mr = mask.reshape(-1, factor)
        counts = mr.sum(axis=1)
        out = np.divide(
            (xr * mr).sum(axis=1),
            counts,
            out=np.zeros(len(counts), dtype=np.float32),
            where=counts > 0,
        )
        return out

    fhr2 = _masked_block_mean(fhr_filled, fhr_mask_native)
    uc2 = _masked_block_mean(uc_filled, uc_mask_native)
    fhr_mask2 = _block_mean(fhr_mask_native)
    uc_mask2 = _block_mean(uc_mask_native)

    return {
        "fhr": fhr2.astype(np.float32),
        "uc": uc2.astype(np.float32),
        "fhr_mask": fhr_mask2.astype(np.float32),
        "uc_mask": uc_mask2.astype(np.float32),
        "fhr_missing_fraction": float(1.0 - fhr_mask2.mean()),
        "uc_missing_fraction": float(1.0 - uc_mask2.mean()),
        "padded": padded,
        "duration_min_available": duration_min_available,
    }


# ---------------------------------------------------------------------------
# Fitted signal scaler
# ---------------------------------------------------------------------------
@dataclass
class SignalPreprocessor:
    fhr_mean: float = 0.0
    fhr_std: float = 1.0
    uc_mean: float = 0.0
    uc_std: float = 1.0
    window_min: int = WINDOW_MIN
    target_fs: int = TARGET_FS

    def fit(self, windows: Iterable[dict]) -> "SignalPreprocessor":
        fhr_values, uc_values = [], []
        for window in windows:
            f = window["fhr"][window["fhr_mask"] > 0]
            u = window["uc"][window["uc_mask"] > 0]
            if f.size:
                fhr_values.append(f)
            if u.size:
                uc_values.append(u)
        fhr_all = np.concatenate(fhr_values) if fhr_values else np.array([135.0], dtype=np.float32)
        uc_all = np.concatenate(uc_values) if uc_values else np.array([20.0], dtype=np.float32)
        self.fhr_mean = float(np.mean(fhr_all))
        self.fhr_std = float(np.std(fhr_all)) or 1.0
        self.uc_mean = float(np.mean(uc_all))
        self.uc_std = float(np.std(uc_all)) or 1.0
        return self

    def transform_window(self, window: dict) -> np.ndarray:
        fhr = (window["fhr"] - self.fhr_mean) / self.fhr_std
        uc = (window["uc"] - self.uc_mean) / self.uc_std
        # Missing regions are represented by zero + mask, never by an
        # arbitrary physiological value.
        fhr = np.where(window["fhr_mask"] > 0, fhr, 0.0)
        uc = np.where(window["uc_mask"] > 0, uc, 0.0)
        fhr = np.clip(fhr, -5.0, 5.0)
        uc = np.clip(uc, -5.0, 5.0)
        return np.stack([fhr, uc, window["fhr_mask"], window["uc_mask"]], axis=0).astype(np.float32)

    def transform(self, fhr: np.ndarray, uc: np.ndarray, fs: float = FS_NATIVE) -> Tuple[np.ndarray, dict]:
        window = extract_window(fhr, uc, fs=fs, window_min=self.window_min, target_fs=self.target_fs)
        return self.transform_window(window), window

    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str) -> "SignalPreprocessor":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected SignalPreprocessor, got {type(obj)}")
        return obj


# ---------------------------------------------------------------------------
# Fitted clinical preprocessor
# ---------------------------------------------------------------------------
@dataclass
class ClinicalPreprocessor:
    numeric_features: List[str] = field(default_factory=lambda: list(NUMERIC_FEATURES))
    binary_features: List[str] = field(default_factory=lambda: list(BINARY_FEATURES))
    categorical_features: List[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    medians: Dict[str, float] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)
    modes: Dict[str, float] = field(default_factory=dict)
    categories: List[float] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "ClinicalPreprocessor":
        for col in self.numeric_features:
            values = pd.to_numeric(df[col], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            self.medians[col] = median
            self.means[col] = float(filled.mean())
            self.stds[col] = float(filled.std(ddof=0)) or 1.0
        for col in self.binary_features:
            values = pd.to_numeric(df[col], errors="coerce")
            self.modes[col] = 0.0
        for col in self.categorical_features:
            values = pd.to_numeric(df[col], errors="coerce")
            mode = float(values.mode().iloc[0]) if values.notna().any() else 1.0
            self.modes[col] = mode
            self.categories = sorted(float(v) for v in values.dropna().unique())
            if not self.categories:
                self.categories = [1.0]
        self._build_output_names()
        return self

    def _build_output_names(self) -> None:
        names = list(self.numeric_features) + list(self.binary_features)
        for col in self.categorical_features:
            names.extend([f"{col}_{int(c)}" for c in self.categories])
            names.append(f"{col}_missing")
        self.output_names = names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.output_names:
            raise RuntimeError("ClinicalPreprocessor is not fitted")
        parts = []
        numeric = np.empty((len(df), len(self.numeric_features)), dtype=np.float32)
        for j, col in enumerate(self.numeric_features):
            values = pd.to_numeric(df[col], errors="coerce").fillna(self.medians[col]).astype(float)
            numeric[:, j] = (values - self.means[col]) / self.stds[col]
        parts.append(numeric)

        binary = np.empty((len(df), len(self.binary_features)), dtype=np.float32)
        for j, col in enumerate(self.binary_features):
            binary[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
        parts.append(binary)

        for col in self.categorical_features:
            values = pd.to_numeric(df[col], errors="coerce")
            missing = values.isna().astype(np.float32).to_numpy()[:, None]
            filled = values.fillna(self.modes[col])
            onehot = np.stack(
                [(filled == category).astype(np.float32).to_numpy() for category in self.categories],
                axis=1,
            )
            parts.append(onehot)
            parts.append(missing)

        out = np.concatenate(parts, axis=1).astype(np.float32)
        if out.shape[1] != len(self.output_names):
            raise RuntimeError("Clinical feature layout changed between fit and transform")
        return out

    def transform_one(self, values: Dict[str, float]) -> np.ndarray:
        df = pd.DataFrame([{name: values.get(name, np.nan) for name in
                            self.numeric_features + self.binary_features + self.categorical_features}])
        return self.transform(df)[0]

    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str) -> "ClinicalPreprocessor":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ClinicalPreprocessor, got {type(obj)}")
        return obj


def signal_array_from_window(window: dict, preprocessor: SignalPreprocessor) -> np.ndarray:
    """Convenience helper used by the API and tests."""
    return preprocessor.transform_window(window)
