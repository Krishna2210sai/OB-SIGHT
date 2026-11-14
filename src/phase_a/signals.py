"""Window extraction with anti-aliasing and explicit duration policy.

Differences from the production preprocessing are intentional and isolated:

* records that cannot supply the requested window are excluded (default) or
  left-padded with an explicit mask; they are never silently padded;
* downsampling uses a zero-phase Butterworth low-pass filter before
  decimation, instead of linear interpolation;
* only samples inside the requested window are filtered, so no information
  after the window end can leak into the input;
* long gaps remain missing (mask 0); short gaps are interpolated exactly as
  in the production pipeline, and both decisions are recorded in the mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.config import FHR_INTERP_MAX_SEC, FHR_VALID_MAX, FHR_VALID_MIN, FS_NATIVE, UC_INTERP_MAX_SEC

WINDOW_POLICIES = ("exclude", "pad_masked")


# ---------------------------------------------------------------------------
# Gap handling
# ---------------------------------------------------------------------------
def interpolate_short_gaps(
    values: np.ndarray,
    valid: np.ndarray,
    fs: float,
    max_gap_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of short gaps; long gaps remain missing.

    Returns ``(filled, still_missing)`` where missing samples are 0.0 in
    ``filled`` and 1.0 in ``still_missing``.
    """
    values = np.asarray(values, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=bool)
    filled = values.copy()
    filled[~valid] = 0.0
    still_missing = (~valid).astype(np.float32)
    if valid.sum() < 2 or valid.all():
        return filled, still_missing

    idx = np.flatnonzero(valid)
    interp = np.interp(np.arange(len(values)), idx, values[valid]).astype(np.float32)
    changes = np.diff(np.r_[0, (~valid).astype(np.int8), 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    max_len = max(1, int(round(max_gap_sec * fs)))
    for start, end in zip(starts, ends):
        if (end - start) <= max_len:
            filled[start:end] = interp[start:end]
            still_missing[start:end] = 0.0
    return filled, still_missing


def trim_recording(fhr: np.ndarray, uc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove leading/trailing samples where both channels are missing."""
    fhr = np.asarray(fhr, dtype=np.float32).reshape(-1)
    uc = np.asarray(uc, dtype=np.float32).reshape(-1)
    if fhr.shape != uc.shape:
        raise ValueError("FHR and UC must have the same length")
    valid = (fhr > 0) | (uc > 0)
    if not valid.any():
        return fhr[:0], uc[:0]
    first = int(np.argmax(valid))
    last = len(valid) - int(np.argmax(valid[::-1]))
    return fhr[first:last], uc[first:last]


# ---------------------------------------------------------------------------
# Anti-aliased resampling
# ---------------------------------------------------------------------------
def _lowpass(x: np.ndarray, cutoff_hz: float, fs_in: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass, applied inside the window only."""
    nyquist = fs_in / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        return x
    sos = butter(4, cutoff_hz / nyquist, btype="low", output="sos")
    # sosfiltfilt needs a signal longer than its padding; skip when very short.
    padlen = 3 * (2 * len(sos) + 1)
    if len(x) <= padlen:
        return x
    return sosfiltfilt(sos, x)


def resample_channel(
    values: np.ndarray,
    valid: np.ndarray,
    fs_in: float,
    fs_out: float,
    channel: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decimate a masked channel with an anti-aliasing filter.

    Missing samples are filled by linear interpolation only for the numerical
    filtering step; the output mask still records which output samples were
    supported by a majority of valid native samples.  Long gaps therefore stay
    missing in the model input.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if values.shape != valid.shape:
        raise ValueError("values and valid must have the same length")
    factor = int(round(fs_in / fs_out))
    if factor < 1 or abs(fs_in / factor - fs_out) > 1e-9:
        raise ValueError(f"{fs_out} Hz is not an integer decimation of {fs_in} Hz")

    if factor == 1:
        return values.astype(np.float32), valid.astype(np.float32)

    if valid.sum() < 2:
        return np.zeros(len(values) // factor, dtype=np.float32), np.zeros(len(values) // factor, dtype=np.float32)

    idx = np.flatnonzero(valid)
    interp = np.interp(np.arange(len(values)), idx, values[valid])
    cutoff = (0.40 if channel == "fhr" else 0.20) * fs_out
    filtered = _lowpass(interp, cutoff, fs_in)

    n_out = len(filtered) // factor
    filtered = filtered[: n_out * factor].reshape(n_out, factor)
    valid_blocks = valid[: n_out * factor].reshape(n_out, factor)
    mask = valid_blocks.mean(axis=1) >= 0.5
    out = filtered.mean(axis=1)
    out[~mask] = 0.0
    return out.astype(np.float32), mask.astype(np.float32)


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------
def extract_recent_window(
    fhr: np.ndarray,
    uc: np.ndarray,
    fs: float = FS_NATIVE,
    window_min: int = 30,
    target_fs: float = 1.0,
    policy: str = "exclude",
    record_id: Optional[str] = None,
) -> Optional[Dict]:
    """Extract the most recent ``window_min`` minutes at ``target_fs``.

    Returns ``None`` when the recording is shorter than the window and
    ``policy='exclude'`` (the default).  ``policy='pad_masked'`` left-pads with
    zeros and sets the mask to 0; this is explicit and recorded in the output.
    """
    if policy not in WINDOW_POLICIES:
        raise ValueError(f"Unknown window policy: {policy}")
    fhr = np.asarray(fhr, dtype=np.float32).reshape(-1)
    uc = np.asarray(uc, dtype=np.float32).reshape(-1)
    if fhr.shape != uc.shape:
        raise ValueError("FHR and UC must have the same length")
    if len(fhr) == 0:
        return None

    fhr, uc = trim_recording(fhr, uc)
    if len(fhr) == 0:
        return None
    duration_min_available = float(len(fhr) / fs / 60.0)
    n_native = int(round(window_min * 60 * fs))
    padded = False
    if len(fhr) < n_native:
        if policy == "exclude":
            return None
        pad = n_native - len(fhr)
        fhr = np.pad(fhr, (pad, 0))
        uc = np.pad(uc, (pad, 0))
        padded = True
    else:
        fhr = fhr[-n_native:]
        uc = uc[-n_native:]
        pad = 0

    # Masks are defined on the cropped window, before resampling.
    fhr_valid = (fhr > 0) & (fhr >= FHR_VALID_MIN) & (fhr <= FHR_VALID_MAX)
    uc_valid = uc > 0
    fhr_filled, fhr_still_missing = interpolate_short_gaps(fhr, fhr_valid, fs, FHR_INTERP_MAX_SEC)
    uc_filled, uc_still_missing = interpolate_short_gaps(uc, uc_valid, fs, UC_INTERP_MAX_SEC)
    fhr_mask_native = (1.0 - fhr_still_missing).astype(np.float32)
    uc_mask_native = (1.0 - uc_still_missing).astype(np.float32)

    fhr_out, fhr_mask = resample_channel(fhr_filled, fhr_mask_native > 0, fs, target_fs, "fhr")
    uc_out, uc_mask = resample_channel(uc_filled, uc_mask_native > 0, fs, target_fs, "uc")

    expected = int(round(window_min * 60 * target_fs))
    if len(fhr_out) != expected or len(uc_out) != expected:
        raise RuntimeError(
            f"Unexpected window length for {record_id}: {len(fhr_out)} / {len(uc_out)} (expected {expected})"
        )

    return {
        "record_id": record_id,
        "fhr": fhr_out,
        "uc": uc_out,
        "fhr_mask": fhr_mask,
        "uc_mask": uc_mask,
        "window_min": int(window_min),
        "target_fs": float(target_fs),
        "duration_min_available": duration_min_available,
        "padded": padded,
        "pad_samples_min": float(pad / fs / 60.0),
        "fhr_missing_fraction": float(1.0 - fhr_mask.mean()),
        "uc_missing_fraction": float(1.0 - uc_mask.mean()),
    }


# ---------------------------------------------------------------------------
# Fold-specific signal scaling
# ---------------------------------------------------------------------------
@dataclass
class SignalScaler:
    fhr_mean: float = 0.0
    fhr_std: float = 1.0
    uc_mean: float = 0.0
    uc_std: float = 1.0

    def fit(self, windows: Iterable[Dict]) -> "SignalScaler":
        fhr_values: List[np.ndarray] = []
        uc_values: List[np.ndarray] = []
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

    def transform(self, window: Dict) -> np.ndarray:
        fhr = (window["fhr"].astype(np.float32) - self.fhr_mean) / self.fhr_std
        uc = (window["uc"].astype(np.float32) - self.uc_mean) / self.uc_std
        fhr = np.where(window["fhr_mask"] > 0, fhr, 0.0)
        uc = np.where(window["uc_mask"] > 0, uc, 0.0)
        fhr = np.clip(fhr, -5.0, 5.0)
        uc = np.clip(uc, -5.0, 5.0)
        return np.stack([fhr, uc, window["fhr_mask"], window["uc_mask"]], axis=0).astype(np.float32)

    def to_dict(self) -> Dict[str, float]:
        return {
            "fhr_mean": self.fhr_mean,
            "fhr_std": self.fhr_std,
            "uc_mean": self.uc_mean,
            "uc_std": self.uc_std,
        }

    def save(self, path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "SignalScaler":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected SignalScaler, got {type(obj)}")
        return obj


def stack_windows(windows: List[Dict], scaler: SignalScaler) -> np.ndarray:
    return np.stack([scaler.transform(w) for w in windows], axis=0).astype(np.float32)
