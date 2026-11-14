"""Hand-crafted CTG descriptors used by the baseline models and the API.

These are deliberately simple engineering approximations of descriptive CTG
features (baseline level, variability, accelerations, decelerations,
contractions).  They are *not* a validated FIGO scoring system and must not be
interpreted clinically.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

from src.config import TARGET_FS

FEATURE_DESCRIPTIONS = {
    "fhr_mean": "Mean FHR in the 30-minute window (bpm)",
    "fhr_median": "Median FHR (approximate baseline, bpm)",
    "fhr_std": "Standard deviation of FHR (bpm)",
    "fhr_min": "Minimum FHR (bpm)",
    "fhr_max": "Maximum FHR (bpm)",
    "fhr_p5": "5th percentile of FHR (bpm)",
    "fhr_p95": "95th percentile of FHR (bpm)",
    "fhr_range": "p95 - p5 FHR range (bpm)",
    "pct_fhr_below_110": "Fraction of time with FHR < 110 bpm",
    "pct_fhr_above_160": "Fraction of time with FHR > 160 bpm",
    "pct_fhr_below_100": "Fraction of time with FHR < 100 bpm",
    "pct_fhr_above_180": "Fraction of time with FHR > 180 bpm",
    "fhr_missing_fraction": "Fraction of the window without usable FHR",
    "fhr_variability_sd_1min": "Mean within-minute FHR standard deviation (bpm)",
    "acceleration_count": "Approximate accelerations (>15 bpm, >=15 s)",
    "deceleration_count": "Approximate decelerations (<-15 bpm, >=15 s)",
    "deep_deceleration_count": "Approximate deep decelerations (<-30 bpm, >=15 s)",
    "acceleration_fraction": "Fraction of time in approximate accelerations",
    "deceleration_fraction": "Fraction of time in approximate decelerations",
    "uc_mean": "Mean uterine contraction signal (a.u.)",
    "uc_std": "Standard deviation of uterine contraction signal",
    "uc_max": "Maximum uterine contraction signal (a.u.)",
    "uc_p90": "90th percentile of uterine contraction signal",
    "uc_contraction_count": "Approximate contractions (peaks, >=60 s apart)",
    "uc_missing_fraction": "Fraction of the window without usable UC",
}

FEATURE_NAMES: List[str] = list(FEATURE_DESCRIPTIONS)


def _safe(values: np.ndarray, default: float = 0.0) -> float:
    return float(values) if np.isfinite(values) else default


def _count_runs(mask: np.ndarray, min_len: int) -> int:
    if not mask.any():
        return 0
    changes = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.sum((ends - starts) >= min_len))


def extract_ctg_features(window: Dict[str, np.ndarray], fs: float = TARGET_FS) -> Dict[str, float]:
    """Compute descriptors from a window produced by ``extract_window``.

    ``fs`` may be fractional (for example 0.5 Hz in the Phase A sampling
    experiments); filter sizes and peak distances are rounded to whole samples.
    """
    fs = float(fs)
    fhr = np.asarray(window["fhr"], dtype=np.float64)
    uc = np.asarray(window["uc"], dtype=np.float64)
    fhr_mask = np.asarray(window["fhr_mask"], dtype=np.float64) > 0
    uc_mask = np.asarray(window["uc_mask"], dtype=np.float64) > 0

    out: Dict[str, float] = {}
    if fhr_mask.sum() >= 2:
        valid = fhr[fhr_mask]
        out.update(
            fhr_mean=_safe(valid.mean()),
            fhr_median=_safe(np.median(valid)),
            fhr_std=_safe(valid.std()),
            fhr_min=_safe(valid.min()),
            fhr_max=_safe(valid.max()),
            fhr_p5=_safe(np.percentile(valid, 5)),
            fhr_p95=_safe(np.percentile(valid, 95)),
            pct_fhr_below_110=_safe(np.mean(valid < 110)),
            pct_fhr_above_160=_safe(np.mean(valid > 160)),
            pct_fhr_below_100=_safe(np.mean(valid < 100)),
            pct_fhr_above_180=_safe(np.mean(valid > 180)),
        )
        out["fhr_range"] = out["fhr_p95"] - out["fhr_p5"]
    else:
        for name in [
            "fhr_mean", "fhr_median", "fhr_std", "fhr_min", "fhr_max", "fhr_p5",
            "fhr_p95", "fhr_range", "pct_fhr_below_110", "pct_fhr_above_160",
            "pct_fhr_below_100", "pct_fhr_above_180",
        ]:
            out[name] = 0.0
    out["fhr_missing_fraction"] = float(1.0 - fhr_mask.mean())

    # Interpolate for episode detection only (the DL input keeps masks).
    if fhr_mask.sum() >= 2:
        idx = np.flatnonzero(fhr_mask)
        fhr_i = np.interp(np.arange(len(fhr)), idx, fhr[idx])
    else:
        fhr_i = np.full_like(fhr, 135.0)

    # 1-minute variability: mean within-minute SD of valid samples.
    minute = max(1, int(round(fs * 60)))
    n_min = len(fhr) // minute
    sds = []
    for k in range(n_min):
        seg = fhr[k * minute : (k + 1) * minute]
        seg_mask = fhr_mask[k * minute : (k + 1) * minute]
        if seg_mask.sum() > 5:
            sds.append(float(seg[seg_mask].std()))
    out["fhr_variability_sd_1min"] = float(np.mean(sds)) if sds else 0.0

    baseline = median_filter(fhr_i, size=max(1, int(round(fs * 300))), mode="nearest")
    deviation = fhr_i - baseline
    smooth = uniform_filter1d(deviation, size=max(1, int(round(fs * 3))))
    min_len = max(1, int(round(fs * 15)))  # 15 seconds
    acc = smooth > 15
    dec = smooth < -15
    deep = smooth < -30
    out["acceleration_count"] = float(_count_runs(acc, min_len))
    out["deceleration_count"] = float(_count_runs(dec, min_len))
    out["deep_deceleration_count"] = float(_count_runs(deep, min_len))
    out["acceleration_fraction"] = float(acc.mean())
    out["deceleration_fraction"] = float(dec.mean())

    if uc_mask.sum() >= 2:
        valid_uc = uc[uc_mask]
        out.update(
            uc_mean=_safe(valid_uc.mean()),
            uc_std=_safe(valid_uc.std()),
            uc_max=_safe(valid_uc.max()),
            uc_p90=_safe(np.percentile(valid_uc, 90)),
        )
        idx = np.flatnonzero(uc_mask)
        uc_i = np.interp(np.arange(len(uc)), idx, uc[idx])
        peaks, _ = find_peaks(
            uniform_filter1d(uc_i, size=max(1, int(round(fs * 5)))),
            height=20.0,
            distance=max(1, int(round(fs * 60))),
        )
        out["uc_contraction_count"] = float(len(peaks))
    else:
        for name in ["uc_mean", "uc_std", "uc_max", "uc_p90", "uc_contraction_count"]:
            out[name] = 0.0
    out["uc_missing_fraction"] = float(1.0 - uc_mask.mean())

    return {name: float(out.get(name, 0.0)) for name in FEATURE_NAMES}
