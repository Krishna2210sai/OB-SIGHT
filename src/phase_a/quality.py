"""Record-level CTG data-quality audit.

The dataset encodes missing signal samples as exact zeros.  Evidence for that
encoding is documented in the Phase A report and re-checked here:

* FHR = 0 bpm is not physiological;
* the smallest positive UC value is 0.5 (no genuine 0.0 samples were found);
* zero runs are long, contiguous and partly coincide across channels, which is
  characteristic of signal loss rather than uterine quiescence.

Consequently ``value <= 0`` is treated as missing.  The audit never deletes a
record; it labels records and records the reason for every label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import FHR_VALID_MAX, FHR_VALID_MIN, FS_NATIVE
from src.phase_a.config import QualityThresholds

QUALITY_CATEGORIES = ("good", "acceptable", "poor")


# ---------------------------------------------------------------------------
# Low-level signal helpers
# ---------------------------------------------------------------------------
def _runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    changes = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)


def longest_run_min(mask: np.ndarray, fs: float = FS_NATIVE) -> float:
    """Length of the longest contiguous True run, in minutes."""
    starts, ends = _runs(mask)
    if not len(starts):
        return 0.0
    return float((ends - starts).max()) / float(fs) / 60.0


def n_runs(mask: np.ndarray, min_len_samples: int = 1) -> int:
    starts, ends = _runs(mask)
    if not len(starts):
        return 0
    return int(np.sum((ends - starts) >= min_len_samples))


def trim_zero_padding(fhr: np.ndarray, uc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove leading/trailing samples where both channels are missing."""
    fhr = np.asarray(fhr, dtype=np.float64).reshape(-1)
    uc = np.asarray(uc, dtype=np.float64).reshape(-1)
    if fhr.shape != uc.shape:
        raise ValueError("FHR and UC must have the same length")
    valid = (fhr > 0) | (uc > 0)
    if not valid.any():
        return fhr[:0], uc[:0]
    first = int(np.argmax(valid))
    last = len(valid) - int(np.argmax(valid[::-1]))
    return fhr[first:last], uc[first:last]


def fhr_flatline_stats(
    fhr: np.ndarray,
    valid: np.ndarray,
    fs: float = FS_NATIVE,
    window_min: int = 5,
    sd_threshold: float = 2.0,
    min_valid_fraction: float = 0.5,
    step_min: int = 1,
) -> Dict[str, float]:
    """Fraction and longest run of near-constant FHR windows.

    A window is assessable when at least ``min_valid_fraction`` of its native
    samples are valid.  Missing values are interpolated only for this
    descriptive statistic; model inputs are never filled this way.
    """
    fhr = np.asarray(fhr, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    n = len(fhr)
    win = int(round(window_min * 60 * fs))
    step = int(round(step_min * 60 * fs))
    if n < win or valid.sum() < 10:
        return {"flatline_fraction": 0.0, "longest_flatline_min": 0.0, "flatline_windows": 0, "assessable_windows": 0}

    idx = np.flatnonzero(valid)
    if len(idx) >= 2:
        interp = np.interp(np.arange(n), idx, fhr[idx])
    else:
        interp = np.zeros(n, dtype=np.float64)

    # Cumulative sums give O(n) rolling statistics.
    c1 = np.r_[0.0, np.cumsum(interp)]
    c2 = np.r_[0.0, np.cumsum(interp * interp)]
    cv = np.r_[0.0, np.cumsum(valid.astype(np.float64))]
    starts = np.arange(0, n - win + 1, step)
    assessable = 0
    flat = 0
    flat_flags = []
    for start in starts:
        end = start + win
        vfrac = (cv[end] - cv[start]) / win
        if vfrac < min_valid_fraction:
            flat_flags.append(False)
            continue
        mean = (c1[end] - c1[start]) / win
        var = max((c2[end] - c2[start]) / win - mean * mean, 0.0)
        assessable += 1
        is_flat = float(np.sqrt(var)) < sd_threshold
        flat += int(is_flat)
        flat_flags.append(bool(is_flat))
    if assessable == 0:
        return {"flatline_fraction": 0.0, "longest_flatline_min": 0.0, "flatline_windows": 0, "assessable_windows": 0}
    flags = np.asarray(flat_flags, dtype=bool)
    # One flag per step_min minutes: express the flag rate in samples/second.
    run_min = float(longest_run_min(flags, fs=1.0 / (step_min * 60.0)))
    return {
        "flatline_fraction": float(flat / assessable),
        "longest_flatline_min": run_min,
        "flatline_windows": int(flat),
        "assessable_windows": int(assessable),
    }


def discontinuity_stats(
    values: np.ndarray,
    valid: np.ndarray,
    fs: float = FS_NATIVE,
    jump_bpm: float = 25.0,
) -> Dict[str, float]:
    """Count implausibly large sample-to-sample FHR changes."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if len(values) < 2:
        return {"jump_count": 0, "jump_rate_per_min": 0.0, "max_abs_jump": 0.0, "abs_diff_mean": 0.0}
    both = valid[:-1] & valid[1:]
    if not both.any():
        return {"jump_count": 0, "jump_rate_per_min": 0.0, "max_abs_jump": 0.0, "abs_diff_mean": 0.0}
    diffs = np.abs(np.diff(values))[both]
    minutes = float(both.sum()) / float(fs) / 60.0
    return {
        "jump_count": int(np.sum(diffs > jump_bpm)),
        "jump_rate_per_min": float(np.sum(diffs > jump_bpm) / minutes) if minutes > 0 else 0.0,
        "jump_fraction": float(np.mean(diffs > jump_bpm)),
        "max_abs_jump": float(diffs.max()),
        "abs_diff_mean": float(diffs.mean()),
    }


def _gap_metrics(valid: np.ndarray, fs: float) -> Dict[str, float]:
    missing = ~np.asarray(valid, dtype=bool)
    return {
        "longest_gap_min": longest_run_min(missing, fs),
        "n_gaps_gt_10s": n_runs(missing, int(round(10 * fs))),
        "n_gaps_gt_60s": n_runs(missing, int(round(60 * fs))),
    }


# ---------------------------------------------------------------------------
# Record audit
# ---------------------------------------------------------------------------
def audit_record(
    record_id: str,
    fhr: np.ndarray,
    uc: np.ndarray,
    label: Optional[int] = None,
    fs: float = FS_NATIVE,
    thresholds: Optional[QualityThresholds] = None,
    window_min: int = 30,
) -> Dict:
    """Compute the quality report for one record.

    Metrics are reported for the full trimmed recording and for the most
    recent ``window_min`` minutes.  Eligibility for 30/45/60-minute windows is
    reported separately.
    """
    thresholds = thresholds or QualityThresholds()
    fhr_raw = np.asarray(fhr, dtype=np.float64).reshape(-1)
    uc_raw = np.asarray(uc, dtype=np.float64).reshape(-1)
    raw_duration = float(len(fhr_raw) / fs / 60.0)
    fhr_t, uc_t = trim_zero_padding(fhr_raw, uc_raw)
    trimmed_duration = float(len(fhr_t) / fs / 60.0)

    row: Dict = {
        "record_id": str(record_id),
        "label": int(label) if label is not None and not pd.isna(label) else -1,
        "raw_duration_min": raw_duration,
        "trimmed_duration_min": trimmed_duration,
        "fhr_zero_fraction_recording": float(np.mean(fhr_t <= 0)) if len(fhr_t) else 1.0,
        "uc_zero_fraction_recording": float(np.mean(uc_t <= 0)) if len(uc_t) else 1.0,
        "fhr_min_positive": float(np.min(fhr_t[fhr_t > 0])) if np.any(fhr_t > 0) else np.nan,
        "fhr_max_positive": float(np.max(fhr_t[fhr_t > 0])) if np.any(fhr_t > 0) else np.nan,
        "uc_min_positive": float(np.min(uc_t[uc_t > 0])) if np.any(uc_t > 0) else np.nan,
        "uc_max_positive": float(np.max(uc_t[uc_t > 0])) if np.any(uc_t > 0) else np.nan,
        "n_positive_uc_values_below_0_1": int(np.sum((uc_t > 0) & (uc_t < 0.1))),
    }

    for minutes in (30, 45, 60):
        row[f"supports_{minutes}m"] = int(trimmed_duration + 1e-9 >= minutes)

    # Metrics for the primary (30-minute by default) window.
    n_window = int(round(window_min * 60 * fs))
    if len(fhr_t) >= n_window:
        fw, uw = fhr_t[-n_window:], uc_t[-n_window:]
        padded = 0
    else:
        # For the audit only, assess the whole available recording.  This does
        # not silently pad model input; it marks the window unsupported.
        fw, uw = fhr_t, uc_t
        padded = 1

    fhr_valid = fw > 0
    uc_valid = uw > 0
    row.update(
        {
            f"window_minutes": int(window_min),
            "window_supported": int(not padded),
            "window_padded_for_audit": int(padded),
            f"fhr_missing_fraction_{window_min}m": float(1.0 - fhr_valid.mean()) if len(fw) else 1.0,
            f"uc_missing_fraction_{window_min}m": float(1.0 - uc_valid.mean()) if len(uw) else 1.0,
            f"fhr_valid_fraction_{window_min}m": float(fhr_valid.mean()) if len(fw) else 0.0,
            f"uc_valid_fraction_{window_min}m": float(uc_valid.mean()) if len(uw) else 0.0,
        }
    )

    # Gaps.
    for name, valid in (("fhr", fhr_valid), ("uc", uc_valid)):
        gaps = _gap_metrics(valid, fs)
        row[f"longest_{name}_gap_min_{window_min}m"] = gaps["longest_gap_min"]
        row[f"n_{name}_gaps_gt_10s_{window_min}m"] = gaps["n_gaps_gt_10s"]
        row[f"n_{name}_gaps_gt_60s_{window_min}m"] = gaps["n_gaps_gt_60s"]

    # FHR plausibility and discontinuity.
    positive = fw > 0
    implausible = positive & ((fw < FHR_VALID_MIN) | (fw > FHR_VALID_MAX))
    row[f"implausible_fhr_fraction_{window_min}m"] = float(implausible.mean()) if len(fw) else 0.0
    row[f"implausible_fhr_count_{window_min}m"] = int(implausible.sum())
    jumps = discontinuity_stats(fw, fhr_valid, fs=fs)
    row.update({f"{k}_{window_min}m": v for k, v in jumps.items() if k in ("jump_count", "jump_rate_per_min", "jump_fraction", "max_abs_jump", "abs_diff_mean")})

    flat = fhr_flatline_stats(fw, fhr_valid, fs=fs)
    row.update({f"{k}_{window_min}m": v for k, v in flat.items()})

    # Missing-run summary (the dataset's "zero-value segments").
    row[f"longest_fhr_zero_run_min_{window_min}m"] = longest_run_min(~fhr_valid, fs)
    row[f"longest_uc_zero_run_min_{window_min}m"] = longest_run_min(~uc_valid, fs)

    # Quality category.  The 30-minute window is the primary usability unit.
    reasons: List[str] = []
    good, acceptable = True, True

    def check(value: float, good_lim: float, acceptable_lim: float, label_text: str, higher_is_bad: bool = True) -> None:
        nonlocal good, acceptable
        if not np.isfinite(value):
            acceptable = False
            reasons.append(f"{label_text}: not finite")
            return
        bad_good = value > good_lim if higher_is_bad else value < good_lim
        bad_acc = value > acceptable_lim if higher_is_bad else value < acceptable_lim
        if bad_good:
            good = False
        if bad_acc:
            acceptable = False
            reasons.append(f"{label_text}={value:.4g} beyond acceptable limit {acceptable_lim:.4g}")

    check(row[f"fhr_missing_fraction_{window_min}m"], thresholds.fhr_missing_good, thresholds.fhr_missing_acceptable, "FHR missing fraction")
    check(row[f"uc_missing_fraction_{window_min}m"], thresholds.uc_missing_good, thresholds.uc_missing_acceptable, "UC missing fraction")
    check(row[f"longest_fhr_gap_min_{window_min}m"], thresholds.longest_gap_good_min, thresholds.longest_gap_acceptable_min, "longest FHR gap (min)")
    check(row[f"longest_uc_gap_min_{window_min}m"], thresholds.longest_gap_good_min, thresholds.longest_gap_acceptable_min, "longest UC gap (min)")
    check(row[f"implausible_fhr_fraction_{window_min}m"], thresholds.implausible_good, thresholds.implausible_acceptable, "implausible FHR fraction")
    check(row[f"jump_fraction_{window_min}m"], thresholds.jump_rate_good, thresholds.jump_rate_acceptable, "FHR jump fraction")
    check(row[f"flatline_fraction_{window_min}m"], thresholds.flatline_good, thresholds.flatline_acceptable, "FHR flatline fraction")
    check(trimmed_duration, thresholds.min_duration_good_min, thresholds.min_duration_acceptable_min, "trimmed duration (min)", higher_is_bad=False)

    if not row["window_supported"]:
        acceptable = False
        reasons.append(f"recording shorter than {window_min} minutes")
    if len(fw) == 0:
        good = acceptable = False
        reasons.append("no usable samples")

    if good:
        category = "good"
    elif acceptable:
        category = "acceptable"
    else:
        category = "poor"
    row["quality_category"] = category
    row["quality_reasons"] = "; ".join(dict.fromkeys(reasons)) if reasons else ""
    return row


def audit_dataset(
    records: Iterable[Dict],
    thresholds: Optional[QualityThresholds] = None,
    window_min: int = 30,
) -> pd.DataFrame:
    """Audit an iterable of records with ``record_id``, ``signals``, ``label``."""
    rows = []
    for record in records:
        signals = np.asarray(record["signals"], dtype=np.float64)
        rows.append(
            audit_record(
                record_id=str(record["record_id"]),
                fhr=signals[:, 0],
                uc=signals[:, 1],
                label=record.get("label"),
                fs=float(record.get("fs", FS_NATIVE)),
                thresholds=thresholds,
                window_min=window_min,
            )
        )
    return pd.DataFrame(rows)


def summarise_quality(report: pd.DataFrame) -> Dict:
    """Aggregate quality findings for the report."""
    summary: Dict = {
        "n_records": int(len(report)),
        "categories": report["quality_category"].value_counts().to_dict(),
        "category_by_label": pd.crosstab(report["quality_category"], report["label"]).to_dict(),
        "class_counts": report["label"].value_counts().sort_index().to_dict(),
        "duration_min": {
            "mean": float(report["trimmed_duration_min"].mean()),
            "median": float(report["trimmed_duration_min"].median()),
            "min": float(report["trimmed_duration_min"].min()),
            "max": float(report["trimmed_duration_min"].max()),
        },
        "support_counts": {
            f"{m}m": int(report[f"supports_{m}m"].sum()) for m in (30, 45, 60)
        },
        "support_by_label": {
            f"{m}m": pd.crosstab(report[f"supports_{m}m"], report["label"]).to_dict()
            for m in (30, 45, 60)
        },
        "missingness": {},
        "std_positive_uc_below_0_1": int(report["n_positive_uc_values_below_0_1"].sum()),
    }
    for col in (
        "fhr_missing_fraction_30m",
        "uc_missing_fraction_30m",
        "fhr_zero_fraction_recording",
        "uc_zero_fraction_recording",
        "longest_fhr_gap_min_30m",
        "longest_uc_gap_min_30m",
        "implausible_fhr_fraction_30m",
        "jump_fraction_30m",
        "flatline_fraction_30m",
    ):
        summary["missingness"][col] = {
            "mean": float(report[col].mean()),
            "median": float(report[col].median()),
            "p90": float(report[col].quantile(0.90)),
            "max": float(report[col].max()),
        }
    summary["missingness_by_label"] = (
        report.groupby("label")[
            [
                "fhr_missing_fraction_30m",
                "uc_missing_fraction_30m",
                "implausible_fhr_fraction_30m",
                "jump_fraction_30m",
                "flatline_fraction_30m",
            ]
        ]
        .mean()
        .round(4)
        .to_dict()
    )
    return summary


def standardised_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute standardised mean difference between two samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float(abs(np.mean(a) - np.mean(b)) / pooled)


def eligibility_mask(report: pd.DataFrame, include: Sequence[str]) -> np.ndarray:
    """Boolean mask for a documented quality-inclusion policy."""
    return report["quality_category"].isin(list(include)).to_numpy()
