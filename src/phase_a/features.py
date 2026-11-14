"""Feature assembly for Phase A: clinical variables + CTG descriptors.

The production feature set is preserved.  The only change is that descriptors
are recomputed on the Phase A window/sampling configuration, so the effect of
preprocessing can be measured without changing the models.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.data.load_dataset import read_signals
from src.phase_a.config import PROTECTED_FEATURES
from src.phase_a.signals import extract_recent_window
from src.preprocessing.ctg_features import FEATURE_NAMES, extract_ctg_features

CLINICAL_RAW = [
    "age",
    "gravidity",
    "parity",
    "gest_weeks",
    "diabetes",
    "hypertension",
    "preeclampsia",
    "liquor_praecox",
    "pyrexia",
    "meconium",
    "presentation",
    "induced",
]
CLINICAL_NUMERIC = ["age", "gravidity", "parity", "gest_weeks"]
CLINICAL_BINARY = [
    "diabetes",
    "hypertension",
    "preeclampsia",
    "liquor_praecox",
    "pyrexia",
    "meconium",
    "induced",
]
PRESENTATION_CATEGORIES = (1, 2, 3)
PRESENTATION_FEATURES = [f"presentation_{c}" for c in PRESENTATION_CATEGORIES] + ["presentation_missing"]


def load_development_metadata() -> pd.DataFrame:
    """Fixed split metadata for the development set (test excluded)."""
    dataset_path = PROCESSED_DIR / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "data/processed/dataset.csv is missing. Run python -m src.data.prepare_dataset first."
        )
    frame = pd.read_csv(dataset_path)
    frame = frame[frame["split"].isin(("train", "val"))].copy()
    frame["record_id"] = frame["record_id"].astype(str)
    frame = frame.sort_values("record_num").reset_index(drop=True)
    if frame["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs in the development metadata")
    if frame["label"].isna().any():
        raise ValueError("Missing target labels in the development metadata")
    return frame


def load_raw_signals(record_ids: Iterable[str]) -> Dict[str, np.ndarray]:
    """Load raw 4 Hz signals once for a set of records."""
    return {str(rid): read_signals(str(rid)) for rid in record_ids}


def clinical_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    """Raw clinical variables with a stable one-hot presentation encoding."""
    out = metadata[["record_id", "label", "split"] + CLINICAL_RAW].copy()
    presentation = pd.to_numeric(out["presentation"], errors="coerce")
    for category in PRESENTATION_CATEGORIES:
        out[f"presentation_{category}"] = (presentation == category).astype(int)
    out["presentation_missing"] = presentation.isna().astype(int)
    return out


def build_feature_dataset(
    metadata: pd.DataFrame,
    raw_signals: Dict[str, np.ndarray],
    window_min: int,
    target_fs: float,
    policy: str = "exclude",
) -> Tuple[pd.DataFrame, Dict[str, Dict], List[Dict]]:
    """Build one row per record that supports the requested window.

    Returns ``(frame, windows, insufficient)``.  ``insufficient`` lists every
    excluded record with its available duration so the inclusion policy is
    auditable.
    """
    base = clinical_frame(metadata)
    rows: List[Dict] = []
    windows: Dict[str, Dict] = {}
    insufficient: List[Dict] = []

    for _, row in base.iterrows():
        rid = str(row["record_id"])
        signals = np.asarray(raw_signals[rid], dtype=np.float32)
        window = extract_recent_window(
            signals[:, 0],
            signals[:, 1],
            fs=4.0,
            window_min=window_min,
            target_fs=target_fs,
            policy=policy,
            record_id=rid,
        )
        if window is None:
            insufficient.append(
                {
                    "record_id": rid,
                    "label": int(row["label"]),
                    "split": row["split"],
                    "window_min": int(window_min),
                    "target_fs": float(target_fs),
                    "reason": f"recording shorter than {window_min} minutes",
                }
            )
            continue
        windows[rid] = window
        features = extract_ctg_features(window, fs=int(round(target_fs)) if target_fs >= 1 else target_fs)
        item = row.to_dict()
        item.update({name: float(features[name]) for name in FEATURE_NAMES})
        item["window_min"] = int(window_min)
        item["target_fs"] = float(target_fs)
        item["padded"] = int(window["padded"])
        item["fhr_missing_fraction_window"] = float(window["fhr_missing_fraction"])
        item["uc_missing_fraction_window"] = float(window["uc_missing_fraction"])
        rows.append(item)

    frame = pd.DataFrame(rows)
    return frame, windows, insufficient


def candidate_feature_names(frame: pd.DataFrame) -> List[str]:
    """Ordered candidate features used by the feature-selection experiments."""
    clinical = CLINICAL_NUMERIC + CLINICAL_BINARY + PRESENTATION_FEATURES
    names = [name for name in clinical if name in frame.columns]
    names += [name for name in FEATURE_NAMES if name in frame.columns]
    return names


def feature_matrix(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    names = candidate_feature_names(frame)
    X = frame[names].to_numpy(dtype=np.float64)
    y = frame["label"].to_numpy(dtype=np.int64)
    return X, y, names


def _safe_auc(y: np.ndarray, x: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    mask = np.isfinite(x)
    if mask.sum() < 5 or len(np.unique(y[mask])) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y[mask], x[mask]))
    except ValueError:
        return float("nan")


def descriptive_feature_report(frame: pd.DataFrame) -> Dict:
    """Correlation, variance and univariate summaries on the development set.

    These are descriptive only.  Feature selection itself is always fitted
    inside each training fold.
    """
    names = candidate_feature_names(frame)
    X = frame[names].astype(float)
    y = frame["label"].to_numpy(int)
    corr = X.corr().abs()
    pairs = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            value = float(corr.loc[a, b]) if np.isfinite(corr.loc[a, b]) else np.nan
            if np.isfinite(value) and value >= 0.90:
                pairs.append({"feature_a": a, "feature_b": b, "abs_correlation": value})
    pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)

    rows = []
    for name in names:
        values = X[name]
        uniq = values.dropna().nunique()
        rows.append(
            {
                "feature": name,
                "family": "clinical" if name in CLINICAL_NUMERIC + CLINICAL_BINARY + PRESENTATION_FEATURES else "ctg",
                "protected": int(name in PROTECTED_FEATURES),
                "missing_fraction": float(values.isna().mean()),
                "n_unique": int(uniq),
                "mean": float(values.mean()) if values.notna().any() else float("nan"),
                "std": float(values.std(ddof=0)) if values.notna().any() else float("nan"),
                "univariate_auc": _safe_auc(y, values.to_numpy(float)),
                "zero_fraction": float((values == 0).mean()),
            }
        )
    info = pd.DataFrame(rows)
    info["abs_auc_deviation"] = (info["univariate_auc"] - 0.5).abs()
    low_info = info.sort_values(["abs_auc_deviation", "std"], ascending=[True, True]).head(10).to_dict("records")
    return {
        "n_features": len(names),
        "n_clinical": int(sum(1 for n in names if n in CLINICAL_NUMERIC + CLINICAL_BINARY + PRESENTATION_FEATURES)),
        "n_ctg": int(sum(1 for n in names if n in FEATURE_NAMES)),
        "correlated_pairs_ge_0_90": pairs,
        "feature_table": info.to_dict("records"),
        "low_information_features": low_info,
        "protected_features": list(PROTECTED_FEATURES),
    }


def split_xy(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, names = feature_matrix(frame)
    if not np.isfinite(X).all():
        # Clinical missingness is expected; median imputation is fitted inside
        # each training fold.  Keep NaNs here.
        pass
    return X, y, names
