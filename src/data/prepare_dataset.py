"""Build the processed modelling tables from the read-only WFDB dataset.

Run:  python -m src.data.prepare_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_NAMES,
    TARGET_FS,
    MODEL_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
    WINDOW_SAMPLES,
)
from src.data.load_dataset import build_metadata_table, read_signals
from src.preprocessing.ctg_features import FEATURE_NAMES, extract_ctg_features
from src.preprocessing.preprocess import (
    ClinicalPreprocessor,
    SignalPreprocessor,
    extract_window,
)

CLINICAL_COLUMNS = [
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


def make_splits(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Record-level stratified split (each record is one pregnancy)."""
    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs would leak between splits")
    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=df["label"],
        random_state=seed,
    )
    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - relative_val,
        stratify=df.loc[temp_idx, "label"],
        random_state=seed,
    )
    split = pd.Series("train", index=df.index)
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test"
    return pd.DataFrame({"record_id": df["record_id"], "label": df["label"], "split": split})


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing headers ...")
    meta = build_metadata_table()
    if meta["label"].isna().any():
        missing = meta.loc[meta["label"].isna(), "record_id"].tolist()
        raise ValueError(f"Records without a verified delivery-type label: {missing}")

    splits = make_splits(meta)
    df = meta.merge(splits[["record_id", "split"]], on="record_id", how="left")
    df = df.sort_values("record_num").reset_index(drop=True)
    print(df["split"].value_counts().to_dict())
    print(pd.crosstab(df["split"], df["label"]))

    windows = {}
    clinical_rows = []
    signal_rows = []
    for i, row in df.iterrows():
        rid = row["record_id"]
        signals = read_signals(rid)
        fhr, uc = signals[:, 0], signals[:, 1]
        window = extract_window(fhr, uc, fs=4)
        windows[rid] = window
        signal_rows.append(
            {
                "record_id": rid,
                "fhr_missing_fraction": window["fhr_missing_fraction"],
                "uc_missing_fraction": window["uc_missing_fraction"],
                "window_padded": int(window["padded"]),
                "duration_min_available": window["duration_min_available"],
            }
        )
        clinical_rows.append({col: row.get(col, np.nan) for col in CLINICAL_COLUMNS})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)} records")

    ctg_df = pd.DataFrame(signal_rows)
    ctg_features = pd.DataFrame(
        [extract_ctg_features(windows[rid]) for rid in df["record_id"]]
    )
    ctg_features.insert(0, "record_id", df["record_id"].values)

    clinical_raw = pd.DataFrame(clinical_rows)
    clinical_raw.insert(0, "record_id", df["record_id"].values)

    train_mask = (df["split"] == "train").to_numpy()
    signal_pre = SignalPreprocessor().fit(windows[rid] for rid in df.loc[train_mask, "record_id"])
    clinical_pre = ClinicalPreprocessor().fit(clinical_raw.loc[train_mask].reset_index(drop=True))

    X_signal = np.stack(
        [signal_pre.transform_window(windows[rid]) for rid in df["record_id"]], axis=0
    ).astype(np.float32)
    X_clinical = clinical_pre.transform(clinical_raw).astype(np.float32)

    if X_signal.shape != (len(df), 4, WINDOW_SAMPLES):
        raise RuntimeError(f"Unexpected signal tensor shape: {X_signal.shape}")
    if X_clinical.shape != (len(df), len(clinical_pre.output_names)):
        raise RuntimeError(f"Unexpected clinical matrix shape: {X_clinical.shape}")
    if not np.isfinite(X_signal).all() or not np.isfinite(X_clinical).all():
        raise RuntimeError("Non-finite values in processed features")

    np.savez_compressed(
        PROCESSED_DIR / "signals.npz",
        X=X_signal,
        y=df["label"].to_numpy(np.int64),
        record_ids=df["record_id"].to_numpy(),
        split=df["split"].to_numpy(),
    )
    np.savez_compressed(
        PROCESSED_DIR / "clinical.npz",
        X=X_clinical,
        y=df["label"].to_numpy(np.int64),
        record_ids=df["record_id"].to_numpy(),
        split=df["split"].to_numpy(),
        feature_names=np.array(clinical_pre.output_names),
    )
    ctg_features.to_csv(PROCESSED_DIR / "ctg_features.csv", index=False)
    clinical_raw.to_csv(PROCESSED_DIR / "clinical_raw.csv", index=False)
    ctg_df.to_csv(PROCESSED_DIR / "signal_quality.csv", index=False)
    df.to_csv(PROCESSED_DIR / "dataset.csv", index=False)
    splits.to_csv(PROCESSED_DIR / "splits.csv", index=False)

    joblib.dump(
        {
            "signal": signal_pre,
            "clinical": clinical_pre,
            "clinical_columns": CLINICAL_COLUMNS,
            "ctg_feature_names": FEATURE_NAMES,
        },
        MODEL_DIR / "preprocessing.joblib",
    )
    with open(MODEL_DIR / "preprocessing_meta.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "window_min": signal_pre.window_min,
                "target_fs": signal_pre.target_fs,
                "window_samples": WINDOW_SAMPLES,
                "signal_channels": ["fhr", "uc", "fhr_mask", "uc_mask"],
                "fhr_mean": signal_pre.fhr_mean,
                "fhr_std": signal_pre.fhr_std,
                "uc_mean": signal_pre.uc_mean,
                "uc_std": signal_pre.uc_std,
                "clinical_features": clinical_pre.output_names,
                "class_names": CLASS_NAMES,
                "random_seed": RANDOM_SEED,
            },
            fh,
            indent=2,
        )

    summary = {
        "n_records": int(len(df)),
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "class_counts": df["label"].value_counts().sort_index().to_dict(),
        "class_counts_by_split": {
            split: df.loc[df["split"] == split, "label"].value_counts().sort_index().to_dict()
            for split in ["train", "val", "test"]
        },
        "signal_shape": list(X_signal.shape),
        "clinical_shape": list(X_clinical.shape),
        "fhr_mean": signal_pre.fhr_mean,
        "fhr_std": signal_pre.fhr_std,
        "uc_mean": signal_pre.uc_mean,
        "uc_std": signal_pre.uc_std,
    }
    with open(PROCESSED_DIR / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
