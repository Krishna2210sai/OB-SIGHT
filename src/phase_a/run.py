"""Phase A experiment runner.

Examples
--------
Full Phase A pipeline (quality -> windows -> sampling -> features -> compare
-> deep-learning screening -> report)::

    python -m src.phase_a.run --stage all

Fast development pass without deep learning::

    python -m src.phase_a.run --stage all --no-dl

Individual stages::

    python -m src.phase_a.run --stage quality --force
    python -m src.phase_a.run --stage window
    python -m src.phase_a.run --stage sampling
    python -m src.phase_a.run --stage features
    python -m src.phase_a.run --stage compare
    python -m src.phase_a.run --stage deep --epochs 10
    python -m src.phase_a.run --stage report

All cross-validation uses the fixed development set (``split in {train,val}``).
The 83 held-out test records are never loaded by this module.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src.config import DATASET_DIR, PROCESSED_DIR, RANDOM_SEED
from src.data.load_dataset import read_signals
from src.phase_a.config import (
    CACHE_DIR,
    COMPARISON_DIR,
    CURRENT_FS,
    CURRENT_WINDOW_MINUTES,
    DATA_QUALITY_DIR,
    DL_BATCH_SIZE,
    DL_EPOCHS,
    DL_LEARNING_RATE,
    DL_PATIENCE,
    DL_WEIGHT_DECAY,
    EXPERIMENTS_DIR,
    FEATURE_DIR,
    LOG_DIR,
    REPORT_DIR,
    SAMPLING_DIR,
    WINDOW_DIR,
    PhaseAConfig,
    QualityThresholds,
    ensure_directories,
)
from src.phase_a.experiment import (
    cv_baseline,
    cv_deep,
    make_folds,
    save_cv_result,
)
from src.phase_a.features import (
    build_feature_dataset,
    candidate_feature_names,
    descriptive_feature_report,
    feature_matrix,
    load_development_metadata,
)
from src.phase_a.quality import (
    audit_record,
    eligibility_mask,
    standardised_mean_difference,
    summarise_quality,
)
from src.phase_a.selection import SELECTOR_LABELS, selection_stability
from src.phase_a.signals import extract_recent_window
from src.phase_a import plots

METRIC_KEYS = (
    "auroc",
    "auprc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "mcc",
)

LOGGER = logging.getLogger("phase_a")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _setup_logging(stage: str, force: bool = False) -> None:
    ensure_directories()
    existing = LOGGER.handlers[:]
    for handler in existing:
        LOGGER.removeHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    logfile = LOG_DIR / f"{stage}.log"
    file_handler = logging.FileHandler(logfile, mode="w" if force else "a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def _tag(window_min: int, fs: float) -> str:
    rate = f"{fs:g}".replace(".", "p")
    return f"w{window_min}_fs{rate}"


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def _full_metadata() -> pd.DataFrame:
    path = PROCESSED_DIR / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            "data/processed/dataset.csv is missing. Run python -m src.data.prepare_dataset first."
        )
    frame = pd.read_csv(path).sort_values("record_num").reset_index(drop=True)
    frame["record_id"] = frame["record_id"].astype(str)
    if frame["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs in data/processed/dataset.csv")
    if frame["label"].isna().any():
        raise ValueError("Missing target labels in data/processed/dataset.csv")
    return frame


def _load_signals(record_ids: Iterable[str]) -> Dict[str, np.ndarray]:
    return {str(rid): read_signals(str(rid)) for rid in record_ids}


# ---------------------------------------------------------------------------
# Stage 1: data quality
# ---------------------------------------------------------------------------
def run_quality(config: PhaseAConfig, force: bool = False) -> pd.DataFrame:
    _setup_logging("quality", force)
    out_dir = DATA_QUALITY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "record_quality.csv"
    metadata = _full_metadata()

    if report_path.exists() and not force:
        LOGGER.info("Loading cached quality report %s", report_path)
        report = pd.read_csv(report_path)
        report["record_id"] = report["record_id"].astype(str)
    else:
        LOGGER.info("Auditing %d records at native 4 Hz", len(metadata))
        rows = []
        for i, row in enumerate(metadata.itertuples(index=False), start=1):
            signals = read_signals(str(row.record_id))
            rows.append(
                audit_record(
                    record_id=str(row.record_id),
                    fhr=signals[:, 0],
                    uc=signals[:, 1],
                    label=int(row.label),
                    fs=4.0,
                    thresholds=config.quality,
                    window_min=CURRENT_WINDOW_MINUTES,
                )
            )
            if i % 100 == 0:
                LOGGER.info("  audited %d/%d", i, len(metadata))
        report = pd.DataFrame(rows)
        report = report.merge(
            metadata[["record_id", "split"]], on="record_id", how="left", validate="one_to_one"
        )
        order = ["record_id", "split", "label"] + [
            c for c in report.columns if c not in ("record_id", "split", "label")
        ]
        report = report[order]
        report.to_csv(report_path, index=False)

    summary = summarise_quality(report)
    summary["thresholds"] = config.quality.to_dict()
    summary["zero_encoding_evidence"] = {
        "positive_uc_values_below_0.1": int(report["n_positive_uc_values_below_0_1"].sum()),
        "minimum_positive_uc": float(report["uc_min_positive"].min()),
        "interpretation": (
            "Exact zeros are treated as missing: FHR=0 is not physiological, the "
            "smallest positive UC value is 0.5, and zero runs are long and "
            "contiguous. No genuine 0.0 UC sample was observed."
        ),
    }
    summary["signal_availability"] = {
        "records_with_uc_fully_missing_30m": int(
            (report["uc_missing_fraction_30m"] >= 0.999).sum()
        ),
        "records_with_fhr_fully_missing_30m": int(
            (report["fhr_missing_fraction_30m"] >= 0.999).sum()
        ),
        "records_with_fhr_missing_gt_50pct_30m": int(
            (report["fhr_missing_fraction_30m"] > 0.50).sum()
        ),
        "records_with_uc_missing_gt_50pct_30m": int(
            (report["uc_missing_fraction_30m"] > 0.50).sum()
        ),
    }
    summary["clinical_missing_fraction"] = {
        col: float(metadata[col].isna().mean())
        for col in (
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
        )
    }
    summary["target"] = {
        "n_missing": int(metadata["label"].isna().sum()),
        "counts": {str(k): int(v) for k, v in metadata["label"].value_counts().sort_index().items()},
        "prevalence": float(metadata["label"].mean()),
    }
    summary["quality_reason_counts"] = (
        report.loc[report["quality_category"] != "good", "quality_reasons"]
        .str.split("; ")
        .explode()
        .replace("", np.nan)
        .dropna()
        .str.split("=")
        .str[0]
        .value_counts()
        .head(20)
        .to_dict()
    )
    _write_json(out_dir / "quality_summary.json", summary)
    report.to_csv(report_path, index=False)
    plots.plot_quality(report, out_dir / "figures")

    LOGGER.info(
        "Quality categories: %s; window support: %s",
        summary["categories"],
        summary["support_counts"],
    )
    return report


# ---------------------------------------------------------------------------
# Cached window/feature construction
# ---------------------------------------------------------------------------
def _cache_key(window_min: int, fs: float) -> Path:
    return CACHE_DIR / _tag(window_min, fs)


def build_configuration(
    metadata: pd.DataFrame,
    raw_signals: Dict[str, np.ndarray],
    window_min: int,
    fs: float,
    force: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict], List[Dict]]:
    """Build (or load) the feature frame and window cache for one configuration."""
    directory = _cache_key(window_min, fs)
    directory.mkdir(parents=True, exist_ok=True)
    features_path = directory / "features.csv"
    insufficient_path = directory / "insufficient_records.csv"
    windows_path = directory / "windows.npz"

    if features_path.exists() and windows_path.exists() and not force:
        frame = pd.read_csv(features_path)
        frame["record_id"] = frame["record_id"].astype(str)
        insufficient = []
        if insufficient_path.exists():
            try:
                insufficient = pd.read_csv(insufficient_path).to_dict("records")
            except pd.errors.EmptyDataError:
                insufficient = []
        with np.load(windows_path) as data:
            record_ids = [str(x) for x in data["record_id"]]
            # Materialise each array once: indexing an NpzFile member
            # re-decompresses the whole array on every access.
            fhr_all = data["fhr"]
            uc_all = data["uc"]
            fhr_mask_all = data["fhr_mask"]
            uc_mask_all = data["uc_mask"]
            windows = {
                rid: {
                    "record_id": rid,
                    "fhr": fhr_all[i],
                    "uc": uc_all[i],
                    "fhr_mask": fhr_mask_all[i],
                    "uc_mask": uc_mask_all[i],
                    "window_min": int(window_min),
                    "target_fs": float(fs),
                }
                for i, rid in enumerate(record_ids)
            }
        LOGGER.info("Loaded cached configuration %s (%d records)", _tag(window_min, fs), len(frame))
        return frame, windows, insufficient

    LOGGER.info("Building configuration %s", _tag(window_min, fs))
    frame, windows, insufficient = build_feature_dataset(
        metadata, raw_signals, window_min=window_min, target_fs=fs
    )
    frame.to_csv(features_path, index=False)
    if insufficient:
        pd.DataFrame(insufficient).to_csv(insufficient_path, index=False)
    elif insufficient_path.exists():
        insufficient_path.unlink()
    record_ids = list(windows)
    np.savez_compressed(
        windows_path,
        record_id=np.array(record_ids),
        fhr=np.stack([windows[rid]["fhr"] for rid in record_ids]),
        uc=np.stack([windows[rid]["uc"] for rid in record_ids]),
        fhr_mask=np.stack([windows[rid]["fhr_mask"] for rid in record_ids]),
        uc_mask=np.stack([windows[rid]["uc_mask"] for rid in record_ids]),
    )
    LOGGER.info(
        "  %d records usable, %d excluded", len(frame), len(insufficient)
    )
    return frame, windows, insufficient


def _row_from_result(configuration: str, result: Dict) -> Dict:
    metrics = result["metrics"]
    pooled = metrics["pooled"]
    mean = metrics["fold_mean"]
    std = metrics["fold_std"]
    row = {
        "configuration": configuration,
        "model": result.get("model", result.get("variant", "")),
        "method": result.get("method", "none"),
        "n_records": result["n_records"],
        "n_features_input": result.get("n_features_input", np.nan),
        "n_features_selected": result.get("n_features_mean_selected", np.nan),
    }
    for key in METRIC_KEYS:
        row[f"{key}_mean"] = mean.get(key, np.nan)
        row[f"{key}_std"] = std.get(key, np.nan)
    row["auroc_pooled"] = pooled.get("auroc", np.nan)
    row["auprc_pooled"] = pooled.get("auprc", np.nan)
    row["mcc_pooled"] = pooled.get("mcc", np.nan)
    row["auroc_ci_low"] = pooled.get("auroc_ci_low", np.nan)
    row["auroc_ci_high"] = pooled.get("auroc_ci_high", np.nan)
    row["auprc_ci_low"] = pooled.get("auprc_ci_low", np.nan)
    row["auprc_ci_high"] = pooled.get("auprc_ci_high", np.nan)
    row["mcc_ci_low"] = pooled.get("mcc_ci_low", np.nan)
    row["mcc_ci_high"] = pooled.get("mcc_ci_high", np.nan)
    row["n_cesarean"] = int(pooled.get("n_positive", 0))
    row["prevalence"] = float(pooled.get("prevalence", np.nan))
    row["window_minutes"] = result.get("window_minutes", np.nan)
    row["target_fs"] = result.get("target_fs", np.nan)
    return row


def _run_baselines(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    methods: Sequence[str],
    models: Sequence[str],
    directory: Path,
    prefix: str,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    rows: List[Dict] = []
    results: Dict[str, Dict] = {}
    for method in methods:
        for model_name in models:
            name = f"{prefix}_{model_name}_{method}"
            LOGGER.info("  CV %s", name)
            output = cv_baseline(frame, folds, method=method, model_name=model_name, seed=seed)
            save_cv_result(directory, name, output["result"], output["oof_frame"])
            rows.append(_row_from_result(f"{prefix}:{model_name}:{method}", output["result"]))
            results[f"{model_name}:{method}"] = output
    table = pd.DataFrame(rows)
    return table, results


# ---------------------------------------------------------------------------
# Stage 2: window duration
# ---------------------------------------------------------------------------
def run_window(
    config: PhaseAConfig,
    raw_signals: Dict[str, np.ndarray],
    metadata: pd.DataFrame,
    folds: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    _setup_logging("window", force)
    tables = []
    for window_min in config.windows:
        directory = WINDOW_DIR / _tag(window_min, CURRENT_FS)
        directory.mkdir(parents=True, exist_ok=True)
        frame, _, insufficient = build_configuration(
            metadata, raw_signals, window_min, CURRENT_FS, force=force
        )
        _write_json(
            directory / "config.json",
            {
                "window_minutes": window_min,
                "target_fs": CURRENT_FS,
                "policy": "exclude",
                "n_records": int(len(frame)),
                "n_cesarean": int(frame["label"].sum()),
                "n_excluded": len(insufficient),
                "excluded": insufficient,
                "seed": config.seed,
            },
        )
        if insufficient:
            pd.DataFrame(insufficient).to_csv(directory / "excluded_records.csv", index=False)
        table, _ = _run_baselines(
            frame,
            folds,
            methods=["none"],
            models=["logistic", "gradient_boosting"],
            directory=directory,
            prefix=_tag(window_min, CURRENT_FS),
            seed=config.seed,
        )
        table["window_minutes"] = window_min
        table["target_fs"] = CURRENT_FS
        table["n_excluded_insufficient"] = len(insufficient)
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(WINDOW_DIR / "window_comparison.csv", index=False)
    return combined


# ---------------------------------------------------------------------------
# Stage 3: sampling frequency
# ---------------------------------------------------------------------------
def run_sampling(
    config: PhaseAConfig,
    raw_signals: Dict[str, np.ndarray],
    metadata: pd.DataFrame,
    folds: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    _setup_logging("sampling", force)
    tables = []
    for fs in config.sampling_rates:
        directory = SAMPLING_DIR / _tag(CURRENT_WINDOW_MINUTES, fs)
        directory.mkdir(parents=True, exist_ok=True)
        frame, _, insufficient = build_configuration(
            metadata, raw_signals, CURRENT_WINDOW_MINUTES, fs, force=force
        )
        _write_json(
            directory / "config.json",
            {
                "window_minutes": CURRENT_WINDOW_MINUTES,
                "target_fs": fs,
                "anti_aliasing": "4th-order zero-phase Butterworth low-pass before decimation",
                "cutoff_hz": {"fhr": 0.4 * fs, "uc": 0.2 * fs} if fs < 4 else None,
                "n_records": int(len(frame)),
                "n_cesarean": int(frame["label"].sum()),
                "n_excluded": len(insufficient),
                "excluded": insufficient,
                "seed": config.seed,
            },
        )
        if insufficient:
            pd.DataFrame(insufficient).to_csv(directory / "excluded_records.csv", index=False)
        table, _ = _run_baselines(
            frame,
            folds,
            methods=["none"],
            models=["logistic", "gradient_boosting"],
            directory=directory,
            prefix=_tag(CURRENT_WINDOW_MINUTES, fs),
            seed=config.seed,
        )
        table["window_minutes"] = CURRENT_WINDOW_MINUTES
        table["target_fs"] = fs
        table["n_excluded_insufficient"] = len(insufficient)
        table["window_samples"] = int(round(CURRENT_WINDOW_MINUTES * 60 * fs))
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(SAMPLING_DIR / "sampling_comparison.csv", index=False)
    return combined


# ---------------------------------------------------------------------------
# Stage 4: feature selection
# ---------------------------------------------------------------------------
def run_features(
    config: PhaseAConfig,
    metadata: pd.DataFrame,
    folds: pd.DataFrame,
    raw_signals: Dict[str, np.ndarray],
    target_configs: Optional[Sequence[Tuple[int, float]]] = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    _setup_logging("features", force)
    target_configs = target_configs or [(CURRENT_WINDOW_MINUTES, CURRENT_FS)]
    methods = ["none", "correlation", "mutual_info", "rfe", "l1", "rf_importance"]
    tables = []
    stability_tables = []
    summaries: Dict[str, Dict] = {}

    for window_min, fs in target_configs:
        frame, _, insufficient = build_configuration(metadata, raw_signals, window_min, fs, force=force)
        directory = FEATURE_DIR / _tag(window_min, fs)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            directory / "config.json",
            {
                "window_minutes": window_min,
                "target_fs": fs,
                "n_records": int(len(frame)),
                "n_cesarean": int(frame["label"].sum()),
                "n_excluded": len(insufficient),
                "excluded": insufficient,
                "feature_names": candidate_feature_names(frame),
                "methods": methods,
                "seed": config.seed,
            },
        )
        if insufficient:
            pd.DataFrame(insufficient).to_csv(directory / "excluded_records.csv", index=False)

        descriptive = descriptive_feature_report(frame)
        _write_json(directory / "feature_report.json", descriptive)

        table, results = _run_baselines(
            frame,
            folds,
            methods=methods,
            models=["logistic", "gradient_boosting"],
            directory=directory,
            prefix=_tag(window_min, fs),
            seed=config.seed,
        )
        table["window_minutes"] = window_min
        table["target_fs"] = fs
        tables.append(table)

        all_features = candidate_feature_names(frame)
        for model_name in ("logistic", "gradient_boosting"):
            selections = {
                method: [
                    results[f"{model_name}:{method}"]["selected_by_fold"][fold]
                    for fold in results[f"{model_name}:{method}"]["selected_by_fold"]
                ]
                for method in methods
            }
            stability, method_summary = selection_stability(selections, all_features)
            stability["model"] = model_name
            stability["window_minutes"] = window_min
            stability["target_fs"] = fs
            stability_tables.append(stability)
            summaries[f"{_tag(window_min, fs)}:{model_name}"] = method_summary
            for method in methods:
                plots.plot_selection_stability(
                    stability,
                    directory / "figures",
                    f"stability_{model_name}_{method}.png",
                    f"{method} ({model_name})",
                )
        (directory / "selection_stability.json").write_text(
            json.dumps(summaries, indent=2, default=_json_default), encoding="utf-8"
        )

    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(FEATURE_DIR / "feature_selection_comparison.csv", index=False)
    if stability_tables:
        pd.concat(stability_tables, ignore_index=True).to_csv(
            FEATURE_DIR / "selection_stability.csv", index=False
        )
    _write_json(FEATURE_DIR / "selection_stability_summary.json", summaries)
    return combined, summaries


# ---------------------------------------------------------------------------
# Stage 5: deep-learning screening
# ---------------------------------------------------------------------------
def run_deep(
    config: PhaseAConfig,
    metadata: pd.DataFrame,
    folds: pd.DataFrame,
    raw_signals: Dict[str, np.ndarray],
    force: bool = False,
    epochs: Optional[int] = None,
    configs: Optional[Sequence[Tuple[int, float]]] = None,
) -> pd.DataFrame:
    _setup_logging("deep", force)
    epochs = int(epochs or config.dl_epochs)
    if configs is None:
        configs = [(30, 1.0), (45, 1.0), (60, 1.0), (30, 4.0), (30, 2.0), (30, 0.5)]
    directory = EXPERIMENTS_DIR / "deep_screening"
    directory.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for window_min, fs in configs:
        tag = _tag(window_min, fs)
        result_path = directory / f"{tag}_multimodal.json"
        if result_path.exists() and not force:
            LOGGER.info("Reusing %s", result_path.name)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(_row_from_result(f"{tag}:multimodal", payload))
            continue
        frame, windows, insufficient = build_configuration(
            metadata, raw_signals, window_min, fs, force=force
        )
        LOGGER.info("Deep screening %s (%d records, %d epochs)", tag, len(frame), epochs)
        started = time.time()
        output = cv_deep(
            frame,
            windows,
            folds,
            variant="multimodal",
            seed=config.seed,
            epochs=epochs,
            patience=config.dl_patience,
            batch_size=config.dl_batch_size,
            learning_rate=config.dl_learning_rate,
            weight_decay=config.dl_weight_decay,
        )
        output["result"]["window_minutes"] = window_min
        output["result"]["target_fs"] = fs
        output["result"]["window_samples"] = int(round(window_min * 60 * fs))
        output["result"]["n_excluded_insufficient"] = len(insufficient)
        output["result"]["wall_clock_seconds"] = float(time.time() - started)
        _write_json(result_path, output["result"])
        output["oof_frame"].to_csv(directory / f"{tag}_multimodal_oof.csv", index=False)
        output["history"].to_csv(directory / f"{tag}_multimodal_history.csv", index=False)
        rows.append(_row_from_result(f"{tag}:multimodal", output["result"]))
        LOGGER.info("  AUROC %.3f (%.0fs)", output["result"]["metrics"]["pooled"]["auroc"], time.time() - started)
    # Aggregate every completed screening run in the directory so that a
    # partial re-run (for example resuming after an interruption) still
    # produces a complete comparison table.
    rows = []
    for result_path in sorted(directory.glob("*_multimodal.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        window_min = int(payload.get("window_minutes", 0))
        fs = float(payload.get("target_fs", 0.0))
        rows.append(_row_from_result(f"{_tag(window_min, fs)}:multimodal", payload))
    table = pd.DataFrame(rows)
    table.to_csv(directory / "deep_screening_comparison.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# Stage 6: quality sensitivity + comparison assembly
# ---------------------------------------------------------------------------
def run_quality_sensitivity(
    config: PhaseAConfig,
    metadata: pd.DataFrame,
    folds: pd.DataFrame,
    raw_signals: Dict[str, np.ndarray],
    force: bool = False,
) -> Dict:
    """Does restricting the development set to higher-quality records bias results?"""
    _setup_logging("quality_sensitivity", force)
    report_path = DATA_QUALITY_DIR / "record_quality.csv"
    if not report_path.exists():
        report = run_quality(config, force=force)
    else:
        report = pd.read_csv(report_path)
        report["record_id"] = report["record_id"].astype(str)
    frame, _, _ = build_configuration(metadata, raw_signals, CURRENT_WINDOW_MINUTES, CURRENT_FS)
    merged = frame.merge(
        report[
            [
                "record_id",
                "quality_category",
                "trimmed_duration_min",
                "fhr_missing_fraction_30m",
                "uc_missing_fraction_30m",
            ]
        ],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    subsets = {
        "all": merged,
        "good_or_acceptable": merged[merged["quality_category"].isin(["good", "acceptable"])],
        "good_only": merged[merged["quality_category"] == "good"],
    }
    rows = []
    for name, subset in subsets.items():
        excluded = merged[~merged["record_id"].isin(subset["record_id"])]
        table = [[int((subset["label"] == 1).sum()), int((subset["label"] == 0).sum())],
                 [int((excluded["label"] == 1).sum()), int((excluded["label"] == 0).sum())]]
        try:
            fisher_p = float(fisher_exact(table)[1]) if excluded.shape[0] else float("nan")
        except Exception:
            fisher_p = float("nan")
        row = {
            "subset": name,
            "n_records": int(len(subset)),
            "n_cesarean": int(subset["label"].sum()),
            "prevalence": float(subset["label"].mean()),
            "n_excluded": int(len(excluded)),
            "n_excluded_cesarean": int(excluded["label"].sum()),
            "fisher_exact_p_class_balance": fisher_p,
            "smd_duration": standardised_mean_difference(
                subset["trimmed_duration_min"], excluded["trimmed_duration_min"]
            )
            if len(excluded)
            else float("nan"),
            "smd_fhr_missing": standardised_mean_difference(
                subset["fhr_missing_fraction_30m"], excluded["fhr_missing_fraction_30m"]
            )
            if len(excluded)
            else float("nan"),
        }
        if len(subset) >= 20 and subset["label"].nunique() == 2:
            output = cv_baseline(subset, folds, method="none", model_name="logistic", seed=config.seed)
            row["cv_auroc"] = float(output["result"]["metrics"]["pooled"]["auroc"])
            row["cv_auprc"] = float(output["result"]["metrics"]["pooled"]["auprc"])
        else:
            row["cv_auroc"] = float("nan")
            row["cv_auprc"] = float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(DATA_QUALITY_DIR / "quality_sensitivity.csv", index=False)
    payload = {
        "note": (
            "Descriptive sensitivity analysis only. No record is removed from the main "
            "experiments. Fisher's exact test compares outcome counts inside and outside "
            "each subset; SMD is the absolute standardised mean difference."
        ),
        "subsets": rows,
    }
    _write_json(DATA_QUALITY_DIR / "quality_sensitivity.json", payload)
    LOGGER.info("Quality sensitivity: %s", table[["subset", "n_records", "n_cesarean", "cv_auroc"]].to_dict("records"))
    return payload


def run_compare(
    config: PhaseAConfig,
    metadata: pd.DataFrame,
    raw_signals: Dict[str, np.ndarray],
    folds: pd.DataFrame,
    force: bool = False,
) -> Dict:
    _setup_logging("compare", force)
    frames = {}
    for name, path in (
        ("window", WINDOW_DIR / "window_comparison.csv"),
        ("sampling", SAMPLING_DIR / "sampling_comparison.csv"),
        ("features", FEATURE_DIR / "feature_selection_comparison.csv"),
    ):
        if path.exists():
            frames[name] = pd.read_csv(path)
    if "window" not in frames or "sampling" not in frames:
        raise RuntimeError("Window and sampling comparisons must be run before compare.")

    combined = pd.concat([frames["window"], frames["sampling"]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["configuration"])
    deep_path = EXPERIMENTS_DIR / "deep_screening" / "deep_screening_comparison.csv"
    deep = pd.read_csv(deep_path) if deep_path.exists() else pd.DataFrame()
    if not deep.empty:
        combined = pd.concat([combined, deep], ignore_index=True)
    combined.to_csv(COMPARISON_DIR / "all_configurations.csv", index=False)

    plots.plot_comparison(
        combined[combined["model"] == "logistic"],
        COMPARISON_DIR / "figures",
        "window_and_sampling_auroc.png",
        "Window and sampling frequency (logistic regression, 5-fold CV)",
        metric="auroc_mean",
        error="auroc_std",
    )
    plots.plot_comparison(
        combined[combined["model"] == "gradient_boosting"],
        COMPARISON_DIR / "figures",
        "window_and_sampling_auroc_gb.png",
        "Window and sampling frequency (gradient boosting, 5-fold CV)",
        metric="auroc_mean",
        error="auroc_std",
    )

    # Select using development-set CV only; the test split is untouched.
    baseline = combined[combined["model"].isin(["logistic", "gradient_boosting"])]
    baseline = baseline[baseline["method"] == "none"]
    if baseline.empty:
        raise RuntimeError("No baseline results available for configuration selection")
    baseline = baseline.sort_values(["auroc_mean", "mcc_mean"], ascending=False)
    best = baseline.iloc[0]
    n_folds = max(1, int(config.n_splits))
    standard_error = float(best["auroc_std"]) / np.sqrt(n_folds)
    one_se_threshold = float(best["auroc_mean"]) - standard_error
    eligible = baseline[baseline["auroc_mean"] >= one_se_threshold].copy()
    # One-standard-error rule: among statistically indistinguishable options,
    # keep the current production configuration (30 min, 1 Hz, no selection)
    # rather than chasing a difference smaller than fold-to-fold noise.
    production = eligible[
        (eligible["window_minutes"].astype(float) == CURRENT_WINDOW_MINUTES)
        & (eligible["target_fs"].astype(float) == CURRENT_FS)
        & (eligible["method"] == "none")
    ].sort_values(["auroc_mean", "mcc_mean"], ascending=False)
    selected_row = production.iloc[0] if not production.empty else eligible.iloc[0]
    decision = {
        "selected_configuration": selected_row["configuration"],
        "selection_rule": (
            "One-standard-error rule: configurations whose mean out-of-fold ROC-AUC is "
            "within one standard error of the best are considered statistically "
            "indistinguishable; among them the current production configuration "
            "(30 min, 1 Hz, no feature selection) is preferred. The held-out test split "
            "was not evaluated."
        ),
        "best_mean_auroc": float(best["auroc_mean"]),
        "best_standard_error": standard_error,
        "one_se_threshold": one_se_threshold,
        "n_eligible": int(len(eligible)),
        "selected": {key: _json_default(value) for key, value in selected_row.to_dict().items()},
        "ranked_top10": [
            {key: _json_default(value) for key, value in row.items()}
            for row in baseline.head(10).to_dict("records")
        ],
    }
    _write_json(COMPARISON_DIR / "selection_decision.json", decision)
    LOGGER.info(
        "Selected configuration by development CV: %s (best %s, one-SE threshold %.3f)",
        selected_row["configuration"],
        best["configuration"],
        one_se_threshold,
    )

    # ROC/PR and confusion matrices for the best baseline per model family and
    # for the strongest deep configuration, if available.
    curves = {}
    for model_name in ("logistic", "gradient_boosting"):
        subset = baseline[baseline["model"] == model_name].sort_values(
            ["auroc_mean", "mcc_mean"], ascending=False
        )
        if subset.empty:
            continue
        row = subset.iloc[0]
        tag = _tag(int(row["window_minutes"]), float(row["target_fs"]))
        oof_name = f"{tag}_{model_name}_none_oof.csv"
        oof_path = None
        for candidate in (
            WINDOW_DIR / tag / oof_name,
            SAMPLING_DIR / tag / oof_name,
        ):
            if candidate.exists():
                oof_path = candidate
                break
        if oof_path is None:
            continue
        oof = pd.read_csv(oof_path)
        curves[f"{model_name} ({row['configuration']})"] = (
            oof["y_true"].to_numpy(),
            oof["prob"].to_numpy(),
        )
        metrics = json.loads(
            (oof_path.parent / oof_path.name.replace("_oof.csv", ".json")).read_text(encoding="utf-8")
        )
        plots.plot_confusion(
            metrics["metrics"]["pooled"]["confusion_matrix"],
            COMPARISON_DIR / "figures",
            f"confusion_{model_name}.png",
            f"Confusion matrix — {model_name}",
        )
    plots.plot_roc_pr(
        curves,
        COMPARISON_DIR / "figures",
        "roc_pr_best_baselines.png",
        "Best Phase A baselines (development-set out-of-fold predictions)",
    )

    if "features" in frames:
        feature_table = frames["features"]
        plots.plot_comparison(
            feature_table,
            COMPARISON_DIR / "figures",
            "feature_selection_auroc.png",
            "Feature selection comparison (pooled out-of-fold ROC-AUC)",
            label_col="configuration",
            metric="auroc_mean",
            error="auroc_std",
        )
        feature_table.to_csv(COMPARISON_DIR / "feature_selection_summary.csv", index=False)

    if not deep.empty:
        plots.plot_comparison(
            deep,
            COMPARISON_DIR / "figures",
            "deep_screening_auroc.png",
            "Deep multimodal screening (35 epochs, identical folds)",
            metric="auroc_mean",
            error="auroc_std",
        )
    return decision


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_configs(values: Optional[Sequence[str]]) -> Optional[List[Tuple[int, float]]]:
    if not values:
        return None
    configs = []
    for value in values:
        window, fs = value.split("@")
        configs.append((int(window), float(fs.rstrip("hz"))))
    return configs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase A data-handling experiments for OB-SIGHT")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["quality", "window", "sampling", "features", "deep", "compare", "report", "all"],
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached artefacts")
    parser.add_argument("--no-dl", action="store_true", help="Skip deep-learning screening")
    parser.add_argument("--epochs", type=int, default=None, help="Deep screening epochs")
    parser.add_argument("--quick", action="store_true", help="Minimal smoke run (2 epochs, 30 min / 1 Hz)")
    parser.add_argument("--windows", type=int, nargs="*", default=None)
    parser.add_argument("--rates", type=float, nargs="*", default=None)
    parser.add_argument(
        "--feature-configs",
        nargs="*",
        default=None,
        help="Configurations for feature selection, formatted as 30@1, 45@1, ...",
    )
    parser.add_argument(
        "--deep-configs",
        nargs="*",
        default=None,
        help="Configurations for deep screening, formatted as 30@1, 45@1, ...",
    )
    args = parser.parse_args(argv)

    config = PhaseAConfig()
    if args.quick:
        config = replace(
            config,
            windows=(30,),
            sampling_rates=(1.0,),
            dl_epochs=2,
        )
        args.no_dl = args.no_dl or False
    if args.windows:
        config.windows = tuple(args.windows)
    if args.rates:
        config.sampling_rates = tuple(args.rates)
    ensure_directories()
    config.save(EXPERIMENTS_DIR / "config.json")

    metadata = load_development_metadata()
    folds = make_folds(metadata, n_splits=config.n_splits, seed=config.seed)
    folds.to_csv(EXPERIMENTS_DIR / "folds.csv", index=False)
    _write_json(
        EXPERIMENTS_DIR / "dataset_manifest.json",
        {
            "dataset": "CTU-CHB Intrapartum Cardiotocography Database v1.0.0",
            "source_directory": str(DATASET_DIR),
            "native_sampling_hz": 4.0,
            "channels": ["FHR", "UC"],
            "processed_table": str(PROCESSED_DIR / "dataset.csv"),
            "processed_table_sha256": _sha256(PROCESSED_DIR / "dataset.csv"),
            "development_set": "split in {train, validation}; test excluded",
            "n_development_records": int(len(metadata)),
            "n_development_cesarean": int(metadata["label"].sum()),
            "split_counts": {str(k): int(v) for k, v in metadata["split"].value_counts().items()},
            "record_ids": metadata["record_id"].astype(str).tolist(),
            "seed": config.seed,
        },
    )
    raw_signals = _load_signals(metadata["record_id"])
    report = None

    if args.stage in ("quality", "all"):
        report = run_quality(config, force=args.force)

    if args.stage in ("window", "all"):
        run_window(config, raw_signals, metadata, folds, force=args.force)

    if args.stage in ("sampling", "all"):
        run_sampling(config, raw_signals, metadata, folds, force=args.force)

    if args.stage in ("features", "all"):
        feature_configs = _parse_configs(args.feature_configs)
        if feature_configs is None:
            # Evaluate every window at the production sampling rate so the
            # feature-selection comparison is not tied to a single window.
            feature_configs = [(int(window), 1.0) for window in config.windows]
        run_features(
            config,
            metadata,
            folds,
            raw_signals,
            target_configs=feature_configs,
            force=args.force,
        )

    if args.stage in ("deep", "all") and not args.no_dl:
        deep_configs = _parse_configs(args.deep_configs)
        if args.quick:
            deep_configs = [(30, 1.0)]
        run_deep(
            config,
            metadata,
            folds,
            raw_signals,
            force=args.force,
            epochs=args.epochs,
            configs=deep_configs,
        )

    if args.stage in ("compare", "all"):
        run_quality_sensitivity(
            config, metadata, folds, raw_signals, force=args.force
        )
        run_compare(config, metadata, raw_signals, folds, force=args.force)

    if args.stage in ("report", "all"):
        from src.phase_a.report import generate_report

        _setup_logging("report", force=args.force)
        path = generate_report(config)
        LOGGER.info("Report written to %s", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
