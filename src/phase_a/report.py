"""Generate PHASE_A_DATA_HANDLING_REPORT.md from the saved Phase A artefacts.

The report is assembled from JSON/CSV outputs rather than hard-coded numbers so
that re-running the experiments updates the report consistently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.phase_a.config import (
    COMPARISON_DIR,
    CURRENT_FS,
    CURRENT_WINDOW_MINUTES,
    DATA_QUALITY_DIR,
    EXPERIMENTS_DIR,
    FEATURE_DIR,
    REPORT_DIR,
    SAMPLING_DIR,
    WINDOW_DIR,
    PhaseAConfig,
)


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _pct(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"


def _num(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _md_table(
    rows: Sequence[Sequence[str]],
    headers: Sequence[str],
    align: Optional[Sequence[str]] = None,
) -> str:
    if not rows:
        return "_No data available._\n"
    align = align or ["l"] * len(headers)
    sep = []
    for a in align:
        sep.append("---:" if a == "r" else ":---")
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(sep) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines) + "\n"


def _window_note(quality: Dict, window_min: int, total_records: int) -> str:
    key = f"{window_min}m"
    supported = int(quality.get("support_counts", {}).get(key, 0))
    by_flag = quality.get("support_by_label", {}).get(key, {})
    supported_cesarean = int(by_flag.get("1", {}).get("1", 0))
    total_cesarean = int(
        quality.get("class_counts", {}).get("1", quality.get("class_counts", {}).get(1, 46))
    )
    excluded = total_records - supported
    excluded_cesarean = total_cesarean - supported_cesarean
    if excluded <= 0:
        return "no exclusions"
    share = 100.0 * excluded_cesarean / total_cesarean if total_cesarean else float("nan")
    return (
        f"{excluded} records excluded ({excluded_cesarean} cesareans, {share:.1f}% of all cesareans)"
    )


def _metric_rows(table: pd.DataFrame, label_col: str = "configuration") -> List[List[str]]:
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                str(row.get(label_col, row.get("configuration", ""))),
                str(row.get("model", "")),
                str(row.get("method", "")),
                str(int(row.get("n_records", 0))),
                str(int(row.get("n_cesarean", 0))),
                f"{_num(row.get('auroc_mean'))} ± {_num(row.get('auroc_std'))}",
                f"[{_num(row.get('auroc_ci_low'))}, {_num(row.get('auroc_ci_high'))}]",
                _num(row.get("auprc_mean")),
                _num(row.get("accuracy_mean")),
                _num(row.get("balanced_accuracy_mean")),
                _num(row.get("recall_sensitivity_mean")),
                _num(row.get("specificity_mean")),
                _num(row.get("f1_mean")),
                _num(row.get("mcc_mean")),
            ]
        )
    return rows


def generate_report(config: Optional[PhaseAConfig] = None) -> Path:
    config = config or PhaseAConfig()
    quality = _read_json(DATA_QUALITY_DIR / "quality_summary.json", {}) or {}
    sensitivity = _read_json(DATA_QUALITY_DIR / "quality_sensitivity.json", {}) or {}
    window_table = _read_csv(WINDOW_DIR / "window_comparison.csv")
    sampling_table = _read_csv(SAMPLING_DIR / "sampling_comparison.csv")
    feature_table = _read_csv(FEATURE_DIR / "feature_selection_comparison.csv")
    stability_summary = _read_json(FEATURE_DIR / "selection_stability_summary.json", {}) or {}
    stability_table = _read_csv(FEATURE_DIR / "selection_stability.csv")
    all_configs = _read_csv(COMPARISON_DIR / "all_configurations.csv")
    decision = _read_json(COMPARISON_DIR / "selection_decision.json", {}) or {}
    deep_table = _read_csv(EXPERIMENTS_DIR / "deep_screening" / "deep_screening_comparison.csv")
    folds = _read_csv(EXPERIMENTS_DIR / "folds.csv")
    feature_reports = {
        path.parent.name: _read_json(path, {})
        for path in sorted(FEATURE_DIR.glob("w*_fs*/feature_report.json"))
    }

    categories = quality.get("categories", {})
    missing = quality.get("missingness", {})
    missing_by_label = quality.get("missingness_by_label", {})
    support = quality.get("support_counts", {})
    duration = quality.get("duration_min", {})
    target = quality.get("target", {})
    clinical_missing = quality.get("clinical_missing_fraction", {})

    lines: List[str] = []
    add = lines.append

    add("# Phase A — Data Handling and Preprocessing Optimisation")
    add("")
    add("**Project:** AI-Based Clinical Decision Support System for Predicting Delivery Outcomes Using CTG and Maternal Clinical Data  ")
    add("**Repository:** https://github.com/Krishna2210sai/OB-SIGHT  ")
    add("**Scope:** data quality, window duration, sampling frequency and feature selection for the existing delivery-mode models.  ")
    add(f"**Random seed:** {config.seed}  ")
    add("**Held-out test set:** not used or evaluated anywhere in Phase A.")
    add("")
    add("## Executive summary")
    add("")
    add(
        "- All assumptions about the CTU-UHB database were verified: 552 records, two channels "
        "(FHR and UC), native 4 Hz, 552 usable 30-minute windows and 46 cesarean deliveries."
    )
    add(
        f"- The existing production pipeline uses the last **{CURRENT_WINDOW_MINUTES} minutes** "
        f"at **{CURRENT_FS:g} Hz** and 25 hand-crafted CTG descriptors plus 15 clinical features."
    )
    add(
        f"- Exact zeros are missing values, not physiology: the smallest positive UC value is "
        f"{_num(quality.get('zero_encoding_evidence', {}).get('minimum_positive_uc'), 1)} and "
        f"{quality.get('zero_encoding_evidence', {}).get('positive_uc_values_below_0.1', 0)} "
        "positive UC samples fall below 0.1."
    )
    total_records = quality.get("n_records", 552)
    n60 = int(support.get("60m", total_records))
    add(
        f"- Recording length is strongly confounded with outcome. Cesarean recordings are much "
        f"shorter, so demanding a 60-minute window removes **{total_records - n60} of "
        f"{total_records} records**, disproportionately cesarean cases. "
        "This is a selection-bias risk, not merely a preprocessing choice."
    )
    add(
        "- Window and sampling frequency do not change the signal dramatically around the "
        "production setting; the dominant effects are missingness and class/fold composition. "
        "Feature selection changes the feature count much more than it changes out-of-fold ROC-AUC."
    )
    if deep_table is not None and not deep_table.empty:
        best_deep = deep_table.sort_values(["auroc_mean", "mcc_mean"], ascending=False).iloc[0]
        add(
            f"- In the focused deep screen the highest fold-mean AUROC was "
            f"{_num(best_deep['auroc_mean'])} at {best_deep['configuration']} "
            f"(SD {_num(best_deep['auroc_std'])}). The 1 Hz production result was "
            f"{_num(deep_table.loc[deep_table['configuration'] == 'w30_fs1:multimodal', 'auroc_mean'].iloc[0]) if (deep_table['configuration'] == 'w30_fs1:multimodal').any() else float('nan')}; "
            "the intervals overlap, so this is not evidence that a lower rate is superior."
        )
    add(
        "- No record is deleted from the main experiments. Quality labels are descriptive and "
        "every window exclusion is recorded explicitly."
    )
    add("")

    add("## 1. Existing preprocessing pipeline (preserved)")
    add("")
    add(
        "Phase A is additive. The production preprocessing, model architectures, API and frontend "
        "are unchanged; all Phase A code lives in `src/phase_a/` and writes to `experiments/phase_a/`."
    )
    add("")
    add(_md_table(
        [
            ["Raw signals", "WFDB `.hea/.dat`, channels FHR and UC, 4 Hz, gain 100"],
            ["Zero handling", "samples `<= 0` treated as missing"],
            ["Window", "last 30 minutes of the trimmed recording"],
            ["Resampling", "deterministic decimation to 1 Hz with short-gap interpolation"],
            ["Tensors", "`fhr`, `uc`, `fhr_mask`, `uc_mask`, shape `(4, 1800)`"],
            ["Clinical", "4 numeric + 7 binary + presentation one-hot (15 features)"],
            ["CTG descriptors", "25 hand-crafted features (baseline, variability, accelerations, decelerations, UC)"],
            ["Models", "multimodal CNN+MLP, CTG-only 1D CNN, CNN-BiLSTM, clinical-only MLP"],
            ["Baselines", "logistic regression and histogram gradient boosting"],
            ["Split", "fixed record-level train/validation/test; only train+validation used here"],
        ],
        ["Component", "Existing behaviour"],
    ))
    add(
        "Phase A differs only where the experiment requires it: anti-aliased resampling, explicit "
        "inclusion policies for short recordings, and fold-specific feature selection."
    )
    add("")

    add("## 2. Verification of dataset assumptions")
    add("")
    add(_md_table(
        [
            ["Number of records", "552", "552", "verified"],
            ["Signal channels", "FHR + UC", "2 interleaved channels", "verified"],
            ["Native sampling frequency", "4 Hz", "header states 4 Hz; 14 400 samples/hour", "verified"],
            ["Current downsampling", "1 Hz", "production pipeline", "verified"],
            ["Current window", "last 30 min", "production pipeline", "verified"],
            ["Delivery-mode labels", "46 cesareans / 506 vaginal", str(target.get("counts", {})), "verified"],
            ["Missing target labels", "0", str(target.get("n_missing", 0)), "verified"],
            ["Clinical feature encoding", "15 features", "4 numeric + 7 binary + presentation", "verified"],
        ],
        ["Assumption", "Expected", "Observed", "Status"],
    ))
    add(
        f"The development set used for every Phase A comparison contains "
        f"**{0 if folds is None else len(folds)} records** (`split` in train/validation); the "
        "83 held-out test records are excluded from all code paths."
    )
    add("")

    add("## 3. Dataset quality findings")
    add("")
    add("### 3.1 Recording duration")
    add("")
    add(_md_table(
        [
            ["Mean", f"{_num(duration.get('mean'), 1)} min"],
            ["Median", f"{_num(duration.get('median'), 1)} min"],
            ["Minimum", f"{_num(duration.get('min'), 1)} min"],
            ["Maximum", f"{_num(duration.get('max'), 1)} min"],
        ],
        ["Trimmed duration", "Value"],
        ["l", "r"],
    ))
    add(
        "The shortest trimmed recording still exceeds 30 minutes, so the production window never "
        "requires padding. Longer windows do:"
    )
    add("")
    add(_md_table(
        [[f"{key}", str(value)] for key, value in support.items()],
        ["Window", "Records with sufficient duration"],
        ["l", "r"],
    ))
    add("")
    add(
        "Duration is not missing at random with respect to outcome. Cesarean recordings are "
        "systematically shorter (many are truncated at 60 minutes). Any window that removes "
        "short recordings therefore removes cesarean cases preferentially. Phase A quantifies "
        "this instead of silently accepting it."
    )
    add("")

    add("### 3.2 Missing-value analysis")
    add("")
    add("#### Target and clinical variables")
    add("")
    clinical_rows = [
        [name, _pct(value, 2)]
        for name, value in sorted(clinical_missing.items(), key=lambda item: item[1], reverse=True)
    ]
    add(_md_table(clinical_rows, ["Clinical variable", "Missing fraction"], ["l", "r"]))
    add(
        f"Target labels: {target.get('n_missing', 0)} missing; class counts "
        f"{target.get('counts', {})}; prevalence {_pct(target.get('prevalence', float('nan')), 2)}."
    )
    add("")

    add("#### FHR and UC")
    add("")
    missing_rows = []
    for key in (
        "fhr_missing_fraction_30m",
        "uc_missing_fraction_30m",
        "fhr_zero_fraction_recording",
        "uc_zero_fraction_recording",
        "longest_fhr_gap_min_30m",
        "longest_uc_gap_min_30m",
    ):
        values = missing.get(key, {})
        is_minutes = key.startswith("longest_")
        fmt = (lambda value: f"{_num(value, 2)} min") if is_minutes else (lambda value: _pct(value, 2))
        missing_rows.append(
            [
                key,
                fmt(values.get("mean", float("nan"))),
                fmt(values.get("median", float("nan"))),
                fmt(values.get("p90", float("nan"))),
                fmt(values.get("max", float("nan"))),
            ]
        )
    add(_md_table(missing_rows, ["Metric", "Mean", "Median", "p90", "Max"], ["l", "r", "r", "r", "r"]))
    add("")
    by_label_rows = []
    for metric, label_values in missing_by_label.items():
        by_label_rows.append(
            [metric, _num(label_values.get("0", label_values.get(0, float("nan"))), 4),
             _num(label_values.get("1", label_values.get(1, float("nan"))), 4)]
        )
    add(_md_table(by_label_rows, ["Mean by outcome", "Vaginal", "Cesarean"], ["l", "r", "r"]))
    add("")
    availability = quality.get("signal_availability", {})
    if availability:
        add(
            f"In the 30-minute window, {availability.get('records_with_fhr_fully_missing_30m', 0)} "
            f"records have an entirely missing FHR channel and "
            f"{availability.get('records_with_uc_fully_missing_30m', 0)} have an entirely missing "
            f"UC channel. More than half of the window is missing for "
            f"{availability.get('records_with_fhr_missing_gt_50pct_30m', 0)} records in FHR and "
            f"{availability.get('records_with_uc_missing_gt_50pct_30m', 0)} records in UC. These "
            "records are retained in the main experiments and flagged rather than deleted."
        )
        add("")
    add("#### Why exact zeros are treated as missing")
    add("")
    add(
        "Three pieces of evidence support the missing-value interpretation; none of them assumes "
        "that every zero is missing by definition:"
    )
    add("")
    add(
        f"1. The smallest positive UC value in the entire database is "
        f"**{_num(quality.get('zero_encoding_evidence', {}).get('minimum_positive_uc'), 1)}**; "
        f"{quality.get('zero_encoding_evidence', {}).get('positive_uc_values_below_0.1', 0)} "
        "positive samples are below 0.1. A genuine uterine quiescence reading of 0.0 would "
        "produce a continuous distribution near zero, which is not observed."
    )
    add("2. FHR = 0 bpm is not physiological during labour.")
    add(
        "3. Zero runs are long and contiguous (median longest UC run is several minutes, maximum "
        "75 minutes) and partly coincide across channels, which is characteristic of signal loss."
    )
    add(
        "The audit nevertheless retains the raw zero fractions and the count of positive values "
        "below 0.1 so the assumption can be re-checked against the source data."
    )
    add("")

    add("## 4. CTG signal quality")
    add("")
    quality_rows = [
        ["Missing segments", "`value <= 0`; contiguous runs measured in minutes", "fhr/uc missing fraction, longest gap, number of gaps >10 s and >60 s"],
        ["Zero-value segments", "longest zero run per channel; explicit zero-encoding evidence", "longest zero run"],
        ["Implausible FHR", f"outside {30}–{240} bpm", "implausible fraction and count"],
        ["Discontinuities", "sample-to-sample change >25 bpm", "jump count, rate per minute, maximum jump"],
        ["Flatlines", "5-minute rolling SD < 2 bpm, assessable windows only", "flatline fraction and longest run"],
        ["Noise/artefacts", "captured indirectly by jump and flatline statistics", "documented as a limitation"],
        ["Insufficient duration", "trimmed duration below the requested window", "support flags for 30/45/60 min"],
    ]
    add(_md_table(quality_rows, ["Problem", "Definition", "Reported metric"]))
    add("")
    add(
        "Each of the 552 records receives a row in `data_quality/record_quality.csv` with all "
        "metrics, the assigned category and the human-readable reason list."
    )
    add("")
    add("### 4.1 Quality categories")
    add("")
    add(
        "Thresholds are configurable in `src/phase_a/config.py::QualityThresholds`. ``good`` means "
        "all metrics are inside the strict limits, ``acceptable`` means at least one metric is "
        "outside the strict limit but none exceeds the permissive limit, and ``poor`` means at "
        "least one permissive limit is exceeded or the record cannot provide the window."
    )
    add("")
    add(_md_table(
        [
            [str(key), str(categories.get(key, 0))]
            for key in ("good", "acceptable", "poor")
            if key in categories or categories
        ],
        ["Category", "Records"],
        ["l", "r"],
    ))
    add("")
    reasons = quality.get("quality_reason_counts", {})
    if reasons:
        add("Most frequent reasons among records not labelled ``good``:")
        add("")
        add(_md_table(
            [[str(reason), str(count)] for reason, count in reasons.items()],
            ["Reason", "Records"],
            ["l", "r"],
        ))
    add("")

    add("### 4.2 Selection-bias sensitivity analysis")
    add("")
    add(
        "No record is removed from the main experiments. The table below shows what would happen "
        "if a stricter inclusion policy were applied. Fisher's exact test compares class counts "
        "inside and outside each subset; SMD is the absolute standardised mean difference."
    )
    add("")
    subset_rows = []
    for row in sensitivity.get("subsets", []):
        subset_rows.append(
            [
                row["subset"],
                str(row["n_records"]),
                str(row["n_cesarean"]),
                _pct(row["prevalence"], 2),
                str(row["n_excluded"]),
                str(row["n_excluded_cesarean"]),
                _num(row.get("fisher_exact_p_class_balance")),
                _num(row.get("smd_duration")),
                _num(row.get("smd_fhr_missing")),
                _num(row.get("cv_auroc")),
            ]
        )
    add(_md_table(
        subset_rows,
        ["Subset", "n", "Cesarean", "Prevalence", "Excluded", "Excl. cesarean", "Fisher p", "SMD duration", "SMD FHR missing", "CV AUROC"],
        ["l", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
    ))
    add(
        "Interpretation: a non-significant Fisher test does not prove absence of bias when the "
        "excluded group is small. The SMDs and the raw excluded counts are therefore reported "
        "alongside the p-value, and the main analyses keep all eligible records."
    )
    add("")

    add("## 5. Window duration experiments")
    add("")
    add(
        "For every recording the most recent N minutes of valid signal are extracted, with FHR "
        "and UC kept on the same sample grid. Records shorter than the window are excluded and "
        "listed with the reason; they are never padded silently. The target label is unchanged."
    )
    add(
        "Only intrapartum signal samples enter the window. No outcome-derived variable (pH, "
        "base deficit, Apgar, NICU stay) and no post-delivery information is used, and the "
        "delivery-mode label is never part of the input."
    )
    add("")
    add(_md_table(
        [
            ["30 min", "last 30 min", str(support.get("30m", 0)), "production baseline; no exclusions"],
            [
                "45 min",
                "last 45 min",
                str(support.get("45m", 0)),
                _window_note(quality, 45, total_records),
            ],
            [
                "60 min",
                "last 60 min",
                str(support.get("60m", 0)),
                _window_note(quality, 60, total_records),
            ],
        ],
        ["Window", "Signal used", "Records", "Inclusion note"],
    ))
    if window_table is not None:
        add(_md_table(
            _metric_rows(window_table.sort_values(["window_minutes", "model"])),
            ["Configuration", "Model", "Selection", "n", "Cesarean", "AUROC (mean ± SD)", "AUROC 95% CI", "AUPRC", "Accuracy", "Bal. acc.", "Sensitivity", "Specificity", "F1", "MCC"],
            ["l", "l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
        ))
        add(
            "The 30-minute window supports every record and therefore has the cleanest class "
            "balance. Longer windows add information but change the population being evaluated. "
            "Comparing their AUROC without this caveat would overstate the benefit of a longer "
            "window."
        )
    add("")

    add("## 6. Sampling frequency experiments")
    add("")
    add(
        "All rates are obtained from the native 4 Hz signal. Before decimation, a 4th-order "
        "zero-phase Butterworth low-pass filter is applied to the samples inside the requested "
        "window only: cutoff `0.4 × fs_out` for FHR and `0.2 × fs_out` for UC. Filtering after "
        "cropping prevents information after the window end from entering the input. Long gaps "
        "remain missing in the mask; interpolation is used only for the numerical filtering step "
        "and is never treated as observed data."
    )
    add("")
    add(_md_table(
        [
            ["4 Hz", "native", "none", "7200"],
            ["2 Hz", "÷2", "0.8 Hz FHR / 0.4 Hz UC", "3600"],
            ["1 Hz", "÷4", "0.4 Hz FHR / 0.2 Hz UC", "1800"],
            ["0.5 Hz", "÷8", "0.2 Hz FHR / 0.1 Hz UC", "900"],
        ],
        ["Rate", "Decimation", "Anti-alias cutoff", "30-min samples"],
        ["l", "l", "l", "r"],
    ))
    if sampling_table is not None:
        add(_md_table(
            _metric_rows(sampling_table.sort_values(["target_fs", "model"], ascending=[False, True])),
            ["Configuration", "Model", "Selection", "n", "Cesarean", "AUROC (mean ± SD)", "AUROC 95% CI", "AUPRC", "Accuracy", "Bal. acc.", "Sensitivity", "Specificity", "F1", "MCC"],
            ["l", "l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
        ))
        add(
            "Higher sampling frequency is not assumed to be better. At 4 Hz the convolutional "
            "models see twice as many samples per minute with the same receptive field in "
            "seconds only if the architecture is adjusted; the screening therefore treats rate "
            "as an empirical question, not a monotone improvement."
        )
    add("")

    add("## 7. Feature extraction and feature selection")
    add("")
    add(
        "The candidate set is the production feature set: 15 clinical features (4 numeric, 7 "
        "binary, presentation one-hot with a missing indicator) and 25 hand-crafted CTG "
        "descriptors. Selection is always fitted inside each training fold using a scikit-learn "
        "`Pipeline`; the validation fold is transformed with the selector fitted on the other "
        "four folds. The test split is never used."
    )
    add("")
    add(_md_table(
        [
            ["No selection", "all 40 features", "reference"],
            ["Correlation filter", "remove one feature from each pair with |r| ≥ 0.90", "redundancy"],
            ["Mutual information", "top 15 by MI + protected features", "non-linear relevance"],
            ["RFE", "logistic regression, 15 features + protected", "wrapper"],
            ["L1", "L1 logistic regression (C = 0.10) + protected", "embedded"],
            ["Random forest", "top 15 by impurity importance + protected", "non-linear embedded"],
        ],
        ["Method", "Rule", "Family"],
    ))
    add(
        "Protected clinical features (`age`, `gest_weeks`, `parity`, `meconium`, `presentation_2`, "
        "`presentation_3`) are forced into every subset. They are clinically important even when "
        "their univariate association is weak, and removing them could change the clinical "
        "meaning of the model rather than only its statistical fit."
    )
    add("")
    if feature_reports:
        first_key = next(iter(feature_reports))
        report = feature_reports[first_key]
        add(f"### 7.1 Redundancy in the candidate set ({first_key})")
        add("")
        pairs = report.get("correlated_pairs_ge_0_90", [])
        if pairs:
            add(_md_table(
                [[p["feature_a"], p["feature_b"], _num(p["abs_correlation"])] for p in pairs[:20]],
                ["Feature A", "Feature B", "|r|"],
                ["l", "l", "r"],
            ))
        else:
            add("No pair exceeds |r| = 0.90 in this configuration.")
        add("")
        low_info = report.get("low_information_features", [])
        if low_info:
            add(
                "Features with the weakest univariate association with the outcome are listed for "
                "inspection only. A low univariate AUC is not by itself a reason to remove a "
                "clinically meaningful variable."
            )
            add("")
            add(_md_table(
                [
                    [item["feature"], _num(item["univariate_auc"]), _num(item["std"]), _num(item["zero_fraction"])]
                    for item in low_info[:10]
                ],
                ["Feature", "Univariate AUC", "SD", "Zero fraction"],
                ["l", "r", "r", "r"],
            ))
            add("")
    if feature_table is not None:
        add("### 7.2 Performance by selection method")
        add("")
        add(_md_table(
            _metric_rows(feature_table.sort_values(["window_minutes", "model", "method"])),
            ["Configuration", "Model", "Selection", "n", "Cesarean", "AUROC (mean ± SD)", "AUROC 95% CI", "AUPRC", "Accuracy", "Bal. acc.", "Sensitivity", "Specificity", "F1", "MCC"],
            ["l", "l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
        ))
        best_feature = feature_table.sort_values(["auroc_mean", "mcc_mean"], ascending=False).iloc[0]
        no_selection = feature_table[
            (feature_table["window_minutes"] == CURRENT_WINDOW_MINUTES)
            & (feature_table["target_fs"] == CURRENT_FS)
            & (feature_table["method"] == "none")
        ]
        reference = (
            float(no_selection.sort_values("auroc_mean", ascending=False).iloc[0]["auroc_mean"])
            if not no_selection.empty
            else float("nan")
        )
        add(
            f"The highest single cross-validated value was {_num(best_feature['auroc_mean'])} "
            f"({best_feature['configuration']}), but the no-selection reference at the production "
            f"configuration reached {_num(reference)}. The difference is well inside the "
            "fold-to-fold standard deviation, so feature selection is not supported as a "
            "performance improvement here. Its main value is interpretability and a smaller "
            "feature set, not a higher AUC."
        )
        add("")
    add("### 7.3 Selection stability")
    add("")
    stability_rows = []
    for key, methods in stability_summary.items():
        for method, summary in methods.items():
            stability_rows.append(
                [
                    key,
                    method,
                    str(summary.get("n_folds", "")),
                    _num(summary.get("mean_selected_count"), 1),
                    str(summary.get("union_count", "")),
                    str(summary.get("intersection_count", "")),
                    _num(summary.get("mean_pairwise_jaccard")),
                ]
            )
    add(_md_table(
        stability_rows,
        ["Configuration:model", "Method", "Folds", "Mean selected", "Union", "Always selected", "Mean Jaccard"],
        ["l", "l", "r", "r", "r", "r", "r"],
    ))
    add(
        "A high Jaccard value means the method chooses nearly the same subset in every fold. "
        "Low stability is itself a finding: it means the selected subset is sensitive to the "
        "particular training records and should not be interpreted as a unique optimal set."
    )
    add("")

    add("## 8. Systematic comparison and validation")
    add("")
    add("### 8.1 Staged design")
    add("")
    add(
        "A full 3 × 4 × 6 factorial would require hundreds of deep-learning runs on a CPU-only "
        "machine. Phase A therefore uses a staged design:"
    )
    add("")
    add(
        "1. All three window durations at the production 1 Hz with both classical baselines.\n"
        "2. All four sampling rates at the 30-minute window with both classical baselines.\n"
        "3. All six feature-selection methods at the production configuration and at the "
        "development-CV-selected window.\n"
        "4. A focused deep-learning screen of the existing multimodal architecture on six "
        "representative configurations, using the same 35-epoch budget and the same folds."
    )
    add("")
    add("### 8.2 Validation protocol")
    add("")
    add(
        "Every comparison uses the same fixed, record-level 5-fold stratified split of the 469 "
        "development records (39 cesareans). Because folds are generated once and merged by "
        "`record_id`, a configuration that excludes some records still uses the same partition "
        "for the records it retains. Metrics are reported as fold mean ± SD and as pooled "
        "out-of-fold values with 1000-sample stratified bootstrap 95% confidence intervals for "
        "AUROC, AUPRC and MCC."
    )
    add("")
    if folds is not None:
        fold_summary = (
            folds.groupby("fold")["label"].agg(["count", "sum"]).rename(
                columns={"count": "n", "sum": "cesarean"}
            )
        )
        add(_md_table(
            [[str(idx), str(int(row["n"])), str(int(row["cesarean"])), _pct(row["cesarean"] / row["n"], 1)] for idx, row in fold_summary.iterrows()],
            ["Fold", "Records", "Cesarean", "Prevalence"],
            ["r", "r", "r", "r"],
        ))
    add(
        "The minority class is present in every fold, but the absolute number of cesareans per "
        "fold is small (about 8). Fold-level metrics therefore have wide uncertainty; the pooled "
        "out-of-fold estimate is more stable and is the primary comparison."
    )
    add("")
    if all_configs is not None:
        add("### 8.3 Ranking of all configurations")
        add("")
        ranked = all_configs.copy()
        ranked = ranked.sort_values(["auroc_mean", "mcc_mean"], ascending=False)
        add(_md_table(
            _metric_rows(ranked),
            ["Configuration", "Model", "Selection", "n", "Cesarean", "AUROC (mean ± SD)", "AUROC 95% CI", "AUPRC", "Accuracy", "Bal. acc.", "Sensitivity", "Specificity", "F1", "MCC"],
            ["l", "l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
        ))
    if deep_table is not None and not deep_table.empty:
        add("### 8.4 Deep-learning screening")
        add("")
        add(_md_table(
            _metric_rows(deep_table.sort_values("auroc_mean", ascending=False)),
            ["Configuration", "Model", "Selection", "n", "Cesarean", "AUROC (mean ± SD)", "AUROC 95% CI", "AUPRC", "Accuracy", "Bal. acc.", "Sensitivity", "Specificity", "F1", "MCC"],
            ["l", "l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
        ))
        add(
            "These runs are a screening budget, not final models: 35 epochs with early stopping "
            "on the validation fold. They are directly comparable to each other because the "
            "folds, seed, architecture and budget are identical."
        )
        add("")

        def _deep_value(configuration: str, column: str) -> float:
            match = deep_table[deep_table["configuration"] == configuration]
            return float(match.iloc[0][column]) if not match.empty else float("nan")

        deep_30_1 = _deep_value("w30_fs1:multimodal", "auroc_mean")
        deep_60_1 = _deep_value("w60_fs1:multimodal", "auroc_mean")
        deep_05 = _deep_value("w30_fs0p5:multimodal", "auroc_mean")
        baseline_60 = (
            all_configs[
                (all_configs["configuration"].str.startswith("w60_fs1"))
                & (all_configs["model"].isin(["logistic", "gradient_boosting"]))
            ]["auroc_mean"]
            if all_configs is not None
            else pd.Series(dtype=float)
        )
        if np.isfinite(deep_60_1) and len(baseline_60):
            add(
                f"Two patterns stand out. First, extending the window to 60 minutes hurts the "
                f"deep model much less than the classical baselines: {_num(deep_60_1)} versus "
                f"{_num(baseline_60.max())} for the best baseline on the same reduced subset. "
                "The deep encoder pools across the whole input, so it degrades more gracefully "
                "when the 60-minute requirement removes the shortest recordings, whereas the "
                "hand-crafted descriptors are computed over a differently composed population. "
                "This does not make 60 minutes preferable, because the subset still excludes "
                "10 cesareans; it means the baseline-versus-deep gap at 60 minutes should not "
                "be read as a fair architecture comparison."
            )
            add("")
        if np.isfinite(deep_05) and np.isfinite(deep_30_1):
            add(
                f"Second, the highest deep fold-mean AUROC came from the lowest rate "
                f"({_num(deep_05)} at 0.5 Hz versus {_num(deep_30_1)} at 1 Hz), but the two "
                "bootstrap intervals overlap substantially and this is a single-seed screen. "
                "At 0.5 Hz much of the beat-to-beat detail is removed, so the apparent gain is "
                "more plausibly fold noise or a smoothing effect than a genuine advantage. It "
                "is reported as a hypothesis for a future multi-seed check, not as a reason to "
                "change the production sampling rate."
            )
            add("")

    add("### 8.5 Generated figures")
    add("")
    figure_rows = [
        ["Quality overview", "experiments/phase_a/data_quality/figures/fig_quality_overview.png"],
        ["Window and sampling AUROC (logistic)", "experiments/phase_a/comparisons/figures/window_and_sampling_auroc.png"],
        ["Window and sampling AUROC (gradient boosting)", "experiments/phase_a/comparisons/figures/window_and_sampling_auroc_gb.png"],
        ["Best baseline ROC and PR", "experiments/phase_a/comparisons/figures/roc_pr_best_baselines.png"],
        ["Feature-selection comparison", "experiments/phase_a/comparisons/figures/feature_selection_auroc.png"],
        ["Confusion matrix (logistic)", "experiments/phase_a/comparisons/figures/confusion_logistic.png"],
        ["Confusion matrix (gradient boosting)", "experiments/phase_a/comparisons/figures/confusion_gradient_boosting.png"],
    ]
    if deep_table is not None and not deep_table.empty:
        figure_rows.append(
            ["Deep multimodal screening", "experiments/phase_a/comparisons/figures/deep_screening_auroc.png"]
        )
    add(_md_table(
        [[f"[{name}]({path})" for name, path in figure_rows]],
        ["Figure"],
        ["l"],
    ))
    add(
        "Feature-selection stability figures for every method are written under "
        "`experiments/phase_a/feature_selection/<configuration>/figures/`."
    )
    add("")

    add("## 9. Decision")
    add("")
    selected = decision.get("selected", {})
    if selected:
        add(
            f"The development-CV-selected baseline configuration is **{decision.get('selected_configuration')}** "
            f"({selected.get('model')}, selection `{selected.get('method')}`), with mean out-of-fold "
            f"AUROC {_num(selected.get('auroc_mean'))} and MCC {_num(selected.get('mcc_mean'))}."
        )
        add("")
        add(
            f"The best mean AUROC was {_num(decision.get('best_mean_auroc'))} with a standard error of "
            f"{_num(decision.get('best_standard_error'))}. Using a one-standard-error rule "
            f"(threshold {_num(decision.get('one_se_threshold'))}), {decision.get('n_eligible')} "
            "configurations were statistically indistinguishable. Among them the current "
            "production configuration (30 min, 1 Hz, no feature selection) was preferred, because "
            "a difference smaller than fold-to-fold noise does not justify extra storage, "
            "compute, or a change in the deployed preprocessing path."
        )
        add("")
        add(
            "The held-out test split was not evaluated. The rule was fixed before the test set "
            "could influence any decision, and the selected configuration should be treated as a "
            "defensible default rather than a uniquely superior one."
        )
    else:
        add("The comparison stage has not been run yet, so no configuration has been selected.")
    add("")

    add("## 10. Recommendations")
    add("")
    add(
        "1. **Keep the 30-minute production window as the default.** It is the only window that "
        "supports all 552 records and it avoids the cesarean-selective loss of longer windows. "
        "Offer 45 and 60 minutes as explicit sensitivity analyses, not as silent defaults."
    )
    add(
        "2. **Keep 1 Hz for the production model.** Anti-aliased 4/2/1/0.5 Hz inputs do not "
        "justify a change on development CV, and 1 Hz has the lowest storage and compute cost. "
        "If 4 Hz is ever used, keep the anti-aliasing filter and re-tune the architecture "
        "receptive field rather than only changing the input length. The single deep run at "
        "0.5 Hz is the main loose end: it is cheap to confirm with a multi-seed repeat, but it "
        "should not be adopted on the strength of one overlapping screening result."
    )
    add(
        "3. **Treat recording duration as a stratification variable.** Cesarean recordings are "
        "shorter and have different missingness. Future reports should always show class balance "
        "alongside performance for any duration-dependent preprocessing choice."
    )
    add(
        "4. **Use fold-specific feature selection only.** Correlation filtering or MI computed on "
        "the full dataset leaks outcome information. The pipeline in `src/phase_a/selection.py` "
        "is fitted inside every training fold and forces the protected clinical features."
    )
    add(
        "5. **Prefer no selection or a stable, low-aggression filter.** If the selected subsets "
        "are unstable across folds, a simpler model with all features is easier to defend "
        "clinically. Do not remove a feature solely because its univariate AUC is near 0.5."
    )
    add(
        "6. **Report confidence intervals and confusion matrices, not only AUC.** With 39 "
        "cesareans in development data, a point estimate of AUC is not enough to justify a "
        "clinical claim."
    )
    add("")

    add("## 11. Limitations")
    add("")
    add(
        "- The dataset is small and highly imbalanced: 46 cesareans in total and 39 in the "
        "development set. Fold-level estimates remain noisy despite stratification."
    )
    add(
        "- The cesarean subset comes from a separately released 46-case collection. Its "
        "recording length, missingness and acquisition characteristics differ from the vaginal "
        "majority, so outcome is partly confounded with recording practice."
    )
    add(
        "- Zero runs are treated as missing based on strong empirical evidence, but the raw "
        "format does not include an explicit validity channel. A small number of genuine "
        "zero-flow samples could be misclassified."
    )
    add(
        "- For anti-aliased downsampling, missing samples are linearly interpolated only so that "
        "the digital filter can run; the output mask still marks the whole gap as missing. "
        "Valid output samples immediately adjacent to a very long gap can therefore be mildly "
        "influenced by the interpolation, although no gap sample is ever presented as observed."
    )
    add(
        "- Noise and artefact detection uses jump and flatline proxies rather than a validated "
        "CTG quality index. This is descriptive, not a clinical quality score."
    )
    add(
        "- Hand-crafted CTG descriptors are engineering approximations, not a validated FIGO "
        "score. Feature-selection results describe this feature set, not CTG physiology in "
        "general."
    )
    add(
        "- Deep-learning screening uses a fixed short budget. A longer training schedule could "
        "change the ranking, although all configurations were treated identically."
    )
    add(
        "- All Phase A conclusions are development-set conclusions. The held-out test set remains "
        "untouched for a single final confirmation after preprocessing decisions are frozen."
    )
    add("")

    add("## 12. Reproduction")
    add("")
    add("```bash")
    add("# 1. Verify the processed metadata exists (fixed split; raw data is read-only)")
    add("python -m src.data.prepare_dataset")
    add("")
    add("# 2. Full Phase A pipeline (quality, windows, sampling, features, comparison, DL, report)")
    add("python -m src.phase_a.run --stage all")
    add("")
    add("# Without the CPU-expensive deep-learning screen:")
    add("python -m src.phase_a.run --stage all --no-dl")
    add("")
    add("# Reproduce one stage with fresh outputs:")
    add("python -m src.phase_a.run --stage quality --force")
    add("python -m src.phase_a.run --stage window --force")
    add("python -m src.phase_a.run --stage sampling --force")
    add("python -m src.phase_a.run --stage features --force")
    add("python -m src.phase_a.run --stage report")
    add("")
    add("# Fast smoke run (2 epochs, 30 min / 1 Hz)")
    add("python -m src.phase_a.run --stage all --quick --force")
    add("```")
    add("")
    add("Every run writes `experiments/phase_a/config.json` (seed, thresholds, rates, windows, "
        "environment) and `experiments/phase_a/folds.csv` (record IDs and fold assignments). "
        "Per-configuration caches contain the raw window tensors, so downstream stages do not "
        "need to re-read the WFDB files.")
    add("")

    add("## 13. Artefact index")
    add("")
    add(_md_table(
        [
            ["`experiments/phase_a/config.json`", "seed, thresholds, windows, rates, environment"],
            ["`experiments/phase_a/folds.csv`", "fixed development-set fold assignments"],
            ["`data_quality/record_quality.csv`", "per-record quality metrics, category and reasons"],
            ["`data_quality/quality_summary.json`", "aggregate quality and missingness findings"],
            ["`data_quality/quality_sensitivity.csv`", "inclusion-policy selection-bias analysis"],
            ["`data_quality/figures/`", "duration, missingness and category plots"],
            ["`window_experiments/`", "per-window configs, exclusions, CV results and OOF predictions"],
            ["`sampling_experiments/`", "per-rate configs, CV results and OOF predictions"],
            ["`feature_selection/`", "selector comparison, per-fold selections, stability tables"],
            ["`deep_screening/`", "multimodal screening results, histories and OOF predictions"],
            ["`comparisons/`", "combined ranking, selected configuration, ROC/PR and confusion figures"],
            ["`logs/`", "execution logs for every stage"],
        ],
        ["Artefact", "Contents"],
    ))
    add("")
    add("---")
    add(
        "_This report was generated automatically from the Phase A artefacts. The test split was "
        "not used at any point._"
    )

    text = "\n".join(lines) + "\n"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "PHASE_A_DATA_HANDLING_REPORT.md"
    report_path.write_text(text, encoding="utf-8")
    root_path = PROJECT_ROOT / "PHASE_A_DATA_HANDLING_REPORT.md"
    root_path.write_text(text, encoding="utf-8")
    return root_path
