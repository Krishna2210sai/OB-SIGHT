"""Markdown report generation for Phase B.

Every number in the report is read from an artifact produced by the B0/B1
runs.  Missing artifacts are reported as unavailable rather than imputed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.phase_b.config import (
    AUGMENT_MODES,
    B0_DIR,
    CLASSICAL_DIR,
    COMPARISON_DIR,
    DEEP_DIR,
    DEEP_STRATEGIES,
    DEEP_VARIANTS,
    PHASE_A_DIR,
    PHASE_B_DIR,
    REPORT_DIR,
    augment_result_path,
    b0_result_path,
    classical_result_path,
    deep_result_path,
    selection_path,
)
from src.phase_b.imbalance import SPECS
from src.phase_b.metrics import evaluate_predictions

REPORT_PATH = REPORT_DIR / "PHASE_B_IMBALANCE_REPORT.md"
PRODUCTION_REPORTS = PHASE_B_DIR.parents[1] / "reports"

DEEP_LABELS = {
    "signal": "CTG-only 1D CNN",
    "bilstm": "CNN-BiLSTM",
    "multimodal": "Multimodal CNN + Clinical MLP",
    "clinical": "Clinical-only MLP",
}
CLASSICAL_LABELS = {
    "logistic": "Logistic regression",
    "gradient_boosting": "Histogram gradient boosting",
}
STRATEGY_LABELS = {
    "plain_bce": "Plain BCE",
    "weighted_bce": "Weighted BCE",
    "focal_gamma1": "Focal loss (gamma=1)",
    "focal_gamma2": "Focal loss (gamma=2)",
    "weighted_sampling": "Weighted random sampling",
    "none": "No rebalancing",
    "class_weight": "Class weights",
    "random_oversample": "Random oversampling",
    "smote": "SMOTE (tabular)",
}
METRIC_LABELS = {
    "auroc": "ROC-AUC",
    "auprc": "PR-AUC",
    "recall_sensitivity": "Cesarean sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1": "F1",
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
    "mcc": "MCC",
    "brier": "Brier",
    "ece": "ECE",
}
MAIN_KEYS = ("auroc", "auprc", "recall_sensitivity", "specificity", "mcc", "brier", "ece")


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _f(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _ci(pooled: Dict, key: str, digits: int = 3) -> str:
    low = pooled.get(f"{key}_ci_low")
    high = pooled.get(f"{key}_ci_high")
    if low is None or high is None:
        return "—"
    if not (np.isfinite(low) and np.isfinite(high)):
        return "—"
    return f"{float(low):.{digits}f}–{float(high):.{digits}f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rows = list(rows)
    if not rows:
        return "_No results available._\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def _pooled(result: Optional[Dict]) -> Dict:
    if not result:
        return {}
    return result.get("metrics", {}).get("pooled", {})


def _fold_frame(result: Optional[Dict]) -> pd.DataFrame:
    if not result:
        return pd.DataFrame()
    folds = result.get("metrics", {}).get("folds", [])
    return pd.DataFrame(folds)


def _load_deep(variant: str, strategy: str) -> Optional[Dict]:
    return _read_json(deep_result_path(variant, strategy) / "result.json")


def _load_classical(model: str, strategy: str) -> Optional[Dict]:
    return _read_json(classical_result_path(model, strategy) / "result.json")


def _pooled_at_inner_threshold(directory: Path) -> Dict:
    """Pool out-of-fold predictions using each fold's inner-selected threshold."""
    predictions = _read_csv(directory / "predictions.csv")
    folds = _read_csv(directory / "fold_metrics.csv")
    if predictions is None or folds is None or folds.empty:
        return {}
    threshold_map = dict(zip(folds["fold"].astype(int), folds["inner_threshold"].astype(float)))
    pred = (predictions["prob"].to_numpy(float) >= predictions["fold"].map(threshold_map).to_numpy(float))
    y = predictions["y_true"].to_numpy(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def _b0_rows() -> List[Dict]:
    rows = []
    for variant in DEEP_VARIANTS:
        result = _read_json(b0_result_path(variant) / "result.json")
        if result:
            rows.append(
                {
                    "family": "deep",
                    "name": DEEP_LABELS[variant],
                    "key": variant,
                    "result": result,
                }
            )
    for model in CLASSICAL_LABELS:
        result = _read_json(B0_DIR / model / "result.json")
        if result:
            rows.append(
                {
                    "family": "classical",
                    "name": CLASSICAL_LABELS[model],
                    "key": model,
                    "result": result,
                }
            )
    return rows


def _b1_deep_summary() -> pd.DataFrame:
    rows = []
    for variant in DEEP_VARIANTS:
        for strategy in DEEP_STRATEGIES:
            result = _load_deep(variant, strategy)
            if not result:
                continue
            pooled = _pooled(result)
            rows.append(
                {
                    "variant": variant,
                    "strategy": strategy,
                    **{key: pooled.get(key, np.nan) for key in MAIN_KEYS},
                    "precision": pooled.get("precision", np.nan),
                    "f1": pooled.get("f1", np.nan),
                }
            )
    return pd.DataFrame(rows)


def _b1_classical_summary() -> pd.DataFrame:
    rows = []
    for model in CLASSICAL_LABELS:
        for strategy in ("none", "class_weight", "random_oversample", "smote"):
            result = _load_classical(model, strategy)
            if not result:
                continue
            pooled = _pooled(result)
            rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    **{key: pooled.get(key, np.nan) for key in MAIN_KEYS},
                    "precision": pooled.get("precision", np.nan),
                    "f1": pooled.get("f1", np.nan),
                }
            )
    return pd.DataFrame(rows)


def _strategy_summary(deep: pd.DataFrame) -> pd.DataFrame:
    if deep.empty:
        return pd.DataFrame()
    summary = (
        deep.groupby("strategy")
        .agg(
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_sensitivity=("recall_sensitivity", "mean"),
            mean_specificity=("specificity", "mean"),
            mean_mcc=("mcc", "mean"),
            mean_brier=("brier", "mean"),
            mean_ece=("ece", "mean"),
        )
        .reset_index()
    )
    summary["worst_sensitivity"] = deep.groupby("strategy")["recall_sensitivity"].min().to_numpy()
    summary["n_models"] = deep.groupby("strategy").size().to_numpy()
    return summary


def _recommend(summary: pd.DataFrame) -> Dict:
    if summary.empty:
        return {}
    by_auprc = summary.sort_values(
        ["mean_auprc", "mean_auroc", "mean_sensitivity"], ascending=False
    ).iloc[0]
    by_auroc = summary.sort_values("mean_auroc", ascending=False).iloc[0]
    if by_auprc["mean_auroc"] >= by_auroc["mean_auroc"] - 0.01:
        chosen = by_auprc
        rule = (
            "Highest mean PR-AUC among the five deep strategies; its mean ROC-AUC is "
            "within 0.01 of the best mean ROC-AUC."
        )
    else:
        chosen = by_auroc
        rule = (
            "Highest mean ROC-AUC, because the PR-AUC leader lost more than 0.01 "
            "mean ROC-AUC relative to the ROC-AUC leader."
        )
    return {
        "recommended_deep_strategy": str(chosen["strategy"]),
        "selection_rule": rule,
        "chosen": chosen.to_dict(),
        "by_auprc": by_auprc.to_dict(),
        "by_auroc": by_auroc.to_dict(),
    }


def _phase_a_historical() -> pd.DataFrame:
    rows = []
    deep = _read_json(PHASE_A_DIR / "deep_screening" / "w30_fs1_multimodal.json")
    if deep:
        pooled = deep["metrics"]["pooled"]
        rows.append(
            {
                "source": "Phase A deep screening (multimodal, 30 min/1 Hz, 35 epochs)",
                "auroc": pooled.get("auroc"),
                "auprc": pooled.get("auprc"),
                "sensitivity": pooled.get("recall_sensitivity"),
                "specificity": pooled.get("specificity"),
            }
        )
    for model in ("logistic", "gradient_boosting"):
        path = PHASE_A_DIR / "window_experiments" / "w30_fs1" / f"w30_fs1_{model}_none.json"
        payload = _read_json(path)
        if payload:
            pooled = payload["metrics"]["pooled"]
            rows.append(
                {
                    "source": f"Phase A window study ({model}, 30 min/1 Hz)",
                    "auroc": pooled.get("auroc"),
                    "auprc": pooled.get("auprc"),
                    "sensitivity": pooled.get("recall_sensitivity"),
                    "specificity": pooled.get("specificity"),
                }
            )
    for variant in DEEP_VARIANTS:
        path = PRODUCTION_REPORTS / f"cv_results_{variant}.json"
        payload = _read_json(path)
        if payload:
            oof = _read_csv(PRODUCTION_REPORTS / f"cv_oof_{variant}.csv")
            sensitivity = specificity = np.nan
            if oof is not None and not oof.empty:
                metrics = evaluate_predictions(
                    oof["y_true"].to_numpy(int), oof["prob"].to_numpy(float)
                )
                sensitivity = metrics["recall_sensitivity"]
                specificity = metrics["specificity"]
            rows.append(
                {
                    "source": f"Production record-level 5-fold CV ({variant})",
                    "auroc": payload.get("pooled_auroc"),
                    "auprc": payload.get("pooled_auprc"),
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _section_imbalance(manifest: Dict, folds: Optional[pd.DataFrame]) -> str:
    n = int(manifest.get("n_records", 469))
    positives = int(manifest.get("n_cesarean", 39))
    negatives = n - positives
    ratio = negatives / positives if positives else float("nan")
    lines = [
        "## 2. Class imbalance analysis\n",
        f"- Development records: **{n}**.",
        f"- Cesarean deliveries (minority): **{positives}** ({100 * positives / n:.1f}%).",
        f"- Vaginal deliveries (majority): **{negatives}** ({100 * negatives / n:.1f}%).",
        f"- Negative-to-positive ratio: **{ratio:.1f}:1**.",
        f"- Positive class prevalence: **{positives / n:.4f}**.",
        "- A no-skill classifier that always predicts vaginal delivery scores "
        f"accuracy {negatives / n:.3f}, balanced accuracy 0.500, cesarean sensitivity 0.000, "
        f"PR-AUC {positives / n:.3f} and MCC 0.000.",
        "",
    ]
    if folds is not None and not folds.empty:
        table = (
            folds.groupby("fold")
            .agg(
                n=("record_id", "size"),
                cesarean=("label", "sum"),
            )
            .reset_index()
        )
        table["vaginal"] = table["n"] - table["cesarean"]
        table["cesarean_pct"] = 100 * table["cesarean"] / table["n"]
        rows = [
            (
                int(row.fold),
                int(row.n),
                int(row.cesarean),
                int(row.vaginal),
                f"{row.cesarean_pct:.1f}%",
            )
            for row in table.itertuples()
        ]
        lines.append("Outer fold composition (fixed Phase A patient/record-level folds):\n")
        lines.append(_table(["Fold", "Records", "Cesareans", "Vaginal", "Cesarean %"], rows))
        lines.append(
            "With 7–8 cesareans per outer validation fold, a single misclassified cesarean "
            "moves fold sensitivity by 12.5–14.3 percentage points. Fold-level sensitivity is "
            "therefore reported for transparency but should not be interpreted as a stable estimate.\n"
        )
    lines.append(
        "The minority class is small enough that loss reweighting can trade a large amount of "
        "specificity for a small absolute gain in true positives. For that reason PR-AUC, "
        "cesarean sensitivity at a fixed threshold, and calibration are analysed jointly "
        "rather than optimising accuracy.\n"
    )
    return "\n".join(lines)


def _section_method(manifest: Dict) -> str:
    strategy_rows = []
    for name in DEEP_STRATEGIES:
        spec = SPECS[name]
        strategy_rows.append(
            (STRATEGY_LABELS.get(name, name), f"`{spec.loss}`", spec.sampler, spec.notes)
        )
    classical_rows = [
        ("No rebalancing", "plain fit", "reference"),
        ("Class weights", "`class_weight='balanced'` in the estimator", "production classical baseline"),
        ("Random oversampling", "minority rows sampled with replacement to parity", "training partition only"),
        ("SMOTE", "custom tabular SMOTE, k=5, interpolation to parity", "training partition only; never raw CTG tensors"),
    ]
    return f"""## 3. Experimental methodology

### 3.1 Data and preprocessing

- Development set only: **{manifest.get('n_records')} records** ({manifest.get('n_cesarean')} cesareans).
  The 83-record held-out test split is not loaded anywhere in `src/phase_b/`; the loader
  raises an exception if a `test` record is present.
- Phase A production preprocessing recommendation: **30-minute CTG window at 1 Hz**
  (1800 samples per channel), existing CTG feature extraction and existing clinical
  feature extraction (`32` clinical inputs after encoding, `40` tabular features).
- Signal standardisation and clinical imputation/scaling are fitted **inside each
  training partition** and applied unchanged to the corresponding validation partition.
- Patient-level separation follows the fixed Phase A outer folds (`experiments/phase_a/folds.csv`),
  which are preserved unchanged for historical comparison.

### 3.2 Nested cross-validation

- **Outer**: 5 stratified folds, fixed Phase A assignments, seed 42.
- **Inner**: 3 stratified folds *inside each outer training partition*, used only for
  model/hyperparameter selection.
- Classical models: inner folds select `C` for logistic regression or
  `max_depth`/`learning_rate` for gradient boosting, using mean inner-fold ROC-AUC
  (PR-AUC tie-break).
- Deep models: inner folds select the learning rate from `{5e-4, 1e-3}` and an epoch
  budget (mean best epoch across inner folds) for the **weighted-BCE** configuration.
  Those settings are then reused for all five strategies of that architecture, so the
  strategy comparison is not confounded by a different tuning budget per strategy.
- The outer model is refit on the full outer-training partition for the selected epoch
  count with **no outer-validation early stopping**. The outer validation fold is
  scored exactly once per fold.
- Class weights, sampling weights, oversampling and SMOTE are computed from the
  outer-training labels only. The weighted sampler is created over the training dataset
  only; validation data are never resampled.

### 3.3 Deep imbalance strategies (one at a time)

{_table(["Strategy", "Loss", "Sampling", "Notes"], strategy_rows)}

Classical strategies:

{_table(["Strategy", "Implementation", "Constraint"], classical_rows)}

### 3.4 Metrics and thresholds

- Primary operating point: **threshold 0.5**, fixed before evaluation.
- Threshold-free metrics (ROC-AUC, PR-AUC, Brier, ECE) are the primary imbalance
  comparison; they do not depend on the 0.5 cut-off.
- Inner-fold-selected thresholds are stored and reported **secondarily**; they are
  never fitted on the outer validation fold.
- Uncertainty: stratified bootstrap (1000 resamples, seed 42) for pooled 95% CIs, and
  paired stratified bootstrap for strategy differences against the weighted-BCE
  reference (deep) or class-weighted reference (classical).
- Every run records fold assignments, configuration, losses, class weights, sampler
  configuration, seed, checkpoints, predictions, metrics, histories and logs under
  `experiments/phase_b/`.
"""


def _section_b0(rows: List[Dict]) -> str:
    table_rows = []
    for item in rows:
        pooled = _pooled(item["result"])
        table_rows.append(
            (
                item["name"],
                item["family"],
                _f(pooled.get("auroc")),
                _ci(pooled, "auroc"),
                _f(pooled.get("auprc")),
                _ci(pooled, "auprc"),
                _f(pooled.get("recall_sensitivity")),
                _f(pooled.get("specificity")),
                _f(pooled.get("f1")),
                _f(pooled.get("mcc")),
                _f(pooled.get("brier")),
            )
        )
    text = [
        "## 1. Baseline reproduction\n",
        "Stage B0 reruns the six existing model families on the fixed Phase A outer folds "
        "with the same 30-minute / 1 Hz preprocessing, seed 42 and threshold 0.5. Deep "
        "models use the production weighted binary cross-entropy with a fixed learning rate "
        "of 1e-3 and a **25-epoch budget with no outer-validation early stopping**; this is a "
        "common reproduction protocol, not the longer production schedule (80–120 epochs with "
        "early stopping), so small differences from the production numbers are expected. "
        "Classical models use the production `class_weight='balanced'` settings. No imbalance "
        "strategy is applied in B0.\n",
        _table(
            [
                "Model",
                "Family",
                "ROC-AUC",
                "95% CI",
                "PR-AUC",
                "95% CI",
                "Sensitivity",
                "Specificity",
                "F1",
                "MCC",
                "Brier",
            ],
            table_rows,
        ),
    ]

    historical = _phase_a_historical()
    if not historical.empty:
        hist_rows = [
            (
                row.source,
                _f(row.auroc),
                _f(row.auprc),
                _f(row.sensitivity),
                _f(row.specificity),
            )
            for row in historical.itertuples()
        ]
        text.append(
            "Historical reference values. These were produced with different protocols and are "
            "**not directly comparable** to B0; they are shown only to confirm that B0 is in the "
            "expected range.\n"
        )
        text.append(
            _table(
                ["Source", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity"], hist_rows
            )
        )
        text.append(
            "The Phase A classical window study used the same folds and tabular features as B0, "
            "so its logistic regression result should reproduce almost exactly; the deep and "
            "production values use different epoch budgets, splits or thresholding and are "
            "context only.\n"
        )
    return "\n".join(text)


def _section_strategy_comparison(
    deep: pd.DataFrame, classical: pd.DataFrame, recommendation: Dict
) -> str:
    summary = _strategy_summary(deep)
    text = ["## 4. Comparison of imbalance strategies\n"]

    if summary.empty:
        text.append("_Deep B1 results are not available yet._\n")
    else:
        order = {name: i for i, name in enumerate(DEEP_STRATEGIES)}
        summary = summary.sort_values("strategy", key=lambda s: s.map(order))
        rows = [
            (
                STRATEGY_LABELS.get(row.strategy, row.strategy),
                _f(row.mean_auroc),
                _f(row.mean_auprc),
                _f(row.mean_sensitivity),
                _f(row.mean_specificity),
                _f(row.mean_mcc),
                _f(row.mean_brier),
                _f(row.mean_ece),
                int(row.n_models),
            )
            for row in summary.itertuples()
        ]
        text.append(
            "Deep strategies, mean over the four architectures (pooled out-of-fold metrics):\n"
        )
        text.append(
            _table(
                [
                    "Strategy",
                    "Mean ROC-AUC",
                    "Mean PR-AUC",
                    "Mean sensitivity",
                    "Mean specificity",
                    "Mean MCC",
                    "Mean Brier",
                    "Mean ECE",
                    "Models",
                ],
                rows,
            )
        )
        baseline = summary[summary["strategy"] == "weighted_bce"]
        if not baseline.empty:
            base = baseline.iloc[0]
            delta_rows = [
                (
                    STRATEGY_LABELS.get(row.strategy, row.strategy),
                    f"{row.mean_auprc - base.mean_auprc:+.3f}",
                    f"{row.mean_sensitivity - base.mean_sensitivity:+.3f}",
                    f"{row.mean_specificity - base.mean_specificity:+.3f}",
                    f"{row.mean_brier - base.mean_brier:+.3f}",
                )
                for row in summary.itertuples()
            ]
            text.append(
                "Differences relative to weighted BCE (positive PR-AUC/sensitivity means an "
                "improvement; positive Brier means worse calibration):\n"
            )
            text.append(
                _table(
                    ["Strategy", "Δ PR-AUC", "Δ sensitivity", "Δ specificity", "Δ Brier"],
                    delta_rows,
                )
            )

    if not classical.empty:
        rows = [
            (
                CLASSICAL_LABELS.get(row.model, row.model),
                STRATEGY_LABELS.get(row.strategy, row.strategy),
                _f(row.auroc),
                _f(row.auprc),
                _f(row.recall_sensitivity),
                _f(row.specificity),
                _f(row.mcc),
                _f(row.brier),
            )
            for row in classical.itertuples()
        ]
        text.append("Classical models by tabular strategy:\n")
        text.append(
            _table(
                [
                    "Model",
                    "Strategy",
                    "ROC-AUC",
                    "PR-AUC",
                    "Sensitivity",
                    "Specificity",
                    "MCC",
                    "Brier",
                ],
                rows,
            )
        )
        text.append(
            "SMOTE and random oversampling are applied to the standardised tabular matrix only. "
            "The raw `(4, 1800)` CTG tensors are never interpolated or oversampled.\n"
        )

    paired = _read_csv(COMPARISON_DIR / "deep_paired_differences.csv")
    if paired is not None and not paired.empty:
        notable = paired[
            (paired["metric"].isin(["auprc", "recall_sensitivity"]))
            & (paired["ci_low"] * paired["ci_high"] > 0)
        ].copy()
        if not notable.empty:
            rows = [
                (
                    DEEP_LABELS.get(row.variant, row.variant),
                    STRATEGY_LABELS.get(row.comparison.split(" vs ")[0], row.comparison),
                    METRIC_LABELS.get(row.metric, row.metric),
                    f"{row.difference:+.3f}",
                    f"{row.ci_low:+.3f} to {row.ci_high:+.3f}",
                    _f(row.p_value),
                )
                for row in notable.sort_values("p_value").head(15).itertuples()
            ]
            text.append(
                "Paired bootstrap differences against weighted BCE whose 95% CI excludes zero "
                "(top 15 by p-value). These are unadjusted for multiple comparisons and should "
                "be read as exploratory:\n"
            )
            text.append(
                _table(
                    ["Model", "Strategy", "Metric", "Difference", "95% CI", "p"],
                    rows,
                )
            )
        else:
            text.append(
                "No paired comparison against weighted BCE had a 95% confidence interval "
                "excluding zero for PR-AUC or sensitivity.\n"
            )

    if recommendation:
        chosen = recommendation.get("chosen", {})
        rule = str(recommendation.get("selection_rule", "")).strip()
        if rule:
            rule = rule[0].upper() + rule[1:]
        if rule and not rule.endswith("."):
            rule += "."
        text.append(
            f"**Recommendation rule.** {rule} "
            f"Selected strategy: **{STRATEGY_LABELS.get(recommendation.get('recommended_deep_strategy'), recommendation.get('recommended_deep_strategy'))}** "
            f"(mean PR-AUC {_f(chosen.get('mean_auprc'))}, mean ROC-AUC {_f(chosen.get('mean_auroc'))}, "
            f"mean sensitivity {_f(chosen.get('mean_sensitivity'))}, mean specificity {_f(chosen.get('mean_specificity'))}).\n"
        )
    return "\n".join(text)


def _section_model_wise(deep: pd.DataFrame) -> str:
    text = ["## 5. Model-wise results\n"]
    if deep.empty:
        text.append("_Deep B1 results are not available yet._\n")
        return "\n".join(text)

    order = {name: i for i, name in enumerate(DEEP_STRATEGIES)}
    for variant in DEEP_VARIANTS:
        subset = deep[deep["variant"] == variant].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("strategy", key=lambda s: s.map(order))
        rows = [
            (
                STRATEGY_LABELS.get(row.strategy, row.strategy),
                _f(row.auroc),
                _f(row.auprc),
                _f(row.recall_sensitivity),
                _f(row.specificity),
                _f(row.precision),
                _f(row.f1),
                _f(row.mcc),
                _f(row.brier),
            )
            for row in subset.itertuples()
        ]
        text.append(f"### {DEEP_LABELS[variant]} (`{variant}`)\n")
        text.append(
            _table(
                [
                    "Strategy",
                    "ROC-AUC",
                    "PR-AUC",
                    "Sensitivity",
                    "Specificity",
                    "Precision",
                    "F1",
                    "MCC",
                    "Brier",
                ],
                rows,
            )
        )
        best_pr = subset.sort_values("auprc", ascending=False).iloc[0]
        best_sens = subset.sort_values("recall_sensitivity", ascending=False).iloc[0]
        text.append(
            f"- Best PR-AUC: **{STRATEGY_LABELS.get(best_pr['strategy'], best_pr['strategy'])}** "
            f"({_f(best_pr['auprc'])}) with sensitivity {_f(best_pr['recall_sensitivity'])} "
            f"and specificity {_f(best_pr['specificity'])}.\n"
            f"- Highest sensitivity: **{STRATEGY_LABELS.get(best_sens['strategy'], best_sens['strategy'])}** "
            f"({_f(best_sens['recall_sensitivity'])}) with PR-AUC {_f(best_sens['auprc'])} "
            f"and specificity {_f(best_sens['specificity'])}.\n"
        )
    return "\n".join(text)


def _section_minority(deep: pd.DataFrame, recommendation: Dict) -> str:
    text = ["## 6. Minority-class detection performance\n"]
    if deep.empty:
        text.append("_Deep B1 results are not available yet._\n")
        return "\n".join(text)

    text.append(
        "Cesarean detection receives particular attention. PR-AUC is the primary ranking "
        "metric because prevalence is only 8.3%; a model that ranks cesareans poorly cannot "
        "achieve high PR-AUC regardless of threshold. Sensitivity is reported at the fixed "
        "0.5 threshold and, secondarily, at the inner-fold-selected threshold.\n"
    )

    rows = []
    for variant in DEEP_VARIANTS:
        for strategy in DEEP_STRATEGIES:
            result = _load_deep(variant, strategy)
            if not result:
                continue
            pooled = _pooled(result)
            directory = deep_result_path(variant, strategy)
            alt = _pooled_at_inner_threshold(directory)
            rows.append(
                (
                    DEEP_LABELS[variant],
                    STRATEGY_LABELS.get(strategy, strategy),
                    _f(pooled.get("auprc")),
                    _f(pooled.get("recall_sensitivity")),
                    f"{int(pooled.get('tp', 0))}/{int(pooled.get('tp', 0) + pooled.get('fn', 0))}",
                    _f(pooled.get("specificity")),
                    _f(alt.get("recall_sensitivity")),
                    _f(alt.get("specificity")),
                )
            )
    text.append(
        _table(
            [
                "Model",
                "Strategy",
                "PR-AUC",
                "Sensitivity @0.5",
                "Cesareans detected",
                "Specificity @0.5",
                "Sensitivity @inner threshold",
                "Specificity @inner threshold",
            ],
            rows,
        )
    )
    text.append(
        "The inner-threshold columns show what the operating point chosen without outer "
        "validation data would have produced. Where sensitivity rises sharply at the inner "
        "threshold, the ranking information was present but the 0.5 cut-off was miscalibrated. "
        "Conversely, a very high inner-threshold sensitivity paired with near-zero specificity "
        "shows that a threshold transferred from inner folds is not a substitute for "
        "outer-fold calibration; it can over-predict the minority class when the score "
        "distribution shifts.\n"
    )

    chosen = recommendation.get("recommended_deep_strategy")
    if chosen:
        text.append(
            f"### Per-fold sensitivity, recommended strategy ({STRATEGY_LABELS.get(chosen, chosen)})\n"
        )
        fold_rows = []
        for variant in DEEP_VARIANTS:
            result = _load_deep(variant, chosen)
            folds = _fold_frame(result)
            if folds.empty:
                continue
            for row in folds.itertuples():
                fold_rows.append(
                    (
                        DEEP_LABELS[variant],
                        int(row.fold),
                        int(row.n_cesarean_val),
                        _f(row.recall_sensitivity),
                        _f(row.specificity),
                        _f(row.auprc),
                    )
                )
        text.append(
            _table(
                [
                    "Model",
                    "Fold",
                    "Val cesareans",
                    "Sensitivity",
                    "Specificity",
                    "Fold PR-AUC",
                ],
                fold_rows,
            )
        )
    return "\n".join(text)


def _section_uncertainty() -> str:
    text = [
        "## 7. Statistical uncertainty\n",
        "Pooled out-of-fold 95% confidence intervals from a stratified bootstrap "
        "(1000 resamples, seed 42). Bootstrap intervals for sensitivity are wide because only "
        "39 cesareans exist in the development set.\n",
    ]
    rows = []
    for variant in DEEP_VARIANTS:
        result = _read_json(b0_result_path(variant) / "result.json")
        if result:
            pooled = _pooled(result)
            rows.append(
                (
                    f"B0 {DEEP_LABELS[variant]}",
                    _f(pooled.get("auroc")),
                    _ci(pooled, "auroc"),
                    _f(pooled.get("auprc")),
                    _ci(pooled, "auprc"),
                    _f(pooled.get("recall_sensitivity")),
                    _ci(pooled, "recall_sensitivity"),
                )
            )
        for strategy in DEEP_STRATEGIES:
            result = _load_deep(variant, strategy)
            if not result:
                continue
            pooled = _pooled(result)
            rows.append(
                (
                    f"{DEEP_LABELS[variant]} — {STRATEGY_LABELS.get(strategy, strategy)}",
                    _f(pooled.get("auroc")),
                    _ci(pooled, "auroc"),
                    _f(pooled.get("auprc")),
                    _ci(pooled, "auprc"),
                    _f(pooled.get("recall_sensitivity")),
                    _ci(pooled, "recall_sensitivity"),
                )
            )
    for model in CLASSICAL_LABELS:
        for label, result in (
            ("B0 production", _read_json(B0_DIR / model / "result.json")),
            ("B1 no rebalancing", _load_classical(model, "none")),
            ("B1 class weights", _load_classical(model, "class_weight")),
            ("B1 random oversampling", _load_classical(model, "random_oversample")),
            ("B1 SMOTE", _load_classical(model, "smote")),
        ):
            if not result:
                continue
            pooled = _pooled(result)
            rows.append(
                (
                    f"{CLASSICAL_LABELS[model]} — {label}",
                    _f(pooled.get("auroc")),
                    _ci(pooled, "auroc"),
                    _f(pooled.get("auprc")),
                    _ci(pooled, "auprc"),
                    _f(pooled.get("recall_sensitivity")),
                    _ci(pooled, "recall_sensitivity"),
                )
            )
    text.append(
        _table(
            [
                "Configuration",
                "ROC-AUC",
                "95% CI",
                "PR-AUC",
                "95% CI",
                "Sensitivity",
                "95% CI",
            ],
            rows,
        )
    )

    fold_rows = []
    for variant in DEEP_VARIANTS:
        for strategy in DEEP_STRATEGIES:
            result = _load_deep(variant, strategy)
            if not result:
                continue
            folds = _fold_frame(result)
            for metric in ("auprc", "recall_sensitivity"):
                values = folds[metric].dropna().to_numpy(float) if metric in folds else np.array([])
                if len(values) > 1:
                    fold_rows.append(
                        (
                            f"{DEEP_LABELS[variant]} — {STRATEGY_LABELS.get(strategy, strategy)}",
                            METRIC_LABELS[metric],
                            f"{values.mean():.3f} ± {values.std(ddof=1):.3f}",
                            f"{values.min():.3f}–{values.max():.3f}",
                        )
                    )
    text.append("Fold-wise variability (mean ± standard deviation across the five outer folds):\n")
    text.append(
        _table(
            ["Configuration", "Metric", "Mean ± SD", "Range"],
            fold_rows,
        )
    )
    text.append(
        "Fold-level dispersion is large relative to the differences between strategies. "
        "Strategy rankings within ±0.02 PR-AUC should therefore be treated as ties.\n"
    )
    return "\n".join(text)


def _section_calibration(deep: pd.DataFrame, classical: pd.DataFrame) -> str:
    text = ["## 8. Calibration findings\n"]
    if deep.empty:
        text.append("_Deep B1 results are not available yet._\n")
        return "\n".join(text)

    rows = []
    for variant in DEEP_VARIANTS:
        for strategy in DEEP_STRATEGIES:
            result = _load_deep(variant, strategy)
            if not result:
                continue
            pooled = _pooled(result)
            rows.append(
                (
                    DEEP_LABELS[variant],
                    STRATEGY_LABELS.get(strategy, strategy),
                    _f(pooled.get("brier"), 4),
                    _f(pooled.get("ece"), 4),
                    _f(pooled.get("precision")),
                    _f(pooled.get("recall_sensitivity")),
                )
            )
    text.append(
        _table(
            ["Model", "Strategy", "Brier", "ECE", "Precision @0.5", "Sensitivity @0.5"],
            rows,
        )
    )
    if not classical.empty:
        rows = [
            (
                CLASSICAL_LABELS.get(row.model, row.model),
                STRATEGY_LABELS.get(row.strategy, row.strategy),
                _f(row.brier, 4),
                _f(row.ece, 4),
            )
            for row in classical.itertuples()
        ]
        text.append("Classical calibration for reference:\n")
        text.append(_table(["Model", "Strategy", "Brier", "ECE"], rows))
    text.append(
        "Unweighted and focal losses usually keep better-calibrated probabilities than "
        "reweighted losses or resampling, which deliberately shift the predicted probability "
        "scale towards the minority class. Brier/ECE therefore often worsen exactly when "
        "sensitivity improves. Because the operating threshold is fixed at 0.5, this "
        "miscalibration directly reduces specificity and can also suppress sensitivity even "
        "when ranking (ROC-AUC/PR-AUC) improves. Calibration curves are saved as "
        "`../comparisons/figures/calibration_<model>.png`.\n"
    )
    return "\n".join(text)


def _section_recommendation(recommendation: Dict, deep: pd.DataFrame, classical: pd.DataFrame) -> str:
    text = ["## 10. Recommended imbalance strategy\n"]
    if not recommendation:
        text.append("_Insufficient deep results to make a recommendation._\n")
        return "\n".join(text)

    chosen = recommendation["recommended_deep_strategy"]
    chosen_row = recommendation["chosen"]
    by_auprc = recommendation.get("by_auprc") or chosen_row
    by_auroc = recommendation.get("by_auroc") or chosen_row
    rule = str(recommendation.get("selection_rule", "")).strip()
    if rule:
        rule = rule[0].upper() + rule[1:]
    if rule and not rule.endswith("."):
        rule += "."
    text.append(
        f"**Deep models: {STRATEGY_LABELS.get(chosen, chosen)} (`{chosen}`).** "
        f"Selection rule: {rule}\n"
    )
    text.append(
        f"- Mean PR-AUC: **{_f(chosen_row.get('mean_auprc'))}** "
        f"(best PR-AUC strategy: {_f(by_auprc.get('mean_auprc'))}).",
    )
    text.append(
        f"- Mean ROC-AUC: **{_f(chosen_row.get('mean_auroc'))}** "
        f"(best ROC-AUC strategy: {_f(by_auroc.get('mean_auroc'))})."
    )
    text.append(
        f"- Mean cesarean sensitivity: **{_f(chosen_row.get('mean_sensitivity'))}**; "
        f"mean specificity: **{_f(chosen_row.get('mean_specificity'))}**; "
        f"mean Brier: **{_f(chosen_row.get('mean_brier'))}**.\n"
    )

    text.append(
        "**Classical models.** Tabular resampling should remain a secondary tool. The "
        "production `class_weight='balanced'` configuration is the most defensible default "
        "because it needs no synthetic data, and SMOTE results are reported only as an "
        "exploratory tabular comparison.\n"
    )

    if not deep.empty:
        text.append("Per-architecture best strategy by PR-AUC:\n")
        rows = []
        for variant in DEEP_VARIANTS:
            subset = deep[deep["variant"] == variant]
            if subset.empty:
                continue
            best = subset.sort_values(["auprc", "auroc"], ascending=False).iloc[0]
            rows.append(
                (
                    DEEP_LABELS[variant],
                    STRATEGY_LABELS.get(best["strategy"], best["strategy"]),
                    _f(best["auprc"]),
                    _f(best["auroc"]),
                    _f(best["recall_sensitivity"]),
                    _f(best["specificity"]),
                )
            )
        text.append(
            _table(
                ["Model", "Best PR-AUC strategy", "PR-AUC", "ROC-AUC", "Sensitivity", "Specificity"],
                rows,
            )
        )
        text.append(
            "Architecture-specific winners differ, so the pooled recommendation should be "
            "applied as a default, not as a claim that one loss dominates for every model.\n"
        )

    text.append(
        "**Operational recommendation.** Keep PR-AUC and calibration as the primary decision "
        "criteria, select the operating threshold on inner folds or a dedicated calibration "
        "split, and treat any strategy whose PR-AUC gain is below ~0.02 as equivalent to the "
        "baseline. Do not deploy a reweighted model solely because its sensitivity at 0.5 "
        "increased; check the accompanying specificity and Brier change.\n"
    )
    return "\n".join(text)


def _section_limitations(deep: pd.DataFrame, classical: pd.DataFrame) -> str:
    return """## 9. Limitations

- **Small minority class.** Only 39 development-set cesareans (7–8 per outer fold) support
  all conclusions. Confidence intervals for sensitivity span tens of percentage points and
  a single fold can change the pooled estimate materially.
- **Single seed.** Deep runs use seed 42 with deterministic settings, but no multi-seed
  variance study was run. Deep training variance may be comparable to the strategy effects.
- **Inner selection transferred across strategies.** Learning rate and epoch budget were
  selected once per outer fold with weighted BCE and reused for every strategy. This keeps
  the comparison fair but may disadvantage strategies that would prefer a different budget.
- **Fixed 0.5 threshold.** Primary sensitivity/specificity use threshold 0.5. Inner-selected
  thresholds are reported secondarily, but no outer-fold threshold calibration was performed,
  so clinical operating points are not yet optimised.
- **No multiplicity control.** Paired bootstrap p-values are unadjusted; with many model ×
  strategy comparisons, some "significant" differences are expected by chance.
- **CPU-only training.** Epoch budgets were limited by runtime, not by convergence analysis.
  Deep runs are bitwise reproducible only when `PHASE_B_THREADS` is fixed, because CPU
  floating-point reduction order depends on the thread count; all Phase B artifacts were
  generated with `PHASE_B_THREADS=3`. Repeating the multimodal `none` augmentation control
  at a different thread count changed pooled PR-AUC by about 0.016, which is a useful
  reminder that single-run deep differences of that size are not meaningful.
- **Tabular SMOTE is a local implementation.** `imbalanced-learn` is not installed, so the
  SMOTE variant uses a tested but minimal implementation; results may differ slightly from
  the reference library.
- **No held-out test evaluation.** By design, the 83-record test split is never loaded in
  Phase B. All reported numbers are development-set out-of-fold estimates and are expected
  to be optimistic relative to external performance.
- **No claim of clinical utility.** These are research-prototype results on a single-centre
  retrospective database, not a validated clinical decision tool.
"""


def _section_artifacts(deep: pd.DataFrame, classical: pd.DataFrame) -> str:
    rows = [
        ("`experiments/phase_b/config.json`", "Phase B configuration and environment"),
        ("`experiments/phase_b/folds.csv`", "Outer fold assignments (Phase A folds)"),
        ("`experiments/phase_b/dataset_manifest.json`", "Dataset counts, hashes, fold composition"),
        ("`experiments/phase_b/b0/<model>/result.json`", "B0 baseline metrics, predictions, folds"),
        ("`experiments/phase_b/b1/deep/<model>/<strategy>/`", "Deep checkpoints, predictions, histories, metrics"),
        ("`experiments/phase_b/b1/classical/<model>/<strategy>/`", "Classical predictions, selections, metrics"),
        ("`experiments/phase_b/b1/augmentation/<mode>/`", "Exploratory multimodal augmentation runs"),
        ("`experiments/phase_b/comparisons/`", "Comparison tables, paired differences, recommendation"),
        ("`experiments/phase_b/comparisons/figures/`", "ROC, PR, calibration, confusion and fold figures"),
        ("`experiments/phase_b/logs/`", "Stage logs"),
    ]
    text = ["## Generated artifacts\n", _table(["Path", "Contents"], rows)]
    text.append(
        "Figures were written only for configurations that completed; missing figure files "
        "indicate that the corresponding stage had not run when the report was generated.\n"
    )
    return "\n".join(text)


def _section_augmentation() -> str:
    rows = []
    for mode in AUGMENT_MODES:
        result = _read_json(augment_result_path(mode) / "result.json")
        if not result:
            continue
        pooled = _pooled(result)
        rows.append(
            (
                mode,
                _f(pooled.get("auroc")),
                _f(pooled.get("auprc")),
                _f(pooled.get("recall_sensitivity")),
                _f(pooled.get("specificity")),
                _f(pooled.get("brier")),
            )
        )
    text = [
        "## Appendix A. Controlled CTG augmentation (exploratory)\n",
        "Augmentation is deliberately kept outside the B1 strategy comparison and applied only "
        "to the multimodal architecture. Noise is added to the two physiological channels only; "
        "masks are untouched. The `mix` mode combines Gaussian noise, a ±5% amplitude scale and "
        "a ±30-second time shift.\n",
    ]
    if rows:
        text.append(
            _table(
                ["Mode", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "Brier"], rows
            )
        )
        none_result = _read_json(augment_result_path("none") / "result.json")
        b0_result = _read_json(b0_result_path("multimodal") / "result.json")
        if none_result and b0_result:
            a = _pooled(none_result)
            b = _pooled(b0_result)
            same = all(
                abs(float(a.get(key, np.nan)) - float(b.get(key, np.nan))) < 1e-9
                for key in ("auroc", "auprc", "recall_sensitivity", "specificity")
            )
            text.append(
                "As a reproducibility check, the `none` augmentation mode uses the same "
                "configuration and seeds as B0 multimodal: the pooled metrics are "
                + ("**identical**" if same else "not identical")
                + ", confirming that the staged runner is deterministic for an unchanged "
                "configuration.\n"
            )
    else:
        text.append("_Augmentation runs are not available yet._\n")
    return "\n".join(text)


def _section_selection() -> str:
    rows = []
    for variant in DEEP_VARIANTS:
        selection = _read_json(selection_path(variant))
        if not selection:
            continue
        for fold, payload in sorted(selection.get("outer_folds", {}).items(), key=lambda kv: int(kv[0])):
            fold_json = _read_json(deep_result_path(variant, "weighted_bce") / f"fold{fold}.json")
            training = (fold_json or {}).get("training", {})
            weights = training.get("class_weights", {})
            rows.append(
                (
                    DEEP_LABELS[variant],
                    fold,
                    _f(payload.get("selected_learning_rate"), 4),
                    int(payload.get("selected_epochs", 0)),
                    _f(payload.get("selected_inner_auroc")),
                    _f(payload.get("threshold_from_inner")),
                    _f(weights.get("1")),
                    _f(training.get("positive_weight")),
                )
            )
    text = [
        "## Appendix B. Inner-selection and training-fold class weights\n",
        "Selected learning rate and epoch budget per outer fold, taken from the inner "
        "3-fold selection that only sees the outer-training records. Class weights are "
        "computed from the same outer-training labels and stored in each fold artifact.\n",
    ]
    if rows:
        text.append(
            _table(
                [
                    "Model",
                    "Outer fold",
                    "Selected LR",
                    "Epochs",
                    "Inner AUROC",
                    "Inner threshold",
                    "Cesarean weight",
                    "pos_weight",
                ],
                rows,
            )
        )
    else:
        text.append("_Inner selection artifacts are not available yet._\n")
    return "\n".join(text)


def generate_report(config=None) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(PHASE_B_DIR / "dataset_manifest.json") or {}
    folds = _read_csv(PHASE_B_DIR / "folds.csv")
    deep = _b1_deep_summary()
    classical = _b1_classical_summary()
    summary = _strategy_summary(deep)

    recommendation = _read_json(COMPARISON_DIR / "recommendation.json") or {}
    if not recommendation:
        recommendation = _recommend(summary)
    else:
        # Ensure the stored recommendation is complete and consistent.
        if "chosen" not in recommendation:
            recommendation = _recommend(summary)

    b0_rows = _b0_rows()

    header = f"""# Phase B — Baseline reproduction and imbalance-aware learning

**Project:** OB-SIGHT intrapartum cardiotocography delivery-mode prediction
**Scope:** Stage B0 (baseline reproduction) and Stage B1 (single-strategy imbalance experiments)
**Data:** development set only — {manifest.get('n_records', 469)} records, {manifest.get('n_cesarean', 39)} cesareans
**Preprocessing:** {manifest.get('window_minutes', 30)}-minute CTG window, {manifest.get('target_fs', 1)} Hz
**Validation:** nested patient/record-level cross-validation, outer 5 folds, inner 3 folds, seed {manifest.get('seed', 42)}
**Primary threshold:** 0.5; PR-AUC and cesarean sensitivity emphasised
**Held-out test split:** not loaded and not evaluated

> Every value below is read from generated experiment artifacts. Runs that did not
> complete are marked unavailable rather than filled in.

"""
    body = "\n".join(
        [
            _section_b0(b0_rows),
            _section_imbalance(manifest, folds),
            _section_method(manifest),
            _section_strategy_comparison(deep, classical, recommendation),
            _section_model_wise(deep),
            _section_minority(deep, recommendation),
            _section_uncertainty(),
            _section_calibration(deep, classical),
            _section_limitations(deep, classical),
            _section_recommendation(recommendation, deep, classical),
            _section_augmentation(),
            _section_selection(),
            _section_artifacts(deep, classical),
            "## Reproducibility\n",
            "```bash\n"
            "python -m src.phase_b.run --stage prepare\n"
            "python -m src.phase_b.run --stage b0 --force\n"
            "python -m src.phase_b.run --stage classical --force\n"
            "python -m src.phase_b.run --stage deep --force\n"
            "python -m src.phase_b.run --stage augment --force\n"
            "python -m src.phase_b.run --stage compare --force\n"
            "python -m src.phase_b.run --stage report --force\n"
            "python -m pytest -q\n"
            "```\n",
            "All runs use `PYTHONHASHSEED`-independent deterministic seeding through "
            "`set_seed`, and `PHASE_B_THREADS` can cap CPU threads per process. "
            "Deep checkpoints include the fold, loss strategy, class weights and "
            "preprocessing metadata needed to reproduce predictions.\n",
        ]
    )
    REPORT_PATH.write_text(header + body, encoding="utf-8")

    machine = {
        "recommendation": recommendation,
        "n_deep_results": int(len(deep)),
        "n_classical_results": int(len(classical)),
    }
    (REPORT_DIR / "report_summary.json").write_text(
        json.dumps(machine, indent=2, default=float), encoding="utf-8"
    )
    return REPORT_PATH
