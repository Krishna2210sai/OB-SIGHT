"""Phase B command-line orchestration.

Examples
--------
    python -m src.phase_b.run --stage prepare
    python -m src.phase_b.run --stage b0 --models signal --classical-models logistic
    python -m src.phase_b.run --stage deep --models multimodal
    python -m src.phase_b.run --stage classical --classical-models logistic
    python -m src.phase_b.run --stage augment
    python -m src.phase_b.run --stage compare
    python -m src.phase_b.run --stage report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.phase_b.classical import run_b0_classical, run_classical_strategy
from src.phase_b.config import (
    AUGMENT_MODES,
    B0_DIR,
    B1_DIR,
    BOOTSTRAP_SAMPLES,
    CLASSICAL_DIR,
    CLASSICAL_MODELS,
    CLASSICAL_STRATEGIES,
    COMPARISON_DIR,
    DEEP_DIR,
    DEEP_OUTER_EPOCHS,
    DEEP_STRATEGIES,
    DEEP_VARIANTS,
    LOG_DIR,
    PHASE_B_DIR,
    PhaseBConfig,
    augment_result_path,
    b0_result_path,
    classical_result_path,
    deep_result_path,
    ensure_directories,
)
from src.phase_b.data import load_development_bundle, save_phase_b_folds, write_manifest
from src.phase_b.metrics import paired_bootstrap_difference
from src.phase_b.training import run_augmentation, run_b0_deep, run_deep_strategy
from src.phase_b import plots

LOGGER = logging.getLogger("phase_b")


def _setup_logging(stage: str, force: bool = False, suffix: str = "") -> None:
    ensure_directories()
    for handler in LOGGER.handlers[:]:
        LOGGER.removeHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    name = f"{stage}{('_' + suffix) if suffix else ''}.log"
    file_handler = logging.FileHandler(LOG_DIR / name, mode="w" if force else "a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _result_row(result: Dict, family: str, label: str) -> Dict:
    pooled = result["metrics"]["pooled"]
    row = {
        "family": family,
        "label": label,
        "variant": result.get("variant", result.get("model", "")),
        "strategy": result.get("strategy", ""),
        "n_records": result.get("n_records", np.nan),
        "n_cesarean": result.get("n_cesarean", np.nan),
    }
    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "auroc",
        "auprc",
        "mcc",
        "brier",
        "ece",
    ):
        row[key] = pooled.get(key, np.nan)
    for key in ("auroc", "auprc", "recall_sensitivity", "specificity", "mcc", "brier"):
        row[f"{key}_ci_low"] = pooled.get(f"{key}_ci_low", np.nan)
        row[f"{key}_ci_high"] = pooled.get(f"{key}_ci_high", np.nan)
    return row


def stage_prepare(config: PhaseBConfig, force: bool = False) -> None:
    _setup_logging("prepare", force)
    ensure_directories()
    config.save(PHASE_B_DIR / "config.json")
    bundle = load_development_bundle()
    save_phase_b_folds(bundle, PHASE_B_DIR / "folds.csv")
    manifest = write_manifest(bundle, PHASE_B_DIR / "dataset_manifest.json")
    LOGGER.info(
        "Development bundle: %d records, %d cesareans, folds=%s",
        manifest["n_records"],
        manifest["n_cesarean"],
        manifest["outer_fold_cesareans"],
    )


def stage_b0(
    config: PhaseBConfig,
    models: Sequence[str],
    classical_models: Sequence[str],
    force: bool = False,
    epochs_outer: int = DEEP_OUTER_EPOCHS,
) -> None:
    _setup_logging("b0", force, suffix="-".join(models))
    bundle = load_development_bundle()
    for variant in models:
        started = time.time()
        result = run_b0_deep(
            bundle, variant, seed=config.seed, outer_epochs=epochs_outer, force=force
        )
        LOGGER.info(
            "B0 %s AUROC=%.3f AUPRC=%.3f sensitivity=%.3f (%.0fs)",
            variant,
            result["metrics"]["pooled"]["auroc"],
            result["metrics"]["pooled"]["auprc"],
            result["metrics"]["pooled"]["recall_sensitivity"],
            time.time() - started,
        )
    for model in classical_models:
        result = run_b0_classical(bundle, model, seed=config.seed, force=force)
        LOGGER.info(
            "B0 %s AUROC=%.3f AUPRC=%.3f sensitivity=%.3f",
            model,
            result["metrics"]["pooled"]["auroc"],
            result["metrics"]["pooled"]["auprc"],
            result["metrics"]["pooled"]["recall_sensitivity"],
        )


def stage_deep(
    config: PhaseBConfig,
    models: Sequence[str],
    strategies: Sequence[str],
    force: bool = False,
    epochs_outer: int = DEEP_OUTER_EPOCHS,
) -> None:
    _setup_logging("deep", force, suffix="-".join(models))
    bundle = load_development_bundle()
    for variant in models:
        for strategy in strategies:
            started = time.time()
            result = run_deep_strategy(
                bundle,
                variant,
                strategy,
                seed=config.seed,
                outer_epochs=epochs_outer,
                patience=config.deep_patience,
                force=force,
            )
            LOGGER.info(
                "B1 %s/%s AUROC=%.3f AUPRC=%.3f sensitivity=%.3f (%.0fs)",
                variant,
                strategy,
                result["metrics"]["pooled"]["auroc"],
                result["metrics"]["pooled"]["auprc"],
                result["metrics"]["pooled"]["recall_sensitivity"],
                time.time() - started,
            )


def stage_classical(
    config: PhaseBConfig,
    models: Sequence[str],
    strategies: Sequence[str],
    force: bool = False,
) -> None:
    _setup_logging("classical", force, suffix="-".join(models))
    bundle = load_development_bundle()
    for model in models:
        for strategy in strategies:
            started = time.time()
            result = run_classical_strategy(
                bundle, model, strategy, seed=config.seed, force=force
            )
            LOGGER.info(
                "B1 %s/%s AUROC=%.3f AUPRC=%.3f sensitivity=%.3f (%.0fs)",
                model,
                strategy,
                result["metrics"]["pooled"]["auroc"],
                result["metrics"]["pooled"]["auprc"],
                result["metrics"]["pooled"]["recall_sensitivity"],
                time.time() - started,
            )


def stage_augment(
    config: PhaseBConfig,
    modes: Sequence[str],
    force: bool = False,
    epochs_outer: int = DEEP_OUTER_EPOCHS,
) -> None:
    _setup_logging("augment", force)
    bundle = load_development_bundle()
    for mode in modes:
        started = time.time()
        result = run_augmentation(
            bundle, mode, seed=config.seed, outer_epochs=epochs_outer, force=force
        )
        LOGGER.info(
            "Augmentation %s AUROC=%.3f AUPRC=%.3f sensitivity=%.3f (%.0fs)",
            mode,
            result["metrics"]["pooled"]["auroc"],
            result["metrics"]["pooled"]["auprc"],
            result["metrics"]["pooled"]["recall_sensitivity"],
            time.time() - started,
        )


def _load_deep_results() -> List[Dict]:
    rows = []
    for variant in DEEP_VARIANTS:
        payload = _read_json(b0_result_path(variant) / "result.json")
        if payload:
            rows.append(_result_row(payload, "b0_deep", f"{variant}:production"))
        for strategy in DEEP_STRATEGIES:
            payload = _read_json(deep_result_path(variant, strategy) / "result.json")
            if payload:
                rows.append(_result_row(payload, "b1_deep", f"{variant}:{strategy}"))
    return rows


def _load_classical_results() -> List[Dict]:
    rows = []
    for model in CLASSICAL_MODELS:
        payload = _read_json(B0_DIR / model / "result.json")
        if payload:
            rows.append(_result_row(payload, "b0_classical", f"{model}:production"))
        for strategy in CLASSICAL_STRATEGIES:
            payload = _read_json(classical_result_path(model, strategy) / "result.json")
            if payload:
                rows.append(_result_row(payload, "b1_classical", f"{model}:{strategy}"))
    return rows


def _paired_for_deep(output_dir: Path) -> pd.DataFrame:
    rows = []
    for variant in DEEP_VARIANTS:
        baseline_path = deep_result_path(variant, "weighted_bce") / "predictions.csv"
        if not baseline_path.exists():
            continue
        baseline = pd.read_csv(baseline_path).sort_values("record_id").reset_index(drop=True)
        for strategy in DEEP_STRATEGIES:
            if strategy == "weighted_bce":
                continue
            candidate_path = deep_result_path(variant, strategy) / "predictions.csv"
            if not candidate_path.exists():
                continue
            candidate = pd.read_csv(candidate_path).sort_values("record_id").reset_index(drop=True)
            if not np.array_equal(
                candidate["record_id"].astype(str), baseline["record_id"].astype(str)
            ):
                continue
            diffs = paired_bootstrap_difference(
                baseline["y_true"].to_numpy(int),
                candidate["prob"].to_numpy(float),
                baseline["prob"].to_numpy(float),
                n_boot=BOOTSTRAP_SAMPLES,
            )
            for metric, payload in diffs.items():
                rows.append(
                    {
                        "family": "deep",
                        "variant": variant,
                        "comparison": f"{strategy} vs weighted_bce",
                        "metric": metric,
                        **payload,
                    }
                )
    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(output_dir / "deep_paired_differences.csv", index=False)
    return table


def _paired_for_classical(output_dir: Path) -> pd.DataFrame:
    rows = []
    for model in CLASSICAL_MODELS:
        baseline_path = classical_result_path(model, "class_weight") / "predictions.csv"
        if not baseline_path.exists():
            continue
        baseline = pd.read_csv(baseline_path).sort_values("record_id").reset_index(drop=True)
        for strategy in CLASSICAL_STRATEGIES:
            if strategy == "class_weight":
                continue
            candidate_path = classical_result_path(model, strategy) / "predictions.csv"
            if not candidate_path.exists():
                continue
            candidate = pd.read_csv(candidate_path).sort_values("record_id").reset_index(drop=True)
            if not np.array_equal(
                candidate["record_id"].astype(str), baseline["record_id"].astype(str)
            ):
                continue
            diffs = paired_bootstrap_difference(
                baseline["y_true"].to_numpy(int),
                candidate["prob"].to_numpy(float),
                baseline["prob"].to_numpy(float),
                n_boot=BOOTSTRAP_SAMPLES,
            )
            for metric, payload in diffs.items():
                rows.append(
                    {
                        "family": "classical",
                        "variant": model,
                        "comparison": f"{strategy} vs class_weight",
                        "metric": metric,
                        **payload,
                    }
                )
    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(output_dir / "classical_paired_differences.csv", index=False)
    return table


def stage_compare(config: PhaseBConfig, force: bool = False) -> Dict:
    _setup_logging("compare", force)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    deep_rows = _load_deep_results()
    classical_rows = _load_classical_results()
    deep = pd.DataFrame(deep_rows)
    classical = pd.DataFrame(classical_rows)
    if not deep.empty:
        deep.to_csv(COMPARISON_DIR / "deep_comparison.csv", index=False)
    if not classical.empty:
        classical.to_csv(COMPARISON_DIR / "classical_comparison.csv", index=False)
    combined = pd.concat([deep, classical], ignore_index=True) if (not deep.empty or not classical.empty) else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(COMPARISON_DIR / "all_results.csv", index=False)

    deep_paired = _paired_for_deep(COMPARISON_DIR)
    classical_paired = _paired_for_classical(COMPARISON_DIR)

    # Strategy summary across architectures (B1 only).
    b1 = deep[deep["family"] == "b1_deep"] if not deep.empty else pd.DataFrame()
    recommendation = {}
    if not b1.empty:
        summary = (
            b1.groupby("strategy")
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
            .sort_values(["mean_auprc", "mean_sensitivity"], ascending=False)
        )
        summary.to_csv(COMPARISON_DIR / "deep_strategy_summary.csv", index=False)
        best_auprc = summary.iloc[0]
        best_auroc = summary.sort_values("mean_auroc", ascending=False).iloc[0]
        if best_auprc["mean_auroc"] >= best_auroc["mean_auroc"] - 0.01:
            chosen = best_auprc
            rule = "highest mean PR-AUC with mean ROC-AUC within 0.01 of the best"
        else:
            chosen = best_auroc
            rule = "highest mean ROC-AUC because the PR-AUC leader lost more than 0.01 AUROC"
        recommendation = {
            "recommended_deep_strategy": str(chosen["strategy"]),
            "selection_rule": rule,
            "summary": summary.to_dict("records"),
            "chosen": chosen.to_dict(),
        }
        (COMPARISON_DIR / "recommendation.json").write_text(
            json.dumps(recommendation, indent=2, default=float), encoding="utf-8"
        )

        plots.plot_strategy_comparison(
            b1,
            COMPARISON_DIR / "figures",
            "deep_strategy_auprc.png",
            "auprc",
            "Deep models: pooled PR-AUC by imbalance strategy",
        )
        plots.plot_strategy_comparison(
            b1,
            COMPARISON_DIR / "figures",
            "deep_strategy_sensitivity.png",
            "recall_sensitivity",
            "Deep models: cesarean sensitivity at 0.5 by strategy",
        )
        bar = b1[["label", "auroc", "auprc", "recall_sensitivity", "specificity"]].copy()
        bar["label"] = bar["label"].str.replace(":", " / ", regex=False)
        plots.plot_metric_bars(
            bar,
            COMPARISON_DIR / "figures",
            "deep_metrics_all.png",
            ("auroc", "auprc", "recall_sensitivity", "specificity"),
            "Deep screening: discrimination and operating-point metrics",
        )

        for variant in DEEP_VARIANTS:
            curves = {}
            for strategy in DEEP_STRATEGIES:
                path = deep_result_path(variant, strategy) / "predictions.csv"
                if path.exists():
                    frame = pd.read_csv(path)
                    curves[strategy] = (
                        frame["y_true"].to_numpy(int),
                        frame["prob"].to_numpy(float),
                    )
            if curves:
                plots.plot_roc_pr(
                    curves,
                    COMPARISON_DIR / "figures",
                    f"roc_pr_{variant}.png",
                    f"{variant}: imbalance strategies (development OOF)",
                )
                plots.plot_calibration(
                    curves,
                    COMPARISON_DIR / "figures",
                    f"calibration_{variant}.png",
                    f"{variant}: calibration by strategy",
                )
            chosen_strategy = recommendation.get("recommended_deep_strategy")
            chosen_path = deep_result_path(variant, chosen_strategy or "") / "predictions.csv"
            if chosen_path.exists():
                frame = pd.read_csv(chosen_path)
                from src.phase_b.metrics import evaluate_predictions

                metrics = evaluate_predictions(
                    frame["y_true"].to_numpy(int), frame["prob"].to_numpy(float)
                )
                plots.plot_confusion(
                    metrics["confusion_matrix"],
                    COMPARISON_DIR / "figures",
                    f"confusion_{variant}_{chosen_strategy}.png",
                    f"{variant} — {chosen_strategy} at 0.5",
                )
                fold_table = pd.read_csv(deep_result_path(variant, chosen_strategy) / "fold_metrics.csv")
                plots.plot_fold_metrics(
                    fold_table,
                    COMPARISON_DIR / "figures",
                    f"fold_metrics_{variant}_{chosen_strategy}.png",
                    f"{variant} — {chosen_strategy}",
                )

    if not classical.empty:
        b1c = classical[classical["family"] == "b1_classical"]
        summary_c = (
            b1c.groupby(["variant", "strategy"])
            .agg(
                auroc=("auroc", "mean"),
                auprc=("auprc", "mean"),
                sensitivity=("recall_sensitivity", "mean"),
                specificity=("specificity", "mean"),
                mcc=("mcc", "mean"),
                brier=("brier", "mean"),
            )
            .reset_index()
        )
        summary_c.to_csv(COMPARISON_DIR / "classical_strategy_summary.csv", index=False)
        for model in CLASSICAL_MODELS:
            curves = {}
            for strategy in CLASSICAL_STRATEGIES:
                path = classical_result_path(model, strategy) / "predictions.csv"
                if path.exists():
                    frame = pd.read_csv(path)
                    curves[strategy] = (
                        frame["y_true"].to_numpy(int),
                        frame["prob"].to_numpy(float),
                    )
            if curves:
                plots.plot_roc_pr(
                    curves,
                    COMPARISON_DIR / "figures",
                    f"roc_pr_classical_{model}.png",
                    f"{model}: classical imbalance strategies",
                )
                plots.plot_calibration(
                    curves,
                    COMPARISON_DIR / "figures",
                    f"calibration_classical_{model}.png",
                    f"{model}: calibration by strategy",
                )

    augment_rows = []
    for mode in AUGMENT_MODES:
        payload = _read_json(augment_result_path(mode) / "result.json")
        if payload:
            augment_rows.append(_result_row(payload, "augmentation", f"multimodal:{mode}"))
    if augment_rows:
        pd.DataFrame(augment_rows).to_csv(COMPARISON_DIR / "augmentation_comparison.csv", index=False)
        aug = pd.DataFrame(augment_rows)
        aug["label"] = aug["label"].str.replace(":", " / ", regex=False)
        plots.plot_metric_bars(
            aug,
            COMPARISON_DIR / "figures",
            "augmentation_metrics.png",
            ("auroc", "auprc", "recall_sensitivity", "specificity"),
            "Controlled CTG augmentation (exploratory, multimodal)",
        )

    LOGGER.info("Recommendation: %s", recommendation.get("recommended_deep_strategy", "n/a"))
    return {
        "deep": deep,
        "classical": classical,
        "deep_paired": deep_paired,
        "classical_paired": classical_paired,
        "recommendation": recommendation,
    }


def stage_report(config: PhaseBConfig, force: bool = False) -> Path:
    _setup_logging("report", force)
    from src.phase_b.report import generate_report

    path = generate_report(config)
    LOGGER.info("Report written to %s", path)
    return path


def _parse_configs(values: Optional[Sequence[str]], default: Sequence[str]) -> List[str]:
    return list(values) if values else list(default)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase B imbalance experiments for OB-SIGHT")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["prepare", "b0", "deep", "classical", "augment", "compare", "report", "all"],
    )
    parser.add_argument("--models", nargs="*", default=None, choices=list(DEEP_VARIANTS))
    parser.add_argument("--strategies", nargs="*", default=None, choices=list(DEEP_STRATEGIES))
    parser.add_argument("--classical-models", nargs="*", default=None, choices=list(CLASSICAL_MODELS))
    parser.add_argument(
        "--classical-strategies", nargs="*", default=None, choices=list(CLASSICAL_STRATEGIES)
    )
    parser.add_argument("--augment-modes", nargs="*", default=None, choices=list(AUGMENT_MODES))
    parser.add_argument("--epochs-outer", type=int, default=DEEP_OUTER_EPOCHS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-deep", action="store_true", help="Skip deep models")
    parser.add_argument("--no-classical", action="store_true", help="Skip classical models")
    args = parser.parse_args(argv)

    config = PhaseBConfig()
    if args.quick:
        config.deep_outer_epochs = 3
        config.deep_inner_epochs = 2
        args.epochs_outer = 3
    ensure_directories()
    models = [] if args.no_deep else _parse_configs(args.models, DEEP_VARIANTS)
    strategies = _parse_configs(args.strategies, DEEP_STRATEGIES)
    classical_models = (
        [] if args.no_classical else _parse_configs(args.classical_models, CLASSICAL_MODELS)
    )
    classical_strategies = _parse_configs(args.classical_strategies, CLASSICAL_STRATEGIES)
    augment_modes = _parse_configs(args.augment_modes, AUGMENT_MODES)

    if args.stage in ("prepare", "all"):
        stage_prepare(config, force=args.force)
    if args.stage in ("b0", "all"):
        stage_b0(
            config,
            models,
            classical_models,
            force=args.force,
            epochs_outer=args.epochs_outer,
        )
    if args.stage in ("classical", "all"):
        stage_classical(
            config, classical_models, classical_strategies, force=args.force
        )
    if args.stage in ("deep", "all"):
        stage_deep(
            config,
            models,
            strategies,
            force=args.force,
            epochs_outer=args.epochs_outer,
        )
    if args.stage in ("augment", "all"):
        stage_augment(
            config, augment_modes, force=args.force, epochs_outer=args.epochs_outer
        )
    if args.stage in ("compare", "all"):
        stage_compare(config, force=args.force)
    if args.stage in ("report", "all"):
        stage_report(config, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
