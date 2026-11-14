"""Configuration for the Phase A data-handling experiments.

All thresholds are explicit, serialisable and can be overridden from the
command line.  No quality threshold is used to delete records silently.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import PROJECT_ROOT, RANDOM_SEED

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "phase_a"
DATA_QUALITY_DIR = EXPERIMENTS_DIR / "data_quality"
WINDOW_DIR = EXPERIMENTS_DIR / "window_experiments"
SAMPLING_DIR = EXPERIMENTS_DIR / "sampling_experiments"
FEATURE_DIR = EXPERIMENTS_DIR / "feature_selection"
COMPARISON_DIR = EXPERIMENTS_DIR / "comparisons"
REPORT_DIR = EXPERIMENTS_DIR / "reports"
LOG_DIR = EXPERIMENTS_DIR / "logs"
CACHE_DIR = EXPERIMENTS_DIR / "cache"

WINDOW_DURATIONS_MINUTES: Tuple[int, ...] = (30, 45, 60)
# Explicit alias matching the Phase A specification; the production constant in
# ``src.config.WINDOW_MIN`` remains 30 so deployed behaviour is unchanged.
WINDOW_DURATION_MINUTES = list(WINDOW_DURATIONS_MINUTES)
SAMPLING_FREQUENCIES_HZ: Tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)

# The current production pipeline, kept as a reference point.
CURRENT_WINDOW_MINUTES = 30
CURRENT_FS = 1.0
CURRENT_NATIVE_FS = 4.0

# Fixed, record-level development set.  The production held-out test records
# are excluded from all Phase A cross-validation.
DEV_SPLITS = ("train", "val")
N_SPLITS = 5

# Feature-selection settings.
PROTECTED_FEATURES: Tuple[str, ...] = (
    "age",
    "gest_weeks",
    "parity",
    "meconium",
    "presentation_2",  # breech
    "presentation_3",  # other / abnormal
)
MI_K = 15
RFE_K = 15
RF_K = 15
L1_C = 0.10
CORRELATION_THRESHOLD = 0.90

# Deep-learning screening configuration.  It is intentionally shorter than
# the production final-training run; all compared configurations use exactly
# the same budget and the same folds.
DL_EPOCHS = 35
DL_PATIENCE = 8
DL_BATCH_SIZE = 32
DL_LEARNING_RATE = 1e-3
DL_WEIGHT_DECAY = 1e-4

# Eligibility thresholds for the quality categories.  These are descriptive:
# a "poor" label does not remove a record from the main experiments.
@dataclass(frozen=True)
class QualityThresholds:
    fhr_missing_good: float = 0.10
    fhr_missing_acceptable: float = 0.30
    uc_missing_good: float = 0.10
    uc_missing_acceptable: float = 0.35
    longest_gap_good_min: float = 2.0
    longest_gap_acceptable_min: float = 10.0
    implausible_good: float = 0.001
    implausible_acceptable: float = 0.010
    jump_rate_good: float = 0.0030
    jump_rate_acceptable: float = 0.0060
    flatline_good: float = 0.05
    flatline_acceptable: float = 0.20
    min_duration_good_min: float = 60.0
    min_duration_acceptable_min: float = 30.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PhaseAConfig:
    seed: int = RANDOM_SEED
    windows: Tuple[int, ...] = WINDOW_DURATIONS_MINUTES
    sampling_rates: Tuple[float, ...] = SAMPLING_FREQUENCIES_HZ
    n_splits: int = N_SPLITS
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    dl_epochs: int = DL_EPOCHS
    dl_patience: int = DL_PATIENCE
    dl_batch_size: int = DL_BATCH_SIZE
    dl_learning_rate: float = DL_LEARNING_RATE
    dl_weight_decay: float = DL_WEIGHT_DECAY
    run_deep_learning: bool = True
    run_test_confirmation: bool = False

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["windows"] = list(self.windows)
        data["sampling_rates"] = list(self.sampling_rates)
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": "A",
            "config": self.to_dict(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_directories() -> None:
    for directory in (
        EXPERIMENTS_DIR,
        DATA_QUALITY_DIR,
        WINDOW_DIR,
        SAMPLING_DIR,
        FEATURE_DIR,
        COMPARISON_DIR,
        REPORT_DIR,
        LOG_DIR,
        CACHE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
