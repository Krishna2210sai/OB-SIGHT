"""Phase B constants, paths and experimental configuration.

Phase B is additive: it reuses the production architectures, the Phase A
30-minute / 1 Hz preprocessing and the fixed Phase A development folds.  The
held-out test split is never loaded.
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import PROJECT_ROOT, RANDOM_SEED

PHASE_B_DIR = PROJECT_ROOT / "experiments" / "phase_b"
B0_DIR = PHASE_B_DIR / "b0"
B1_DIR = PHASE_B_DIR / "b1"
CLASSICAL_DIR = B1_DIR / "classical"
DEEP_DIR = B1_DIR / "deep"
AUGMENT_DIR = B1_DIR / "augmentation"
COMPARISON_DIR = PHASE_B_DIR / "comparisons"
REPORT_DIR = PHASE_B_DIR / "reports"
LOG_DIR = PHASE_B_DIR / "logs"
CACHE_DIR = PHASE_B_DIR / "cache"

PHASE_A_DIR = PROJECT_ROOT / "experiments" / "phase_a"
PHASE_A_FOLDS = PHASE_A_DIR / "folds.csv"
PHASE_A_CACHE = PHASE_A_DIR / "cache" / "w30_fs1"

WINDOW_MIN = 30
TARGET_FS = 1.0
OUTER_FOLDS = 5
INNER_FOLDS = 3
SEED = RANDOM_SEED

DEEP_VARIANTS = ("signal", "bilstm", "multimodal", "clinical")
DEEP_STRATEGIES = (
    "plain_bce",
    "weighted_bce",
    "focal_gamma1",
    "focal_gamma2",
    "weighted_sampling",
)
CLASSICAL_MODELS = ("logistic", "gradient_boosting")
CLASSICAL_STRATEGIES = ("none", "class_weight", "random_oversample", "smote")

# Deep-learning budget.  Inner selection deliberately uses a shorter schedule
# than the outer refit; the selected epoch count is transferred to the outer
# model, which is trained on the full outer-training partition without touching
# the outer validation fold.
DEEP_INNER_EPOCHS = 15
DEEP_OUTER_EPOCHS = 25
DEEP_PATIENCE = 5
DEEP_BATCH_SIZE = 32
DEEP_WEIGHT_DECAY = 1e-4
DEEP_LR_GRID: Tuple[float, ...] = (5e-4, 1e-3)
DEEP_MIN_EPOCHS = 5
# CPU float reductions depend on the thread count, so fix the default to keep
# reruns bitwise reproducible.  Override with PHASE_B_THREADS when needed.
DEFAULT_TORCH_THREADS = 3

CLASSICAL_LR_C = (0.1, 0.5, 1.0)
CLASSICAL_GB_GRID = (
    {"max_depth": 2, "learning_rate": 0.05},
    {"max_depth": 3, "learning_rate": 0.05},
    {"max_depth": 3, "learning_rate": 0.1},
)

AUGMENT_MODES = ("none", "gaussian", "mix")
DEFAULT_THRESHOLD = 0.5
BOOTSTRAP_SAMPLES = 1000


@dataclass
class PhaseBConfig:
    seed: int = SEED
    window_min: int = WINDOW_MIN
    target_fs: float = TARGET_FS
    outer_folds: int = OUTER_FOLDS
    inner_folds: int = INNER_FOLDS
    deep_inner_epochs: int = DEEP_INNER_EPOCHS
    deep_outer_epochs: int = DEEP_OUTER_EPOCHS
    deep_patience: int = DEEP_PATIENCE
    deep_batch_size: int = DEEP_BATCH_SIZE
    deep_weight_decay: float = DEEP_WEIGHT_DECAY
    deep_lr_grid: Tuple[float, ...] = DEEP_LR_GRID
    deep_variants: Tuple[str, ...] = DEEP_VARIANTS
    deep_strategies: Tuple[str, ...] = DEEP_STRATEGIES
    classical_models: Tuple[str, ...] = CLASSICAL_MODELS
    classical_strategies: Tuple[str, ...] = CLASSICAL_STRATEGIES
    augment_modes: Tuple[str, ...] = AUGMENT_MODES
    default_threshold: float = DEFAULT_THRESHOLD
    bootstrap_samples: int = BOOTSTRAP_SAMPLES

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": "B",
            "config": asdict(self),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def ensure_directories() -> None:
    for directory in (
        PHASE_B_DIR,
        B0_DIR,
        B1_DIR,
        CLASSICAL_DIR,
        DEEP_DIR,
        AUGMENT_DIR,
        COMPARISON_DIR,
        REPORT_DIR,
        LOG_DIR,
        CACHE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def deep_result_path(variant: str, strategy: str) -> Path:
    return DEEP_DIR / variant / strategy


def classical_result_path(model: str, strategy: str) -> Path:
    return CLASSICAL_DIR / model / strategy


def augment_result_path(mode: str) -> Path:
    return AUGMENT_DIR / mode


def b0_result_path(variant: str) -> Path:
    return B0_DIR / variant


def selection_path(variant: str) -> Path:
    return DEEP_DIR / variant / "inner_selection.json"


def config_by_name() -> Dict[str, object]:
    return {name: value for name, value in globals().items() if name.isupper()}
