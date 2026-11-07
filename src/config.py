"""Project-wide constants and paths.

The original dataset directory is treated as read-only.  Everything produced by
the pipeline is written to data/, models/, or reports/.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
SAMPLE_DIR = PROJECT_ROOT / "samples"

# --- Signal constants -------------------------------------------------------
FS_NATIVE = 4                 # Hz, as documented by PhysioNet
WINDOW_MIN = 30               # minutes of CTG used by the model
TARGET_FS = 1                 # Hz after deterministic down-sampling
WINDOW_SAMPLES = WINDOW_MIN * 60 * TARGET_FS   # 1800
SIGNAL_CHANNELS = ["fhr", "uc", "fhr_mask", "uc_mask"]

# Physiologically plausible / valid ranges used for basic artefact screening.
FHR_VALID_MIN = 30.0
FHR_VALID_MAX = 240.0
FHR_INTERP_MAX_SEC = 5.0      # interpolate short drop-outs only
UC_INTERP_MAX_SEC = 30.0

# --- Target ----------------------------------------------------------------
# Verified from the .hea metadata and the PhysioNet documentation:
#   Deliv. type 1 = vaginal delivery (spontaneous or operative vaginal)
#   Deliv. type 2 = cesarean section
LABEL_MAP = {1: 0, 2: 1}
CLASS_NAMES = ["Vaginal delivery", "Cesarean delivery"]
CLASS_SHORT = ["Vaginal", "Cesarean"]
TARGET_FIELD = "Deliv. type"

# --- Clinical features -----------------------------------------------------
# Only variables that can reasonably be available during labour, before the
# delivery mode is known, are used.  Post-delivery and decision variables
# (pH, Apgar, II.stage, NoProgress, CK/KP, birth weight, ...) are excluded.
NUMERIC_FEATURES = ["age", "gravidity", "parity", "gest_weeks"]
BINARY_FEATURES = [
    "diabetes",
    "hypertension",
    "preeclampsia",
    "liquor_praecox",
    "pyrexia",
    "meconium",
    "induced",
]
CATEGORICAL_FEATURES = ["presentation"]
PRESENTATION_LABELS = {1: "Vertex", 2: "Breech", 3: "Other / abnormal"}

RANDOM_SEED = 42

# --- Evaluation ------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15
