"""Loading and parsing of the CTU-CHB Intrapartum CTG Database.

The dataset is distributed in WFDB format.  Each record has:
  * ``<id>.hea`` -- WFDB header + clinical metadata in ``#`` comments
  * ``<id>.dat`` -- 16-bit little-endian, two interleaved channels (FHR, UC)

The files are never modified.  ``wfdb`` is used when available as a reference
parser, but the fast path reads the binary file directly (format 16, two
channels, gain 100) after validating the header.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from src.config import DATASET_DIR, LABEL_MAP, TARGET_FIELD

# Metadata keys present in the .hea comments.  Parsing is deliberately strict:
# a missing field becomes NaN rather than a fabricated value.
META_KEYS = [
    "pH",
    "BDecf",
    "pCO2",
    "BE",
    "Apgar1",
    "Apgar5",
    "NICU days",
    "Seizures",
    "HIE",
    "Intubation",
    "Main diag.",
    "Other diag.",
    "Gest. weeks",
    "Weight(g)",
    "Sex",
    "Age",
    "Gravidity",
    "Parity",
    "Diabetes",
    "Hypertension",
    "Preeclampsia",
    "Liq. praecox",
    "Pyrexia",
    "Meconium",
    "Presentation",
    "Induced",
    "I.stage",
    "NoProgress",
    "CK/KP",
    "II.stage",
    "Deliv. type",
    "dbID",
    "Rec. type",
    "Pos. II.st.",
    "Sig2Birth",
]

# Rename to stable snake_case names used throughout the project.
META_RENAME = {
    "pH": "ph",
    "BDecf": "bdecf",
    "pCO2": "pco2",
    "BE": "be",
    "Apgar1": "apgar1",
    "Apgar5": "apgar5",
    "NICU days": "nicu_days",
    "Seizures": "seizures",
    "HIE": "hie",
    "Intubation": "intubation",
    "Main diag.": "main_diag",
    "Other diag.": "other_diag",
    "Gest. weeks": "gest_weeks",
    "Weight(g)": "weight_g",
    "Sex": "sex",
    "Age": "age",
    "Gravidity": "gravidity",
    "Parity": "parity",
    "Diabetes": "diabetes",
    "Hypertension": "hypertension",
    "Preeclampsia": "preeclampsia",
    "Liq. praecox": "liquor_praecox",
    "Pyrexia": "pyrexia",
    "Meconium": "meconium",
    "Presentation": "presentation",
    "Induced": "induced",
    "I.stage": "first_stage_min",
    "NoProgress": "no_progress",
    "CK/KP": "ck_kp",
    "II.stage": "second_stage_min",
    "Deliv. type": "deliv_type",
    "dbID": "db_id",
    "Rec. type": "rec_type",
    "Pos. II.st.": "pos_second_stage",
    "Sig2Birth": "sig_to_birth",
}

SIGNAL_FS = 4
GAIN = 100.0


@dataclass
class RecordHeader:
    record_id: str
    n_signals: int
    fs: float
    n_samples: int
    signal_specs: List[dict] = field(default_factory=list)
    meta: Dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> Optional[int]:
        value = self.meta.get("deliv_type")
        if value is None or np.isnan(value):
            return None
        return LABEL_MAP.get(int(value))


def _parse_number(text: str) -> Optional[float]:
    text = text.strip()
    if text in {"", "-", "nan", "NaN"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_header(path: Path | str) -> RecordHeader:
    """Parse a WFDB ``.hea`` file, including the clinical metadata comments."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty header: {path}")

    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Malformed WFDB header line in {path}: {lines[0]!r}")
    record_id, n_signals, fs, n_samples = first[0], int(first[1]), float(first[2]), int(first[3])

    specs = []
    for line in lines[1 : 1 + n_signals]:
        parts = line.split()
        if len(parts) < 9:
            raise ValueError(f"Malformed signal specification in {path}: {line!r}")
        specs.append(
            {
                "file": parts[0],
                "format": parts[1],
                "gain": parts[2],
                "adc_res": int(parts[3]),
                "adc_zero": int(parts[4]),
                "init_value": int(parts[5]),
                "checksum": int(parts[6]),
                "block_size": int(parts[7]),
                "description": parts[8],
            }
        )

    meta: Dict[str, float] = {}
    for key in META_KEYS:
        # values are separated from the key by one or more spaces; allow a
        # trailing comment marker on the same line (none occur in practice).
        pattern = re.compile(r"(?m)^#\s*" + re.escape(key) + r"\s+(-?\d+(?:\.\d+)?)\s*$")
        match = pattern.search(path.read_text(encoding="utf-8", errors="replace"))
        value = _parse_number(match.group(1)) if match else None
        meta[META_RENAME.get(key, key)] = np.nan if value is None else value

    return RecordHeader(record_id, n_signals, fs, n_samples, specs, meta)


def read_signals(record_id: str | int, dataset_dir: Path = DATASET_DIR) -> np.ndarray:
    """Read the two channels of a record as float32 physical values.

    Returns an array of shape ``(n_samples, 2)`` with columns ``[FHR, UC]``.
    Values are *raw*, including the 0 codes that denote missing samples.
    """
    record_id = str(record_id)
    header = parse_header(dataset_dir / f"{record_id}.hea")

    if header.n_signals != 2 or header.fs != SIGNAL_FS:
        raise ValueError(f"Unexpected signal layout for {record_id}: {header.n_signals} signals at {header.fs} Hz")
    for spec in header.signal_specs:
        if spec["format"] != "16":
            raise ValueError(f"Unsupported WFDB format {spec['format']} in {record_id}")

    raw = np.fromfile(dataset_dir / f"{record_id}.dat", dtype="<i2")
    expected = header.n_samples * header.n_signals
    if raw.size != expected:
        raise ValueError(
            f"{record_id}: expected {expected} samples, found {raw.size}. "
            "The .dat file may be truncated or not WFDB format 16."
        )
    signals = raw.reshape(header.n_samples, header.n_signals).astype(np.float32)
    # gain is 100(0) for both channels and baseline is 0 (verified in headers)
    return signals / GAIN


def load_record(record_id: str | int, dataset_dir: Path = DATASET_DIR) -> dict:
    header = parse_header(dataset_dir / f"{record_id}.hea")
    return {
        "record_id": header.record_id,
        "fs": header.fs,
        "signals": read_signals(record_id, dataset_dir),
        "meta": header.meta,
        "label": header.label,
    }


def iter_record_ids(dataset_dir: Path = DATASET_DIR) -> List[str]:
    ids = sorted(p.stem for p in dataset_dir.glob("*.hea"))
    return ids


def build_metadata_table(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Parse every header into one tidy table (no signal data)."""
    rows = []
    for rid in iter_record_ids(dataset_dir):
        header = parse_header(dataset_dir / f"{rid}.hea")
        row = {"record_id": rid, "label": header.label, **header.meta}
        rows.append(row)
    df = pd.DataFrame(rows)
    df["record_num"] = df["record_id"].astype(int)
    df = df.sort_values("record_num").reset_index(drop=True)
    return df


def iter_records(dataset_dir: Path = DATASET_DIR) -> Iterator[dict]:
    for rid in iter_record_ids(dataset_dir):
        yield load_record(rid, dataset_dir)
