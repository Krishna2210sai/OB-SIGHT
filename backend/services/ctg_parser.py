"""Validation and parsing of user-supplied CTG files.

Supported formats
-----------------
1. CSV with a header row containing ``fhr`` and ``uc`` columns, optionally a
   ``time`` column in seconds.  If no time column is present, the sampling
   frequency is taken from the optional ``fs`` form field (default 4 Hz).
2. JSON object: ``{"fs": 4, "fhr": [...], "uc": [...]}``.

The original WFDB ``.dat``/``.hea`` pair is **not** an end-user upload format;
those files are research data and require both files to be parsed together.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

MAX_FILE_BYTES = 5 * 1024 * 1024
MIN_DURATION_MIN = 10.0
MAX_DURATION_MIN = 240.0
MAX_SAMPLES = 4 * 60 * 240  # 240 minutes at 4 Hz


class CTGFileError(ValueError):
    """Raised for a user-correctable file problem (HTTP 400)."""


@dataclass
class ParsedCTG:
    fhr: np.ndarray
    uc: np.ndarray
    fs: float
    source_format: str
    warnings: list[str]


def _validate_arrays(fhr: np.ndarray, uc: np.ndarray, fs: float) -> ParsedCTG:
    fhr = np.asarray(fhr, dtype=np.float64).reshape(-1)
    uc = np.asarray(uc, dtype=np.float64).reshape(-1)
    if fhr.size != uc.size:
        raise CTGFileError("FHR and UC arrays must have the same number of samples")
    if fhr.size < 10:
        raise CTGFileError("Too few samples; at least 10 samples are required")
    if fhr.size > MAX_SAMPLES:
        raise CTGFileError(f"Too many samples (maximum {MAX_SAMPLES})")
    if not (np.isfinite(fhr).all() and np.isfinite(uc).all()):
        raise CTGFileError("The file contains NaN or infinite values")
    if fs <= 0 or fs > 100:
        raise CTGFileError("Sampling frequency must be between 0 and 100 Hz")
    duration = fhr.size / fs / 60.0
    if duration < MIN_DURATION_MIN:
        raise CTGFileError(
            f"Recording is only {duration:.1f} minutes; at least {MIN_DURATION_MIN:.0f} minutes are required"
        )
    if duration > MAX_DURATION_MIN:
        raise CTGFileError(f"Recording is longer than {MAX_DURATION_MIN:.0f} minutes")
    if (fhr > 0).sum() < 0.05 * fhr.size:
        raise CTGFileError("The FHR channel contains almost no positive values")
    if (uc > 0).sum() == 0:
        raise CTGFileError("The UC channel is entirely zero; no uterine activity signal was found")

    warnings = []
    if (fhr > 0).mean() < 0.5:
        warnings.append("More than half of the FHR samples are missing or coded as zero.")
    if duration < 30:
        warnings.append(
            f"Recording is {duration:.1f} minutes; the model window is 30 minutes, so the start is padded."
        )
    implausible = ((fhr > 0) & ((fhr < 30) | (fhr > 240))).mean()
    if implausible > 0.02:
        warnings.append("More than 2% of FHR values are outside 30–240 bpm and were treated as artefacts.")
    return ParsedCTG(fhr=fhr, uc=uc, fs=fs, source_format="", warnings=warnings)


def _parse_csv(data: bytes, fs_default: float) -> ParsedCTG:
    text = data.decode("utf-8-sig", errors="strict")
    # Allow a comment line such as: # fs=4
    fs = fs_default
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if "fs=" in stripped.replace(" ", ""):
                try:
                    fs = float(stripped.split("fs=")[1].split(",")[0].strip())
                except ValueError:
                    raise CTGFileError("Could not read the 'fs=' comment in the CSV header")
            continue
        lines.append(line)
    if not lines:
        raise CTGFileError("The CSV file is empty")
    try:
        frame = pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception as exc:
        raise CTGFileError(f"Could not parse the CSV file: {exc}") from exc

    normalised = {str(c).strip().lower(): c for c in frame.columns}
    if "fhr" not in normalised or "uc" not in normalised:
        raise CTGFileError("CSV must contain columns named 'fhr' and 'uc' (one row per sample)")
    fhr = pd.to_numeric(frame[normalised["fhr"]], errors="coerce").to_numpy(dtype=float)
    uc = pd.to_numeric(frame[normalised["uc"]], errors="coerce").to_numpy(dtype=float)

    if "time" in normalised:
        time = pd.to_numeric(frame[normalised["time"]], errors="coerce").to_numpy(dtype=float)
        if len(time) > 2 and np.isfinite(time).all():
            diffs = np.diff(time)
            median_dt = float(np.median(diffs))
            if median_dt > 0:
                fs = 1.0 / median_dt

    parsed = _validate_arrays(fhr, uc, fs)
    parsed.source_format = "csv"
    return parsed


def _parse_json(data: bytes, fs_default: float) -> ParsedCTG:
    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except Exception as exc:
        raise CTGFileError(f"Could not parse the JSON file: {exc}") from exc
    if not isinstance(payload, dict):
        raise CTGFileError("JSON must be an object with 'fhr', 'uc' and optional 'fs'")
    if "fhr" not in payload or "uc" not in payload:
        raise CTGFileError("JSON must contain 'fhr' and 'uc' arrays")
    fs = float(payload.get("fs", fs_default))
    try:
        fhr = np.asarray(payload["fhr"], dtype=float)
        uc = np.asarray(payload["uc"], dtype=float)
    except Exception as exc:
        raise CTGFileError(f"'fhr' and 'uc' must be numeric arrays: {exc}") from exc
    parsed = _validate_arrays(fhr, uc, fs)
    parsed.source_format = "json"
    return parsed


def parse_ctg_file(filename: str, data: bytes, fs_default: float = 4.0) -> ParsedCTG:
    if not data:
        raise CTGFileError("The uploaded file is empty")
    if len(data) > MAX_FILE_BYTES:
        raise CTGFileError(f"File is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
    lower = filename.lower()
    if lower.endswith(".json"):
        return _parse_json(data, fs_default)
    if lower.endswith(".csv") or lower.endswith(".txt"):
        return _parse_csv(data, fs_default)
    raise CTGFileError("Unsupported file type. Upload a .csv or .json CTG file.")
