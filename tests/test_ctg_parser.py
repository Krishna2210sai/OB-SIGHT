"""Tests for the user-facing CTG file parser (CSV/JSON validation)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from backend.services.ctg_parser import (
    MAX_FILE_BYTES,
    MIN_DURATION_MIN,
    CTGFileError,
    parse_ctg_file,
)

FS = 4.0
MINUTES = 10
N = int(MINUTES * 60 * FS)  # minimum accepted duration


def _signal(n: int = N, fhr: float = 140.0, uc: float = 20.0):
    return np.full(n, fhr, dtype=float), np.full(n, uc, dtype=float)


def _csv_bytes(fhr, uc, time: bool = False, fs_comment: float | None = None) -> bytes:
    lines = []
    if fs_comment is not None:
        lines.append(f"# fs={fs_comment}")
    header = "time,fhr,uc" if time else "fhr,uc"
    lines.append(header)
    for i, (f, u) in enumerate(zip(fhr, uc)):
        if time:
            lines.append(f"{i / FS:.4f},{f},{u}")
        else:
            lines.append(f"{f},{u}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _json_bytes(fhr, uc, fs: float | None = FS) -> bytes:
    payload = {"fhr": list(map(float, fhr)), "uc": list(map(float, uc))}
    if fs is not None:
        payload["fs"] = fs
    return json.dumps(payload).encode("utf-8")


def test_valid_csv_without_time_uses_default_fs():
    fhr, uc = _signal()
    parsed = parse_ctg_file("recording.csv", _csv_bytes(fhr, uc), fs_default=FS)
    assert parsed.source_format == "csv"
    assert parsed.fs == pytest.approx(FS)
    assert parsed.fhr.shape == (N,)
    assert np.allclose(parsed.uc, 20.0)
    # Exactly the minimum accepted duration is fine but must warn about padding.
    assert any("30 minutes" in w for w in parsed.warnings)


def test_csv_with_time_column_infers_sampling_frequency():
    # 12 minutes of samples timestamped at 2 Hz; the default (4 Hz) is deliberately wrong.
    fhr, uc = _signal(int(12 * 60 * 2))
    lines = ["time,fhr,uc"] + [f"{i / 2.0:.4f},{f},{u}" for i, (f, u) in enumerate(zip(fhr, uc))]
    parsed = parse_ctg_file("recording.csv", ("\n".join(lines) + "\n").encode(), fs_default=4.0)
    assert parsed.fs == pytest.approx(2.0, abs=1e-6)
    assert any("30 minutes" in w for w in parsed.warnings)


def test_csv_fs_comment_overrides_default():
    fhr, uc = _signal()
    parsed = parse_ctg_file("recording.csv", _csv_bytes(fhr, uc, fs_comment=2.0), fs_default=4.0)
    assert parsed.fs == pytest.approx(2.0)


def test_json_payload_with_fs():
    fhr, uc = _signal()
    parsed = parse_ctg_file("recording.json", _json_bytes(fhr, uc, fs=2.0))
    assert parsed.source_format == "json"
    assert parsed.fs == pytest.approx(2.0)
    assert parsed.fhr.size == N


def test_json_without_fs_uses_form_default():
    fhr, uc = _signal()
    parsed = parse_ctg_file("recording.json", _json_bytes(fhr, uc, fs=None), fs_default=4.0)
    assert parsed.fs == pytest.approx(4.0)


def test_valid_30_minute_recording_has_no_duration_warning():
    fhr, uc = _signal(int(30 * 60 * FS))
    parsed = parse_ctg_file("recording.csv", _csv_bytes(fhr, uc))
    assert not any("30 minutes" in w for w in parsed.warnings)


def test_missing_column_is_rejected():
    data = b"fhr,other\n140,1\n141,2\n"
    with pytest.raises(CTGFileError, match="fhr"):
        parse_ctg_file("bad.csv", data)


def test_non_numeric_values_are_rejected_as_nan():
    payload = _csv_bytes(*_signal()).decode().splitlines()
    payload[1] = "not-a-number,20"
    with pytest.raises(CTGFileError, match="NaN"):
        parse_ctg_file("bad.csv", ("\n".join(payload) + "\n").encode())


def test_mismatched_lengths_are_rejected():
    fhr, uc = _signal()
    with pytest.raises(CTGFileError, match="same number"):
        parse_ctg_file("bad.json", _json_bytes(fhr, uc[:-1]))


def test_too_short_recording_is_rejected():
    n = int((MIN_DURATION_MIN - 1) * 60 * FS)
    fhr, uc = _signal(n)
    with pytest.raises(CTGFileError, match="at least"):
        parse_ctg_file("short.csv", _csv_bytes(fhr, uc))


def test_too_few_samples_is_rejected():
    fhr, uc = _signal(5)
    with pytest.raises(CTGFileError, match="Too few samples"):
        parse_ctg_file("tiny.json", _json_bytes(fhr, uc))


def test_all_zero_uc_is_rejected():
    fhr, uc = _signal()
    with pytest.raises(CTGFileError, match="uterine activity"):
        parse_ctg_file("flat_uc.csv", _csv_bytes(fhr, np.zeros_like(uc)))


def test_no_positive_fhr_is_rejected():
    fhr, uc = _signal(fhr=0.0)
    with pytest.raises(CTGFileError, match="positive values"):
        parse_ctg_file("flat_fhr.json", _json_bytes(fhr, uc))


def test_non_finite_values_are_rejected():
    fhr, uc = _signal()
    fhr[10] = np.inf
    with pytest.raises(CTGFileError, match="infinite"):
        parse_ctg_file("inf.json", _json_bytes(fhr, uc))


def test_unsupported_extension_is_rejected():
    fhr, uc = _signal()
    with pytest.raises(CTGFileError, match="Unsupported"):
        parse_ctg_file("recording.pdf", _csv_bytes(fhr, uc))


def test_empty_file_is_rejected():
    with pytest.raises(CTGFileError, match="empty"):
        parse_ctg_file("recording.csv", b"")


def test_oversized_file_is_rejected():
    with pytest.raises(CTGFileError, match="larger"):
        parse_ctg_file("recording.csv", b"0" * (MAX_FILE_BYTES + 1))


def test_json_must_be_object_with_arrays():
    with pytest.raises(CTGFileError, match="object"):
        parse_ctg_file("bad.json", b"[1, 2, 3]")
    with pytest.raises(CTGFileError, match="fhr"):
        parse_ctg_file("bad.json", b'{"fhr": [140, 141]}')


def test_invalid_json_syntax_is_rejected():
    with pytest.raises(CTGFileError, match="Could not parse the JSON"):
        parse_ctg_file("bad.json", b"{not json")


def test_implausible_values_raise_warning_but_parse():
    fhr, uc = _signal()
    fhr[: int(0.1 * N)] = 400.0  # > 2% outside 30-240 bpm
    parsed = parse_ctg_file("outliers.csv", _csv_bytes(fhr, uc))
    assert any("outside 30" in w for w in parsed.warnings)


def test_txt_extension_is_accepted_as_csv():
    fhr, uc = _signal()
    parsed = parse_ctg_file("recording.txt", _csv_bytes(fhr, uc))
    assert parsed.source_format == "csv"
