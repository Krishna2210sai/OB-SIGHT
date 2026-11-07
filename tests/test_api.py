"""End-to-end API tests using FastAPI's TestClient and the real local artefacts.

These tests exercise the deployed research prototype; they do not assert any
particular clinical performance, only that the contract and safety disclosures
are honoured.
"""
from __future__ import annotations

import io
import json

import pytest


def _form(clinical: dict) -> dict:
    return {
        "age": clinical["age"],
        "gravidity": clinical["gravidity"],
        "parity": clinical["parity"],
        "gest_weeks": clinical["gest_weeks"],
        "diabetes": str(clinical["diabetes"]).lower(),
        "hypertension": str(clinical["hypertension"]).lower(),
        "preeclampsia": str(clinical["preeclampsia"]).lower(),
        "liquor_praecox": str(clinical["liquor_praecox"]).lower(),
        "pyrexia": str(clinical["pyrexia"]).lower(),
        "meconium": str(clinical["meconium"]).lower(),
        "induced": str(clinical["induced"]).lower(),
        "presentation": clinical["presentation"],
        "fs": "4.0",
    }


def _csv_payload(minutes: int = 30, fs: int = 4, fhr: float = 140.0, uc: float = 20.0) -> bytes:
    n = minutes * 60 * fs
    rows = ["fhr,uc"] + [f"{fhr},{uc}"] * n
    return ("\n".join(rows) + "\n").encode()


def _json_payload(minutes: int = 30, fs: int = 4) -> bytes:
    n = minutes * 60 * fs
    return json.dumps({"fs": fs, "fhr": [140.0] * n, "uc": [20.0] * n}).encode()


# --- Service metadata -------------------------------------------------------

def test_health_reports_loaded_models(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "research" in body["task"].lower()


def test_model_info_exposes_governance_sections(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"]["n_records"] == 552
    assert body["dataset"]["n_cesarean"] == 46
    assert len(body["limitations"]) >= 3
    assert body["intended_use"] and body["prohibited_use"]
    assert "not" in body["disclaimer"].lower()


def test_metrics_are_held_out_and_have_baselines(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["test_set"]["n"] == 83
    assert body["test_set"]["n_positive"] == 7
    assert "multimodal_dl" in body["models"]
    assert "bilstm_dl" in body["models"]
    assert "dummy_prior" not in body["models"]
    assert "bilstm_vs_signal" in body["comparisons"]


# --- Clinical-only prediction ----------------------------------------------

def test_predict_returns_valid_contract(client, synthetic_clinical, model_card):
    response = client.post("/predict", json=synthetic_clinical)
    assert response.status_code == 200
    body = response.json()
    threshold = model_card["evaluation"]["operating_points"]["clinical"]

    assert body["model_variant"] == "clinical"
    assert body["decision_threshold"] == pytest.approx(threshold)
    assert 0 <= body["cesarean_probability"] <= 1
    assert body["vaginal_probability"] == pytest.approx(1 - body["cesarean_probability"], abs=1e-6)
    assert body["predicted_class"] in (0, 1)
    expected = "Cesarean delivery" if body["cesarean_probability"] >= threshold else "Vaginal delivery"
    assert body["prediction"] == expected
    assert body["clinical_contributions"]
    assert body["signal_contributions"] == []
    assert any("Research prototype" in w for w in body["warnings"])
    assert "not a diagnosis" in body["disclaimer"].lower()


def test_predict_is_deterministic(client, synthetic_clinical):
    first = client.post("/predict", json=synthetic_clinical).json()
    second = client.post("/predict", json=synthetic_clinical).json()
    assert first["cesarean_probability"] == second["cesarean_probability"]


def test_predict_accepts_boundary_values(client, synthetic_clinical):
    payload = dict(synthetic_clinical, age=12, gest_weeks=45, gravidity=20, parity=20)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "update",
    [
        {"age": 5},
        {"age": 90},
        {"gest_weeks": 10},
        {"gravidity": -1},
        {"parity": 5, "gravidity": 2},  # parity cannot exceed gravidity
        {"presentation": 4},
        {"diabetes": "maybe"},
    ],
)
def test_predict_rejects_invalid_payloads(client, synthetic_clinical, update):
    payload = dict(synthetic_clinical, **update)
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_missing_fields(client):
    assert client.post("/predict", json={"age": 30}).status_code == 422


# --- CTG upload -------------------------------------------------------------

def test_predict_ctg_with_csv(client, synthetic_clinical, model_card):
    response = client.post(
        "/predict-ctg",
        files={"file": ("recording.csv", _csv_payload(), "text/csv")},
        data=_form(synthetic_clinical),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    threshold = model_card["evaluation"]["operating_points"]["multimodal"]
    assert body["model_variant"] == "multimodal"
    assert body["decision_threshold"] == pytest.approx(threshold)
    assert body["calibrated"] in (True, False)
    summary = body["ctg_summary"]
    assert summary["duration_minutes"] == pytest.approx(30.0, abs=1e-6)
    assert summary["sampling_frequency_hz"] == pytest.approx(4.0)
    assert summary["fhr_missing_fraction"] == pytest.approx(0.0)
    assert summary["window_padded"] is False
    assert body["signal_contributions"]


def test_predict_ctg_can_select_the_cnn_bilstm(client, synthetic_clinical, model_card):
    data = _form(synthetic_clinical)
    data["model"] = "bilstm"
    response = client.post(
        "/predict-ctg",
        files={"file": ("recording.csv", _csv_payload(), "text/csv")},
        data=data,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    threshold = model_card["evaluation"]["operating_points"]["bilstm"]
    assert body["model_variant"] == "bilstm"
    assert body["decision_threshold"] == pytest.approx(threshold)
    assert body["signal_contributions"]


def test_predict_ctg_rejects_unknown_model(client, synthetic_clinical):
    data = _form(synthetic_clinical)
    data["model"] = "not-a-model"
    response = client.post(
        "/predict-ctg",
        files={"file": ("recording.csv", _csv_payload(), "text/csv")},
        data=data,
    )
    assert response.status_code == 422
    assert "Unknown CTG model" in response.json()["detail"]


def test_predict_ctg_with_json_and_time_inference(client, synthetic_clinical):
    response = client.post(
        "/predict-ctg",
        files={"file": ("recording.json", _json_payload(minutes=20, fs=2), "application/json")},
        data=_form(synthetic_clinical),
    )
    assert response.status_code == 200, response.text
    assert response.json()["ctg_summary"]["sampling_frequency_hz"] == pytest.approx(2.0)
    assert response.json()["ctg_summary"]["window_padded"] is True


def test_predict_ctg_short_recording_is_rejected(client, synthetic_clinical):
    response = client.post(
        "/predict-ctg",
        files={"file": ("short.csv", _csv_payload(minutes=3), "text/csv")},
        data=_form(synthetic_clinical),
    )
    assert response.status_code == 400
    assert "at least" in response.json()["detail"].lower()


def test_predict_ctg_rejects_unsupported_extension(client, synthetic_clinical):
    response = client.post(
        "/predict-ctg",
        files={"file": ("recording.pdf", _csv_payload(), "application/pdf")},
        data=_form(synthetic_clinical),
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_predict_ctg_rejects_invalid_form_values(client):
    data = _form(
        {
            "age": 30,
            "gravidity": 2,
            "parity": 1,
            "gest_weeks": 39,
            "diabetes": False,
            "hypertension": False,
            "preeclampsia": False,
            "liquor_praecox": False,
            "pyrexia": False,
            "meconium": False,
            "induced": False,
            "presentation": 1,
        }
    )
    data["age"] = "999"
    response = client.post(
        "/predict-ctg",
        files={"file": ("recording.csv", _csv_payload(), "text/csv")},
        data=data,
    )
    assert response.status_code == 422


def test_predict_ctg_rejects_oversized_upload(client, synthetic_clinical):
    response = client.post(
        "/predict-ctg",
        files={"file": ("huge.csv", b"0" * (5 * 1024 * 1024 + 1), "text/csv")},
        data=_form(synthetic_clinical),
    )
    assert response.status_code == 413


def test_predict_ctg_requires_a_file(client, synthetic_clinical):
    response = client.post("/predict-ctg", data=_form(synthetic_clinical))
    assert response.status_code == 422


# --- Demo records -----------------------------------------------------------

def test_demo_cases_are_held_out_and_labelled(client):
    response = client.get("/demo-cases")
    assert response.status_code == 200
    cases = response.json()
    assert len(cases) == 14
    assert len({case["record_id"] for case in cases}) == 14
    assert all(case["split"] == "test" for case in cases)
    assert {case["label"] for case in cases} == {"Vaginal delivery", "Cesarean delivery"}
    assert sum(case["label"] == "Vaginal delivery" for case in cases) == 7
    assert sum(case["label"] == "Cesarean delivery" for case in cases) == 7
    assert all(case["clinical"]["parity"] <= case["clinical"]["gravidity"] for case in cases)


def test_demo_prediction_includes_true_label_only_for_teaching(client):
    cases = client.get("/demo-cases").json()
    case = cases[0]
    response = client.post(f"/demo/{case['record_id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["demo"] is True
    assert body["model_variant"] == "multimodal"
    assert body["true_label"] == case["label"]
    assert "ctg_summary" in body
    assert body["disclaimer"]


def test_demo_prediction_can_use_the_cnn_bilstm(client, model_card):
    cases = client.get("/demo-cases").json()
    response = client.post(f"/demo/{cases[0]['record_id']}?model=bilstm")
    assert response.status_code == 200
    body = response.json()
    assert body["model_variant"] == "bilstm"
    assert body["decision_threshold"] == pytest.approx(
        model_card["evaluation"]["operating_points"]["bilstm"]
    )


def test_demo_prediction_rejects_unknown_model(client):
    cases = client.get("/demo-cases").json()
    response = client.post(f"/demo/{cases[0]['record_id']}?model=nope")
    assert response.status_code == 422


def test_unknown_demo_record_returns_404(client):
    assert client.post("/demo/does-not-exist").status_code == 404


# --- Sample downloads and static hosting ------------------------------------

@pytest.mark.parametrize(
    "name, media",
    [
        ("sample_ctg.csv", "text/csv"),
        ("synthetic_ctg.csv", "text/csv"),
        ("sample_ctg.json", "application/json"),
    ],
)
def test_sample_files_download(client, name, media):
    response = client.get(f"/api/sample/{name}")
    assert response.status_code == 200
    assert media in response.headers["content-type"]
    assert response.content


def test_unknown_sample_returns_404(client):
    assert client.get("/api/sample/secrets.csv").status_code == 404


def test_root_serves_frontend_or_json(client):
    response = client.get("/")
    assert response.status_code == 200
    if "text/html" in response.headers.get("content-type", ""):
        assert b"<!doctype html" in response.content.lower()
    else:
        assert response.json()["health"] == "/health"


def test_api_prefix_is_not_swallowed_by_spa_fallback(client):
    assert client.get("/api/does-not-exist").status_code == 404
