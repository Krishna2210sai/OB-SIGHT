"""FastAPI backend for the delivery-mode clinical decision-support prototype.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend.schemas import (
    ClinicalInput,
    CTGPredictionResponse,
    CTGSummary,
    DemoCaseSummary,
    DemoPredictionResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from backend.services.ctg_parser import CTGFileError, MAX_FILE_BYTES, parse_ctg_file
from backend.services.model_service import DISCLAIMER, ModelService, ModelUnavailableError
from src.config import PROJECT_ROOT

# Models that consume a CTG tensor and can be selected for an upload.
CTG_MODEL_VARIANTS = {"multimodal", "bilstm", "signal"}

app = FastAPI(
    title="Delivery Outcome CDSS — Research Prototype",
    version="1.1.0",
    description=(
        "Research prototype for predicting cesarean versus vaginal delivery from "
        "intrapartum cardiotocography and maternal clinical data. Not for clinical use."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_service: ModelService | None = None


def get_service() -> ModelService:
    global _service
    if _service is None:
        try:
            _service = ModelService()
        except Exception as exc:  # pragma: no cover - depends on local artefacts
            raise HTTPException(status_code=503, detail=f"Model artifacts could not be loaded: {exc}") from exc
    return _service


@app.exception_handler(CTGFileError)
async def ctg_error_handler(request, exc: CTGFileError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ModelUnavailableError)
async def model_error_handler(request, exc: ModelUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready = all(
        (PROJECT_ROOT / "models" / f"{v}.pt").exists()
        for v in ["clinical", "multimodal", "signal", "bilstm"]
    )
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=ready,
        task="Cesarean vs vaginal delivery prediction (research prototype)",
        version="1.1.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        card = get_service().load_card()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ModelInfoResponse(**card)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    try:
        payload = get_service().load_metrics()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return MetricsResponse(**payload)


@app.post("/predict", response_model=PredictionResponse)
def predict(clinical: ClinicalInput) -> PredictionResponse:
    """Clinical-only prediction (no CTG uploaded)."""
    try:
        result = get_service().predict_clinical(clinical.as_feature_dict())
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)


def _clinical_from_form(
    age: float,
    gravidity: int,
    parity: int,
    gest_weeks: float,
    diabetes: bool,
    hypertension: bool,
    preeclampsia: bool,
    liquor_praecox: bool,
    pyrexia: bool,
    meconium: bool,
    induced: bool,
    presentation: int,
) -> ClinicalInput:
    try:
        return ClinicalInput(
            age=age,
            gravidity=gravidity,
            parity=parity,
            gest_weeks=gest_weeks,
            diabetes=diabetes,
            hypertension=hypertension,
            preeclampsia=preeclampsia,
            liquor_praecox=liquor_praecox,
            pyrexia=pyrexia,
            meconium=meconium,
            induced=induced,
            presentation=presentation,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@app.post("/predict-ctg", response_model=CTGPredictionResponse)
async def predict_ctg(
    file: UploadFile = File(..., description="CSV with fhr,uc columns or JSON {fs,fhr,uc}"),
    age: float = Form(...),
    gravidity: int = Form(...),
    parity: int = Form(...),
    gest_weeks: float = Form(...),
    diabetes: bool = Form(...),
    hypertension: bool = Form(...),
    preeclampsia: bool = Form(...),
    liquor_praecox: bool = Form(...),
    pyrexia: bool = Form(...),
    meconium: bool = Form(...),
    induced: bool = Form(...),
    presentation: int = Form(...),
    fs: float = Form(4.0, description="Sampling frequency, used only for CSV without a time column"),
    model: str = Form("multimodal", description="multimodal, bilstm or signal"),
) -> CTGPredictionResponse:
    if model not in CTG_MODEL_VARIANTS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown CTG model '{model}'. Choose one of: "
                f"{', '.join(sorted(CTG_MODEL_VARIANTS))}."
            ),
        )
    clinical = _clinical_from_form(
        age, gravidity, parity, gest_weeks, diabetes, hypertension, preeclampsia,
        liquor_praecox, pyrexia, meconium, induced, presentation,
    )
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 5 MB limit")
    parsed = parse_ctg_file(file.filename or "upload.csv", data, fs_default=fs)
    try:
        result, summary = get_service().predict_ctg(
            clinical.as_feature_dict(), parsed.fhr, parsed.uc, parsed.fs, variant=model
        )
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result["warnings"] = list(dict.fromkeys(parsed.warnings + result.get("warnings", [])))
    return CTGPredictionResponse(**result, ctg_summary=CTGSummary(**summary))


@app.get("/demo-cases", response_model=List[DemoCaseSummary])
def demo_cases() -> List[DemoCaseSummary]:
    try:
        return [DemoCaseSummary(**case) for case in get_service().demo_cases()]
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/demo/{record_id}", response_model=DemoPredictionResponse)
def demo(record_id: str) -> DemoPredictionResponse:
    try:
        result = get_service().predict_demo(record_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Demo record not found in the held-out test split")
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return DemoPredictionResponse(**result)


@app.get("/api/sample/{name}")
def sample_file(name: str):
    """Download a labelled sample CTG file for the upload screen."""
    allowed = {
        "sample_ctg.csv": PROJECT_ROOT / "samples" / "sample_ctg.csv",
        "synthetic_ctg.csv": PROJECT_ROOT / "samples" / "synthetic_ctg.csv",
        "sample_ctg.json": PROJECT_ROOT / "samples" / "sample_ctg.json",
    }
    path = allowed.get(name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path, media_type="text/csv" if name.endswith(".csv") else "application/json", filename=name)


# --- Static frontend (optional) --------------------------------------------
DIST = PROJECT_ROOT / "frontend" / "dist"
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/")
def index():
    index_file = DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Backend is running. Frontend build not found.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str):
    if path.startswith(("api/", "docs", "openapi", "redoc")):
        raise HTTPException(status_code=404, detail="Not found")
    index_file = DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Not found")
