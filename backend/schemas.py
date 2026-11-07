"""Pydantic request/response schemas for the clinical decision-support API."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ClinicalInput(BaseModel):
    age: float = Field(..., ge=12, le=60, description="Maternal age in years")
    gravidity: int = Field(..., ge=0, le=20, description="Total number of pregnancies")
    parity: int = Field(..., ge=0, le=20, description="Number of previous deliveries")
    gest_weeks: float = Field(..., ge=20, le=45, description="Gestational age in completed weeks")
    diabetes: bool = Field(..., description="Pre-existing or gestational diabetes")
    hypertension: bool = Field(..., description="Chronic or pregnancy-induced hypertension")
    preeclampsia: bool = Field(..., description="Preeclampsia / eclampsia")
    liquor_praecox: bool = Field(..., description="Prolonged rupture of membranes / liquor praecox")
    pyrexia: bool = Field(..., description="Maternal pyrexia during labour")
    meconium: bool = Field(..., description="Meconium-stained amniotic fluid")
    induced: bool = Field(..., description="Labour was induced")
    presentation: Literal[1, 2, 3] = Field(..., description="1=vertex, 2=breech, 3=other/abnormal")

    @model_validator(mode="after")
    def check_parity(self):
        if self.parity > self.gravidity:
            raise ValueError("Parity cannot exceed gravidity")
        return self

    def as_feature_dict(self) -> Dict[str, float]:
        return {
            "age": float(self.age),
            "gravidity": float(self.gravidity),
            "parity": float(self.parity),
            "gest_weeks": float(self.gest_weeks),
            "diabetes": float(self.diabetes),
            "hypertension": float(self.hypertension),
            "preeclampsia": float(self.preeclampsia),
            "liquor_praecox": float(self.liquor_praecox),
            "pyrexia": float(self.pyrexia),
            "meconium": float(self.meconium),
            "induced": float(self.induced),
            "presentation": float(self.presentation),
        }


class ClinicalContribution(BaseModel):
    feature: str
    display_name: str
    input_value: Optional[float] = None
    contribution: float
    direction: Literal["toward_cesarean", "toward_vaginal"]


class SignalContribution(BaseModel):
    window: str
    importance: float
    note: str = "Gradient-based attribution; not a clinical interpretation."


class PredictionResponse(BaseModel):
    task: str
    prediction: Literal["Vaginal delivery", "Cesarean delivery"]
    predicted_class: int
    cesarean_probability: float = Field(..., ge=0, le=1)
    vaginal_probability: float = Field(..., ge=0, le=1)
    decision_threshold: float
    model_variant: str
    model_version: str
    calibrated: bool
    clinical_contributions: List[ClinicalContribution] = []
    signal_contributions: List[SignalContribution] = []
    warnings: List[str] = []
    disclaimer: str


class ModelInfoResponse(BaseModel):
    project: str
    version: str
    task: str
    target: str
    prediction_time: str
    dataset: dict
    architecture: dict
    evaluation: dict
    limitations: List[str]
    intended_use: List[str]
    prohibited_use: List[str]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    task: str
    version: str


class MetricsResponse(BaseModel):
    generated_at: str
    task: dict
    test_set: dict
    models: dict
    comparisons: dict = {}
    interpretation: List[str]


class CTGSummary(BaseModel):
    sampling_frequency_hz: float
    duration_minutes: float
    samples_used: int
    mean_fhr_bpm: Optional[float]
    fhr_missing_fraction: float
    uc_missing_fraction: float
    window_padded: bool


class CTGPredictionResponse(PredictionResponse):
    ctg_summary: CTGSummary


class DemoCaseSummary(BaseModel):
    record_id: str
    label: str
    split: str
    clinical: ClinicalInput
    note: str


class DemoPredictionResponse(CTGPredictionResponse):
    demo: bool = True
    true_label: Optional[str] = None
    true_label_note: str = (
        "True outcome is shown only for teaching/demonstration of this public "
        "research dataset and is never available at inference time."
    )
