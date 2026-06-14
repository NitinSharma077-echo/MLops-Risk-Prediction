from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from mlops_risk_platform.api.schema import (
    BatchCustomerRiskRequest,
    BatchCustomerRiskResponse,
    CustomerRiskRequest,
    CustomerRiskResponse,
    HealthResponse,
    ModelInfoResponse,
)
from mlops_risk_platform.utils import get_project_root
from mlops_risk_platform.utils.logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = get_project_root()
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"

MODEL = None
MODEL_NAME = "best_model"

INPUT_FEATURES = [
    "account_age_days",
    "transaction_count_30d",
    "avg_transaction_amount",
    "failed_login_count_7d",
    "support_tickets_90d",
    "monthly_charges",
    "total_spend",
    "credit_score",
    "country",
    "device_type",
]

app = FastAPI(
    title="Customer Risk Prediction API",
    description="Predict customer fraud risk using machine learning",
    version="1.0.0",
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    logger.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
    return model


@app.on_event("startup")
def startup_event():
    global MODEL
    try:
        MODEL = load_model()
        logger.info("Startup complete. API is ready.")
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        raise e


def get_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def get_recommendation(risk_level: str) -> str:
    if risk_level == "High Risk":
        return "Flag for manual review and apply stricter verification."
    if risk_level == "Medium Risk":
        return "Monitor activity and request additional verification if needed."
    return "Allow normal processing."


def build_response(probability: float, label: int) -> CustomerRiskResponse:
    risk_level = get_risk_level(probability)
    return CustomerRiskResponse(
        risk_probability=round(float(probability), 4),
        risk_label=int(label),
        risk_level=risk_level,
        recommendation=get_recommendation(risk_level),
        model_name=MODEL_NAME,
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="healthy", model_loaded=True, model_path=str(MODEL_PATH))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_path=str(MODEL_PATH),
        input_features=INPUT_FEATURES,
        output_description="Binary risk classification: 0 = No Risk, 1 = Risk",
    )


@app.post("/predict", response_model=CustomerRiskResponse, tags=["Prediction"])
def predict(request: CustomerRiskRequest) -> CustomerRiskResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame([request.model_dump()])[INPUT_FEATURES]
        label = int(MODEL.predict(df)[0])
        probability = float(MODEL.predict_proba(df)[0][1])
        return build_response(probability, label)
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict-batch", response_model=BatchCustomerRiskResponse, tags=["Prediction"])
def predict_batch(request: BatchCustomerRiskRequest) -> BatchCustomerRiskResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        records = [r.model_dump() for r in request.records]
        df = pd.DataFrame(records)[INPUT_FEATURES]
        labels = MODEL.predict(df)
        probabilities = MODEL.predict_proba(df)[:, 1]
        predictions = [
            build_response(prob, label)
            for prob, label in zip(probabilities, labels)
        ]
        return BatchCustomerRiskResponse(
            predictions=predictions,
            total_records=len(predictions),
        )
    except Exception as e:
        logger.error("Batch prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Batch prediction failed")
