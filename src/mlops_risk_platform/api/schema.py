from typing import List

from pydantic import BaseModel, Field


class CustomerRiskRequest(BaseModel):
    """
    Request schema for one customer risk prediction.
    """

    account_age_days: int = Field(..., ge=0, example=730)
    transaction_count_30d: int = Field(..., ge=0, example=45)
    avg_transaction_amount: float = Field(..., ge=0, example=250.75)
    failed_login_count_7d: int = Field(..., ge=0, example=3)
    support_tickets_90d: int = Field(..., ge=0, example=2)
    monthly_charges: float = Field(..., ge=0, example=1299.0)
    total_spend: float = Field(..., ge=0, example=45000.0)
    credit_score: int = Field(..., ge=300, le=900, example=640)
    country: str = Field(..., example="India")
    device_type: str = Field(..., example="Mobile")


class CustomerRiskResponse(BaseModel):
    """
    Response schema for one customer risk prediction.
    """

    risk_probability: float
    risk_label: int
    risk_level: str
    recommendation: str
    model_name: str


class BatchCustomerRiskRequest(BaseModel):
    """
    Request schema for batch prediction.
    """

    records: List[CustomerRiskRequest]


class BatchCustomerRiskResponse(BaseModel):
    """
    Response schema for batch prediction.
    """

    predictions: List[CustomerRiskResponse]
    total_records: int


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """

    status: str
    model_loaded: bool
    model_path: str


class ModelInfoResponse(BaseModel):
    """
    Model information response schema.
    """

    model_name: str
    model_path: str
    input_features: List[str]
    output_description: str