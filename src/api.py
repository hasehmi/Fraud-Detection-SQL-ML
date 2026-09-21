"""
Fraud Detection — Scoring API

A minimal FastAPI service that loads the trained model and scores a single
transaction in real time. This is what gets containerized in the Dockerfile —
the same trained model from the notebook, served behind a REST endpoint
instead of only living inside a notebook.

Run locally:
    uvicorn src.api:app --reload --port 8000

Then POST a transaction to http://localhost:8000/predict (see README for an example).
"""
from datetime import datetime
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import (
    validate_transaction,
    validate_customer,
    compute_amount_vs_own_avg_ratio,
    is_odd_hour,
    is_away_from_home_city,
    is_unusual_category,
    minutes_since,
    apply_decision_threshold,
    SchemaValidationError,
)

MODEL_DIR = "models"

app = FastAPI(
    title="Fraud Detection API",
    description="Scores a single transaction for fraud risk using a trained XGBoost model.",
    version="1.0.0",
)

# Loaded once at startup, not per-request — this matters for response latency
model = joblib.load(f"{MODEL_DIR}/fraud_model.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
model_columns = joblib.load(f"{MODEL_DIR}/model_columns.pkl")
best_threshold = joblib.load(f"{MODEL_DIR}/best_threshold.pkl")

NUMERIC_COLS = [
    "amount", "txn_hour", "txn_day_of_week", "is_away_from_home_city",
    "is_unusual_category", "txns_last_24h", "txns_last_1h",
    "customer_rolling_avg_amount", "amount_vs_own_avg_ratio",
    "minutes_since_last_txn", "is_odd_hour",
]


class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp of the transaction")
    merchant_category: str = Field(..., description="e.g. grocery, electronics, travel")
    city: str = Field(..., description="City where the transaction occurred")

    customer_home_city: str
    customer_avg_spend: float = Field(..., ge=0)
    customer_favorite_category: str
    customer_risk_profile: str = Field(..., pattern="^(low|medium|high)$")
    customer_rolling_avg_amount: Optional[float] = Field(
        None, description="Customer's rolling average spend before this transaction; omit for a first-ever transaction"
    )
    txns_last_24h: int = Field(1, ge=0)
    txns_last_1h: int = Field(1, ge=0)
    previous_txn_timestamp: Optional[datetime] = Field(
        None, description="Timestamp of the customer's previous transaction, if any"
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    decision: str
    threshold_used: float


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionRequest):
    transaction_dict = {
        "amount": txn.amount,
        "timestamp": txn.timestamp,
        "merchant_category": txn.merchant_category,
        "city": txn.city,
    }
    customer_dict = {
        "home_city": txn.customer_home_city,
        "avg_spend": txn.customer_avg_spend,
        "favorite_category": txn.customer_favorite_category,
        "risk_profile": txn.customer_risk_profile,
    }

    try:
        validate_transaction(transaction_dict)
        validate_customer(customer_dict)
    except SchemaValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    rolling_avg = txn.customer_rolling_avg_amount if txn.customer_rolling_avg_amount else txn.customer_avg_spend

    features = {
        "amount": txn.amount,
        "txn_hour": txn.timestamp.hour,
        "txn_day_of_week": txn.timestamp.weekday(),
        "is_away_from_home_city": int(is_away_from_home_city(txn.city, txn.customer_home_city)),
        "is_unusual_category": int(is_unusual_category(txn.merchant_category, txn.customer_favorite_category)),
        "txns_last_24h": txn.txns_last_24h,
        "txns_last_1h": txn.txns_last_1h,
        "customer_rolling_avg_amount": rolling_avg,
        "amount_vs_own_avg_ratio": compute_amount_vs_own_avg_ratio(txn.amount, rolling_avg),
        "minutes_since_last_txn": minutes_since(txn.timestamp, txn.previous_txn_timestamp),
        "is_odd_hour": int(is_odd_hour(txn.timestamp.hour)),
    }

    # One-hot encode categoricals to match the exact training-time column layout
    row = {col: 0 for col in model_columns}
    for feat, val in features.items():
        if feat in row:
            row[feat] = val

    cat_col = f"merchant_category_{txn.merchant_category}"
    if cat_col in row:
        row[cat_col] = 1
    fav_col = f"favorite_category_{txn.customer_favorite_category}"
    if fav_col in row:
        row[fav_col] = 1
    risk_col = f"risk_profile_{txn.customer_risk_profile}"
    if risk_col in row:
        row[risk_col] = 1

    X = pd.DataFrame([row], columns=model_columns)

    fraud_prob = float(model.predict_proba(X)[0, 1])
    decision = apply_decision_threshold(fraud_prob, best_threshold)

    return PredictionResponse(
        fraud_probability=round(fraud_prob, 4),
        decision=decision,
        threshold_used=round(float(best_threshold), 4),
    )
