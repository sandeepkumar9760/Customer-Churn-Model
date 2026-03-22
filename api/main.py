"""
api/main.py
-----------
FastAPI REST API for real-time Customer Churn Scoring
Run: uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import json
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import prepare_single_record

# ── App init ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time churn scoring for Telco customers using XGBoost + SHAP",
    version="1.0.0",
    contact={"name": "Sandeep Kumar", "url": "https://github.com/sandeepkumar9760"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model on startup ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'churn_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"⚠️  Model not found at {MODEL_PATH}. Run notebook 02 first.")


# ── Request schema ─────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    gender: str = Field(..., example="Male", description="Male or Female")
    SeniorCitizen: int = Field(..., example=0, description="1 if senior citizen, else 0")
    Partner: str = Field(..., example="Yes", description="Yes or No")
    Dependents: str = Field(..., example="No", description="Yes or No")
    tenure: int = Field(..., example=12, description="Months as customer")
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic", description="DSL / Fiber optic / No")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month", description="Month-to-month / One year / Two year")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.35)
    TotalCharges: float = Field(..., example=845.5)


# ── Response schema ────────────────────────────────────────────────────────
class ChurnPrediction(BaseModel):
    churn_prediction: int
    churn_label: str
    churn_probability: float
    retention_probability: float
    risk_level: str
    recommendation: str


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=ChurnPrediction, tags=["Prediction"])
def predict_churn(customer: CustomerData):
    """
    Predict churn probability for a single customer.

    Returns churn label, probability score, risk level, and retention recommendation.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run notebook 02 first to train and save the model."
        )

    try:
        # Prepare features
        X = prepare_single_record(customer.dict())

        # Predict
        churn_prob = float(model.predict_proba(X)[0][1])
        churn_pred = int(churn_prob >= 0.5)

        # Risk bucketing
        if churn_prob < 0.3:
            risk_level = "Low"
            recommendation = "Customer is stable. Standard retention programme sufficient."
        elif churn_prob < 0.6:
            risk_level = "Medium"
            recommendation = "Proactively offer loyalty discount or contract upgrade."
        else:
            risk_level = "High"
            recommendation = "Immediate intervention required. Offer personalised retention deal."

        return ChurnPrediction(
            churn_prediction=churn_pred,
            churn_label="Churn" if churn_pred == 1 else "No Churn",
            churn_probability=round(churn_prob, 4),
            retention_probability=round(1 - churn_prob, 4),
            risk_level=risk_level,
            recommendation=recommendation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerData]):
    """
    Predict churn for a batch of customers (max 100).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100.")

    results = []
    for i, customer in enumerate(customers):
        try:
            X = prepare_single_record(customer.dict())
            churn_prob = float(model.predict_proba(X)[0][1])
            churn_pred = int(churn_prob >= 0.5)
            results.append({
                "index": i,
                "churn_prediction": churn_pred,
                "churn_label": "Churn" if churn_pred == 1 else "No Churn",
                "churn_probability": round(churn_prob, 4),
                "risk_level": "High" if churn_prob >= 0.6 else "Medium" if churn_prob >= 0.3 else "Low"
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return {"total": len(results), "predictions": results}


@app.get("/model/info", tags=["Model"])
def model_info():
    """
    Return model metadata and feature list.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "model_type": type(model).__name__,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "features": model.get_booster().feature_names,
        "n_features": len(model.get_booster().feature_names)
    }
