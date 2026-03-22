# 📉 Customer Churn Prediction + FastAPI + Streamlit

**ShyftLabs Placement Portfolio | Data Science + Backend Track**  
**Author:** Sandeep Kumar · B.Tech CSE · Lovely Professional University

---

## Problem Statement

Telecom companies lose significant revenue from customer churn. This project builds
a production-ready churn prediction pipeline using XGBoost with SMOTE for class
imbalance, SHAP for explainability, and a FastAPI REST endpoint for real-time scoring.

---

## Key Results

| Metric | Value |
|---|---|
| Model | XGBoost (n=300, max_depth=4) |
| ROC-AUC | ~0.85+ |
| Imbalance Handling | SMOTE (train only) |
| Explainability | SHAP TreeExplainer |
| API | FastAPI with /predict + /predict/batch |

---

## Project Structure

```
project2-churn/
├── data/
│   ├── raw/                          ← WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/test_predictions.csv
├── notebooks/
│   ├── 01_eda.ipynb                  ← 8-section EDA
│   └── 02_model.ipynb                ← XGBoost + SMOTE + SHAP
├── api/
│   └── main.py                       ← FastAPI REST API
|
├── src/
│   └── features.py                   ← Reusable preprocessing
├── model/
│   ├── churn_model.pkl               ← Saved XGBoost model
│   └── feature_columns.json          ← Feature list for API
├── reports/figures/                  ← Exported plots
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
From Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`

### 3. Run notebooks in order
```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_model.ipynb
```

### 4. Start FastApi server
```bash
uvicorn api.main:app --reload
```

### 5. Test the API
Open browser: http://localhost:8000/docs  
Interactive Swagger UI with all endpoints pre-documented.

### 6. To see the Dashboard 
``` bash
streamlit run .\dashboard\app.py
```

#### Example curl request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.5
  }'
```

#### Example response:
```json
{
  "churn_prediction": 1,
  "churn_label": "Churn",
  "churn_probability": 0.7823,
  "retention_probability": 0.2177,
  "risk_level": "High",
  "recommendation": "Immediate intervention required. Offer personalised retention deal."
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model status |
| POST | `/predict` | Single customer churn score |
| POST | `/predict/batch` | Batch scoring (max 100) |
| GET | `/model/info` | Model metadata |

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML Model | XGBoost |
| Imbalance | SMOTE (imbalanced-learn) |
| Explainability | SHAP |
| API | FastAPI + Uvicorn | Streamlit
| Data | Pandas, NumPy |
| Serialisation | Joblib |

---

## Summary

> *"I built a customer churn prediction system using XGBoost on the Telco dataset.
> The dataset had a 73/27 class imbalance which I handled with SMOTE applied only
> on training data to avoid leakage. The model achieved ~0.85 ROC-AUC. I integrated
> SHAP for explainability — so stakeholders can see why a customer is flagged as high risk.
> I wrapped the whole thing in a FastAPI REST endpoint with batch scoring and a Swagger UI,
> so it can plug directly into any CRM or backend system."*
