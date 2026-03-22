"""
src/features.py
---------------
Reusable feature engineering pipeline for Telco Customer Churn dataset.
Used by both notebooks and the FastAPI endpoint.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the raw Telco churn dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from WA_Fn-UseC_-Telco-Customer-Churn.csv

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features, encoded categoricals, binary target
    """
    df = df.copy()

    # ── 1. Fix TotalCharges dtype ──────────────────────────────────────────
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    # ── 2. Drop ID column ──────────────────────────────────────────────────
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # ── 3. Binary target ───────────────────────────────────────────────────
    if 'Churn' in df.columns:
        df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)

    # ── 4. New engineered features ─────────────────────────────────────────
    # Tenure group
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    # Charges per month (avoid div/0)
    df['charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)

    # New customer flag
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)

    # Count of add-on services
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        if col in df.columns:
            df[col + '_flag'] = (df[col] == 'Yes').astype(int)
    flag_cols = [c + '_flag' for c in service_cols if c in df.columns]
    df['service_count'] = df[flag_cols].sum(axis=1)

    # ── 5. Binary Yes/No columns → 0/1 ────────────────────────────────────
    binary_cols = ['Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'MultipleLines']
    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col] == 'Yes').astype(int)

    # ── 6. Label encode gender ─────────────────────────────────────────────
    if 'gender' in df.columns:
        df['gender'] = (df['gender'] == 'Male').astype(int)

    # ── 7. One-hot encode multi-class categoricals ─────────────────────────
    ohe_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # ── 8. Convert all bool columns to int ────────────────────────────────
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def get_feature_columns() -> list:
    """
    Return the exact feature columns used for model training.
    Must match the columns produced by build_features().
    """
    return [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'PaperlessBilling',
        'MonthlyCharges', 'TotalCharges',
        'tenure_group', 'charges_per_month', 'is_new_customer', 'service_count',
        'OnlineSecurity_flag', 'OnlineBackup_flag', 'DeviceProtection_flag',
        'TechSupport_flag', 'StreamingTV_flag', 'StreamingMovies_flag',
        'InternetService_Fiber optic', 'InternetService_No',
        'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check',
    ]


def prepare_single_record(data: dict) -> pd.DataFrame:
    """
    Prepare a single customer record (from API request) for model inference.

    Parameters
    ----------
    data : dict
        Raw customer fields as received from FastAPI request body

    Returns
    -------
    pd.DataFrame
        Single-row dataframe ready for model.predict_proba()
    """
    df = pd.DataFrame([data])
    df = build_features(df)

    feature_cols = get_feature_columns()

    # Add missing columns with 0 (handles OHE columns not present)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]
