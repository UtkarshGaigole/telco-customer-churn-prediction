"""Data loading, cleaning, feature engineering, and preprocessing.

Single source of truth for every data-preparation step used in the project.
All choices mirror the analysis notebook and are documented there:
- 11 rows with blank TotalCharges are dropped (< 0.2% of the data)
- customerID is removed (identifier, no predictive value)
- Churn is mapped to 1 (Yes) / 0 (No)
- Two engineered features: TenureGroup (bins) and AvgChargesPerMonth
- Stratified 80/20 train-test split with random_state=42
- Scaling + one-hot encoding live inside a ColumnTransformer so they are
  always fitted on training data only (no leakage)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Churn"


def tenure_group(months: float) -> str:
    """Bin tenure into customer-lifetime segments."""
    if months <= 12:
        return "0-1 year"
    if months <= 24:
        return "1-2 years"
    if months <= 48:
        return "2-4 years"
    return "4+ years"


def load_and_clean(csv_path: str = "data/Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Load the raw Telco CSV and apply all cleaning + feature engineering."""
    df = pd.read_csv(csv_path)

    # TotalCharges contains 11 blank strings -> NaN after coercion -> drop
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Identifier column carries no predictive information
    df = df.drop(columns="customerID")

    # Binary target as integers
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    # Engineered features (created BEFORE column-type detection, see split_data)
    df["TenureGroup"] = df["tenure"].apply(tenure_group)
    df["AvgChargesPerMonth"] = (df["TotalCharges"] / df["tenure"]).replace(
        [np.inf, np.nan], 0
    )
    return df


def get_feature_types(X: pd.DataFrame) -> tuple[list, list]:
    """Return (categorical_cols, numerical_cols) detected from X's dtypes."""
    categorical = X.select_dtypes(include=["object", "str"]).columns.tolist()
    numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return categorical, numerical


def split_data(df: pd.DataFrame):
    """Stratified train-test split. Returns X_train, X_test, y_train, y_test."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scaling for numerical features, one-hot encoding for categorical ones.

    Column lists are detected from X *after* feature engineering, so the
    engineered features are always included.
    """
    categorical, numerical = get_feature_types(X)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
