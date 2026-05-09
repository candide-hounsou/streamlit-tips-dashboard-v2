"""
Shared preprocessing pipeline used by all three ML models.

Steps:
    1. Encode categorical columns with OrdinalEncoder
    2. Scale numeric columns with StandardScaler
    3. Return X (features) and y (target)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET = "tip"

NUMERIC_FEATURES = ["total_bill", "size"]
CATEGORICAL_FEATURES = ["sex", "smoker", "day", "time"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORIES = {
    "sex": ["Female", "Male"],
    "smoker": ["No", "Yes"],
    "day": ["Fri", "Sat", "Sun", "Thur"],
    "time": ["Dinner", "Lunch"],
}

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode and scale the full dataframe.
    Returns (X, y) as numpy arrays.
    """
    enc = OrdinalEncoder(
        categories=[CATEGORIES[c] for c in CATEGORICAL_FEATURES]
    )
    cat_encoded = enc.fit_transform(df[CATEGORICAL_FEATURES])
    num_array = df[NUMERIC_FEATURES].values
    X = np.hstack([num_array, cat_encoded])
    y = df[TARGET].values
    return X, y


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full pipeline: split → encode → scale (on training data only).
    Returns (X_train, X_test, y_train, y_test, scaler, encoder).
    
    IMPORTANT: Scaler is fit on training data only to prevent data leakage.
    """
    # First split the data
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Encode categorical features
    enc = OrdinalEncoder(
        categories=[CATEGORIES[c] for c in CATEGORICAL_FEATURES]
    )
    cat_encoded_train = enc.fit_transform(df_train[CATEGORICAL_FEATURES])
    cat_encoded_test = enc.transform(df_test[CATEGORICAL_FEATURES])
    
    # Scale numeric features (fit on training data only)
    num_array_train = df_train[NUMERIC_FEATURES].values
    num_array_test = df_test[NUMERIC_FEATURES].values
    
    scaler = StandardScaler()
    num_scaled_train = scaler.fit_transform(num_array_train)
    num_scaled_test = scaler.transform(num_array_test)
    
    # Combine features
    X_train = np.hstack([num_scaled_train, cat_encoded_train])
    X_test = np.hstack([num_scaled_test, cat_encoded_test])
    
    y_train = df_train[TARGET].values
    y_test = df_test[TARGET].values
    
    return X_train, X_test, y_train, y_test, scaler, enc


def encode_single_input(
    total_bill: float,
    size: int,
    sex: str,
    smoker: str,
    day: str,
    time: str,
    scaler: StandardScaler,
    encoder: OrdinalEncoder,
) -> np.ndarray:
    """
    Encode and scale a single prediction request into a feature vector.
    """
    num = scaler.transform([[total_bill, size]])
    cat_df = pd.DataFrame([[sex, smoker, day, time]], columns=CATEGORICAL_FEATURES)
    cat = encoder.transform(cat_df)
    return np.hstack([num, cat])
