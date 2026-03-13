"""
Random Forest Regressor model — train, evaluate, predict, save/load.
Target: tip amount (continuous).
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.preprocessing import (
    prepare_data,
    encode_single_input,
    ALL_FEATURES,
    SAVED_DIR,
)

MODEL_PATH = os.path.join(SAVED_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(SAVED_DIR, "forest_scaler.pkl")
ENCODER_PATH = os.path.join(SAVED_DIR, "forest_encoder.pkl")


def train(df) -> dict:
    """
    Train a Random Forest Regressor on the tips dataset.
    Saves model, scaler, and encoder to disk.
    Returns evaluation metrics + feature importances.
    """
    X_train, X_test, y_train, y_test, scaler, encoder = prepare_data(df)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    metrics = evaluate(model, X_test, y_test)
    metrics["feature_importances"] = dict(
        zip(ALL_FEATURES, [round(float(i), 4) for i in model.feature_importances_])
    )
    return metrics


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R² on the test set."""
    y_pred = model.predict(X_test)
    return {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }


def load_model():
    """Load model, scaler, and encoder from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Random Forest model not found. Run train() first."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder


def predict(
    total_bill: float,
    size: int,
    sex: str,
    smoker: str,
    day: str,
    time: str,
) -> float:
    """
    Load the saved model and predict the tip for a single input.
    Returns the predicted tip amount (float).
    """
    model, scaler, encoder = load_model()
    X = encode_single_input(total_bill, size, sex, smoker, day, time, scaler, encoder)
    result = model.predict(X)
    return round(float(result[0]), 2)
