"""
Deep Learning model (MLPRegressor) — train, evaluate, predict, save/load.
Architecture: 2 hidden layers (64 → 32 neurons), ReLU activations.
Target: tip amount (continuous).
"""

import os
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.preprocessing import prepare_data, encode_single_input, SAVED_DIR

MODEL_PATH = os.path.join(SAVED_DIR, "deep_learning.pkl")
SCALER_PATH = os.path.join(SAVED_DIR, "deep_scaler.pkl")
ENCODER_PATH = os.path.join(SAVED_DIR, "deep_encoder.pkl")


def train(df) -> dict:
    """
    Train an MLPRegressor (neural network) on the tips dataset.
    Architecture: Dense(64, relu) → Dense(32, relu) → Dense(1, linear)
    Saves model, scaler, and encoder to disk.
    Returns evaluation metrics.
    """
    X_train, X_test, y_train, y_test, scaler, encoder = prepare_data(df)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_train, y_train)

    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    metrics = evaluate(model, X_test, y_test)
    metrics["n_iter"] = int(model.n_iter_)
    metrics["loss"] = round(float(model.loss_), 6)
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
            "Deep Learning model not found. Run train() first."
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
