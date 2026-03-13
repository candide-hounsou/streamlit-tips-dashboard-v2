"""
Tests for FastAPI endpoints using TestClient (no running server needed).
Both GET and POST routes are tested for all three models.
"""

import pytest
import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.endpoints import app

client = TestClient(app)

# ── Shared valid payload ───────────────────────────────────────────────────────
VALID_PAYLOAD = {
    "total_bill": 18.5,
    "size": 3,
    "sex": "Male",
    "smoker": "No",
    "day": "Sat",
    "time": "Dinner",
}

VALID_PARAMS = {
    "total_bill": 18.5,
    "size": 3,
    "sex": "Male",
    "smoker": "No",
    "day": "Sat",
    "time": "Dinner",
}


# ── Health check ───────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


# ── Linear Regression ──────────────────────────────────────────────────────────

class TestLinearEndpoints:
    def test_post_linear_returns_200(self):
        with patch("src.models.linear_regression.load_model") as mock_load:
            mock_load.return_value = _mock_lr_components()
            resp = client.post("/predict/linear", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_post_linear_response_shape(self):
        with patch("src.models.linear_regression.load_model") as mock_load:
            mock_load.return_value = _mock_lr_components()
            resp = client.post("/predict/linear", json=VALID_PAYLOAD)
        data = resp.json()
        assert "predicted_tip" in data
        assert "model" in data
        assert data["model"] == "linear_regression"

    def test_get_linear_returns_200(self):
        with patch("src.models.linear_regression.load_model") as mock_load:
            mock_load.return_value = _mock_lr_components()
            resp = client.get("/predict/linear", params=VALID_PARAMS)
        assert resp.status_code == 200

    def test_post_linear_invalid_sex_returns_422(self):
        bad = {**VALID_PAYLOAD, "sex": "Unknown"}
        resp = client.post("/predict/linear", json=bad)
        assert resp.status_code == 422

    def test_post_linear_negative_bill_returns_422(self):
        bad = {**VALID_PAYLOAD, "total_bill": -5.0}
        resp = client.post("/predict/linear", json=bad)
        assert resp.status_code == 422


# ── Random Forest ──────────────────────────────────────────────────────────────

class TestForestEndpoints:
    def test_post_forest_returns_200(self):
        with patch("src.models.random_forest.load_model") as mock_load:
            mock_load.return_value = _mock_lr_components()
            resp = client.post("/predict/forest", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_post_forest_response_model_name(self):
        with patch("src.models.random_forest.load_model") as mock_load:
            mock_load.return_value = _mock_lr_components()
            resp = client.post("/predict/forest", json=VALID_PAYLOAD)
        assert resp.json()["model"] == "random_forest"

    def test_get_forest_returns_200(self):
        with patch("src.models.random_forest.load_model") as mock_load:
            mock_load.return_value = _mock_lr_components()
            resp = client.get("/predict/forest", params=VALID_PARAMS)
        assert resp.status_code == 200

    def test_post_forest_invalid_day_returns_422(self):
        bad = {**VALID_PAYLOAD, "day": "Monday"}
        resp = client.post("/predict/forest", json=bad)
        assert resp.status_code == 422


# ── Deep Learning ──────────────────────────────────────────────────────────────

class TestDeepEndpoints:
    def test_post_deep_returns_200(self):
        with patch("src.models.deep_learning.load_model") as mock_load:
            mock_load.return_value = _mock_lr_components()
            resp = client.post("/predict/deep", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_post_deep_response_model_name(self):
        with patch("src.models.deep_learning.load_model") as mock_load:
            mock_load.return_value = _mock_lr_components()
            resp = client.post("/predict/deep", json=VALID_PAYLOAD)
        assert resp.json()["model"] == "deep_learning"

    def test_get_deep_returns_200(self):
        with patch("src.models.deep_learning.load_model") as mock_load:
            mock_load.return_value = _mock_lr_components()
            resp = client.get("/predict/deep", params=VALID_PARAMS)
        assert resp.status_code == 200

    def test_post_deep_size_out_of_range_returns_422(self):
        bad = {**VALID_PAYLOAD, "size": 10}
        resp = client.post("/predict/deep", json=bad)
        assert resp.status_code == 422

    def test_predicted_tip_is_numeric(self):
        with patch("src.models.deep_learning.load_model") as mock_load:
            mock_load.return_value = _mock_lr_components()
            resp = client.post("/predict/deep", json=VALID_PAYLOAD)
        assert isinstance(resp.json()["predicted_tip"], (int, float))


# ── Helper ─────────────────────────────────────────────────────────────────────

def _mock_lr_components():
    """Return (model, scaler, encoder) mocks that produce a fixed prediction."""
    from unittest.mock import MagicMock
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([2.50])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[18.5, 3]])
    mock_encoder = MagicMock()
    mock_encoder.transform.return_value = np.array([[0.0, 0.0, 1.0, 0.0]])
    return mock_model, mock_scaler, mock_encoder
