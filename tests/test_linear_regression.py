import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import src.models.linear_regression as lr
from src.models.preprocessing import prepare_data


class TestLinearRegressionTrain:
    def test_train_returns_metrics(self, sample_df):
        """train() should return a dict with MAE, RMSE, R²."""
        with patch("src.models.linear_regression.joblib.dump"):
            with patch("os.makedirs"):
                metrics = lr.train(sample_df)
        assert isinstance(metrics, dict)
        for key in ["mae", "rmse", "r2"]:
            assert key in metrics

    def test_metrics_are_floats(self, sample_df):
        """All metric values should be floats."""
        with patch("src.models.linear_regression.joblib.dump"):
            with patch("os.makedirs"):
                metrics = lr.train(sample_df)
        for key in ["mae", "rmse", "r2"]:
            assert isinstance(metrics[key], float)

    def test_mae_non_negative(self, sample_df):
        """MAE must be >= 0."""
        with patch("src.models.linear_regression.joblib.dump"):
            with patch("os.makedirs"):
                metrics = lr.train(sample_df)
        assert metrics["mae"] >= 0

    def test_rmse_non_negative(self, sample_df):
        """RMSE must be >= 0."""
        with patch("src.models.linear_regression.joblib.dump"):
            with patch("os.makedirs"):
                metrics = lr.train(sample_df)
        assert metrics["rmse"] >= 0


class TestLinearRegressionEvaluate:
    def test_evaluate_returns_correct_keys(self, sample_df):
        """evaluate() must return mae, rmse, r2."""
        from sklearn.linear_model import LinearRegression
        X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_df)
        model = LinearRegression().fit(X_train, y_train)
        result = lr.evaluate(model, X_test, y_test)
        assert set(result.keys()) == {"mae", "rmse", "r2"}

    def test_r2_at_most_one(self, sample_df):
        """R² should never exceed 1."""
        from sklearn.linear_model import LinearRegression
        X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_df)
        model = LinearRegression().fit(X_train, y_train)
        result = lr.evaluate(model, X_test, y_test)
        assert result["r2"] <= 1.0


class TestLinearRegressionPredict:
    def test_predict_returns_float(self):
        """predict() should return a float."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.50])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[18.5, 3]])
        mock_encoder = MagicMock()
        mock_encoder.transform.return_value = np.array([[0.0, 0.0, 1.0, 0.0]])

        with patch("src.models.linear_regression.load_model",
                   return_value=(mock_model, mock_scaler, mock_encoder)):
            result = lr.predict(18.5, 3, "Male", "No", "Sat", "Dinner")
        assert isinstance(result, float)

    def test_predict_positive_value(self):
        """Predicted tip should be > 0 for a standard input."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.50])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[18.5, 3]])
        mock_encoder = MagicMock()
        mock_encoder.transform.return_value = np.array([[0.0, 0.0, 1.0, 0.0]])

        with patch("src.models.linear_regression.load_model",
                   return_value=(mock_model, mock_scaler, mock_encoder)):
            result = lr.predict(18.5, 3, "Male", "No", "Sat", "Dinner")
        assert result > 0

    def test_load_model_raises_if_missing(self, tmp_path, monkeypatch):
        """load_model() should raise FileNotFoundError if .pkl is absent."""
        monkeypatch.setattr(lr, "MODEL_PATH", str(tmp_path / "missing.pkl"))
        monkeypatch.setattr(lr, "SCALER_PATH", str(tmp_path / "missing_s.pkl"))
        monkeypatch.setattr(lr, "ENCODER_PATH", str(tmp_path / "missing_e.pkl"))
        with pytest.raises(FileNotFoundError):
            lr.load_model()
