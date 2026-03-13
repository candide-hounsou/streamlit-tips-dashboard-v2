import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import src.models.random_forest as rf
from src.models.preprocessing import prepare_data, ALL_FEATURES


class TestRandomForestTrain:
    def test_train_returns_metrics(self, sample_df):
        """train() should return a dict with MAE, RMSE, R², and feature_importances."""
        with patch("src.models.random_forest.joblib.dump"):
            with patch("os.makedirs"):
                metrics = rf.train(sample_df)
        assert isinstance(metrics, dict)
        for key in ["mae", "rmse", "r2", "feature_importances"]:
            assert key in metrics

    def test_feature_importances_sum_to_one(self, sample_df):
        """Feature importances must sum to approximately 1.0."""
        with patch("src.models.random_forest.joblib.dump"):
            with patch("os.makedirs"):
                metrics = rf.train(sample_df)
        total = sum(metrics["feature_importances"].values())
        assert abs(total - 1.0) < 1e-4

    def test_feature_importances_keys_match(self, sample_df):
        """Feature importance keys should match ALL_FEATURES."""
        with patch("src.models.random_forest.joblib.dump"):
            with patch("os.makedirs"):
                metrics = rf.train(sample_df)
        assert set(metrics["feature_importances"].keys()) == set(ALL_FEATURES)

    def test_mae_non_negative(self, sample_df):
        """MAE must be >= 0."""
        with patch("src.models.random_forest.joblib.dump"):
            with patch("os.makedirs"):
                metrics = rf.train(sample_df)
        assert metrics["mae"] >= 0


class TestRandomForestEvaluate:
    def test_evaluate_returns_correct_keys(self, sample_df):
        """evaluate() must return mae, rmse, r2."""
        from sklearn.ensemble import RandomForestRegressor
        X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_df)
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_train)
        result = rf.evaluate(model, X_test, y_test)
        assert set(result.keys()) == {"mae", "rmse", "r2"}

    def test_r2_at_most_one(self, sample_df):
        """R² should never exceed 1."""
        from sklearn.ensemble import RandomForestRegressor
        X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_df)
        model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_train)
        result = rf.evaluate(model, X_test, y_test)
        assert result["r2"] <= 1.0


class TestRandomForestPredict:
    def test_predict_returns_float(self):
        """predict() should return a float."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.75])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[18.5, 3]])
        mock_encoder = MagicMock()
        mock_encoder.transform.return_value = np.array([[0.0, 0.0, 1.0, 0.0]])

        with patch("src.models.random_forest.load_model",
                   return_value=(mock_model, mock_scaler, mock_encoder)):
            result = rf.predict(18.5, 3, "Male", "No", "Sat", "Dinner")
        assert isinstance(result, float)

    def test_predict_rounded_to_2_decimals(self):
        """Predicted tip should be rounded to 2 decimal places."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.7512345])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[18.5, 3]])
        mock_encoder = MagicMock()
        mock_encoder.transform.return_value = np.array([[0.0, 0.0, 1.0, 0.0]])

        with patch("src.models.random_forest.load_model",
                   return_value=(mock_model, mock_scaler, mock_encoder)):
            result = rf.predict(18.5, 3, "Male", "No", "Sat", "Dinner")
        assert result == round(result, 2)

    def test_load_model_raises_if_missing(self, tmp_path, monkeypatch):
        """load_model() should raise FileNotFoundError if .pkl is absent."""
        monkeypatch.setattr(rf, "MODEL_PATH", str(tmp_path / "missing.pkl"))
        monkeypatch.setattr(rf, "SCALER_PATH", str(tmp_path / "missing_s.pkl"))
        monkeypatch.setattr(rf, "ENCODER_PATH", str(tmp_path / "missing_e.pkl"))
        with pytest.raises(FileNotFoundError):
            rf.load_model()
