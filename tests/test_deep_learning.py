import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import src.models.deep_learning as dl
from src.models.preprocessing import prepare_data


class TestDeepLearningTrain:
    """
    MLPRegressor with early_stopping=True requires a validation split of at
    least 2 rows, so these tests use a larger synthetic dataframe (30 rows)
    instead of the 8-row shared fixture.
    """

    @pytest.fixture()
    def large_df(self):
        import numpy as np, pandas as pd
        np.random.seed(0)
        n = 30
        return pd.DataFrame({
            "total_bill": np.round(np.random.uniform(5, 50, n), 2),
            "tip": np.round(np.random.uniform(1, 8, n), 2),
            "sex": np.random.choice(["Male", "Female"], n),
            "smoker": np.random.choice(["No", "Yes"], n),
            "day": np.random.choice(["Fri", "Sat", "Sun", "Thur"], n),
            "time": np.random.choice(["Dinner", "Lunch"], n),
            "size": np.random.choice([1, 2, 3, 4, 5, 6], n),
        })

    def test_train_returns_metrics(self, large_df):
        """train() should return a dict with MAE, RMSE, R², n_iter, loss."""
        with patch("src.models.deep_learning.joblib.dump"):
            with patch("os.makedirs"):
                metrics = dl.train(large_df)
        assert isinstance(metrics, dict)
        for key in ["mae", "rmse", "r2", "n_iter", "loss"]:
            assert key in metrics

    def test_n_iter_is_positive(self, large_df):
        """The model should run at least one iteration."""
        with patch("src.models.deep_learning.joblib.dump"):
            with patch("os.makedirs"):
                metrics = dl.train(large_df)
        assert metrics["n_iter"] > 0

    def test_loss_is_non_negative(self, large_df):
        """Training loss must be >= 0."""
        with patch("src.models.deep_learning.joblib.dump"):
            with patch("os.makedirs"):
                metrics = dl.train(large_df)
        assert metrics["loss"] >= 0

    def test_mae_non_negative(self, large_df):
        """MAE must be >= 0."""
        with patch("src.models.deep_learning.joblib.dump"):
            with patch("os.makedirs"):
                metrics = dl.train(large_df)
        assert metrics["mae"] >= 0


class TestDeepLearningEvaluate:
    def test_evaluate_returns_correct_keys(self, sample_df):
        """evaluate() must return mae, rmse, r2."""
        from sklearn.neural_network import MLPRegressor
        X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_df)
        model = MLPRegressor(hidden_layer_sizes=(8,), max_iter=50, random_state=42).fit(X_train, y_train)
        result = dl.evaluate(model, X_test, y_test)
        assert set(result.keys()) == {"mae", "rmse", "r2"}

    def test_r2_at_most_one(self, sample_df):
        """R² should never exceed 1."""
        from sklearn.neural_network import MLPRegressor
        X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_df)
        model = MLPRegressor(hidden_layer_sizes=(8,), max_iter=50, random_state=42).fit(X_train, y_train)
        result = dl.evaluate(model, X_test, y_test)
        assert result["r2"] <= 1.0


class TestDeepLearningPredict:
    def test_predict_returns_float(self):
        """predict() should return a float."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.30])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[18.5, 3]])
        mock_encoder = MagicMock()
        mock_encoder.transform.return_value = np.array([[0.0, 0.0, 1.0, 0.0]])

        with patch("src.models.deep_learning.load_model",
                   return_value=(mock_model, mock_scaler, mock_encoder)):
            result = dl.predict(18.5, 3, "Male", "No", "Sat", "Dinner")
        assert isinstance(result, float)

    def test_predict_rounded_to_2_decimals(self):
        """Predicted tip should be rounded to 2 decimal places."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.123456])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[18.5, 3]])
        mock_encoder = MagicMock()
        mock_encoder.transform.return_value = np.array([[0.0, 0.0, 1.0, 0.0]])

        with patch("src.models.deep_learning.load_model",
                   return_value=(mock_model, mock_scaler, mock_encoder)):
            result = dl.predict(18.5, 3, "Male", "No", "Sat", "Dinner")
        assert result == round(result, 2)

    def test_load_model_raises_if_missing(self, tmp_path, monkeypatch):
        """load_model() should raise FileNotFoundError if .pkl is absent."""
        monkeypatch.setattr(dl, "MODEL_PATH", str(tmp_path / "missing.pkl"))
        monkeypatch.setattr(dl, "SCALER_PATH", str(tmp_path / "missing_s.pkl"))
        monkeypatch.setattr(dl, "ENCODER_PATH", str(tmp_path / "missing_e.pkl"))
        with pytest.raises(FileNotFoundError):
            dl.load_model()
