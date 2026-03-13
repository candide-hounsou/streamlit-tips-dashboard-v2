import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.data_loader import load_data, get_data_summary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tips_df() -> pd.DataFrame:
    """Minimal but realistic mock of the seaborn tips dataset."""
    return pd.DataFrame(
        {
            "total_bill": [16.99, 10.34, 21.01, 23.68, 24.59, 8.77, 26.88, 15.04],
            "tip": [1.01, 1.66, 3.50, 3.31, 3.61, 2.00, 3.12, 1.96],
            "sex": ["Female", "Male", "Male", "Male", "Female", "Male", "Male", "Male"],
            "smoker": ["No", "No", "No", "No", "No", "No", "Yes", "No"],
            "day": ["Sun", "Sun", "Sun", "Sun", "Sun", "Sun", "Sat", "Sat"],
            "time": ["Dinner", "Dinner", "Dinner", "Dinner", "Dinner", "Dinner", "Dinner", "Dinner"],
            "size": [2, 3, 3, 2, 4, 2, 4, 2],
        }
    )


class TestLoadData:
    """
    Tests for load_data().

    seaborn.load_dataset() is mocked to avoid a network call, and
    st.cache_data is bypassed so the function runs in a plain Python context.
    """

    def test_returns_dataframe(self):
        """load_data() should return a pandas DataFrame."""
        with patch("src.data_loader.sns.load_dataset", return_value=_make_tips_df()), \
             patch("src.data_loader.st.cache_data", lambda fn: fn):
            import importlib, src.data_loader as mod
            importlib.reload(mod)
            df = mod.load_data()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        """Dataset must contain all expected columns."""
        with patch("src.data_loader.sns.load_dataset", return_value=_make_tips_df()), \
             patch("src.data_loader.st.cache_data", lambda fn: fn):
            import importlib, src.data_loader as mod
            importlib.reload(mod)
            df = mod.load_data()
        expected = {"total_bill", "tip", "sex", "smoker", "day", "time", "size"}
        assert expected.issubset(set(df.columns))

    def test_not_empty(self):
        """Dataset should not be empty."""
        with patch("src.data_loader.sns.load_dataset", return_value=_make_tips_df()), \
             patch("src.data_loader.st.cache_data", lambda fn: fn):
            import importlib, src.data_loader as mod
            importlib.reload(mod)
            df = mod.load_data()
        assert len(df) > 0

    def test_no_missing_values(self):
        """Tips dataset has no missing values by default."""
        with patch("src.data_loader.sns.load_dataset", return_value=_make_tips_df()), \
             patch("src.data_loader.st.cache_data", lambda fn: fn):
            import importlib, src.data_loader as mod
            importlib.reload(mod)
            df = mod.load_data()
        assert df.isnull().sum().sum() == 0

    def test_tip_column_positive(self):
        """All tip values should be strictly positive."""
        with patch("src.data_loader.sns.load_dataset", return_value=_make_tips_df()), \
             patch("src.data_loader.st.cache_data", lambda fn: fn):
            import importlib, src.data_loader as mod
            importlib.reload(mod)
            df = mod.load_data()
        assert (df["tip"] > 0).all()


class TestGetDataSummary:
    def test_returns_dict(self, sample_df):
        """get_data_summary() should return a dictionary."""
        result = get_data_summary(sample_df)
        assert isinstance(result, dict)

    def test_correct_row_count(self, sample_df):
        """n_rows should match the dataframe length."""
        result = get_data_summary(sample_df)
        assert result["n_rows"] == len(sample_df)

    def test_correct_col_count(self, sample_df):
        """n_cols should match the number of columns."""
        result = get_data_summary(sample_df)
        assert result["n_cols"] == len(sample_df.columns)

    def test_summary_keys_present(self, sample_df):
        """Summary dict must contain all required keys."""
        result = get_data_summary(sample_df)
        for key in ["n_rows", "n_cols", "columns", "dtypes", "missing_values"]:
            assert key in result

    def test_columns_list_matches(self, sample_df):
        """columns key should list exactly the dataframe columns."""
        result = get_data_summary(sample_df)
        assert set(result["columns"]) == set(sample_df.columns)
