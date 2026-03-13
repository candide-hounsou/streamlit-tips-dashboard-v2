import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.filters import apply_sidebar_filters


def make_mock_streamlit(days, times, smokers, size_range):
    """
    Helper to patch all Streamlit sidebar widgets with fixed return values.
    Avoids needing a running Streamlit server during tests.
    """
    mock_st = MagicMock()
    mock_st.sidebar.multiselect.side_effect = [days, times, smokers]
    mock_st.sidebar.slider.return_value = size_range
    return mock_st


class TestApplySidebarFilters:
    def test_no_filter_returns_all_rows(self, sample_df):
        """When all options are selected, all rows should be returned."""
        with patch("src.filters.st") as mock_st:
            mock_st.sidebar.multiselect.side_effect = [
                ["Sun", "Sat"],
                ["Dinner"],
                ["No", "Yes"],
            ]
            mock_st.sidebar.slider.return_value = (2, 4)
            result = apply_sidebar_filters(sample_df)
        assert len(result) == len(sample_df)

    def test_filter_by_day(self, sample_df):
        """Filtering to a single day should only return rows for that day."""
        with patch("src.filters.st") as mock_st:
            mock_st.sidebar.multiselect.side_effect = [
                ["Sat"],          # only Saturday
                ["Dinner"],
                ["No", "Yes"],
            ]
            mock_st.sidebar.slider.return_value = (2, 4)
            result = apply_sidebar_filters(sample_df)
        assert set(result["day"].unique()) == {"Sat"}

    def test_filter_by_smoker(self, sample_df):
        """Filtering to non-smokers only should exclude smoker rows."""
        with patch("src.filters.st") as mock_st:
            mock_st.sidebar.multiselect.side_effect = [
                ["Sun", "Sat"],
                ["Dinner"],
                ["No"],           # non-smokers only
            ]
            mock_st.sidebar.slider.return_value = (2, 4)
            result = apply_sidebar_filters(sample_df)
        assert "Yes" not in result["smoker"].values

    def test_filter_returns_dataframe(self, sample_df):
        """apply_sidebar_filters should always return a DataFrame."""
        with patch("src.filters.st") as mock_st:
            mock_st.sidebar.multiselect.side_effect = [
                ["Sun"],
                ["Dinner"],
                ["No"],
            ]
            mock_st.sidebar.slider.return_value = (2, 4)
            result = apply_sidebar_filters(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_empty_filter_returns_empty_df(self, sample_df):
        """Selecting no days should return an empty dataframe."""
        with patch("src.filters.st") as mock_st:
            mock_st.sidebar.multiselect.side_effect = [
                [],               # no day selected
                ["Dinner"],
                ["No", "Yes"],
            ]
            mock_st.sidebar.slider.return_value = (2, 4)
            result = apply_sidebar_filters(sample_df)
        assert len(result) == 0

    def test_size_filter(self, sample_df):
        """Party size filter should exclude rows outside the selected range."""
        with patch("src.filters.st") as mock_st:
            mock_st.sidebar.multiselect.side_effect = [
                ["Sun", "Sat"],
                ["Dinner"],
                ["No", "Yes"],
            ]
            mock_st.sidebar.slider.return_value = (2, 2)  # only size 2
            result = apply_sidebar_filters(sample_df)
        assert result["size"].max() <= 2
