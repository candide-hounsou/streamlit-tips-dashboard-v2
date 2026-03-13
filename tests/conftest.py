import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Minimal representative sample of the tips dataset.
    Used as a shared fixture across all test files.
    """
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
