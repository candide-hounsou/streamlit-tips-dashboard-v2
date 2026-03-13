import seaborn as sns
import pandas as pd
import streamlit as st


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and cache the tips dataset from seaborn."""
    df = sns.load_dataset("tips")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dictionary of the dataframe."""
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }
