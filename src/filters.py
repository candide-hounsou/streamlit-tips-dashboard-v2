import pandas as pd
import streamlit as st


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render sidebar filters and return the filtered dataframe.

    Filters:
        - Day of the week
        - Time (Lunch / Dinner)
        - Smoker status
        - Party size
    """
    st.sidebar.header("🔍 Filters")

    # --- Day filter ---
    all_days = sorted(df["day"].unique().tolist())
    selected_days = st.sidebar.multiselect(
        "Day of the week",
        options=all_days,
        default=all_days,
    )

    # --- Time filter ---
    all_times = df["time"].unique().tolist()
    selected_times = st.sidebar.multiselect(
        "Time",
        options=all_times,
        default=all_times,
    )

    # --- Smoker filter ---
    smoker_options = df["smoker"].unique().tolist()
    selected_smoker = st.sidebar.multiselect(
        "Smoker",
        options=smoker_options,
        default=smoker_options,
    )

    # --- Party size slider ---
    min_size = int(df["size"].min())
    max_size = int(df["size"].max())
    selected_size = st.sidebar.slider(
        "Party size",
        min_value=min_size,
        max_value=max_size,
        value=(min_size, max_size),
    )

    # --- Apply filters ---
    filtered_df = df[
        (df["day"].isin(selected_days))
        & (df["time"].isin(selected_times))
        & (df["smoker"].isin(selected_smoker))
        & (df["size"].between(selected_size[0], selected_size[1]))
    ]

    return filtered_df
