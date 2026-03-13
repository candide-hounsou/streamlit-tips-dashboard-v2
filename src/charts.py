import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure


def plot_tip_distribution(df: pd.DataFrame) -> Figure:
    """
    Histogram of tip amounts.
    Shows the distribution of tips across all records.
    """
    fig = px.histogram(
        df,
        x="tip",
        nbins=20,
        color_discrete_sequence=["#636EFA"],
        title="Distribution of Tip Amounts",
        labels={"tip": "Tip ($)", "count": "Number of Bills"},
    )
    fig.update_layout(bargap=0.05, xaxis_title="Tip ($)", yaxis_title="Count")
    return fig


def plot_total_bill_vs_tip(df: pd.DataFrame) -> Figure:
    """
    Scatter plot of total bill vs tip, colored by sex.
    Shows the relationship between bill size and tip amount.
    """
    fig = px.scatter(
        df,
        x="total_bill",
        y="tip",
        color="sex",
        size="size",
        hover_data=["day", "time", "smoker"],
        title="Total Bill vs Tip Amount",
        labels={"total_bill": "Total Bill ($)", "tip": "Tip ($)", "sex": "Gender"},
    )
    fig.update_layout(xaxis_title="Total Bill ($)", yaxis_title="Tip ($)")
    return fig


def plot_avg_tip_by_day(df: pd.DataFrame) -> Figure:
    """
    Bar chart of average tip per day.
    Compares tipping behaviour across days of the week.
    """
    avg_tip = (
        df.groupby("day", observed=True)["tip"]
        .mean()
        .reset_index()
        .rename(columns={"tip": "avg_tip"})
    )
    fig = px.bar(
        avg_tip,
        x="day",
        y="avg_tip",
        color="day",
        title="Average Tip by Day of the Week",
        labels={"day": "Day", "avg_tip": "Average Tip ($)"},
        text_auto=".2f",
    )
    fig.update_layout(showlegend=False, yaxis_title="Average Tip ($)")
    return fig


def plot_tip_boxplot_by_time(df: pd.DataFrame) -> Figure:
    """
    Box plot of tip distribution split by lunch vs dinner.
    Highlights spread and outliers per meal period.
    """
    fig = px.box(
        df,
        x="time",
        y="tip",
        color="time",
        points="all",
        title="Tip Distribution: Lunch vs Dinner",
        labels={"time": "Meal Time", "tip": "Tip ($)"},
    )
    fig.update_layout(showlegend=False, xaxis_title="Meal Time", yaxis_title="Tip ($)")
    return fig
