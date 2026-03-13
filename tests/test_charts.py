import pytest
from plotly.graph_objs import Figure
from src.charts import (
    plot_tip_distribution,
    plot_total_bill_vs_tip,
    plot_avg_tip_by_day,
    plot_tip_boxplot_by_time,
)


class TestPlotTipDistribution:
    def test_returns_figure(self, sample_df):
        """Should return a Plotly Figure object."""
        fig = plot_tip_distribution(sample_df)
        assert isinstance(fig, Figure)

    def test_contains_data(self, sample_df):
        """Figure should contain at least one data trace."""
        fig = plot_tip_distribution(sample_df)
        assert len(fig.data) > 0

    def test_correct_title(self, sample_df):
        """Figure title should mention 'Tip'."""
        fig = plot_tip_distribution(sample_df)
        assert "Tip" in fig.layout.title.text


class TestPlotTotalBillVsTip:
    def test_returns_figure(self, sample_df):
        """Should return a Plotly Figure object."""
        fig = plot_total_bill_vs_tip(sample_df)
        assert isinstance(fig, Figure)

    def test_contains_data(self, sample_df):
        """Figure should contain at least one data trace."""
        fig = plot_total_bill_vs_tip(sample_df)
        assert len(fig.data) > 0

    def test_scatter_trace_type(self, sample_df):
        """Should use a scatter trace (scatter plot)."""
        fig = plot_total_bill_vs_tip(sample_df)
        trace_types = [trace.type for trace in fig.data]
        assert "scatter" in trace_types


class TestPlotAvgTipByDay:
    def test_returns_figure(self, sample_df):
        """Should return a Plotly Figure object."""
        fig = plot_avg_tip_by_day(sample_df)
        assert isinstance(fig, Figure)

    def test_contains_data(self, sample_df):
        """Figure should contain at least one data trace."""
        fig = plot_avg_tip_by_day(sample_df)
        assert len(fig.data) > 0

    def test_bar_trace_type(self, sample_df):
        """Should use a bar trace (bar chart)."""
        fig = plot_avg_tip_by_day(sample_df)
        trace_types = [trace.type for trace in fig.data]
        assert "bar" in trace_types

    def test_one_bar_per_day(self, sample_df):
        """Number of bars should match number of unique days in the data."""
        fig = plot_avg_tip_by_day(sample_df)
        n_days = sample_df["day"].nunique()
        total_bars = sum(len(trace.x) for trace in fig.data if trace.type == "bar")
        assert total_bars == n_days


class TestPlotTipBoxplotByTime:
    def test_returns_figure(self, sample_df):
        """Should return a Plotly Figure object."""
        fig = plot_tip_boxplot_by_time(sample_df)
        assert isinstance(fig, Figure)

    def test_contains_data(self, sample_df):
        """Figure should contain at least one data trace."""
        fig = plot_tip_boxplot_by_time(sample_df)
        assert len(fig.data) > 0

    def test_box_trace_type(self, sample_df):
        """Should use a box trace (box plot)."""
        fig = plot_tip_boxplot_by_time(sample_df)
        trace_types = [trace.type for trace in fig.data]
        assert "box" in trace_types
