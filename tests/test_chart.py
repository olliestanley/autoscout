"""
Unit tests for autoscout.vis.chart module.
"""

import pandas as pd
import pytest
from bokeh.plotting._figure import figure as Figure

from autoscout.vis import chart


@pytest.fixture
def chart_data() -> pd.DataFrame:
    """Sample data for chart tests."""
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y1": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        "y2": [100, 81, 64, 49, 36, 25, 16, 9, 4, 1],
        "category": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
        "player": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"],
        "size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    })


class TestLines:
    """Tests for chart.lines function."""

    def test_lines_returns_figure(self, chart_data):
        """Should return a Bokeh Figure."""
        result = chart.lines(
            chart_data,
            x_columns=["x"],
            y_columns=["y1"],
            colors=["blue"],
        )

        assert isinstance(result, Figure)

    def test_lines_single_line(self, chart_data):
        """Should create chart with single line."""
        result = chart.lines(
            chart_data,
            x_columns=["x"],
            y_columns=["y1"],
            colors=["blue"],
        )

        assert result is not None

    def test_lines_multiple_lines(self, chart_data):
        """Should create chart with multiple lines."""
        result = chart.lines(
            chart_data,
            x_columns=["x", "x"],
            y_columns=["y1", "y2"],
            colors=["blue", "red"],
        )

        assert result is not None

    def test_lines_with_legend_labels(self, chart_data):
        """Should accept custom legend labels."""
        result = chart.lines(
            chart_data,
            x_columns=["x"],
            y_columns=["y1"],
            colors=["blue"],
            legend_labels=["My Line"],
        )

        assert result is not None

    def test_lines_with_trends(self, chart_data):
        """Should create trend lines when trends=True."""
        result = chart.lines(
            chart_data,
            x_columns=["x"],
            y_columns=["y1"],
            colors=["blue"],
            trends=True,
        )

        assert result is not None

    def test_lines_with_vshade_int(self, chart_data):
        """Should shade area under line when vshade is int."""
        result = chart.lines(
            chart_data,
            x_columns=["x"],
            y_columns=["y1"],
            colors=["blue"],
            vshade=0,
        )

        assert result is not None

    def test_lines_with_vshade_tuple(self, chart_data):
        """Should shade area between lines when vshade is tuple."""
        result = chart.lines(
            chart_data,
            x_columns=["x", "x"],
            y_columns=["y1", "y2"],
            colors=["blue", "red"],
            vshade=(0, 1),
        )

        assert result is not None

    def test_lines_with_kwargs(self, chart_data):
        """Should pass kwargs to figure."""
        result = chart.lines(
            chart_data,
            x_columns=["x"],
            y_columns=["y1"],
            colors=["blue"],
            title="Test Chart",
            width=800,
            height=600,
        )

        assert result is not None


class TestScatter:
    """Tests for chart.scatter function."""

    def test_scatter_returns_figure(self, chart_data):
        """Should return a Bokeh Figure."""
        result = chart.scatter(chart_data, x="x", y="y1")

        assert isinstance(result, Figure)

    def test_scatter_basic(self, chart_data):
        """Should create basic scatter chart."""
        result = chart.scatter(chart_data, x="x", y="y1")

        assert result is not None

    def test_scatter_with_size(self, chart_data):
        """Should accept size column."""
        result = chart.scatter(chart_data, x="x", y="y1", size="size")

        assert result is not None

    def test_scatter_with_color(self, chart_data):
        """Should accept color column with palette."""
        result = chart.scatter(
            chart_data,
            x="x",
            y="y1",
            color="category",
            palette=["red", "blue", "green", "orange", "purple"],
        )

        assert result is not None

    def test_scatter_color_requires_palette(self, chart_data):
        """Should raise error when color specified without palette."""
        with pytest.raises(ValueError, match="Must pass `palette` with `color`"):
            chart.scatter(chart_data, x="x", y="y1", color="category")

    def test_scatter_with_marker(self, chart_data):
        """Should accept different marker types."""
        result = chart.scatter(chart_data, x="x", y="y1", marker="square")

        assert result is not None

    def test_scatter_without_legend(self, chart_data):
        """Should work without legend."""
        result = chart.scatter(
            chart_data,
            x="x",
            y="y1",
            color="category",
            palette=["red", "blue", "green", "orange", "purple"],
            legend=False,
        )

        assert result is not None


class TestAddMeans:
    """Tests for chart.add_means function."""

    def test_add_means_returns_figure(self, chart_data):
        """Should return the modified Figure."""
        plot = chart.scatter(chart_data, x="x", y="y1")
        result = chart.add_means(plot, chart_data, x="x", y="y1")

        assert isinstance(result, Figure)

    def test_add_means_same_figure(self, chart_data):
        """Should return the same figure object."""
        plot = chart.scatter(chart_data, x="x", y="y1")
        result = chart.add_means(plot, chart_data, x="x", y="y1")

        assert result is plot


class TestAddLabels:
    """Tests for chart.add_labels function."""

    def test_add_labels_returns_figure(self, chart_data):
        """Should return the modified Figure."""
        plot = chart.scatter(chart_data, x="x", y="y1")
        result = chart.add_labels(plot, chart_data, x="x", y="y1", label="player")

        assert isinstance(result, Figure)

    def test_add_labels_same_figure(self, chart_data):
        """Should return the same figure object."""
        plot = chart.scatter(chart_data, x="x", y="y1")
        result = chart.add_labels(plot, chart_data, x="x", y="y1", label="player")

        assert result is plot

    def test_add_labels_default_label(self, chart_data):
        """Should use 'player' as default label column."""
        plot = chart.scatter(chart_data, x="x", y="y1")
        result = chart.add_labels(plot, chart_data, x="x", y="y1")

        assert result is not None
