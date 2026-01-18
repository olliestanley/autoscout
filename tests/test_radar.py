"""
Unit tests for autoscout.vis.radar module.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from mplsoccer import Radar

from autoscout.vis import radar


@pytest.fixture
def radar_data() -> pd.DataFrame:
    """Sample data for radar chart tests."""
    return pd.DataFrame({
        "player": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "team": ["Team A", "Team A", "Team B", "Team B", "Team C"],
        "position": ["FW", "FW", "MF", "DF", "GK"],
        "minutes": [1800, 1620, 1350, 2700, 2520],
        "goals": [15, 12, 8, 2, 0],
        "assists": [6, 9, 12, 3, 1],
        "shots": [60, 50, 40, 10, 2],
        "tackles": [15, 20, 55, 120, 10],
        "interceptions": [8, 12, 30, 80, 5],
    })


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestEstimateLimitsByPosition:
    """Tests for radar.estimate_limits_by_position function."""

    def test_returns_tuple(self, radar_data):
        """Should return tuple of (low, high) values."""
        result = radar.estimate_limits_by_position(
            radar_data, "goals", "FW", alpha=0.1
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_low_less_than_high(self, radar_data):
        """Low percentile should be less than high percentile."""
        low, high = radar.estimate_limits_by_position(
            radar_data, "goals", "FW", alpha=0.1
        )

        assert low <= high

    def test_filters_by_position(self, radar_data):
        """Should only consider players in specified position."""
        # FW players have goals 15, 12 - different from DF player with 2
        low_fw, high_fw = radar.estimate_limits_by_position(
            radar_data, "goals", "FW", alpha=0.1
        )
        low_df, high_df = radar.estimate_limits_by_position(
            radar_data, "goals", "DF", alpha=0.1
        )

        # FW limits should be higher than DF limits
        assert high_fw > high_df

    def test_alpha_affects_range(self, radar_data):
        """Different alpha values should produce different ranges."""
        low_10, high_10 = radar.estimate_limits_by_position(
            radar_data, "goals", "FW", alpha=0.1
        )
        low_25, high_25 = radar.estimate_limits_by_position(
            radar_data, "goals", "FW", alpha=0.25
        )

        # Larger alpha should give narrower range
        assert (high_10 - low_10) >= (high_25 - low_25)


class TestPlotRadar:
    """Tests for radar.plot_radar function."""

    def test_returns_tuple(self, radar_data):
        """Should return tuple of (Radar, Figure, Axes)."""
        result = radar.plot_radar(
            radar_data,
            columns=["goals", "assists", "tackles"],
            index="Alice",
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_radar_object(self, radar_data):
        """First element should be mplsoccer Radar."""
        r, fig, ax = radar.plot_radar(
            radar_data,
            columns=["goals", "assists", "tackles"],
            index="Alice",
        )

        assert isinstance(r, Radar)

    def test_returns_figure(self, radar_data):
        """Second element should be matplotlib Figure."""
        r, fig, ax = radar.plot_radar(
            radar_data,
            columns=["goals", "assists", "tackles"],
            index="Alice",
        )

        assert isinstance(fig, mpl.figure.Figure)

    def test_returns_axes(self, radar_data):
        """Third element should be matplotlib Axes."""
        r, fig, ax = radar.plot_radar(
            radar_data,
            columns=["goals", "assists", "tackles"],
            index="Alice",
        )

        assert isinstance(ax, mpl.axes.Axes)

    def test_with_integer_index(self, radar_data):
        """Should work with integer index."""
        r, fig, ax = radar.plot_radar(
            radar_data,
            columns=["goals", "assists", "tackles"],
            index=0,
        )

        assert r is not None

    def test_with_comparison_player(self, radar_data):
        """Should work with comparison player."""
        r, fig, ax = radar.plot_radar(
            radar_data,
            columns=["goals", "assists", "tackles"],
            index="Alice",
            index_compare="Bob",
        )

        assert r is not None

    def test_with_comparison_average(self, radar_data):
        """Should work with 'average' comparison."""
        r, fig, ax = radar.plot_radar(
            radar_data,
            columns=["goals", "assists", "tackles"],
            index="Alice",
            index_compare="average",
        )

        assert r is not None

    def test_with_display_names(self, radar_data):
        """Should accept custom display names for columns."""
        r, fig, ax = radar.plot_radar(
            radar_data,
            columns=["goals", "assists", "tackles"],
            index="Alice",
            columns_display=["Goals Scored", "Assists Made", "Tackles Won"],
        )

        assert r is not None

    def test_with_lower_is_better(self, radar_data):
        """Should accept lower_is_better columns."""
        r, fig, ax = radar.plot_radar(
            radar_data,
            columns=["goals", "assists", "tackles"],
            index="Alice",
            lower_is_better=["tackles"],
        )

        assert r is not None

    def test_with_manual_min_max(self, radar_data):
        """Should accept manual min/max values."""
        r, fig, ax = radar.plot_radar(
            radar_data,
            columns=["goals", "assists", "tackles"],
            index="Alice",
            min_values=[0, 0, 0],
            max_values=[20, 15, 150],
        )

        assert r is not None

    def test_auto_min_max(self, radar_data):
        """Should calculate min/max automatically with 'auto'."""
        r, fig, ax = radar.plot_radar(
            radar_data,
            columns=["goals", "assists", "tackles"],
            index="Alice",
            min_values="auto",
            max_values="auto",
        )

        assert r is not None


class TestPlotRadarFromConfig:
    """Tests for radar.plot_radar_from_config function."""

    def test_returns_tuple(self, radar_data, sample_radar_config):
        """Should return tuple of (Radar, Figure, Axes)."""
        result = radar.plot_radar_from_config(
            radar_data,
            config=sample_radar_config,
            index="Alice",
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_uses_config_columns(self, radar_data, sample_radar_config):
        """Should use columns specified in config."""
        r, fig, ax = radar.plot_radar_from_config(
            radar_data,
            config=sample_radar_config,
            index="Alice",
        )

        assert r is not None

    def test_with_comparison(self, radar_data, sample_radar_config):
        """Should work with comparison player."""
        r, fig, ax = radar.plot_radar_from_config(
            radar_data,
            config=sample_radar_config,
            index="Alice",
            index_compare="Bob",
        )

        assert r is not None

    def test_normalizes_specified_columns(self, radar_data, sample_radar_config):
        """Should normalize columns marked for normalization in config."""
        r, fig, ax = radar.plot_radar_from_config(
            radar_data,
            config=sample_radar_config,
            index="Alice",
        )

        # Just verify it doesn't raise an error
        assert r is not None
