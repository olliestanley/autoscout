"""
Unit tests for autoscout.util module.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from autoscout import util


class TestGetRecord:
    """Tests for util.get_record function."""

    def test_get_record_by_integer_index(self, sample_player_data):
        """Should return single-row DataFrame when given integer index."""
        result = util.get_record(sample_player_data, 0)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["player"].iloc[0] == "Alice"

    def test_get_record_by_integer_index_middle(self, sample_player_data):
        """Should return correct row for middle index."""
        result = util.get_record(sample_player_data, 3)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["player"].iloc[0] == "Diana"

    def test_get_record_by_player_name_unique(self, sample_player_data):
        """Should return single-row DataFrame when player name is unique."""
        result = util.get_record(sample_player_data, "Bob")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["player"].iloc[0] == "Bob"

    def test_get_record_by_player_name_with_duplicates(self):
        """Should return row with highest minutes when multiple matches exist."""
        df = pd.DataFrame(
            {
                "player": ["John", "John", "Jane"],
                "minutes": [90, 180, 270],
                "goals": [1, 3, 2],
            }
        )
        result = util.get_record(df, "John")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["minutes"].iloc[0] == 180
        assert result["goals"].iloc[0] == 3

    def test_get_record_by_team_name_when_no_player_column(self):
        """Should use team column when player column is absent."""
        df = pd.DataFrame(
            {
                "team": ["Team A", "Team B", "Team A"],
                "minutes": [100, 200, 300],
                "goals": [5, 10, 15],
            }
        )
        result = util.get_record(df, "Team A")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["minutes"].iloc[0] == 300  # highest minutes for Team A

    def test_get_record_returns_dataframe_not_series(self, sample_player_data):
        """Should always return DataFrame, not Series."""
        result_int = util.get_record(sample_player_data, 0)
        result_str = util.get_record(sample_player_data, "Alice")

        assert isinstance(result_int, pd.DataFrame)
        assert isinstance(result_str, pd.DataFrame)


class TestMinMaxScale:
    """Tests for util.min_max_scale function."""

    def test_min_max_scale_basic(self, sample_player_data):
        """Should scale columns to [0, 1] range."""
        result = util.min_max_scale(sample_player_data, ["goals", "assists"])

        assert result["goals"].min() == pytest.approx(0.0)
        assert result["goals"].max() == pytest.approx(1.0)
        assert result["assists"].min() == pytest.approx(0.0)
        assert result["assists"].max() == pytest.approx(1.0)

    def test_min_max_scale_preserves_other_columns(self, sample_player_data):
        """Should not modify columns not specified for scaling."""
        original_minutes = sample_player_data["minutes"].copy()
        result = util.min_max_scale(sample_player_data, ["goals"])

        pd.testing.assert_series_equal(result["minutes"], original_minutes)

    def test_min_max_scale_does_not_modify_original(self, sample_player_data):
        """Should not modify original DataFrame when inplace=False."""
        original_goals = sample_player_data["goals"].copy()
        util.min_max_scale(sample_player_data, ["goals"], inplace=False)

        pd.testing.assert_series_equal(sample_player_data["goals"], original_goals)

    def test_min_max_scale_inplace(self, sample_player_data):
        """Should modify original DataFrame when inplace=True."""
        original_max = sample_player_data["goals"].max()
        util.min_max_scale(sample_player_data, ["goals"], inplace=True)

        assert sample_player_data["goals"].max() == 1.0
        assert original_max != 1.0

    def test_min_max_scale_multiple_columns(self, sample_player_data):
        """Should scale multiple columns correctly."""
        columns = ["goals", "assists", "shots"]
        result = util.min_max_scale(sample_player_data, columns)

        for col in columns:
            assert result[col].min() == pytest.approx(0.0)
            assert result[col].max() == pytest.approx(1.0)

    def test_min_max_scale_single_value_column(self):
        """Should handle column with single unique value."""
        df = pd.DataFrame({"a": [5, 5, 5], "b": [1, 2, 3]})
        result = util.min_max_scale(df, ["a", "b"])

        # Single value column becomes all 0s after min-max scaling
        assert all(result["a"] == 0.0)
        assert result["b"].min() == 0.0
        assert result["b"].max() == 1.0


class TestLoadCsv:
    """Tests for util.load_csv function."""

    def test_load_csv_polars(self, sample_player_data):
        """Should load CSV as polars DataFrame when specified."""
        import polars as pl

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_player_data.to_csv(f.name, index=False)
            result = util.load_csv(f.name, format="polars")

        assert isinstance(result, pl.DataFrame)


class TestWriteCsv:
    """Tests for util.write_csv function."""

    def test_write_csv_creates_directory(self, sample_player_data):
        """Should create output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "path"
            result_path = util.write_csv(sample_player_data, nested_dir, "output")

            assert result_path.exists()
            assert nested_dir.exists()
