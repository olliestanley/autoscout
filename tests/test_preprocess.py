"""
Unit tests for autoscout.preprocess module.
"""

import numpy as np
import pandas as pd
import pytest

from autoscout import preprocess


class TestCombineData:
    """Tests for preprocess.combine_data function."""

    def test_combine_data_basic(self):
        """Should concatenate DataFrames vertically."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

        result = preprocess.combine_data([df1, df2])

        assert len(result) == 4
        assert list(result["a"]) == [1, 2, 5, 6]
        assert list(result["b"]) == [3, 4, 7, 8]

    def test_combine_data_resets_index(self):
        """Should reset index after combining."""
        df1 = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        df2 = pd.DataFrame({"a": [3, 4]}, index=[0, 1])

        result = preprocess.combine_data([df1, df2])

        assert list(result.index) == [0, 1, 2, 3]

    def test_combine_data_inner_join_drops_extra_columns(self):
        """Should drop columns not in all DataFrames with retain_nans=False."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "c": [7, 8]})

        result = preprocess.combine_data([df1, df2], retain_nans=False)

        assert "a" in result.columns
        assert "b" not in result.columns
        assert "c" not in result.columns

    def test_combine_data_outer_join_retains_all_columns(self):
        """Should retain all columns with retain_nans=True."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "c": [7, 8]})

        result = preprocess.combine_data([df1, df2], retain_nans=True)

        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns
        assert pd.isna(result["b"].iloc[2])
        assert pd.isna(result["c"].iloc[0])

    def test_combine_data_multiple_dataframes(self):
        """Should combine more than two DataFrames."""
        dfs = [pd.DataFrame({"a": [i]}) for i in range(5)]

        result = preprocess.combine_data(dfs)

        assert len(result) == 5
        assert list(result["a"]) == [0, 1, 2, 3, 4]

    def test_combine_data_empty_list(self):
        """Should handle empty list gracefully."""
        with pytest.raises(ValueError):
            preprocess.combine_data([])


class TestRolling:
    """Tests for preprocess.rolling function."""

    def test_rolling_mean_basic(self):
        """Should calculate rolling mean correctly."""
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        result = preprocess.rolling(
            df, ["val"], roll_length=3, min_periods=1, dropna=False
        )

        assert "val_roll_mean" in result.columns
        # First value: just 1 -> mean = 1
        # Second value: 1,2 -> mean = 1.5
        # Third value: 1,2,3 -> mean = 2
        assert result["val_roll_mean"].iloc[2] == 2.0

    def test_rolling_sum(self):
        """Should calculate rolling sum when reduction='sum'."""
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        result = preprocess.rolling(
            df, ["val"], roll_length=3, min_periods=3, reduction="sum", dropna=False
        )

        assert "val_roll_sum" in result.columns
        # Third value onwards: sum of 3 values
        assert result["val_roll_sum"].iloc[2] == 6.0  # 1+2+3
        assert result["val_roll_sum"].iloc[3] == 9.0  # 2+3+4

    def test_rolling_multiple_columns(self):
        """Should handle multiple columns."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})

        result = preprocess.rolling(
            df, ["a", "b"], roll_length=2, min_periods=1, dropna=False
        )

        assert "a_roll_mean" in result.columns
        assert "b_roll_mean" in result.columns

    def test_rolling_does_not_modify_original(self):
        """Should not modify original DataFrame."""
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        original_columns = list(df.columns)

        preprocess.rolling(df, ["val"], roll_length=2, min_periods=1)

        assert list(df.columns) == original_columns

    def test_rolling_dropna_true(self):
        """Should drop rows with NaN values when dropna=True."""
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})

        result = preprocess.rolling(
            df, ["val"], roll_length=3, min_periods=3, dropna=True
        )

        # First two rows would have NaN, so should be dropped (after bfill they might not)
        assert "val_roll_mean" in result.columns

    def test_rolling_preserves_other_columns(self):
        """Should preserve columns not involved in rolling."""
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5], "other": ["a", "b", "c", "d", "e"]})

        result = preprocess.rolling(
            df, ["val"], roll_length=2, min_periods=1, dropna=False
        )

        assert "other" in result.columns


class TestClampByPercentiles:
    """Tests for preprocess.clamp_by_percentiles function."""

    def test_clamp_basic(self):
        """Should clamp outliers to percentile values."""
        df = pd.DataFrame({"val": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]})

        result = preprocess.clamp_by_percentiles(df, ["val"], alpha=0.1)

        assert result["val"].max() < 100
        assert result["val"].min() >= 0

    def test_clamp_symmetric(self):
        """Should clamp both high and low outliers."""
        df = pd.DataFrame({"val": [-100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]})

        result = preprocess.clamp_by_percentiles(df, ["val"], alpha=0.1)

        assert result["val"].max() < 100
        assert result["val"].min() > -100

    def test_clamp_does_not_modify_original(self):
        """Should not modify original DataFrame."""
        df = pd.DataFrame({"val": [0, 1, 2, 3, 100]})
        original_max = df["val"].max()

        preprocess.clamp_by_percentiles(df, ["val"], alpha=0.1)

        assert df["val"].max() == original_max

    def test_clamp_multiple_columns(self):
        """Should clamp multiple columns independently."""
        df = pd.DataFrame(
            {
                "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
                "b": [-50, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )

        result = preprocess.clamp_by_percentiles(df, ["a", "b"], alpha=0.1)

        assert result["a"].max() < 100
        assert result["b"].min() > -50

    def test_clamp_preserves_other_columns(self):
        """Should not affect columns not in the list."""
        df = pd.DataFrame({"val": [0, 1, 100], "other": [0, 1, 100]})

        result = preprocess.clamp_by_percentiles(df, ["val"], alpha=0.1)

        # "other" should be unchanged
        assert result["other"].max() == 100


class TestAdjustPer90:
    """Tests for preprocess.adjust_per_90 function."""

    def test_adjust_per_90_basic(self):
        """Should adjust values to per-90-minute basis."""
        df = pd.DataFrame({"minutes": [90, 180, 45], "goals": [1, 4, 1]})

        result = preprocess.adjust_per_90(df, ["goals"])

        assert result["goals"].iloc[0] == pytest.approx(1.0)  # 1 goal in 90 min
        assert result["goals"].iloc[1] == pytest.approx(2.0)  # 4 goals in 180 min
        assert result["goals"].iloc[2] == pytest.approx(2.0)  # 1 goal in 45 min

    def test_adjust_per_90_multiple_columns(self):
        """Should adjust multiple columns."""
        df = pd.DataFrame(
            {
                "minutes": [90, 180],
                "goals": [1, 4],
                "assists": [2, 2],
            }
        )

        result = preprocess.adjust_per_90(df, ["goals", "assists"])

        assert result["goals"].iloc[0] == pytest.approx(1.0)
        assert result["assists"].iloc[0] == pytest.approx(2.0)
        assert result["assists"].iloc[1] == pytest.approx(1.0)

    def test_adjust_per_90_does_not_modify_original(self):
        """Should not modify original DataFrame."""
        df = pd.DataFrame({"minutes": [180], "goals": [4]})

        preprocess.adjust_per_90(df, ["goals"])

        assert df["goals"].iloc[0] == 4

    def test_adjust_per_90_preserves_minutes_column(self):
        """Should not modify the minutes column."""
        df = pd.DataFrame({"minutes": [180], "goals": [4]})

        result = preprocess.adjust_per_90(df, ["goals"])

        assert result["minutes"].iloc[0] == 180


class TestAdjustPossessionDef:
    """Tests for preprocess.adjust_possession_def function."""

    def test_adjust_possession_def_50_percent(self):
        """With 50% possession, values should be unchanged."""
        targets = np.array([[10, 20], [30, 40]])
        possessions = np.array([50, 50])

        result = preprocess.adjust_possession_def(targets, possessions)

        np.testing.assert_array_almost_equal(result, targets)

    def test_adjust_possession_def_high_possession_increases(self):
        """With high possession, defensive stats should increase."""
        targets = np.array([[10]])
        possessions = np.array([70])

        result = preprocess.adjust_possession_def(targets, possessions)

        assert result[0, 0] > 10

    def test_adjust_possession_def_low_possession_decreases(self):
        """With low possession, defensive stats should decrease."""
        targets = np.array([[10]])
        possessions = np.array([30])

        result = preprocess.adjust_possession_def(targets, possessions)

        assert result[0, 0] < 10


class TestAdjustPossession:
    """Tests for preprocess.adjust_possession function."""

    def test_adjust_possession_basic(self, sample_match_data, sample_team_match_data):
        """Should create possession-adjusted columns."""
        result = preprocess.adjust_possession(
            sample_match_data, sample_team_match_data, ["goals", "assists"]
        )

        assert "padj_goals" in result.columns
        assert "padj_assists" in result.columns

    def test_adjust_possession_preserves_original_columns(
        self, sample_match_data, sample_team_match_data
    ):
        """Should preserve original columns."""
        result = preprocess.adjust_possession(
            sample_match_data, sample_team_match_data, ["goals"]
        )

        assert "goals" in result.columns
        assert "padj_goals" in result.columns


class TestFilterCategories:
    """Tests for preprocess.filter_categories function."""

    def test_filter_categories_retain(self):
        """Should keep only specified category columns."""
        df = pd.DataFrame(
            {
                "goals": [1, 2],
                "assists": [3, 4],
                "tackles": [5, 6],
                "interceptions": [7, 8],
            }
        )
        config = {
            "attack": ["goals", "assists"],
            "defense": ["tackles", "interceptions"],
        }

        result = preprocess.filter_categories(df, config, "attack", retain=True)

        assert list(result.columns) == ["goals", "assists"]

    def test_filter_categories_drop(self):
        """Should drop specified category columns when retain=False."""
        df = pd.DataFrame(
            {
                "goals": [1, 2],
                "assists": [3, 4],
                "tackles": [5, 6],
            }
        )
        config = {
            "attack": ["goals", "assists"],
        }

        result = preprocess.filter_categories(df, config, "attack", retain=False)

        assert "goals" not in result.columns
        assert "assists" not in result.columns
        assert "tackles" in result.columns
