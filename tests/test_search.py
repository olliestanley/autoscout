"""
Unit tests for autoscout.search module.
"""

import pandas as pd

from autoscout import search


class TestSearch:
    """Tests for search.search function."""

    def test_search_gte_single_criterion(self):
        """Should filter rows with >= condition."""
        df = pd.DataFrame(
            {
                "player": ["A", "B", "C", "D"],
                "goals": [1, 5, 10, 15],
            }
        )

        result = search.search(df, {"gte": {"goals": 5}})

        assert len(result) == 3
        assert "A" not in result["player"].values

    def test_search_lte_single_criterion(self):
        """Should filter rows with <= condition."""
        df = pd.DataFrame(
            {
                "player": ["A", "B", "C", "D"],
                "goals": [1, 5, 10, 15],
            }
        )

        result = search.search(df, {"lte": {"goals": 10}})

        assert len(result) == 3
        assert "D" not in result["player"].values

    def test_search_eq_single_criterion(self):
        """Should filter rows with == condition."""
        df = pd.DataFrame(
            {
                "player": ["A", "B", "C", "D"],
                "goals": [1, 5, 5, 15],
            }
        )

        result = search.search(df, {"eq": {"goals": 5}})

        assert len(result) == 2
        assert set(result["player"].values) == {"B", "C"}

    def test_search_multiple_criteria_same_operator(self):
        """Should apply multiple conditions with same operator."""
        df = pd.DataFrame(
            {
                "player": ["A", "B", "C", "D"],
                "goals": [1, 5, 10, 15],
                "assists": [10, 8, 6, 4],
            }
        )

        result = search.search(df, {"gte": {"goals": 5, "assists": 6}})

        assert len(result) == 2
        assert set(result["player"].values) == {"B", "C"}

    def test_search_multiple_operators(self):
        """Should apply conditions with different operators."""
        df = pd.DataFrame(
            {
                "player": ["A", "B", "C", "D"],
                "goals": [1, 5, 10, 15],
                "assists": [10, 8, 6, 4],
            }
        )

        result = search.search(
            df,
            {
                "gte": {"goals": 5},
                "lte": {"goals": 10},
            },
        )

        assert len(result) == 2
        assert set(result["player"].values) == {"B", "C"}

    def test_search_empty_criteria(self):
        """Should return all rows with empty criteria."""
        df = pd.DataFrame(
            {
                "player": ["A", "B", "C"],
                "goals": [1, 2, 3],
            }
        )

        result = search.search(df, {})

        assert len(result) == 3

    def test_search_no_matches(self):
        """Should return empty DataFrame when no matches."""
        df = pd.DataFrame(
            {
                "player": ["A", "B", "C"],
                "goals": [1, 2, 3],
            }
        )

        result = search.search(df, {"gte": {"goals": 100}})

        assert len(result) == 0

    def test_search_preserves_columns(self, sample_player_data):
        """Should preserve all columns in result."""
        result = search.search(sample_player_data, {"gte": {"goals": 5}})

        assert list(result.columns) == list(sample_player_data.columns)


class TestSearchSimilar:
    """Tests for search.search_similar function."""

    def test_search_similar_returns_correct_count(self, sample_player_data):
        """Should return specified number of similar records."""
        result = search.search_similar(
            sample_player_data,
            ["goals", "assists"],
            "Alice",
            num=3,
        )

        assert len(result) == 3

    def test_search_similar_includes_target(self, sample_player_data):
        """Should include the target record (most similar to itself)."""
        result = search.search_similar(
            sample_player_data,
            ["goals", "assists"],
            "Alice",
            num=3,
        )

        assert "Alice" in result["player"].values

    def test_search_similar_by_integer_index(self, sample_player_data):
        """Should work with integer index."""
        result = search.search_similar(
            sample_player_data,
            ["goals", "assists"],
            0,  # Alice
            num=3,
        )

        assert len(result) == 3
        assert "Alice" in result["player"].values

    def test_search_similar_uses_euclidean_distance(self):
        """Should find records closest in Euclidean distance."""
        df = pd.DataFrame(
            {
                "player": ["A", "B", "C", "D", "E"],
                "minutes": [900, 900, 900, 900, 900],
                "x": [0, 1, 10, 100, 1000],
                "y": [0, 1, 10, 100, 1000],
            }
        )

        result = search.search_similar(df, ["x", "y"], "A", num=3)

        # A, B, C should be most similar to A (closest in distance)
        assert set(result["player"].values) == {"A", "B", "C"}

    def test_search_similar_multiple_columns(self, sample_player_data):
        """Should work with multiple feature columns."""
        result = search.search_similar(
            sample_player_data,
            ["goals", "assists", "shots", "tackles"],
            "Alice",
            num=3,
        )

        assert len(result) == 3

    def test_search_similar_preserves_all_columns(self, sample_player_data):
        """Should preserve all original columns in result."""
        result = search.search_similar(
            sample_player_data,
            ["goals", "assists"],
            "Alice",
            num=3,
        )

        assert list(result.columns) == list(sample_player_data.columns)

    def test_search_similar_does_not_modify_original(self, sample_player_data):
        """Should not modify original DataFrame."""
        original_shape = sample_player_data.shape
        original_columns = list(sample_player_data.columns)

        search.search_similar(sample_player_data, ["goals", "assists"], "Alice", num=3)

        assert sample_player_data.shape == original_shape
        assert list(sample_player_data.columns) == original_columns

    def test_search_similar_with_num_greater_than_data(self, sample_player_data):
        """Should handle num larger than dataset size."""
        result = search.search_similar(
            sample_player_data,
            ["goals", "assists"],
            "Alice",
            num=100,  # More than data size
        )

        assert len(result) == len(sample_player_data)
