"""
Pytest configuration and shared fixtures for autoscout tests.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_player_data() -> pd.DataFrame:
    """Sample player data for testing."""
    return pd.DataFrame(
        {
            "player": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"],
            "team": ["Team A", "Team A", "Team B", "Team B", "Team C", "Team C", "Team D", "Team D"],
            "position": ["FW", "MF", "FW", "DF", "MF", "GK", "FW", "DF"],
            "minutes": [1800, 1620, 900, 2700, 1350, 2520, 450, 1980],
            "goals": [15, 5, 8, 2, 7, 0, 3, 1],
            "assists": [6, 12, 4, 3, 9, 1, 2, 5],
            "shots": [60, 25, 35, 10, 40, 2, 20, 8],
            "passes": [400, 800, 300, 1200, 600, 150, 200, 900],
            "tackles": [15, 45, 20, 120, 55, 10, 12, 95],
            "interceptions": [8, 25, 12, 80, 30, 5, 6, 60],
        }
    )


@pytest.fixture
def sample_team_data() -> pd.DataFrame:
    """Sample team data for testing."""
    return pd.DataFrame(
        {
            "team": ["Team A", "Team B", "Team C", "Team D"],
            "name": ["Team A", "Team B", "Team C", "Team D"],
            "minutes": [3420, 3600, 3870, 2430],
            "goals": [45, 38, 52, 28],
            "assists": [35, 30, 42, 22],
            "possession": [55, 48, 62, 45],
            "shots": [180, 150, 200, 120],
            "passes": [4500, 3800, 5200, 3000],
        }
    )


@pytest.fixture
def sample_match_data() -> pd.DataFrame:
    """Sample match-level data for testing."""
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"],
            "opponent": ["Team B", "Team C", "Team D", "Team B", "Team C"],
            "squad": ["Team A", "Team A", "Team A", "Team A", "Team A"],
            "goals": [2, 1, 3, 0, 2],
            "assists": [1, 2, 2, 0, 1],
            "shots": [12, 8, 15, 6, 10],
            "possession": [55, 48, 60, 42, 52],
        }
    )


@pytest.fixture
def sample_team_match_data() -> pd.DataFrame:
    """Sample team match-level data for testing possession adjustments."""
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"],
            "name": ["Team A", "Team A", "Team A", "Team A", "Team A"],
            "possession": [55, 48, 60, 42, 52],
        }
    )


@pytest.fixture
def sample_radar_config() -> dict:
    """Sample radar configuration for testing."""
    return {
        "columns": {
            "goals": {
                "display": "Goals",
                "low": 0,
                "high": 20,
                "lower_is_better": False,
                "normalize": True,
            },
            "assists": {
                "display": "Assists",
                "low": 0,
                "high": 15,
                "lower_is_better": False,
                "normalize": True,
            },
            "tackles": {
                "display": "Tackles",
                "low": 0,
                "high": 50,
                "lower_is_better": False,
                "normalize": False,
            },
        }
    }


@pytest.fixture
def sample_rating_config() -> dict:
    """Sample rating configuration for testing style ratings."""
    return {
        "attack": ["goals", "assists", "shots"],
        "defense": ["tackles", "interceptions"],
    }


@pytest.fixture
def large_player_data() -> pd.DataFrame:
    """Larger player dataset for clustering and statistical tests."""
    import numpy as np

    np.random.seed(42)
    n = 100

    return pd.DataFrame(
        {
            "player": [f"Player_{i}" for i in range(n)],
            "team": [f"Team_{i % 10}" for i in range(n)],
            "position": np.random.choice(["FW", "MF", "DF", "GK"], n),
            "minutes": np.random.randint(450, 3600, n),
            "goals": np.random.randint(0, 25, n),
            "assists": np.random.randint(0, 20, n),
            "shots": np.random.randint(5, 100, n),
            "passes": np.random.randint(100, 2000, n),
            "tackles": np.random.randint(5, 150, n),
            "interceptions": np.random.randint(5, 100, n),
        }
    )
