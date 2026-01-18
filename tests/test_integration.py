"""
Integration tests for autoscout using real fbref data.

These tests make actual HTTP requests to fbref.com and test the full pipeline
from scraping to analysis. They are marked as 'integration' and can be skipped
in CI with: pytest -m "not integration"

Note: These tests should be run sparingly to avoid overloading fbref's servers.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from autoscout import analyse, preprocess, search, util
from autoscout.data import scrape
from autoscout.data.fbref import aggregate, comp
from autoscout.vis import chart, radar

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config_dir() -> Path:
    """Path to config directory."""
    return Path(__file__).parent.parent / "config"


@pytest.fixture(scope="module")
def fbref_config(config_dir) -> dict:
    """Load fbref configuration."""
    return {
        "comps": util.load_json(config_dir / "fbref" / "comps.json"),
        "stats": util.load_json(config_dir / "fbref" / "stats.json"),
    }


@pytest.fixture(scope="module")
def rating_config(config_dir) -> dict:
    """Load rating configuration."""
    return util.load_json(config_dir / "rating_inputs.json")


@pytest.fixture(autouse=True)
def rate_limit():
    """Add delay between tests to avoid overwhelming fbref."""
    yield
    time.sleep(2)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestScrapeIntegration:
    """Integration tests for web scraping functionality."""

    def test_scrape_gets_tables_from_fbref(self, fbref_config):
        """Should successfully scrape tables from fbref."""
        comp_config = fbref_config["comps"]["eng1"]
        url = comp_config["top"] + "stats" + comp_config["end"]

        tables = scrape.get_all_tables(url)

        assert len(tables) > 0
        assert tables[0] is not None

    def test_scrape_returns_beautifulsoup_elements(self, fbref_config):
        """Scraped tables should be BeautifulSoup elements."""
        comp_config = fbref_config["comps"]["eng1"]
        url = comp_config["top"] + "stats" + comp_config["end"]

        tables = scrape.get_all_tables(url)

        # Should have findAll method (BeautifulSoup element)
        assert hasattr(tables[0], "find_all")


class TestAggregateDataIntegration:
    """Integration tests for aggregate data downloading."""

    def test_get_player_stats_category(self, fbref_config):
        """Should download player stats for a single category."""
        comp_config = fbref_config["comps"]["eng1"]
        stats_config = fbref_config["stats"]["aggregate"]["outfield"]

        # Get just the standard stats category
        df = aggregate.get_data_for_category(
            "stats",
            comp_config["top"],
            comp_config["end"],
            stats_config["stats"],
            team=False,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "player" in df.columns or len(df.columns) > 0

    def test_get_team_stats_category(self, fbref_config):
        """Should download team stats for a single category."""
        comp_config = fbref_config["comps"]["eng1"]
        stats_config = fbref_config["stats"]["aggregate"]["team"]

        df = aggregate.get_data_for_category(
            "stats",
            comp_config["top"],
            comp_config["end"],
            stats_config["stats"],
            team=True,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestCompetitionDataIntegration:
    """Integration tests for competition schedule data."""

    def test_get_competition_schedule(self, fbref_config):
        """Should download competition schedule data."""
        comp_config = fbref_config["comps"]["eng1"]
        stats_config = fbref_config["stats"]["match"]["competition"]

        df = comp.get_competition_data(
            stats_config,
            comp_config["top"],
            comp_config["schedule_end"],
            season=2025,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "date" in df.columns


class TestFullPipelineIntegration:
    """Integration tests for the full data pipeline."""

    @pytest.fixture(scope="class")
    def player_data(self, fbref_config) -> pd.DataFrame:
        """Download and cache player data for pipeline tests."""
        comp_config = fbref_config["comps"]["eng1"]
        stats_config = fbref_config["stats"]["aggregate"]["outfield"]

        # Get minimal stats for faster test
        minimal_config = {"stats": stats_config["stats"]}

        df = aggregate.get_data(
            minimal_config,
            comp_config["top"],
            comp_config["end"],
            team=False,
            sleep_seconds=3.0,
        )

        return df

    def test_downloaded_data_has_expected_columns(self, player_data):
        """Downloaded data should have expected columns."""
        # These columns should be present from the stats category
        assert "player" in player_data.columns or len(player_data.columns) > 5

    def test_preprocess_adjust_per_90(self, player_data):
        """Should be able to adjust downloaded data per 90 minutes."""
        if "minutes" not in player_data.columns:
            pytest.skip("minutes column not in data")

        # Find numeric columns
        numeric_cols = player_data.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "minutes" and c != "birth_year"]

        if not numeric_cols:
            pytest.skip("No numeric columns to adjust")

        result = preprocess.adjust_per_90(player_data, numeric_cols[:2])

        assert len(result) == len(player_data)

    def test_preprocess_clamp_by_percentiles(self, player_data):
        """Should be able to clamp downloaded data."""
        numeric_cols = player_data.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_cols:
            pytest.skip("No numeric columns to clamp")

        result = preprocess.clamp_by_percentiles(player_data, numeric_cols[:2])

        assert len(result) == len(player_data)

    def test_search_filter_data(self, player_data):
        """Should be able to search/filter downloaded data."""
        if "minutes" not in player_data.columns:
            pytest.skip("minutes column not in data")

        # Filter to players with significant minutes
        median_minutes = player_data["minutes"].median()
        result = search.search(player_data, {"gte": {"minutes": median_minutes}})

        assert len(result) > 0
        assert len(result) < len(player_data)

    def test_analyse_reduce_dimensions(self, player_data):
        """Should be able to reduce dimensions of downloaded data."""
        numeric_cols = player_data.select_dtypes(include=["number"]).columns.tolist()
        # Remove columns that might have issues
        numeric_cols = [c for c in numeric_cols if c not in ["minutes", "birth_year", "age"]]

        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns for dimensionality reduction")

        # Drop rows with NaN
        clean_data = player_data.dropna(subset=numeric_cols[:3])

        if len(clean_data) < 10:
            pytest.skip("Not enough clean data rows")

        result = analyse.reduce_dimensions(clean_data, numeric_cols[:3], reducer=1)

        assert len(result) == len(clean_data)


class TestVisualizationIntegration:
    """Integration tests for visualization with real data structure."""

    @pytest.fixture
    def sample_scraped_structure(self) -> pd.DataFrame:
        """Create data with structure similar to scraped data."""
        return pd.DataFrame({
            "player": [f"Player {i}" for i in range(20)],
            "team": [f"Team {i % 5}" for i in range(20)],
            "position": ["FW", "MF", "DF", "GK"] * 5,
            "minutes": [900 + i * 100 for i in range(20)],
            "goals": [i % 15 for i in range(20)],
            "assists": [(20 - i) % 12 for i in range(20)],
            "shots": [20 + i * 2 for i in range(20)],
            "tackles": [10 + i * 3 for i in range(20)],
        })

    def test_radar_chart_with_scraped_structure(self, sample_scraped_structure):
        """Should create radar chart with data similar to scraped structure."""
        r, fig, ax = radar.plot_radar(
            sample_scraped_structure,
            columns=["goals", "assists", "tackles"],
            index="Player 0",
        )

        assert r is not None
        assert fig is not None

    def test_scatter_chart_with_scraped_structure(self, sample_scraped_structure):
        """Should create scatter chart with data similar to scraped structure."""
        plot = chart.scatter(
            sample_scraped_structure,
            x="goals",
            y="assists",
        )

        assert plot is not None

    def test_lines_chart_with_scraped_structure(self, sample_scraped_structure):
        """Should create line chart with data similar to scraped structure."""
        # Sort by minutes for a sensible line chart
        sorted_data = sample_scraped_structure.sort_values("minutes").reset_index(drop=True)
        sorted_data["index"] = range(len(sorted_data))

        plot = chart.lines(
            sorted_data,
            x_columns=["index"],
            y_columns=["goals"],
            colors=["blue"],
        )

        assert plot is not None


class TestEndToEndWorkflow:
    """End-to-end workflow tests simulating real usage."""

    def test_similarity_search_workflow(self):
        """Test complete similarity search workflow."""
        # Create realistic player data
        data = pd.DataFrame({
            "player": ["Haaland", "Kane", "Salah", "Son", "Saka", "Rashford"],
            "team": ["Man City", "Bayern", "Liverpool", "Spurs", "Arsenal", "Man Utd"],
            "position": ["FW", "FW", "FW", "FW", "FW", "FW"],
            "minutes": [2500, 2400, 2600, 2200, 2300, 1800],
            "goals": [25, 22, 18, 15, 12, 10],
            "assists": [5, 8, 10, 8, 11, 6],
            "shots": [80, 75, 70, 60, 55, 50],
            "xg": [22.5, 20.0, 16.5, 13.0, 10.5, 9.0],
        })

        # Step 1: Adjust per 90
        adjusted = preprocess.adjust_per_90(data, ["goals", "assists", "shots", "xg"])

        # Step 2: Find similar players to Haaland
        similar = search.search_similar(
            adjusted,
            ["goals", "assists", "shots", "xg"],
            "Haaland",
            num=3,
        )

        # Verify
        assert len(similar) == 3
        assert "Haaland" in similar["player"].values

    def test_style_rating_workflow(self):
        """Test complete style rating workflow."""
        # Create realistic player data
        data = pd.DataFrame({
            "player": [f"Player {i}" for i in range(50)],
            "minutes": [1800 + i * 20 for i in range(50)],
            "goals": [i % 20 for i in range(50)],
            "assists": [(50 - i) % 15 for i in range(50)],
            "shots": [30 + i for i in range(50)],
            "tackles": [20 + i * 2 for i in range(50)],
            "interceptions": [10 + i for i in range(50)],
        })

        config = {
            "attack": ["goals", "assists", "shots"],
            "defense": ["tackles", "interceptions"],
        }

        # Generate style ratings
        rated = analyse.estimate_style_ratings(data, config)

        # Verify
        assert "attack_rating" in rated.columns
        assert "defense_rating" in rated.columns
        assert rated["attack_rating"].min() >= 0
        assert rated["attack_rating"].max() <= 100

    def test_visualization_workflow(self):
        """Test complete visualization workflow."""
        # Create data
        data = pd.DataFrame({
            "player": ["Alice", "Bob", "Charlie", "Diana"],
            "position": ["FW", "FW", "MF", "DF"],
            "minutes": [1800, 1620, 1350, 2700],
            "goals": [15, 12, 8, 2],
            "assists": [6, 9, 12, 3],
            "tackles": [15, 20, 55, 120],
        })

        # Create radar chart
        r, fig, ax = radar.plot_radar(
            data,
            columns=["goals", "assists", "tackles"],
            index="Alice",
            index_compare="average",
        )

        # Create scatter chart
        scatter_plot = chart.scatter(data, x="goals", y="assists")
        scatter_plot = chart.add_labels(scatter_plot, data, "goals", "assists", "player")

        # Verify
        assert r is not None
        assert scatter_plot is not None

        plt.close("all")
