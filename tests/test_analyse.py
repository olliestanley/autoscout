"""
Unit tests for autoscout.analyse module.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from autoscout import analyse


class TestEstimateStyleRatings:
    """Tests for analyse.estimate_style_ratings function."""

    def test_estimate_style_ratings_creates_rating_columns(
        self, large_player_data, sample_rating_config
    ):
        """Should create rating columns based on config."""
        result = analyse.estimate_style_ratings(large_player_data, sample_rating_config)

        assert "attack_rating" in result.columns
        assert "defense_rating" in result.columns

    def test_estimate_style_ratings_range(
        self, large_player_data, sample_rating_config
    ):
        """Ratings should be in 0-100 range."""
        result = analyse.estimate_style_ratings(large_player_data, sample_rating_config)

        assert result["attack_rating"].min() >= 0
        assert result["attack_rating"].max() <= 100
        assert result["defense_rating"].min() >= 0
        assert result["defense_rating"].max() <= 100

    def test_estimate_style_ratings_preserves_original_columns(
        self, large_player_data, sample_rating_config
    ):
        """Should preserve original columns."""
        original_columns = list(large_player_data.columns)
        result = analyse.estimate_style_ratings(large_player_data, sample_rating_config)

        for col in original_columns:
            assert col in result.columns

    def test_estimate_style_ratings_does_not_modify_original(
        self, large_player_data, sample_rating_config
    ):
        """Should not modify original DataFrame."""
        original_shape = large_player_data.shape

        analyse.estimate_style_ratings(large_player_data, sample_rating_config)

        assert large_player_data.shape == original_shape
        assert "attack_rating" not in large_player_data.columns


class TestReduceDimensions:
    """Tests for analyse.reduce_dimensions function."""

    def test_reduce_dimensions_default_1d(self, large_player_data):
        """Should reduce to 1 dimension by default."""
        result = analyse.reduce_dimensions(
            large_player_data,
            ["goals", "assists", "shots"],
            reducer=1,
        )

        assert result.shape == (len(large_player_data), 1)

    def test_reduce_dimensions_custom_dimensions(self, large_player_data):
        """Should reduce to specified number of dimensions."""
        result = analyse.reduce_dimensions(
            large_player_data,
            ["goals", "assists", "shots", "tackles"],
            reducer=2,
        )

        assert result.shape == (len(large_player_data), 2)

    def test_reduce_dimensions_with_fitted_estimator(self, large_player_data):
        """Should use provided fitted estimator."""
        pca = PCA(n_components=2)
        columns = ["goals", "assists", "shots"]
        pca.fit(large_player_data[columns])

        result = analyse.reduce_dimensions(large_player_data, columns, reducer=pca)

        assert result.shape == (len(large_player_data), 2)

    def test_reduce_dimensions_returns_numpy_array(self, large_player_data):
        """Should return numpy array."""
        result = analyse.reduce_dimensions(
            large_player_data,
            ["goals", "assists"],
            reducer=1,
        )

        assert isinstance(result, np.ndarray)


class TestFitPca:
    """Tests for analyse.fit_pca function."""

    def test_fit_pca_returns_pca_estimator(self, large_player_data):
        """Should return PCA estimator."""
        result = analyse.fit_pca(
            large_player_data,
            ["goals", "assists", "shots"],
            out_dimensions=2,
        )

        assert isinstance(result, PCA)

    def test_fit_pca_correct_components(self, large_player_data):
        """Should have correct number of components."""
        result = analyse.fit_pca(
            large_player_data,
            ["goals", "assists", "shots"],
            out_dimensions=2,
        )

        assert result.n_components == 2

    def test_fit_pca_is_fitted(self, large_player_data):
        """Returned PCA should be fitted."""
        result = analyse.fit_pca(
            large_player_data,
            ["goals", "assists", "shots"],
            out_dimensions=1,
        )

        # Fitted PCA has components_ attribute
        assert hasattr(result, "components_")
        assert result.components_ is not None


class TestClusterRecords:
    """Tests for analyse.cluster_records function."""

    def test_cluster_records_returns_array(self, large_player_data):
        """Should return array of cluster assignments."""
        # Use explicit n_clusters to avoid elbow test with small data issues
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(large_player_data[["goals", "assists", "shots"]])

        result = analyse.cluster_records(
            large_player_data,
            ["goals", "assists", "shots"],
            estimator=kmeans,
        )

        assert isinstance(result, np.ndarray)
        assert len(result) == len(large_player_data)

    def test_cluster_records_correct_length(self, large_player_data):
        """Should return one cluster assignment per row."""
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(large_player_data[["goals", "assists"]])

        result = analyse.cluster_records(
            large_player_data,
            ["goals", "assists"],
            estimator=kmeans,
        )

        assert len(result) == len(large_player_data)

    def test_cluster_records_cluster_values(self, large_player_data):
        """Cluster assignments should be non-negative integers."""
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(large_player_data[["goals", "assists"]])

        result = analyse.cluster_records(
            large_player_data,
            ["goals", "assists"],
            estimator=kmeans,
        )

        assert all(result >= 0)
        assert all(result < 4)  # 4 clusters


class TestFitKmeans:
    """Tests for analyse.fit_kmeans function."""

    def test_fit_kmeans_returns_kmeans_estimator(self, large_player_data):
        """Should return KMeans estimator."""
        result = analyse.fit_kmeans(
            large_player_data,
            ["goals", "assists", "shots"],
            n_clusters=3,
        )

        assert isinstance(result, KMeans)

    def test_fit_kmeans_correct_clusters(self, large_player_data):
        """Should have correct number of clusters."""
        result = analyse.fit_kmeans(
            large_player_data,
            ["goals", "assists", "shots"],
            n_clusters=5,
        )

        assert result.n_clusters == 5

    def test_fit_kmeans_is_fitted(self, large_player_data):
        """Returned KMeans should be fitted."""
        result = analyse.fit_kmeans(
            large_player_data,
            ["goals", "assists", "shots"],
            n_clusters=3,
        )

        # Fitted KMeans has cluster_centers_ attribute
        assert hasattr(result, "cluster_centers_")
        assert result.cluster_centers_ is not None

    def test_fit_kmeans_auto_clusters(self, large_player_data):
        """Should automatically determine cluster count with 'auto'."""
        result = analyse.fit_kmeans(
            large_player_data,
            ["goals", "assists", "shots"],
            n_clusters="auto",
        )

        assert isinstance(result, KMeans)
        assert result.n_clusters > 0
