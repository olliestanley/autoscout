"""
Apply algorithms to data to identify patterns and insights.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from autoscout.util import min_max_scale


def reduce_dimensions(
    data: pd.DataFrame,
    columns: Sequence[str],
    reducer: Union[BaseEstimator, int] = 1,
) -> np.ndarray:
    """
    Reduce the dimensions specified by `columns` in `data`. A fitted estimator can be
    used for reduction. Otherwise, automatically fitting PCA is supported.

    Input `data` is min-max scaled but not adjusted per-90 minutes prior to reduction.

    Args:
        data: DataFrame containing input data.
        columns: Names of columns to use as features.
        reducer: Existing reducer. If an `int`, a PCA model is fitted automatically
            with the provided number of output dimensions.

    Returns:
        Array of reduced dimensionality feature values, in order of rows in `data`.
        To include this as a column: `data["cluster"] = reduce_dimensions(data, ...)`.
    """

    data = min_max_scale(data, columns)

    if isinstance(reducer, int):
        reducer = fit_pca(data, columns, reducer)

    return reducer.transform(data[columns])


def fit_pca(
    data: pd.DataFrame, columns: Sequence[str], out_dimensions: int = 1
) -> KMeans:
    """
    Fit a PCA dimensionality reduction model to `data`, using `columns` as features.

    Data is not preprocessed. It may be useful to min-max scale the data or adjust
    per 90 minutes prior to passing it to this function.

    Args:
        data: DataFrame containing data to fit estimator.
        columns: Names of columns to use as clustering features.
        out_dimensions: Number of output dimensions for the reduction.

    Returns:
        Fitted PCA estimator.
    """

    estimator = PCA(n_components=out_dimensions)
    estimator.fit(data[columns])
    return estimator


def cluster_records(
    data: pd.DataFrame,
    columns: Sequence[str],
    estimator: Union[BaseEstimator, str] = "auto",
) -> np.ndarray:
    """
    Cluster or classify the records in `data`, using `columns` as features. A fitted
    `estimator` from SciKit-Learn can be passed for classification or clustering with
    an already prepared model. Otherwise, automatically fitting KMeans is supported.

    Input `data` is min-max scaled but not adjusted per-90 minutes, prior to being
    clustered. It is possible, and sometimes beneficial, to apply clustering to data
    which has already been dimensionality reduced. To do this, include the outputs of
    dimensionality reduction as columns in `data`, then pass them as `columns`.

    Args:
        data: DataFrame containing input data.
        columns: Names of columns to use as features.
        estimator: Existing estimator. If "auto", as default, a KMeans estimator is
            fitted automatically.

    Returns:
        Array of cluster assignments, in order of rows in `data`. To include this as a
        column of `data`, use: `data["cluster"] = cluster_records(data, ...)`.
    """

    data = min_max_scale(data, columns)

    if estimator == "auto":
        estimator = fit_kmeans(data, columns)

    return estimator.predict(data[columns])


def fit_kmeans(
    data: pd.DataFrame, columns: Sequence[str], n_clusters: Union[int, str] = "auto"
) -> KMeans:
    """
    Fit a KMeans cluster estimator to `data`, using `columns` as features. Supports
    automatic estimation of number of clusters (`k`).

    Data is not preprocessed. It may be useful to min-max scale the data or adjust
    per 90 minutes prior to passing it to this function.

    Args:
        data: DataFrame containing data to fit estimator.
        columns: Names of columns to use as clustering features.
        n_clusters: Number of clusters to create. If "auto", as default, the number
            will be estimated via an elbow test.

    Returns:
        Fitted KMeans estimator.
    """

    estimator = KMeans()

    if n_clusters == "auto":
        n_clusters = _select_k_by_elbow_test(estimator, data[columns])

    estimator.n_clusters = n_clusters
    estimator.fit(data[columns])
    return estimator


def _select_k_by_elbow_test(
    estimator: KMeans,
    data: ArrayLike,
    k_values: Sequence[int] = range(3, 30),
    relative: bool = True,
) -> int:
    """
    Estimate the best number of clusters (k) for KMeans clustering, based on `data`
    and `estimator`. The strength of each possible k in `k_values` is assessed based
    on the inertia value. See `_get_inertia()` for an explanation of this value.

    An elbow test is used to estimate the strength of each cluster count. This is not
    the only method and may not suit all datasets. It is intended to support other
    estimation methods in the future.

    Args:
        estimator: Estimator, usually KMeans, to estimate optimal k for.
        data: Data to estimate optimal k for.
        k_values: All possible values of k to consider. More values may lead to longer
            runtimes with large volumes of data, but provides greater certainty.
        relative: Assess cluster relative, rather than absolute, strength. This adds
            what is in effect regularisation to cluster count selection, penalising
            larger numbers of clusters by dividing the strength value of all cluster
            counts by the count itself.

    Returns:
        Optimal k value for KMeans clustering using elbow test.
    """

    clusters = np.array(k_values)
    ssd = list(map(lambda k: _get_inertia(estimator, data, k), k_values))

    dx = -np.diff(ssd)
    dx2 = -np.diff(dx)
    dx = np.insert(dx, 0, [0])
    dx2 = np.insert(dx2, 0, [0, 0])

    strength = np.append(np.roll(dx2 - dx, -1)[:-1], [0])
    if not np.any(strength > 0):
        return 0
    if relative:
        return clusters[np.argmax(strength / clusters)]
    return clusters[np.argmax(strength)]


def _get_inertia(estimator: KMeans, data: ArrayLike, k: int) -> float:
    """
    Get the intertia value for `estimator` when fitted to `data` with `k` clusters.
    Inertia is also known as within cluster sum of squared distances, meaning the sum
    of squared distances of data points from their nearest cluster centroid. This can
    be used to assess the goodness of fit of a clustering model.

    Args:
        estimator: Estimator, usually KMeans, to get inertia for.
        data: Data to fit `estimator` to for getting inertia.
        k: Number of clusters to use.

    Returns:
        Inertia value.
    """

    estimator.n_clusters = k
    estimator.fit(data)
    return estimator.inertia_
