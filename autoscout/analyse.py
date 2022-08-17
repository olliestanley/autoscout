"""
Apply algorithms to data to identify patterns and insights.
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale


def cluster_records(
    data: pd.DataFrame,
    columns: Sequence[str],
    estimator: Union[BaseEstimator, str] = "auto",
) -> np.ndarray[int]:
    """
    Cluster or classify the records in `data`, using `columns` as features. A fitted
    `estimator` from SciKit-Learn can be passed for classification or clustering with
    an already prepared model. Otherwise, automatically fitting KMeans is supported.

    Input `data` is min-max scaled but not adjusted per-90 minutes, prior to being
    clustered.

    Args:
        data: DataFrame containing input data.
        columns: Names of columns to use as features.
        estimator: Existing estimator. If "auto", as default, a KMeans estimator is
            fitted automatically.

    Returns:
        Array of cluster assignments, in order of rows in `data`. To include this as a
        column of `data`, use: `data["cluster"] = cluster_records(data, ...)`.
    """

    data = data.copy(deep=True)
    data[columns] = pd.DataFrame(minmax_scale(data[columns]), columns=columns)

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
):
    clusters = np.array(k_values)
    ssd = map(lambda k: _get_inertia(estimator, data, k), k_values)

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


def _get_inertia(estimator: KMeans, data: ArrayLike, k: int):
    estimator.n_clusters = k
    estimator.fit(data)
    return estimator.inertia_
