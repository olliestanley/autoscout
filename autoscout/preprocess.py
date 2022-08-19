"""
Load and preprocess player and team data from tabular (CSV) format.
"""

import itertools
from typing import Dict, Sequence, Union

import pandas as pd


def with_competition_column(
    data: pd.DataFrame,
    competition: str,
) -> pd.DataFrame:
    data = data.copy(deep=True)
    data["competition"] = competition
    return data


def combine_data(
    data: Sequence[pd.DataFrame],
    retain_nans=False,
) -> pd.DataFrame:
    """
    Combine tabular datasets by indices, for example to form a single DataFrame of
    datapoints from multiple competitions, or different seasons of one competition.

    Args:
        data: Sequence of individual datasets.
        retain_nans: Include columns which are not present in all individual datasets
            in the final output data.

    Return:
        Combined data.
    """

    data = pd.concat(data, axis=0, join="outer" if retain_nans else "inner")
    data = data.reset_index(drop=True)
    return data


def clamp_by_percentiles(
    data: pd.DataFrame,
    columns: Sequence[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Clamp values in `columns` within `data` to within percentiles `alpha` and
    `1 - alpha`.

    Args:
        data: DataFrame to clamp values within.
        columns: Columns to apply clamping to.
        alpha: Percentile to clamp at.

    Returns:
        DataFrame with selected columns clamped.
    """

    data = data.copy(deep=True)

    for column in columns:
        vals = data[column]

        data[column] = vals.clip(vals.quantile(alpha), vals.quantile(1 - alpha))

    return data


def adjust_per_90(
    data: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """
    Adjust selected columns from the given DataFrame to be per 90 minutes played.

    Args:
        data: DataFrame to adjust from.
        columns: Columns to apply adjustment to.

    Returns:
        Adjusted DataFrame.
    """

    data = data.copy(deep=True)
    data[columns] = data[columns].div(data.minutes, axis=0).mul(90, axis=0)
    return data


def filter_categories(
    data: pd.DataFrame,
    stats_config: Dict[str, Sequence[str]],
    categories: Union[str, Sequence[str]],
    retain: bool = True,
) -> pd.DataFrame:
    """
    Filter the statistics (columns) in a DataFrame based on their category.

    Args:
        data: DataFrame to filter.
        stats_config: Config defining categories and their associated statistics.
        categories: Categories to filter.
        retain: If `True`, keep statistics included in given categories. If `False`,
            drop statistics included in given categories.

    Returns:
        New DataFrame with filtering criteria applied.
    """

    stats = list(
        itertools.chain([v for k, v in stats_config.items() if k in categories])
    )

    return data[stats] if retain else data.drop(stats, axis=1)
