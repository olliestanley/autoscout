"""
Load and preprocess player and team data from tabular (CSV) format.
"""

import itertools
from typing import Dict, Sequence, Union
import pandas as pd


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
