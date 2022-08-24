"""
Load and preprocess player and team data from tabular (CSV) format.
"""

import itertools
from typing import Callable, Dict, Sequence, Union

import numpy as np
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


def adjust_possession_def(
    targets: np.ndarray, possessions: np.ndarray
) -> np.ndarray:
    """
    Adjust `targets` for `possessions`, where targets is a matrix of defensive stats.
    Defensive stats are harder to accrue with more possession, so the output values are
    larger than the inputs when possession is above 50 and lower when it is below 50.

    See `adjust_possession` for the easiest way to apply this to a `DataFrame`.

    Args:
        targets: Data to adjust for possession, 2D numbers.
        possessions: Relevant possession values, 1D numbers in the range [0, 100].

    Returns:
        Possession-adjusted version of `targets`.
    """

    p_delta = possessions - 50
    exp = np.expand_dims(np.power(np.e, -0.1 * p_delta), 1)
    adjusted = targets * (2 / (1 + exp))
    return adjusted


def adjust_possession(
    data_player: pd.DataFrame,
    data_team: pd.DataFrame,
    columns: Sequence[str],
    adjust_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = adjust_possession_def,
) -> pd.DataFrame:
    """
    Adjust selected columns from the given DataFrame for the level of possession
    enjoyed by the relevant team. This is designed to adjust player statistics, as it
    is fairly trivial to adjust team statistics. Match level data, not aggregate, is
    required.

    This function is dependent on lining up the correct player and team datasets. It
    will match a game in `data_player` to a game in `data_time` via two checks. First,
    `date` column being equal. Second, `squad` in `data_player` being equal to `name`
    in `data_team`. So `date` must be present in both DataFrames, `squad` must be in
    `data_player`, and `name` must be in `data_team` for this function to work.

    Args:
        data_player: DataFrame of player data where rows are matches played by the
            player.
        data_team: DataFrame of team data where rows are matches played by the team.
        columns: Columns to apply adjustments to.
        adjust_fn: Define the exact transformation to use as a possession adjustment.

    Returns:
        Adjusted DataFrame.
    """

    data_player = data_player.copy(deep=True)

    team_possessions = data_player.apply(
        lambda row: _get_team_possession(row, data_team), axis=1
    )

    padj_columns = [f"padj_{col}" for col in columns]

    data_player[padj_columns] = adjust_fn(
        data_player[columns].to_numpy(), team_possessions.to_numpy()
    )

    return data_player


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


def _get_team_possession(player_match: pd.Series, team_data: pd.DataFrame) -> float:
    team_filtered = team_data[team_data["date"] == player_match["date"]]
    team_filtered = team_filtered[team_filtered["name"] == player_match["squad"]]
    return float(team_filtered["possession"])
