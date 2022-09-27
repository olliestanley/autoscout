from typing import Any, Dict, Sequence, Tuple, Union

import matplotlib as mpl
import pandas as pd
from mplsoccer import Radar

from autoscout import preprocess
from autoscout.util import get_record
from autoscout.vis import constant


def estimate_limits_by_position(
    data: pd.DataFrame, column: str, position: str, alpha: float = 0.1
) -> Tuple[float, float]:
    """
    Estimate good outer limits on a radar chart for the given `column` by considering
    percentiles of players in the relevant `position`. For example, it is normal to set
    radar limits as the 10th and 90th percentile of the statistic for players in the
    same position as the subject of the radar.

    Args:
        data: Data for all players to consider.
        column: Column to consider percentiles for.
        position: Position to consider players in when calculating percentiles.
        alpha: Determines percentiles to consider. `alpha` of 0.1 results in 10th and
            90th percentiles.

    Returns:
        Tuple of percentiles, `alpha` and `1 - alpha`.
    """

    data = data[data["position"].str.startswith(position)]
    values = data[column].squeeze()
    return values.quantile(alpha), values.quantile(1 - alpha)


def plot_radar_from_config(
    data: pd.DataFrame,
    config: Dict[str, Any],
    index: Union[str, int],
    index_compare: Union[str, int] = None,
    **kwargs,
) -> Tuple[Radar, mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot a radar chart in `matplotlib` for a single row (team or player) in `data`,
    using specified `columns`. Radar limits are set by 5th and 95th percentile of the
    `columns` over the whole DataFrame.

    Args:
        data: Full DataFrame.
        config: Dict specifying the configuration for the radar chart.
        index: Index of the row to plot data for. Can be integer index, or name of
            player or team.
        index_compare: Index of the second row to plot data for. Follows the same
            behaviour as `index`, but can be `None` for a non-comparison radar.
        **kwargs: Passed to `mplsoccer.Radar.__init__()`.

    Returns:
        Tuple of Radar, PyPlot Figure, and PyPlot Axes.
    """

    columns, display, lib, mins, maxes, normalize = [], [], [], [], [], []

    for stat, stat_config in config["columns"].items():
        columns.append(stat)
        display.append(stat_config["display"])
        mins.append(stat_config["low"])
        maxes.append(stat_config["high"])

        if stat_config["lower_is_better"]:
            lib.append(stat)
        if stat_config["normalize"]:
            normalize.append(stat)

    if normalize:
        data = preprocess.adjust_per_90(data, normalize)

    return plot_radar(
        data,
        columns,
        index,
        index_compare=index_compare,
        columns_display=display,
        lower_is_better=lib,
        min_values=mins,
        max_values=maxes,
        **kwargs,
    )


def plot_radar(
    data: pd.DataFrame,
    columns: Sequence[str],
    index: Union[str, int],
    index_compare: Union[str, int] = None,
    columns_display: Sequence[str] = None,
    lower_is_better: Sequence[str] = None,
    min_values: Union[Sequence[float], str] = "auto",
    max_values: Union[Sequence[float], str] = "auto",
    **kwargs,
) -> Tuple[Radar, mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot a radar chart in `matplotlib` for a single row (team or player) in `data`,
    using specified `columns`. Radar limits are set by 5th and 95th percentile of the
    `columns` over the whole DataFrame.

    Args:
        data: Full DataFrame.
        columns: Names of columns to include in the chart.
        index: Index of the row to plot data for. Can be integer index, or name of
            player or team.
        index_compare: Index of the second row to plot data for. Follows the same
            behaviour as `index`, but can be `None` for a non-comparison radar.
        columns_display: Display names to replace column names with on the chart.
        lower_is_better: Names of columns for which lower should be considered better
            for the radar chart.
        min_values: Min value to display on the chart for each column.
        max_values: Max value to display on the chart for each column.
        **kwargs: Passed to `mplsoccer.Radar.__init__()`.

    Returns:
        Tuple of Radar, PyPlot Figure, and PyPlot Axes.
    """

    if columns_display:
        mapper = dict(zip(columns, columns_display))
        data = data.rename(mapper, axis=1)
        columns = columns_display

        if lower_is_better:
            lower_is_better = [mapper[v] for v in lower_is_better]

    if min_values == "auto":
        min_values = data[columns].quantile(0.05)
    if max_values == "auto":
        max_values = data[columns].quantile(0.95)

    radar = Radar(
        params=columns,
        min_range=min_values,
        max_range=max_values,
        lower_is_better=lower_is_better,
        **kwargs,
    )

    fig, ax = radar.setup_axis(figsize=(10, 10))

    radar.draw_circles(
        ax=ax, facecolor=constant.RADAR_COLOURS[0], edgecolor=constant.RADAR_COLOURS[1]
    )

    # Keep only the columns we want to plot
    values = get_record(data, index)[columns].squeeze()

    if index_compare:
        values_compare = get_record(data, index_compare)[columns].squeeze()

        radar.draw_radar_compare(
            values,
            values_compare,
            ax=ax,
            kwargs_radar={"facecolor": constant.RADAR_COLOURS[2], "alpha": 0.5},
            kwargs_compare={"facecolor": constant.RADAR_COLOURS[4], "alpha": 0.5},
        )

    else:
        radar.draw_radar(
            values,
            ax=ax,
            kwargs_radar={"facecolor": constant.RADAR_COLOURS[2], "alpha": 0.5},
            kwargs_rings={"facecolor": constant.RADAR_COLOURS[3], "alpha": 0.5},
        )

    radar.draw_range_labels(ax=ax)
    radar.draw_param_labels(ax=ax, offset=0.5)

    return radar, fig, ax
