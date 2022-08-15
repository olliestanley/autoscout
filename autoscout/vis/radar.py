from typing import Dict, Sequence, Tuple, Union

import matplotlib as mpl
import pandas as pd
from mplsoccer import Radar

from autoscout import preprocess
from autoscout.vis import constant


def plot_radar_from_config(
    data: pd.DataFrame,
    config: Dict[str, Sequence[str]],
    index: Union[str, int],
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
        **kwargs: Passed to `mplsoccer.Radar.__init__()`.

    Returns:
        Tuple of Radar, PyPlot Figure, and PyPlot Axes.
    """

    if "normalize" in config and len(config["normalize"]):
        data = preprocess.adjust_per_90(data, config["normalize"])

    return plot_radar(
        data,
        config["columns"],
        index,
        config["display"],
        config["lower_is_better"],
        **kwargs,
    )


def plot_radar(
    data: pd.DataFrame,
    columns: Sequence[str],
    index: Union[str, int],
    columns_display: Sequence[str] = None,
    lower_is_better: Sequence[str] = None,
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
        columns_display: Display names to replace column names with on the chart.
        lower_is_better: Names of columns for which lower should be considered better
            for the radar chart.
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

    # Set radar bounds based on quantiles of all rows in the DataFrame
    radar = Radar(
        params=columns,
        min_range=data[columns].quantile(0.05),
        max_range=data[columns].quantile(0.95),
        lower_is_better=lower_is_better,
        **kwargs,
    )

    # Now quantiles are calculated, extract the desired row
    if isinstance(index, str):
        player = "player" in data.columns
        data = data[data["player" if player else "team"] == index]
    else:
        data = data.iloc[index]

    # Keep only the columns we want to plot
    values: pd.Series = data[columns].squeeze()

    fig, ax = radar.setup_axis(figsize=(10, 10))

    radar.draw_circles(
        ax=ax, facecolor=constant.RADAR_COLOURS[0], edgecolor=constant.RADAR_COLOURS[1]
    )
    radar.draw_radar(
        values,
        ax=ax,
        kwargs_radar={"facecolor": constant.RADAR_COLOURS[2], "alpha": 0.5},
        kwargs_rings={"facecolor": constant.RADAR_COLOURS[3], "alpha": 0.5},
    )
    radar.draw_range_labels(ax=ax)
    radar.draw_param_labels(ax=ax, offset=0.5)

    return radar, fig, ax
