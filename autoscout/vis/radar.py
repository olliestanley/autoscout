from typing import Sequence, Tuple, Union

import matplotlib as mpl
import pandas as pd
from mplsoccer import Radar

from autoscout.vis import constant


def plot_radar(
    data: pd.DataFrame,
    columns: Sequence[str],
    index: Union[str, int],
    columns_display: Sequence[str] = None,
) -> Tuple[Radar, mpl.figure.Figure, mpl.axes.Axes]:
    if columns_display:
        mapper = dict(zip(columns, columns_display))
        data = data.rename(mapper, axis=1)
        columns = columns_display

    # Set radar bounds based on quantiles of all rows in the DataFrame
    radar = Radar(
        params=columns,
        min_range=data[columns].quantile(0.05),
        max_range=data[columns].quantile(0.95),
    )

    # Now quantiles are calculated, extract the desired row
    if isinstance(index, str):
        player = "player" in data.columns
        data = data[data["player" if player else "team"] == index]
    else:
        data = data.iloc[index]

    # Keep only the columns we want to plot
    values: pd.Series = data[columns].squeeze()

    fig, ax = radar.setup_axis()

    radar.draw_circles(
        ax=ax, facecolor=constant.RADAR_COLOURS[0], edgecolor=constant.RADAR_COLOURS[1]
    )
    radar.draw_radar(
        values,
        ax=ax,
        kwargs_radar={"facecolor": constant.RADAR_COLOURS[2], "alpha": 0.9},
        kwargs_rings={"facecolor": constant.RADAR_COLOURS[3], "alpha": 0.7},
    )
    radar.draw_range_labels(ax=ax)
    radar.draw_param_labels(ax=ax)

    return radar, fig, ax
