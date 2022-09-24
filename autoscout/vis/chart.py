from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
from bokeh.transform import factor_cmap


def lines(
    data: pd.DataFrame,
    x_columns: Sequence[str],
    y_columns: Sequence[str],
    colors: Sequence[str],
    trends: bool = False,
    vshade: Union[int, Tuple[int, int]] = None,
    legend_labels: Sequence[str] = None,
    **kwargs
) -> Figure:
    """
    Plot one or more lines on a single chart. Supports shading area between lines, and
    plotting trend lines along with the real lines.

    The data in `data[x_columns[0]]` will be plotted against `data[y_columns[0]]` using
    a line of color `colors[0]` and with legend label `legend_labels[0]`, and this is
    repeated to `len(x_columns) - 1`.

    Args:
        data: DataFrame containing all input data.
        x_columns: Column names for x variable(s).
        y_columns: Column names for y variable(s).
        colors: Colors for the lines drawn.
        trends: Plot trend lines as well as real lines.
        vshade: Shade area between the two lines at indices specified in this `tuple`.
            If this is `int`, it will shade the area beneath the single line at the
            index specified instead.
        legend_labels: Label for each line on the legend. Defaults to `y_columns`.
        **kwargs: Passed to bokeh figure. Can alter plot size, title, axis labels, etc.

    Returns:
        Bokeh Figure used for chart.
    """

    if legend_labels is None:
        legend_labels = y_columns

    plot: Figure = figure(**kwargs)

    source = ColumnDataSource(data)

    for x, y, col, leg in zip(x_columns, y_columns, colors, legend_labels):
        plot.line(x, y, color=col, source=source, legend_label=leg)

        if trends:
            regr = np.polyfit(data[x], data[y], deg=1, full=True)
            slope, intercept = regr[0][:2]
            trend = [intercept + slope * v for v in data[x]]
            plot.line(data[x], trend, color=col, line_dash="dashed")

    if isinstance(vshade, int):
        plot.varea(x=x_columns[vshade], y1=y_columns[vshade], y2=0, color=colors[vshade], source=source)

    elif isinstance(vshade, tuple):
        vs1_y, vs2_y = [y_columns[v] for v in vshade]
        vs1_x = x_columns[vshade[0]]

        compare = data[vs1_y] > data[vs2_y]

        diff = compare.diff().fillna(0).abs()
        change_indices = [0,] + [idx for idx, d in enumerate(diff) if d]

        for i, change_idx in enumerate(change_indices):
            x_start = data[vs1_x][change_idx]

            end_idx = (
                change_indices[i + 1]
                if i + 1 < len(change_indices)
                else len(data[vs1_x]) - 1
            )

            x_end = data[vs1_x][end_idx - 1]
            x_values = np.linspace(x_start, x_end, end_idx - change_idx)

            use_2 = compare[change_idx]
            plot.varea(
                x=x_values,
                y1=data[vs2_y if use_2 else vs1_y][change_idx:end_idx],
                y2=data[vs1_y if use_2 else vs2_y][change_idx:end_idx],
                color=colors[vshade[0] if use_2 else vshade[1]],
                fill_alpha=0.2,
            )

    return plot


def scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    size: str = None,
    color: str = None,
    palette: list = None,
    marker: str = "circle",
    **kwargs,
) -> Figure:
    """
    Plot a scatter chart based on `data`, with axes `x` and `y`.

    Args:
        data: DataFrame containing all input data.
        x: Column name for x variable.
        y: Column name for y variable.
        size: Column name determining size of scatter points.
        color: Column name determining color of scatter points.
        palette: Color palette for scatter points. Required if `color` is specified.
        marker: Column name determining marker of scatter points.
        **kwargs: Passed to bokeh figure. Can alter plot size, title, axis labels, etc.

    Returns:
        Bokeh Figure used for chart.
    """

    x_range = (data[x].min(), data[x].max() + 5)
    plot: Figure = figure(x_range=x_range, **kwargs)

    source = ColumnDataSource(data)

    scatter_args = {
        "x": x,
        "y": y,
        "size": size or 5,
        "marker": marker,
        "source": source,
    }

    if color:
        if not palette:
            raise ValueError("Must pass `palette` with `color`")

        cmap = factor_cmap(
            color, palette=palette, factors=data[color].unique()
        )

        scatter_args["color"] = cmap
        scatter_args["legend_field"] = color

    plot.scatter(**scatter_args)

    return plot


def add_means(
    plot: Figure,
    data: pd.DataFrame,
    x: str,
    y: str,
    epsilon: float = 1.0,
) -> Figure:
    """
    Add mean average lines for x and y to a plot.

    Args:
        plot: Existing Bokeh figure to add average lines to.
        data: DataFrame containing all input data.
        x: Column name for x variable.
        y: Column name for y variable.
        average: Type of average to use. Must be in ["mean", "median", "mode"].

    Returns:
        Bokeh Figure used for chart.
    """

    x_avgs = [data[x].mean()] * 2
    x_lims = [data[y].min(), data[y].max()]

    y_avgs = [data[y].mean()] * 2
    y_lims = [data[x].min(), data[x].max()]

    plot.line(x_avgs, x_lims, line_dash="dashed", color="black")
    plot.line(y_lims, y_avgs, line_dash="dashed", color="black")

    return plot


def add_labels(
    plot: Figure,
    data: pd.DataFrame,
    x: str,
    y: str,
    label: str = "player",
) -> Figure:
    """
    Add labels to a plot based on the column `label`. Scatter chart is interactive with
    a hover tool which displays `label` when hovered.

    Args:
        plot: Existing Bokeh figure to add labels to.
        data: DataFrame containing all input data.
        x: Column name for x variable.
        y: Column name for y variable.
        label: Column name determining drawn and hover tool label of scatter points.
        **kwargs: Passed to bokeh LabelSet.

    Returns:
        Bokeh Figure used for chart.
    """

    source = ColumnDataSource(data)

    plot.add_layout(
        LabelSet(
            x=x,
            y=y,
            text=label,
            y_offset=5,
            x_offset=-15,
            source=source,
            render_mode='canvas',
            text_font_size={"value": "12px"}
        )
    )

    plot.add_tools(HoverTool(
        tooltips=[("Name", f"@{label}")]
    ))

    return plot
