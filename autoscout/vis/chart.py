from typing import Sequence

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure


def lines(
    data: pd.DataFrame,
    x_columns: Sequence[str],
    y_columns: Sequence[str],
    colors: Sequence[str],
    trends: bool = False,
    title=None,
    x_label=None,
    y_label=None,
    legend_labels: Sequence[str] = None,
) -> Figure:
    if legend_labels is None:
        legend_labels = y_columns

    plot: Figure = figure(
        plot_width=1000, plot_height=600, title=title,
        x_axis_label=x_label, y_axis_label=y_label
    )

    source = ColumnDataSource(data)

    for x, y, col, leg in zip(x_columns, y_columns, colors, legend_labels):
        plot.line(x, y, color=col, source=source, legend_label=leg)

        if trends:
            regr = np.polyfit(data[x], data[y], deg=1, full=True)
            slope, intercept = regr[0][:2]
            trend = [intercept + slope * v for v in data[x]]
            plot.line(data[x], trend, color=col, line_dash="dashed")

    return plot


def scatter_with_labels(
    data: pd.DataFrame,
    x: str,
    y: str,
    size: str = None,
    color: str = None,
    label: str = "player",
    title=None,
    x_label=None,
    y_label=None,
) -> Figure:
    """
    Plot a scatter chart based on `data`, with axes `x` and `y`, including labels on
    the chart based on the column `label`.
    """

    x_label = x_label or x
    y_label = y_label or y
    title = title or f"{x_label} vs {y_label}"

    x_range = (data[x].min(), data[x].max() + 5)

    plot: Figure = figure(
        plot_width=1000, plot_height=600, title=title, x_range=x_range, x_axis_label=x_label, y_axis_label=y_label
    )

    if size:
        data["plot_size"] = data[size] / 5

    source = ColumnDataSource(data)

    plot.circle(
        x=x,
        y=y,
        size="plot_size" if size else 4,
        color=color,
        source=source,
    )

    if label:
        plot.add_layout(
            LabelSet(
                x=x,
                y=y,
                text=label,
                y_offset=5,
                source=source,
                render_mode='canvas',
                text_font_size={"value": "12px"}
            )
        )

    plot.add_tools(HoverTool(
        tooltips=[("Name", f"@{label}")]
    ))

    return plot
