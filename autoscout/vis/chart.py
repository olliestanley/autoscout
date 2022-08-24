from typing import Sequence, Tuple, Union

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
    vshade: Union[int, Tuple[int, int]] = None,
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

    if isinstance(vshade, int):
        plot.varea(x=x_columns[vshade], y1=y_columns[vshade], y2=0, color=colors[vshade], source=source)

    elif isinstance(vshade, tuple):
        vs1_y, vs2_y = [y_columns[v] for v in vshade]
        vs1_x = x_columns[vshade[0]]

        compare = data[vs1_y] > data[vs2_y]

        diff = compare.diff().fillna(0).abs()
        change_indices = [idx for idx, d in enumerate(diff) if d]
        change_indices.insert(0, 0)

        for i, change_idx in enumerate(change_indices):
            x_start = data[vs1_x][change_idx]

            end_idx = (
                change_indices[i + 1]
                if i + 1 < len(change_indices)
                else len(data[vs1_x]) - 1
            )

            x_end = data[vs1_x][end_idx - 1]
            use_2 = compare[change_idx]

            num_values = end_idx - change_idx
            x_values = np.linspace(x_start, x_end, num_values)

            plot.varea(
                x=x_values,
                y1=data[vs2_y if use_2 else vs1_y][change_idx:end_idx],
                y2=data[vs1_y if use_2 else vs2_y][change_idx:end_idx],
                color=colors[vshade[0] if use_2 else vshade[1]],
                fill_alpha=0.2,
            )

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
        plot_width=1000, plot_height=600, x_range=x_range,
        title=title, x_axis_label=x_label, y_axis_label=y_label
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
