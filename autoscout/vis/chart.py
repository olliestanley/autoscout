import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure


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

    if size:
        data["plot_size"] = data[size] / 5

    source = ColumnDataSource(data)

    plot: Figure = figure(
        plot_width=1000, plot_height=600, title=title, x_range=x_range
    )

    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label

    plot.circle(
        x=x,
        y=y,
        size="plot_size" if size else None,
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
