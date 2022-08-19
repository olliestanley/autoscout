from typing import Any

import pandas as pd
import seaborn as sns


def scatter_with_labels(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    style: str = None,
    size: str = None,
    label: str = "player",
) -> Any:
    """
    Plot a scatter chart based on `data`, with axes `x` and `y`, including labels on
    the chart based on the column `label`.
    """

    plot = sns.scatterplot(
        x=x,
        y=y,
        hue=hue,
        style=style,
        size=size,
        data=data,
        legend=False,
    )

    for idx in data.index:
        plot.text(
            data.at[idx, x] + 0.01,
            data.at[idx, y],
            data.at[idx, label],
            horizontalalignment='left',
            size='medium',
            color='black',
        )

    return plot
