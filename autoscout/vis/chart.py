from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import seaborn as sns
from matplotlib.colors import Colormap


def scatter_with_labels(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    palette: Union[str, List, Dict, Colormap] = None,
    style: str = None,
    size: str = None,
    sizes: Union[Tuple[int, int], Dict, List] = (1, 10),
    label: str = "player",
    label_alpha: float = 1.0,
) -> Any:
    """
    Plot a scatter chart based on `data`, with axes `x` and `y`, including labels on
    the chart based on the column `label`.
    """

    plot = sns.scatterplot(
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        style=style,
        size=size,
        sizes=sizes,
        data=data,
    )

    for idx in data.index:
        xval = data.at[idx, x]
        yval = data.at[idx, y] + label_alpha

        plot.text(
            xval,
            yval,
            data.at[idx, label].split(" ")[-1],
            horizontalalignment='left',
            size='x-large',
            color='black',
        )

    return plot
