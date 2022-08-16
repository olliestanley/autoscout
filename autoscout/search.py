from typing import Dict, Sequence, Union

import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

from autoscout.util import get_record


def search(
    data: pd.DataFrame, criteria: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Search a DataFrame for rows which match numerical `criteria`.

    Example::

        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360

        >>> search(df, { "gte": { "angles": 1 },
                         "eq": { "degrees": 360 }})
                   angles  degrees
        rectangle       4      360

    Args:
        data: DataFrame to search within.
        criteria: Dict mapping operators (`gte`, `eq`, `lte`) to Dicts mapping column
            names to comparison values.
    """

    gte, eq, lte = [
        criteria.get(x, dict())
        for x in ("gte", "eq", "lte")
    ]

    for stat, value in gte.items():
        data = data[data[stat] >= value]

    for stat, value in eq.items():
        data = data[data[stat] == value]

    for stat, value in lte.items():
        data = data[data[stat] <= value]

    return data


def search_similar(
    data: pd.DataFrame, columns: Sequence[str], index: Union[str, int], num: int = 5
) -> pd.DataFrame:
    """
    Search `data` for `num` records which are most similar to the record at the given
    `index`, using `columns` as data points to assess similarity of. This may allow
    identifying players or teams with similar profiles.

    Similarity is assessed by Euclidean distance. Values are min-max scaled to reduce
    bias however they are not adjusted per-90 minutes. To achieve this, it is advised
    to use `autoscout.preprocess.adjust_per_90` first.

    Args:
        data: DataFrame to search within.
        columns: Sequence of column names in `data` which contain values to assess for
            similarity across rows.
        index: Index of the baseline row which is to be used to assess similarity
            against. See `autoscout.util.get_record()` for how this is interpreted.
        num: Number of most similar rows to output. Defaults to 5.

    Returns:
        DataFrame, containing `num` rows from `data` which are most similar to the row
        acquired by `index`.
    """

    data = data.copy(deep=True)
    scaler = MinMaxScaler()

    baseline = get_record(data, index)[columns]
    data_relevant = data[columns]

    data_relevant = pd.DataFrame(scaler.fit_transform(data_relevant), columns=columns)
    baseline = pd.DataFrame(scaler.transform(baseline), columns=columns).squeeze()

    return data.iloc[data_relevant.apply(
        lambda col: distance.euclidean(baseline, col), axis=1
    ).nsmallest(num).index.to_list()]
