from typing import Dict, Sequence, Union

import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

from autoscout.util import get_record


def search(
    data: pd.DataFrame, criteria: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    gte, eq, lte = [
        criteria.get(x, None)
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
    data = data.copy(deep=True)
    scaler = MinMaxScaler()

    baseline = get_record(data, index)[columns]
    data_relevant = data[columns]

    data_relevant = pd.DataFrame(scaler.fit_transform(data_relevant), columns=columns)
    baseline = pd.DataFrame(scaler.transform(baseline), columns=columns).squeeze()

    return data.iloc[data_relevant.apply(
        lambda col: distance.euclidean(baseline, col), axis=1
    ).nsmallest(num).index.to_list()]
