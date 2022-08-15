from typing import Dict, Sequence

import pandas as pd


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
