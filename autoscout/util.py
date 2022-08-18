import json
from pathlib import Path
from time import sleep
from typing import Any, Dict, Union

import pandas as pd
from sklearn.preprocessing import minmax_scale


def get_record(data: pd.DataFrame, index: Union[str, int]) -> pd.DataFrame:
    """
    Get a single row from the given DataFrame, based on a contextually interpreted
    `index` value.

    Args:
        data: Initial DataFrame to get a row from.
        index: Identifier for the row. If `index` is of type `int`, it will be used as
            a row index, so the result will be `data.iloc[index]`. If `index` is of
            type `str` it will be used to obtain a row by matching `index` to a player
            or team name column. If both a player and team column are present, player
            is preferred. If there are multiple matches, the row with the greatest
            value of the `minutes` column is preferred.

    Returns:
        DataFrame containing a single row from the input DataFrame.
    """

    if isinstance(index, int):
        return data.iloc[index]

    player = "player" in data.columns
    data = data[data["player" if player else "team"] == index]

    if len(data.index) > 1:
        return data.iloc[data["minutes"].argmax()]

    return data


def min_max_scale(
    data: pd.DataFrame, columns: str, inplace: bool = False
) -> pd.DataFrame:
    if not inplace:
        data = data.copy(deep=True)
    data[columns] = minmax_scale(data[columns])
    return data


def sleep_and_return(result: Any, sleep_seconds: float) -> Any:
    sleep(sleep_seconds)
    return result


def load_json(
    file_path: Union[str, Path],
) -> Dict[str, Any]:
    file_path = Path(file_path)

    with open(file_path, "r") as f:
        loaded_json = json.load(f)

    return loaded_json


def load_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    file_path = Path(file_path)
    return pd.read_csv(file_path, **kwargs)


def write_csv(
    df: pd.DataFrame,
    out_dir: Union[str, Path],
    basename: str,
    **kwargs,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{basename}.csv"
    df.to_csv(out_path, **kwargs)
    return out_path
