import json
from collections.abc import Sequence
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
import polars as pl
from sklearn.preprocessing import minmax_scale


def get_record(data: pd.DataFrame, index: str | int) -> pd.DataFrame:
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
        return data.iloc[[index]]

    player = "player" in data.columns
    data = data[data["player" if player else "team"] == index]

    if len(data.index) > 1:
        # Use .to_numpy().argmax() for positional index with iloc (pandas 2.x compatibility)
        return data.iloc[[int(data["minutes"].to_numpy().argmax())]]

    return data


def min_max_scale(
    data: pd.DataFrame, columns: Sequence[str], inplace: bool = False
) -> pd.DataFrame:
    """
    Scale selected `columns` in `data` to [0, 1].

    Args:
        data: Original data to scale.
        columns: Columns to perform scaling on.
        inplace: Modify existing DataFrame. Not recommended.

    Returns:
        DataFrame with selected columns scaled.
    """

    if not inplace:
        data = data.copy(deep=True)
    data[columns] = minmax_scale(data[columns])
    return data


def sleep_and_return(result: Any, sleep_seconds: float) -> Any:
    """
    Sleep (do nothing) for `sleep_seconds`, then return `result`.
    """

    sleep(sleep_seconds)
    return result


def load_json(
    file_path: str | Path,
) -> dict[str, Any]:
    """
    Load `file_path` from JSON to dict.
    """

    file_path = Path(file_path)

    with open(file_path) as f:
        loaded_json: dict[str, Any] = json.load(f)

    return loaded_json


def load_csv(
    file_path: str | Path, format: str = "pandas", **kwargs
) -> pd.DataFrame | pl.DataFrame:
    """
    Load `file_path` from CSV to DataFrame.

    Args:
        file_path: Path to CSV file.
        format: Format to load CSV as. Options: "pandas" (default), "polars".
    """

    file_path = Path(file_path)
    load_function = pl.read_csv if format == "polars" else pd.read_csv
    return load_function(file_path, **kwargs)


def write_csv(
    df: pd.DataFrame,
    out_dir: str | Path,
    basename: str,
    **kwargs,
) -> Path:
    """
    Write `df` to a file named `basename`.csv in directory `out_dir`.

    Args:
        df: DataFrame to write.
        out_dir: Directory to write in.
        basename: Stem for the file name to write to.
        **kwargs: Passed to `DataFrame.to_csv()`.

    Returns:
        Path to the written CSV.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{basename}.csv"
    df.to_csv(out_path, **kwargs)
    return out_path
