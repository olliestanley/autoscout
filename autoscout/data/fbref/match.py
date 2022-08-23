from typing import Any, Dict, Sequence

import pandas as pd

from autoscout.data import scrape
from autoscout.util import sleep_and_return


def get_data(
    config: Dict[str, Sequence[str]],
    top: str,
    end: str,
    name: str,
    team: bool = False,
    vs: bool = False,
    sleep_seconds: float = 7.0,
) -> pd.DataFrame:
    """
    Obtain player or team match-level data for statistics specified in the `config`,
    from the fbref website across several categories.

    Args:
        config: Dict defining statistics per-category from fbref.
        top: Start section of the relevant competition URL for fbref.
        end: Final section of the relevant competition URL for fbref.
        team: Obtain team-level data if `True`, else player-level data.
        vs: For team-level data, obtain statistics against each team if `True` or for
            each team if `False`.
        sleep_seconds: Seconds to pause between each request to fbref.

    Returns:
        Downloaded and transformed DataFrame.
    """

    df = pd.concat(
        [
            sleep_and_return(
                get_data_for_category(k, top, end, v, team=team, vs=vs), sleep_seconds
            )
            for k, v in config.items()
        ],
        axis=1,
    )

    combined = df.loc[:, ~df.columns.duplicated()]
    combined["name"] = name
    return combined


def get_data_for_category(
    category: str,
    top: str,
    end: str,
    features: Sequence[str],
    team: bool = False,
    vs: bool = False,
) -> pd.DataFrame:
    """
    Obtain all competition player or team data for statistics specified in `features`, from the fbref
    website, within a single category.

    Args:
        category: ID of the category to obtain statistics from.
        top: Start section of the relevant competition URL for fbref.
        end: Final section of the relevant competition URL for fbref.
        features: IDs of the statistics within the category to obtain.
        team: Obtain team-level data if `True`, else player-level data.
        vs: For team-level data, obtain statistics against each team if `True` or for
            each team if `False`.

    Returns:
        Downloaded DataFrame for the given statistics in this category.
    """

    url = top + category + end
    tables = scrape.get_all_tables(url)
    table = tables[1] if team and vs else tables[0]
    return get_data_from_table(features, table)


def get_data_from_table(
    features: Sequence[str], table
) -> pd.DataFrame:
    """
    Extract data from a single HTML table on the fbref website.

    Args:
        features: IDs of statistics to extract.
        table: HTML table.

    Returns:
        Extracted DataFrame from the table.
    """

    pre_df: Dict[str, Sequence[Any]] = dict()
    rows = table.find_all("tr")

    for row in rows:
        if row.find("th", {"scope": "row"}) is None:
            continue

        opponent = (
            row.find("td", {"data-stat": "opponent"})
            .text.strip()
            .encode()
            .decode("utf-8")
        )

        if "opponent" in pre_df:
            pre_df["opponent"].append(opponent)
        else:
            pre_df["opponent"] = [opponent]

        for feat in features:
            cell = row.find("td", {"data-stat": feat})
            text = cell.text.strip().encode().decode("utf-8")

            if text == "":
                text = "0"
            if feat not in (
                "date",
                "start_time",
                "comp",
                "round",
                "dayofweek",
                "venue",
                "result",
                "opponent",
                "match_report",
                "game_started",
                "position",
            ):
                text = float(text.replace(",", ""))

            if feat in pre_df:
                pre_df[feat].append(text)
            else:
                pre_df[feat] = [text]

    return pd.DataFrame.from_dict(pre_df)
