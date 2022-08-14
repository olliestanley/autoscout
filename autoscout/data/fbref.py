import re
from typing import Any, Dict, Sequence, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

from autoscout.util import sleep_and_return


def get_data(
    config: Dict[str, Sequence[str]],
    top: str,
    end: str,
    team: bool = False,
    vs: bool = False,
    sleep_seconds: float = 5.0,
) -> pd.DataFrame:
    """
    Obtain player or team data for statistics specified in the configuration, from the
    fbref website, across several categories.

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

    return df.loc[:, ~df.columns.duplicated()]


def get_data_for_category(
    category: str,
    top: str,
    end: str,
    features: Sequence[str],
    team: bool = False,
    vs: bool = False,
) -> pd.DataFrame:
    """
    Obtain player or team data for statistics specified in `features`, from the fbref
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
    player_table, team_table = get_tables(url, vs=vs)
    table = team_table if team else player_table
    return get_data_from_table(features, table, team)


def get_data_from_table(
    features: Sequence[str], table, team: bool = False
) -> pd.DataFrame:
    """
    Extract data from a single HTML table on the fbref website.

    Args:
        features: IDs of statistics to extract.
        table: HTML table.
        team: Obtain team-level data if `True`, else player-level data.

    Returns:
        Extracted DataFrame from the table.
    """

    pre_df: Dict[str, Sequence[Any]] = dict()
    rows = table.find_all("tr")

    for row in rows:
        if row.find("th", {"scope": "row"}) is None:
            continue

        if team:
            name = (
                row.find("th", {"data-stat": "team"})
                .text.strip()
                .encode()
                .decode("utf-8")
            )

            if "team" in pre_df:
                pre_df["team"].append(name)
            else:
                pre_df["team"] = [name]

        for feat in features:
            cell = row.find("td", {"data-stat": feat})
            text = cell.text.strip().encode().decode("utf-8")

            if text == "":
                text = "0"
            if feat not in (
                "player",
                "nationality",
                "position",
                "team",
                "age",
                "birth_year",
            ):
                text = float(text.replace(",", ""))

            if feat in pre_df:
                pre_df[feat].append(text)
            else:
                pre_df[feat] = [text]

    return pd.DataFrame.from_dict(pre_df)


def get_tables(url: str, vs: bool = False) -> Tuple:
    """
    Obtain team and player HTML tables from a competition page on the fbref website.

    Args:
        url: URL to the page containing the tables.
        vs: If `True`, obtain statistics against the teams table, instead of for.

    Returns:
        Tuple of two tables, player and team, of statistics.
    """

    res = requests.get(url)
    # avoid issue with comments breaking parsing
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("", res.text), "lxml")
    tables = soup.findAll("tbody")

    team_table, team_vs_table, player_table = tables[:3]

    if vs:
        return player_table, team_vs_table

    return player_table, team_table
