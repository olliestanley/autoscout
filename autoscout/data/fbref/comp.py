from typing import Any, Dict, Sequence

import pandas as pd

from autoscout.data import scrape


def get_competition_data(
    config: Dict[str, Sequence[str]],
    top: str,
    end: str,
    season: int,
) -> pd.DataFrame:
    """
    Obtain match-level schedule, scoreline, and xG data for a whole competition season.

    Args:
        config: Dict defining statistics per-category from fbref. Competition matches
            data only supports "schedule" category.
        top: Start section of the relevant competition URL for fbref.
        end: Final section of the relevant competition URL for fbref.
        season: Competition season.
    """

    season_str = f"{season - 1}-{season}"
    url = f"{top}{season_str}/schedule/{season_str}{end}"
    table = scrape.get_all_tables(url)[0]
    return get_data_from_table(config["schedule"], table)


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

        date = row.find(
            "td", {"data-stat": "date"}
        ).text.strip().encode().decode("utf-8")

        if "date" in pre_df:
            pre_df["date"].append(date)
        else:
            pre_df["date"] = [date]

        for feat in features:
            cell = row.find("td", {"data-stat": feat})
            text = cell.text.strip().encode().decode("utf-8")

            if text == "":
                text = "0"
            if feat not in (
                "date",
                "referee",
                "score",
                "start_time",
                "round",
                "dayofweek",
                "venue",
                "result",
                "match_report",
                "game_started",
                "home_team",
                "away_team",
            ):
                text = float(text.replace(",", ""))

            if feat in pre_df:
                pre_df[feat].append(text)
            else:
                pre_df[feat] = [text]

    return pd.DataFrame.from_dict(pre_df)
