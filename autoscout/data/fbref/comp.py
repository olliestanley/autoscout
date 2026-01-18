from collections.abc import Sequence
from typing import Any

import pandas as pd

from autoscout.data import scrape


def get_competition_data(
    config: dict[str, Sequence[str]],
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
    tables = scrape.get_tables_by_id(url)

    # Find schedule table - ID pattern: sched_{season}_{comp_id}_{number}
    table = get_schedule_table(tables)

    if table is None:
        raise ValueError(
            f"Could not find schedule table. Available tables: {list(tables.keys())}"
        )

    return get_data_from_table(config["schedule"], table)


def get_schedule_table(tables: dict[str, Any]) -> Any | None:
    """
    Get the schedule table from a dict of tables.

    Schedule table IDs follow the pattern: sched_{season}_{comp_id}_{number}
    e.g., sched_2024-2025_9_1 for Premier League 2024-25

    Args:
        tables: Dict mapping table IDs to table elements.

    Returns:
        The schedule table element, or None if not found.
    """
    for table_id, table in tables.items():
        if table_id.startswith("sched_"):
            return table

    # Fallback: return first table if any
    if tables:
        return next(iter(tables.values()))

    return None


def get_data_from_table(features: Sequence[str], table) -> pd.DataFrame:
    """
    Extract data from a single HTML table on the fbref website.

    Args:
        features: IDs of statistics to extract.
        table: HTML table.

    Returns:
        Extracted DataFrame from the table.
    """
    string_features = (
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
        "notes",
    )

    pre_df: dict[str, list[Any]] = {}
    rows = table.find_all("tr")

    for row in rows:
        if row.find("th", {"scope": "row"}) is None:
            continue

        date_cell = row.find("td", {"data-stat": "date"})
        if date_cell is None:
            continue

        date = date_cell.text.strip().encode().decode("utf-8")

        if "date" in pre_df:
            pre_df["date"].append(date)
        else:
            pre_df["date"] = [date]

        for feat in features:
            cell = row.find("td", {"data-stat": feat})

            if cell is None:
                # Feature not found - use default value
                text: Any = "" if feat in string_features else 0
            else:
                text = cell.text.strip().encode().decode("utf-8")

                if text == "":
                    text = "0" if feat not in string_features else ""

                if feat not in string_features:
                    try:
                        text = float(text.replace(",", ""))
                    except ValueError:
                        pass

            if feat in pre_df:
                pre_df[feat].append(text)
            else:
                pre_df[feat] = [text]

    return pd.DataFrame.from_dict(pre_df)
