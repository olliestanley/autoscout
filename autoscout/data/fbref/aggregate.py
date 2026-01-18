import contextlib
from collections.abc import Sequence
from typing import Any

import pandas as pd

from autoscout.data import scrape
from autoscout.util import sleep_and_return


def get_data(
    config: dict[str, Sequence[str]],
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

    return df.loc[:, ~df.columns.duplicated()]  # type: ignore[no-any-return]


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
    tables = scrape.get_tables_by_id(url)

    # Determine which table to use based on category and team/vs flags
    table = get_table_for_category(tables, category, team=team, vs=vs)

    if table is None:
        raise ValueError(
            f"Could not find table for category '{category}' "
            f"(team={team}, vs={vs}). "
            f"Available tables: {list(tables.keys())}"
        )

    return get_data_from_table(features, table, team)


def get_table_for_category(
    tables: dict[str, Any],
    category: str,
    team: bool = False,
    vs: bool = False,
) -> Any | None:
    """
    Get the appropriate table for a given category.

    fbref table IDs follow these patterns:
    - Player tables: stats_{category} (e.g., stats_standard, stats_shooting)
    - Team for tables: stats_squads_{category}_for
    - Team against tables: stats_squads_{category}_against

    Some categories have underscores in table IDs but not in URL paths:
    - playingtime -> playing_time
    - passing_types -> passing_types (same)

    Args:
        tables: Dict mapping table IDs to table elements.
        category: The category name (e.g., "stats", "shooting", "passing").
        team: Whether to get team-level data.
        vs: Whether to get "against" stats for teams.

    Returns:
        The table element, or None if not found.
    """
    # Map URL category names to table ID category names
    category_map = {
        "stats": "standard",
        "playingtime": "playing_time",
        "keepers": "keeper",
        "keepersadv": "keeper_adv",
    }
    table_category = category_map.get(category, category)

    if team:
        # Team tables
        suffix = "against" if vs else "for"
        # Try different possible table ID formats
        possible_ids = [
            f"stats_squads_{table_category}_{suffix}",
            f"stats_{table_category}_squads_{suffix}",
            f"stats_squads_{category}_{suffix}",
        ]
    else:
        # Player tables
        possible_ids = [
            f"stats_{table_category}",
            f"stats_{category}",
            category,  # Some tables might just use the category name
        ]

    for table_id in possible_ids:
        if table_id in tables:
            return tables[table_id]

    # Fallback: try to find a table containing the category name
    for table_id, table in tables.items():
        if category in table_id or table_category in table_id:
            if team:
                if ("squads" in table_id or "squad" in table_id) and (
                    ("against" in table_id) == vs
                ):
                    return table
            elif "squads" not in table_id and "squad" not in table_id:
                return table

    return None


def get_data_from_table(
    features: Sequence[str], table: Any, team: bool = False
) -> pd.DataFrame:
    """
    Extract data from a single HTML table on the fbref website.

    Args:
        features: IDs of statistics to extract.
        table: HTML table body element.
        team: Obtain team-level data if `True`, else player-level data.

    Returns:
        Extracted DataFrame from the table.
    """

    pre_df: dict[str, list[Any]] = {}
    rows = table.find_all("tr")

    for row in rows:
        if row.find("th", {"scope": "row"}) is None:
            continue

        if team:
            name_cell = row.find("th", {"data-stat": "team"})
            if name_cell is None:
                # Try alternative: sometimes it's in a td
                name_cell = row.find("td", {"data-stat": "team"})

            if name_cell:
                name = name_cell.text.strip().encode().decode("utf-8")

                if "team" in pre_df:
                    pre_df["team"].append(name)
                else:
                    pre_df["team"] = [name]

        for feat in features:
            cell = row.find("td", {"data-stat": feat})
            if cell is None:
                # Try th for header cells
                cell = row.find("th", {"data-stat": feat})

            if cell is None:
                # Skip this feature for this row
                continue

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
                with contextlib.suppress(ValueError):
                    text = float(text.replace(",", ""))

            if feat in pre_df:
                pre_df[feat].append(text)
            else:
                pre_df[feat] = [text]

    return pd.DataFrame.from_dict(pre_df)
