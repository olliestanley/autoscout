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

    return df.loc[:, ~df.columns.duplicated()].assign(name=name)  # type: ignore[no-any-return]


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
    tables = scrape.get_tables_by_id(url)

    # Determine which table to use based on team/vs flags
    # fbref match log table IDs:
    # - Player: matchlogs_all
    # - Team for: matchlogs_for
    # - Team against: matchlogs_against
    table = get_match_table(tables, team=team, vs=vs)

    if table is None:
        raise ValueError(
            f"Could not find match log table (team={team}, vs={vs}). "
            f"Available tables: {list(tables.keys())}"
        )

    return get_data_from_table(features, table)


def get_match_table(
    tables: dict[str, Any],
    team: bool = False,
    vs: bool = False,
) -> Any | None:
    """
    Get the appropriate match log table.

    fbref match log table IDs:
    - Player: matchlogs_all
    - Team for: matchlogs_for
    - Team against: matchlogs_against

    Args:
        tables: Dict mapping table IDs to table elements.
        team: Whether this is team-level data.
        vs: Whether to get "against" stats for teams.

    Returns:
        The table element, or None if not found.
    """
    if team:
        table_id = "matchlogs_against" if vs else "matchlogs_for"
    else:
        table_id = "matchlogs_all"

    if table_id in tables:
        return tables[table_id]

    # Fallback: try to find any table with matchlogs in the ID
    for tid, table in tables.items():
        if "matchlogs" in tid:
            return table

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

    pre_df: dict[str, list[Any]] = {}
    rows = table.find_all("tr")

    for row in rows:
        if row.find("th", {"scope": "row"}) is None:
            continue

        goals_cell = row.find("td", {"data-stat": "goals_for"})
        if (
            goals_cell is not None
            and goals_cell.text.strip().encode().decode("utf-8") == ""
        ):
            continue

        # Try to get opponent and date
        opponent_cell = row.find("td", {"data-stat": "opponent"})
        date_cell = row.find("th", {"data-stat": "date"})

        if opponent_cell is None or date_cell is None:
            continue

        opponent = opponent_cell.text.strip().encode().decode("utf-8")
        date = date_cell.text.strip().encode().decode("utf-8")

        if "opponent" in pre_df:
            pre_df["opponent"].append(opponent)
            pre_df["date"].append(date)
        else:
            pre_df["opponent"] = [opponent]
            pre_df["date"] = [date]

        for feat in features:
            cell = row.find("td", {"data-stat": feat})

            if cell is None:
                # Feature not found - use default value
                text: Any = (
                    0
                    if feat
                    not in (
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
                    )
                    else ""
                )
            else:
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
                    with contextlib.suppress(ValueError):
                        text = float(text.replace(",", ""))

            if feat in pre_df:
                pre_df[feat].append(text)
            else:
                pre_df[feat] = [text]

    return pd.DataFrame.from_dict(pre_df)
