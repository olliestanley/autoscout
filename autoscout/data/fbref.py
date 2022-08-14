import re
from typing import Dict, Sequence, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_data(config: Dict[str, Sequence[str]], top, end, team: bool = False, vs: bool = False) -> pd.DataFrame:
    df = pd.concat([
        get_data_for_category(k, top, end, v, team=team, vs=vs)
        for k, v in config.items()
    ], axis=1)

    return df.loc[:, ~df.columns.duplicated()]


def get_data_for_category(category: str, top, end, features: Sequence[str], team: bool = False, vs: bool = False) -> pd.DataFrame:
    url = top + category + end
    player_table, team_table = get_tables(url, vs=vs)

    return (
        get_df_team(features, team_table) if team
        else get_df_player(features, player_table)
    )


def get_df_player(features: Sequence[str], player_table) -> pd.DataFrame:
    pre_df_player = dict()
    rows = player_table.find_all("tr")

    for row in rows:
        if row.find("th", {"scope": "row"}) is None:
            continue

        for feat in features:
            cell = row.find("td", {"data-stat": feat})
            text = cell.text.strip().encode().decode("utf-8")

            if text == "":
                text = "0"
            if feat not in ("player", "nationality", "position", "squad", "age", "birth_year"):
                text = float(text.replace(",", ""))

            if feat in pre_df_player:
                pre_df_player[feat].append(text)
            else:
                pre_df_player[feat] = [text]

    return pd.DataFrame.from_dict(pre_df_player)


def get_df_team(features: Sequence[str], team_table) -> pd.DataFrame:
    pre_df_squad = dict()
    # features does not contain squad name, needs special treatment
    rows = team_table.find_all("tr")

    for row in rows:
        if row.find("th", {"scope": "row"}) is None:
            continue

        name = (
            row.find("th", {"data-stat": "squad"})
            .text.strip().encode().decode("utf-8")
        )

        if "squad" in pre_df_squad:
            pre_df_squad["squad"].append(name)
        else:
            pre_df_squad["squad"] = [name]

        for feat in features:
            cell = row.find("td", {"data-stat": feat})
            text = cell.text.strip().encode().decode("utf-8")

            if text == "":
                text = "0"
            if feat not in ("player", "nationality", "position", "squad", "age", "birth_year"):
                text = float(text.replace(",", ""))

            if feat in pre_df_squad:
                pre_df_squad[feat].append(text)
            else:
                pre_df_squad[feat] = [text]

    return pd.DataFrame.from_dict(pre_df_squad)


def get_tables(url: str, vs: bool = False) -> Tuple:
    res = requests.get(url)
    # avoid issue with comments breaking parsing
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("", res.text), "lxml")
    tables = soup.findAll("tbody")

    team_table, team_vs_table, player_table = tables[:3]

    if vs:
        return player_table, team_vs_table

    return player_table, team_table
