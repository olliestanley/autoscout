"""
Script for downloading data from fbref ready for analysis.

To download outfield player data for the current men's Premier League season::

    python scripts/data/download_fbref.py \
        --config config/fbref/
        --out data/fbref/
        --competition pl
        --type outfield

For goalkeeper data, simply replace `outfield` with `keeper`.

To download team data for the 2021-22 men's Premier League season::

    python scripts/data/download_fbref.py \
        --config config/fbref/
        --out data/fbref/
        --competition pl
        --season 2022
        --type team

For stats against (rather than for) teams, simply append `--vs`.

The default configs (as used in the example usages above) are not mandatory. Using a
file named `comps.json` and another named `stats.json`, you can determine which stats
and competitions are pulled from `fbref.`
"""

import argparse
from pathlib import Path
from typing import Dict, Sequence, Union

from autoscout import util
from autoscout.data.fbref import match


def download_matches_for_player_or_team(
    url_top: str,
    url_end: str,
    stats_config: Dict[str, Dict[str, Sequence[str]]],
    out_dir: Union[str, Path] = "data/fbref/match",
    name: str = "matches",
    team: bool = False,
    vs: bool = False,
) -> None:
    df = match.get_data(
        stats_config["team" if team else "player"],
        url_top,
        url_end,
        team=team,
        vs=vs,
    )

    if team:
        name += "_vs" if vs else "_for"

    util.write_csv(df, out_dir, name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--conf", type=str, default="config/fbref")
    parser.add_argument("--out", type=str, default="data/fbref/match")
    parser.add_argument("--dataset", "--data", type=str, default="mufc_2122")
    parser.add_argument("--vs", action="store_true")
    args = parser.parse_args()

    config_dir = Path(args.config)
    matches_json = util.load_json(config_dir / "matches.json")
    stats_json = util.load_json(config_dir / "stats.json")

    matches: Dict[str, str] = matches_json[args.dataset]
    url_top, url_end = matches["top"], matches["end"]

    team = "squads" in url_top

    download_matches_for_player_or_team(
        url_top,
        url_end,
        stats_json["match"],
        out_dir=f"{args.out}/{args.dataset}",
        team=team,
        vs=args.vs,
    )
