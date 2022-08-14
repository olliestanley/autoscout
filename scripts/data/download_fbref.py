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
from autoscout.data import fbref


def download_player_data_for_comp(
    url_top: str,
    url_end: str,
    stats_config: Dict[str, Dict[str, Sequence[str]]],
    out_dir: Union[str, Path] = "data/fbref",
    keeper: bool = False,
) -> None:
    keeper_str = "keeper" if keeper else "outfield"

    df = fbref.get_data(
        stats_config[keeper_str],
        url_top,
        url_end,
    )

    util.write_dated_csv(df, out_dir, keeper_str, index=False)


def download_team_data_for_comp(
    url_top: str,
    url_end: str,
    stats_config: Dict[str, Dict[str, Sequence[str]]],
    out_dir: Union[str, Path] = "data/fbref",
    vs: bool = False,
) -> None:
    df = fbref.get_data(
        stats_config["team"],
        url_top,
        url_end,
        team=True,
        vs=vs,
    )

    vs_text = "vs" if vs else "for"
    util.write_dated_csv(df, out_dir, f"team_{vs_text}", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--conf", type=str, default="config/fbref")
    parser.add_argument("--out", type=str, default="data/fbref")
    parser.add_argument("--competition", "--comp", type=str, default="pl")
    parser.add_argument("--type", type=str, default="outfield")
    parser.add_argument("--vs", action="store_true")
    parser.add_argument("--season", type=str, default="current")
    args = parser.parse_args()

    config_dir = Path(args.config)
    comps_json = util.load_json(config_dir / "comps.json")
    stats_json = util.load_json(config_dir / "stats.json")

    comp: Dict[str, str] = comps_json[args.competition]
    url_top, url_end = comp["top"], comp["end"]

    if args.season.isdigit():
        season = int(args.season)
        season_str = f"{season - 1}-{season}"

        url_top += f"{season_str}/"
        url_end = f"/{season_str}-{url_end[1:]}"

    if args.type == "team":
        download_team_data_for_comp(
            url_top,
            url_end,
            stats_json,
            args.out,
            vs=args.vs,
        )
    else:
        download_player_data_for_comp(
            url_top,
            url_end,
            stats_json,
            args.out,
            keeper=(args.type == "keeper"),
        )
