"""
Script for downloading match level data from fbref ready for analysis.

It is mandatory to specify the '--data' argument describing the player or team
being pulled, which must match one in the configs.

For stats against (rather than for) teams, simply append `--vs`.

The default configs are not mandatory. Using a file named `matches.json` and another
named `stats.json`, you can determine which stats and competitions are pulled.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from autoscout import util
from autoscout.data.fbref import match


def download_matches_for_player_or_team(
    url_top: str,
    url_end: str,
    name: str,
    stats_config: dict[str, dict[str, Sequence[str]]],
    out_dir: str | Path = "data/fbref/match",
    team: bool = False,
    vs: bool = False,
) -> None:
    df = match.get_data(
        stats_config["team" if team else "player"],
        url_top,
        url_end,
        name,
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
    parser.add_argument("--dataset", "--data", type=str, default="manchester_united")
    parser.add_argument("--season", type=int, default=2022)
    parser.add_argument("--vs", action="store_true")
    args = parser.parse_args()

    config_dir = Path(args.config)
    matches_json = util.load_json(config_dir / "matches.json")
    stats_json = util.load_json(config_dir / "stats.json")

    matches: dict[str, str] = matches_json[args.dataset]
    url_top, url_end, name = matches["top"], matches["end"], matches["name"]

    season = args.season
    url_top = url_top.replace("$season$", f"{season - 1}-{season}")

    team = "squads" in url_top

    download_matches_for_player_or_team(
        url_top,
        url_end,
        name,
        stats_json["match"],
        out_dir=f"{args.out}/{args.dataset}_{season}",
        team=team,
        vs=args.vs,
    )
