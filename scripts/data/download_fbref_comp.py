"""
Script for downloading schedule and scorelines from fbref for one season of a
competition.

It is mandatory to specify the '--name' argument describing the player or team name
being pulled.

The default configs are not mandatory. Using a file named `comps.json` and another
named `stats.json`, you can determine which stats and competitions are pulled.
"""

import argparse
from pathlib import Path
from typing import Dict

from autoscout.data.fbref import comp
from autoscout import util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--conf", type=str, default="config/fbref")
    parser.add_argument("--out", type=str, default="data/fbref")
    parser.add_argument("--comp", type=str, default="eng1")
    parser.add_argument("--season", type=int, default=2022)
    args = parser.parse_args()

    config_dir = Path(args.config)
    comps_json = util.load_json(config_dir / "comps.json")
    stats_json = util.load_json(config_dir / "stats.json")

    comp_config: Dict[str, str] = comps_json[args.comp]
    url_top, url_end = comp_config["top"], comp_config["schedule_end"]

    df = comp.get_competition_data(
        stats_json["match"]["competition"], url_top, url_end, args.season
    )

    out_dir = f"{args.out}/{args.comp}/{args.season}/"

    util.write_csv(df, out_dir, "matches", index=False)
