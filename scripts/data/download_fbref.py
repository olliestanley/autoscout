"""
Script for downloading data from fbref ready for analysis.
"""

import argparse
from pathlib import Path
from typing import Dict, Sequence, Union

from autoscout.data import fbref, util


def download_player_data_for_comp(
    comp_config: Dict[str, str],
    stats_config: Dict[str, Dict[str, Sequence[str]]],
    out_dir: Union[str, Path] = "data/fbref",
    keeper: bool = False,
) -> None:
    keeper_str = "keeper" if keeper else "outfield"

    df = fbref.get_data(
        stats_config[keeper_str],
        comp_config["top"],
        comp_config["end"],
    )

    util.write_dated_csv(df, out_dir, keeper_str, index=False)


def download_team_data_for_comp(
    comp_config: Dict[str, str],
    stats_config: Dict[str, Dict[str, Sequence[str]]],
    out_dir: Union[str, Path] = "data/fbref",
    vs: bool = False,
) -> None:
    df = fbref.get_data(
        stats_config["team"],
        comp_config["top"],
        comp_config["end"],
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
    args = parser.parse_args()

    config_dir = Path(args.config)
    comps_json = util.load_json(config_dir / "comps.json")
    stats_json = util.load_json(config_dir / "stats.json")

    if args.type == "team":
        download_team_data_for_comp(
            comps_json[args.competition],
            stats_json,
            args.out,
            vs=args.vs,
        )
    else:
        download_player_data_for_comp(
            comps_json[args.competition],
            stats_json,
            args.out,
            keeper=(args.type == "keeper"),
        )
