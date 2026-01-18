"""
Download Manchester United 2025-26 Premier League match data and plot a 5-game
rolling average of xG (created) and xG (conceded).

Usage:
    uv run python examples/man_utd_xg_rolling.py
"""

from pathlib import Path

from bokeh.io import output_file, save

from autoscout import preprocess, util
from autoscout.data.fbref import match
from autoscout.vis import chart

# Configuration
TEAM = "manchester_united"
SEASON = 2026  # 2025-26 season
ROLL_LENGTH = 5
OUTPUT_DIR = Path("data/fbref/match")
OUTPUT_CHART = Path("examples/example_team_rolling_xg.html")


def main() -> None:
    # Load configs
    matches_config = util.load_json("config/fbref/matches.json")
    stats_config = util.load_json("config/fbref/stats.json")

    # Get URL components for Manchester United
    team_config = matches_config[TEAM]
    url_top = team_config["top"].replace("$season$", f"{SEASON - 1}-{SEASON}")
    url_end = team_config["end"]
    name = team_config["name"]

    print(f"Downloading {name} match data for {SEASON - 1}-{SEASON} season...")

    # Download team match data (stats for the team)
    df = match.get_data(
        stats_config["match"]["team"],
        url_top,
        url_end,
        name,
        team=True,
        vs=False,
    )

    print(f"Downloaded {len(df)} matches across all competitions")

    # Filter for Premier League matches only
    df = df[df["comp"] == "Premier League"].reset_index(drop=True)
    print(f"Filtered to {len(df)} Premier League matches")

    if len(df) < ROLL_LENGTH:
        print(f"Warning: Only {len(df)} matches available, need at least {ROLL_LENGTH} for rolling average")

    # Save the raw data
    out_dir = OUTPUT_DIR / f"{TEAM}_{SEASON}"
    out_dir.mkdir(parents=True, exist_ok=True)
    util.write_csv(df, out_dir, f"{TEAM}_premier_league", index=False)
    print(f"Saved data to {out_dir}/{TEAM}_premier_league.csv")

    # Add match number for x-axis
    df["match_number"] = range(1, len(df) + 1)

    # Calculate 5-game rolling averages
    df = preprocess.rolling(
        df,
        columns=["xg_for", "xg_against"],
        roll_length=ROLL_LENGTH,
        min_periods=1,  # Start producing values from first match
        reduction="mean",
        dropna=False,
    )

    print(f"Calculated {ROLL_LENGTH}-game rolling averages")

    # Create the plot
    plot = chart.lines(
        df,
        x_columns=["match_number", "match_number"],
        y_columns=["xg_for_roll_mean", "xg_against_roll_mean"],
        colors=["#1f77b4", "#d62728"],  # Blue for xG created, Red for xG conceded
        legend_labels=["xG Created (5-game avg)", "xG Conceded (5-game avg)"],
        trends=True,
        vshade=(0, 1),
        title=f"{name} - {SEASON - 1}/{SEASON % 100} Premier League xG Trend",
        x_axis_label="Match Number",
        y_axis_label="xG (5-game rolling average)",
        width=900,
        height=500,
    )

    # Save the chart
    output_file(OUTPUT_CHART)
    save(plot)
    print(f"Chart saved to {OUTPUT_CHART}")


if __name__ == "__main__":
    main()
