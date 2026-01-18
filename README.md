# autoscout

Football (soccer) scouting and analytics using publicly available data from fbref.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Google Chrome (for web scraping)

## Installation

```shell
git clone https://github.com/olliestanley/autoscout.git
cd autoscout
uv sync
```

Or with pip:

```shell
pip install -e ".[dev]"
```

## Usage

### Getting Data

Download Premier League 2024-25 outfield player data from fbref:

```shell
uv run python scripts/data/download_fbref_aggregate.py --competition eng1 --season 2025 --type outfield
```

Download La Liga current season team data (append `--vs` for stats against each team):

```shell
uv run python scripts/data/download_fbref_aggregate.py --competition spa1 --season current --type team
```

Download Manchester United 2024-25 team match-by-match data:

```shell
uv run python scripts/data/download_fbref_match.py --dataset manchester_united --season 2025
```

Download Premier League 2024-25 fixture schedule and results:

```shell
uv run python scripts/data/download_fbref_comp.py --comp eng1 --season 2025
```

Add to or alter `config/fbref/matches.json` to add extra players or teams to the available list.

**Note on fbref configs:** The provided configurations in `config/fbref/stats.json` define which statistics to scrape from fbref. These may become outdated as fbref updates their website—stat names can change, new stats may be added, or existing ones removed. If scraping fails or returns missing columns, check the fbref website for current stat names and update the config accordingly.

---

Load data into a Pandas or Polars DataFrame:

```python
from autoscout import util

# Pandas (default)
df = util.load_csv("data/fbref/eng1/2025/outfield.csv")

# Polars
df = util.load_csv("data/fbref/eng1/2025/outfield.csv", format="polars")
```

Combine DataFrames from multiple competitions or seasons:

```python
from autoscout import preprocess

combined = preprocess.combine_data([df_1, df_2])
```

### Creating Visualisations

Plot a midfielder radar chart:

```python
from autoscout import util
from autoscout.vis import radar

midfield_config = util.load_json("config/radar/midfield.json")
rdr, fig, ax = radar.plot_radar_from_config(df, midfield_config, "Bruno Fernandes")
```

Radar configurations can be customised by editing the `.json` files in `config/radar`, or plot radars directly with `radar.plot_radar(...)`.

---

Plot rolling xG chart for a team with trend lines:

```python
from autoscout import preprocess
from autoscout.vis import chart

df = preprocess.rolling(df, ["xg_for", "xg_against"])
df["n"] = df.index

plot = chart.lines(
    df, ["n", "n"], ["xg_for_roll_mean", "xg_against_roll_mean"],
    colors=["green", "red"], legend_labels=["xG For", "xG Against"],
    trends=True, vshade=(0, 1), title="10 game rolling average xG",
    x_axis_label="Match", y_axis_label="xG"
)
```

### Searching Data

Find players most similar to a target player:

```python
from autoscout import preprocess, search

columns = ["goals", "npxg", "assists", "xg_assist"]
df = preprocess.adjust_per_90(df, columns)
similar_df = search.search_similar(df, columns, "Bruno Fernandes", num=6)
```

Filter data by criteria:

```python
from autoscout import search

criteria = {
    "gte": {"goals": 5.0, "minutes": 900},
    "lte": {"xg": 15.0}
}
filtered = search.search(df, criteria)
```

### Analysing Data

Create stylistic ratings for players based on statistical profiles:

```python
from autoscout import analyse, util

ratings_config = util.load_json("config/rating_inputs.json")
df = analyse.estimate_style_ratings(df, ratings_config)

df[["player", "progress_rating", "creativity_rating"]]
```

Custom ratings can be defined by adding sections to `rating_inputs.json`.

---

Reduce dimensionality for analysis:

```python
from autoscout import analyse

columns = ["goals", "assists", "xg", "xg_assist"]
df["attack_score"] = analyse.reduce_dimensions(df, columns, reducer=1)
```

Cluster players into groups:

```python
from autoscout import analyse

columns = ["goals", "assists", "xg", "xg_assist"]
df["cluster"] = analyse.cluster_records(df, columns, estimator="auto")
```

## Development

Run tests (excluding integration tests that require network access):

```shell
uv run pytest -m "not integration"
```

Run all tests including integration tests:

```shell
uv run pytest
```

Run linting:

```shell
uv run ruff check .
uv run ruff format --check .
```

Run type checking:

```shell
uv run mypy autoscout
```

## Project Structure

```
├── autoscout/             <- Python source code
│   ├── data/              <- Data acquisition modules
│   │   └── fbref/         <- fbref-specific scrapers
│   └── vis/               <- Visualization modules
│
├── config/                <- Configuration files
│   ├── fbref/             <- fbref scraping configs
│   └── radar/             <- Radar chart configs
│
├── scripts/               <- CLI scripts
│   └── data/              <- Data download scripts
│
├── tests/                 <- Test suite
│
├── data/                  <- Downloaded data (not in git)
├── pyproject.toml         <- Project configuration
└── README.md
```
