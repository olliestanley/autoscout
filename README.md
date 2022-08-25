# autoscout
Football (soccer) scouting via publicly available data.

## Usage

Setup the repository and a virtual environment with requirements:

```shell
$ git clone https://github.com/olliestanley/autoscout.git
$ cd autoscout
$ python -m venv venv
$ source venv/bin/activate
$ python -m pip install -qr requirements.txt
```

### Getting Data

Download Premier League 2021-22 outfield player data from `fbref` via CLI:

```shell
$ python scripts/data/download_fbref_aggregate.py --competition eng1 --season 2022 --type outfield
```

---

Download La Liga current season team data from `fbref` (append `--vs` to get data against the team):

```shell
$ python scripts/data/download_fbref_aggregate.py --competition spa1 --season current --type team
```

---

Download Frenkie de Jong 2021-22 player match-by-match data from `fbref`:

```shell
$ python scripts/data/download_fbref_match.py --dataset frenkie_2223
```

---

Download Manchester United 2022-23 team match-by-match data from `fbref` (append `--vs` to get data against the team):

```shell
$ python scripts/data/download_fbref_match.py --dataset mufc_2122
```

Add to or alter `config/fbref/matches.json` to add extra players or teams to the available list. Note that building a dataset of a large number of players and/or teams may require significant effort as each entity has a unique identifier which you must obtain. In future it may be possible to scrape an ID to player/team mapping but this is not currently supported.

---

Load data into a Pandas `DataFrame`:

```python
from autoscout import util

df = util.load_csv("data/fbref/eng1/2022/outfield.csv")
```

---

Combine `DataFrame`s to create a single dataset, such as from multiple competitions or multiple seasons of the same competition.

```python
from autoscout import preprocess

combined = preprocess.combine_data((df_1, df_2))
```

---

### Creating Visualisations

Plot a Midfielder radar chart, based on a loaded `df`:

```python
from autoscout import util
from autoscout.vis import radar

midfield_config = util.load_json("config/radar/midfield.json")
rdr, fig, ax = radar.plot_radar_from_config(df, midfield_config, "Fred")
```

Radar configurations can be customised and modified by editing the `.json` fles in `config/radar`. It is also possible to plot radars without a `.json` configuration file using `radar.plot_radar(...)`.

---

Plot rolling xG for and against chart for a team with dashed trend lines and shading the gap between xG For and xG Against, using a loaded team match by match `df`:

```python
from autoscout import preprocess
from autoscout.vis import chart

df = preprocess.rolling(df, ["xg_for", "xg_against"])
df["n"] = df.index

plot = chart.lines(
    df, ["n", "n"], ["xg_for_roll_mean", "xg_against_roll_mean"],
    colors=["green", "red"], legend_labels=["xG For", "xG Against"],
    title="10 game rolling average xG", x_label="Date", y_label="xG",
    trends=True, vshade=(0, 1)
)
```

---

### Searching Data

Find 6 players in the dataset most similar to Paul Pogba in the statistics in `columns`, after applying per 90 adjustment to normalize the data:

```python
from autoscout import preprocess, search

columns = ["goals", "npxg", "assists", "xa"]
df = preprocess.adjust_per_90(df, columns)
similar_df = search.search_similar(df, columns, "Paul Pogba", num=6)
```

---

Filter a team dataset to contain only teams which have scored at least 50 goals and have exactly 19 players used:

```python
from autoscout import util, search

criteria = {
    "gte": { "goals": 50.0 },
    "eq": { "players_used": 19.0 }
}

df_teams = util.load_csv("data/fbref/eng1/2022/team_for.csv")
matching_df = search.search(df_teams, criteria)
```

---

### Analysing Data

Create stylistic ratings for all players or teams in a dataset from a loaded `df`, based on pre-existing configuration:

```python
from autoscout import analyse, util

ratings_config = util.load_json("config/rating_inputs.json")
df = analyse.estimate_style_ratings(df, ratings_config)

df["progress_rating"]
```

Ratings based on custom defined sets of statistics can easily be computed by adding sections to `rating_inputs.json`.

---

Reduce the dimensionality of 4 columns of a dataset `df` into 2 columns. This is used by `estimate_style_ratings()` to derive stylistic ratings from raw statistics, but may be useful for other purposes.

```python
from autoscout import analyse

columns = ["goals", "assists", "xg", "xa"]
df["ga_rating"] = analyse.reduce_dimensions(df, columns, reducer=1)
```

A custom reducer from `SciKit-Learn` can be specified in `reduce_dimensions()`, otherwise an integer value for the output number of dimensions can be specified. This defaults to `1` if no value is specified.

---

Cluster players or teams into groups based on statistical similarities in the specified `columns`:

```python
from autoscout import analyse

columns = ["goals", "assists", "xg", "xa"]
df["cluster"] = analyse.cluster_records(df, columns, estimator="auto")
```

Again, a custom estimator from `SciKit-Learn` can be specified in `cluster_records()`, otherwise a `KMeans` estimator is automatically fitted. The appropriate number of clusters is also automatically derived.

---

## Developers

* [Oliver Stanley](https://github.com/olliestanley)

## Suggestions

Adding new functionality to `autoscout`, such as means of obtaining data from new sources or new analytical tools, is always of interest. Feel free to open a [GitHub Issue](https://github.com/olliestanley/autoscout/issues/new) with any suggestions.

## Structure

```
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── setup.py
│
├── autoscout          <- Python source root for autoscout
│   ├── data           <- Code for acquiring data
│   └── vis            <- Code for visualising data
│
├── config             <- Configuration values for feeding to autoscout functions
│
├── scripts            <- Reusable scripts for using autoscout
│   └── data           <- Scripts for acquiring data for analysis via command line
│
├── data               <- Downloaded data, not included in source control
└── notebooks          <- Experimental notebooks, not included in source control
```
