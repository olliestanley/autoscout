# autoscout
Football (soccer) scouting via publicly available data.

### fbref

Currently, `autoscout` has a Python API containing functionality to download data from `fbref.com` into CSV files. Player and team data can be acquired, and the process should work for any competition and season for which `fbref` has data.

Acquiring data can be achieved without the need to write any Python code, by calling the scripts in `scripts/data` from the command line with appropriate arguments. Refer to the docstrings in the individual script `.py` files to understand the required and optional arguments.

In future, functionality to quickly create visualisations and other short-form analytical products based on CSV-format data from `fbref` will be added to `autoscout`. This will aim to allow assessment of players and teams.

### Developers

* [Oliver Stanley](https://github.com/olliestanley)

### Suggestions

Adding new functionality to `autoscout`, such as means of obtaining data from new sources or new analytical tools, is always of interest. Feel free to open a [GitHub Issue](https://github.com/olliestanley/autoscout/issues/new) with any suggestions.
