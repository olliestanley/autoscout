"""
General purpose utilities for web data scraping.
"""

import re
from typing import Sequence

import requests
from bs4 import BeautifulSoup


def get_all_tables(url: str) -> Sequence:
    """
    Obtain all HTML tables from a URL.

    Args:
        url: URL to the page containing the tables.

    Returns:
        Sequence of found tables.
    """

    res = requests.get(url)
    # avoid issue with comments breaking parsing
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("", res.text), "lxml")
    tables = soup.findAll("tbody")
    return tables
