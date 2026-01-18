"""
General purpose utilities for web data scraping.

Uses nodriver to bypass Cloudflare protection on fbref.
"""

import asyncio
import re
from collections.abc import Sequence
from typing import Any

from bs4 import BeautifulSoup, Tag

# Global browser instance for reuse
_browser: Any = None
_browser_lock = asyncio.Lock()


async def _get_browser(headless: bool = False) -> Any:
    """Get or create a browser instance."""
    global _browser

    async with _browser_lock:
        if _browser is None:
            import nodriver as uc

            _browser = await uc.start(headless=headless)

        return _browser


async def close_browser() -> None:
    """Close the global browser instance."""
    global _browser

    async with _browser_lock:
        if _browser is not None:
            # nodriver doesn't have a stop method that needs await
            _browser = None


async def fetch_page_async(url: str, headless: bool = False) -> str:
    """
    Fetch a page using nodriver, waiting for Cloudflare to resolve.

    Args:
        url: URL to fetch.
        headless: Whether to run browser in headless mode. Note: headless mode
            may not work with Cloudflare-protected sites.

    Returns:
        HTML content of the page.
    """
    browser = await _get_browser(headless=headless)
    page = await browser.get(url)

    # Wait for Cloudflare challenge to resolve
    for _ in range(12):  # Max 60 seconds
        await asyncio.sleep(5)
        html = await page.get_content()
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string if soup.title else ""

        # Check if we're past the Cloudflare challenge
        if "Just a moment" not in title:
            return html

    # Return whatever we have after timeout
    return await page.get_content()


def fetch_page(url: str, headless: bool = False) -> str:
    """
    Synchronous wrapper for fetch_page_async.

    Args:
        url: URL to fetch.
        headless: Whether to run browser in headless mode.

    Returns:
        HTML content of the page.
    """
    return asyncio.get_event_loop().run_until_complete(fetch_page_async(url, headless))


def get_all_tables(url: str, headless: bool = False) -> Sequence[Tag]:
    """
    Obtain all HTML table bodies from a URL.

    Args:
        url: URL to the page containing the tables.
        headless: Whether to run browser in headless mode.

    Returns:
        Sequence of found table body elements.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, need to use a new loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, fetch_page_async(url, headless=headless)
                )
                html = future.result()
        else:
            html = loop.run_until_complete(fetch_page_async(url, headless=headless))
    except RuntimeError:
        # No event loop exists
        html = asyncio.run(fetch_page_async(url, headless=headless))

    # Remove HTML comments that might hide tables
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("", html), "lxml")
    tables = soup.find_all("tbody")
    return tables


def get_tables_by_id(url: str, headless: bool = False) -> dict[str, Tag]:
    """
    Obtain HTML tables from a URL, organized by their ID attribute.

    Args:
        url: URL to the page containing the tables.
        headless: Whether to run browser in headless mode.

    Returns:
        Dict mapping table IDs to table elements. Tables without IDs are excluded.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, fetch_page_async(url, headless=headless)
                )
                html = future.result()
        else:
            html = loop.run_until_complete(fetch_page_async(url, headless=headless))
    except RuntimeError:
        html = asyncio.run(fetch_page_async(url, headless=headless))

    # Remove HTML comments
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("", html), "lxml")

    result: dict[str, Tag] = {}
    for table in soup.find_all("table"):
        table_id = table.get("id")
        if table_id:
            tbody = table.find("tbody")
            if tbody:
                result[table_id] = tbody

    return result
