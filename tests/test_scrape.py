"""
Unit tests for autoscout.data.scrape module.
"""

from unittest.mock import patch

from autoscout.data import scrape


# Sample HTML for testing
SAMPLE_HTML = """
<html>
    <body>
        <table id="stats_standard">
            <thead><tr><th>Header</th></tr></thead>
            <tbody><tr><td>Player Data</td></tr></tbody>
        </table>
        <table id="stats_squads_standard_for">
            <thead><tr><th>Team Header</th></tr></thead>
            <tbody><tr><td>Team Data</td></tr></tbody>
        </table>
        <table id="stats_squads_standard_against">
            <thead><tr><th>Team VS Header</th></tr></thead>
            <tbody><tr><td>Team VS Data</td></tr></tbody>
        </table>
    </body>
</html>
"""

SAMPLE_HTML_WITH_COMMENTS = """
<html>
    <body>
        <!--
        <table id="hidden_table">
            <tbody><tr><td>Hidden Data</td></tr></tbody>
        </table>
        -->
        <table id="visible_table">
            <tbody><tr><td>Visible Data</td></tr></tbody>
        </table>
    </body>
</html>
"""

SAMPLE_HTML_NO_TABLES = "<html><body><p>No tables here</p></body></html>"


def mock_async_run(html_content):
    """Helper to mock asyncio.run and get_event_loop for tests."""

    def mock_get_loop():
        class MockLoop:
            def is_running(self):
                return False

            def run_until_complete(self, coro):
                return html_content

        return MockLoop()

    return mock_get_loop


class TestGetAllTables:
    """Tests for scrape.get_all_tables function."""

    def test_get_all_tables_returns_sequence(self):
        """Should return a sequence of table elements."""
        with patch("asyncio.get_event_loop", mock_async_run(SAMPLE_HTML)):
            result = scrape.get_all_tables("http://example.com")

        assert len(result) == 3

    def test_get_all_tables_handles_html_comments(self):
        """Should handle HTML comments that might break parsing."""
        with patch("asyncio.get_event_loop", mock_async_run(SAMPLE_HTML_WITH_COMMENTS)):
            result = scrape.get_all_tables("http://example.com")

        # After removing comment markers, both tables should be found
        assert len(result) == 2

    def test_get_all_tables_empty_page(self):
        """Should return empty sequence for page with no tables."""
        with patch("asyncio.get_event_loop", mock_async_run(SAMPLE_HTML_NO_TABLES)):
            result = scrape.get_all_tables("http://example.com")

        assert len(result) == 0

    def test_get_all_tables_finds_tbody_elements(self):
        """Should find tbody elements specifically."""
        html = """
        <html>
            <body>
                <table>
                    <thead><tr><th>Header</th></tr></thead>
                    <tbody><tr><td>Body Data</td></tr></tbody>
                </table>
            </body>
        </html>
        """
        with patch("asyncio.get_event_loop", mock_async_run(html)):
            result = scrape.get_all_tables("http://example.com")

        # Should find tbody, not thead
        assert len(result) == 1
        assert "Body Data" in str(result[0])

    def test_get_all_tables_removes_comment_markers(self):
        """Should remove <!-- and --> markers from HTML."""
        # This simulates fbref's pattern where tables are in comments
        html = """
        <html>
            <body>
                <!--
                <table><tbody><tr><td>Hidden Table</td></tr></tbody></table>
                -->
            </body>
        </html>
        """
        with patch("asyncio.get_event_loop", mock_async_run(html)):
            result = scrape.get_all_tables("http://example.com")

        # After removing comment markers, table should be found
        assert len(result) == 1


class TestGetTablesById:
    """Tests for scrape.get_tables_by_id function."""

    def test_get_tables_by_id_returns_dict(self):
        """Should return a dict mapping IDs to table elements."""
        with patch("asyncio.get_event_loop", mock_async_run(SAMPLE_HTML)):
            result = scrape.get_tables_by_id("http://example.com")

        assert isinstance(result, dict)
        assert "stats_standard" in result
        assert "stats_squads_standard_for" in result
        assert "stats_squads_standard_against" in result

    def test_get_tables_by_id_excludes_tables_without_id(self):
        """Should exclude tables without ID attribute."""
        html = """
        <html>
            <body>
                <table><tbody><tr><td>No ID</td></tr></tbody></table>
                <table id="has_id"><tbody><tr><td>Has ID</td></tr></tbody></table>
            </body>
        </html>
        """
        with patch("asyncio.get_event_loop", mock_async_run(html)):
            result = scrape.get_tables_by_id("http://example.com")

        assert len(result) == 1
        assert "has_id" in result

    def test_get_tables_by_id_returns_tbody(self):
        """Should return tbody elements, not full table."""
        with patch("asyncio.get_event_loop", mock_async_run(SAMPLE_HTML)):
            result = scrape.get_tables_by_id("http://example.com")

        # Verify we got tbody, not full table (tbody won't have thead)
        for table_id, tbody in result.items():
            assert tbody.name == "tbody"
