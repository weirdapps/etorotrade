#!/usr/bin/env python3
"""
Tests for display utilities package.
Target: Increase coverage for yahoofinance/utils/display/__init__.py
"""

import pytest


class TestDisplayPackageImports:
    """Test display package exports."""

    def test_import_apply_color_formatting(self):
        """Import apply_color_formatting."""
        from yahoofinance.utils.display import apply_color_formatting

        assert apply_color_formatting is not None
        assert callable(apply_color_formatting)

    def test_import_check_confidence_threshold(self):
        """Import check_confidence_threshold."""
        from yahoofinance.utils.display import check_confidence_threshold

        assert check_confidence_threshold is not None
        assert callable(check_confidence_threshold)

    def test_import_display_tabulate_results(self):
        """Import display_tabulate_results."""
        from yahoofinance.utils.display import display_tabulate_results

        assert display_tabulate_results is not None
        assert callable(display_tabulate_results)

    def test_import_format_market_cap(self):
        """Import format_market_cap."""
        from yahoofinance.utils.display import format_market_cap

        assert format_market_cap is not None
        assert callable(format_market_cap)

    def test_import_generate_html_dashboard(self):
        """Import generate_html_dashboard."""
        from yahoofinance.utils.display import generate_html_dashboard

        assert generate_html_dashboard is not None
        assert callable(generate_html_dashboard)

    def test_import_get_output_file_and_title(self):
        """Import get_output_file_and_title."""
        from yahoofinance.utils.display import get_output_file_and_title

        assert get_output_file_and_title is not None
        assert callable(get_output_file_and_title)

    def test_import_handle_manual_input_tickers(self):
        """Import handle_manual_input_tickers."""
        from yahoofinance.utils.display import handle_manual_input_tickers

        assert handle_manual_input_tickers is not None
        assert callable(handle_manual_input_tickers)

    def test_import_parse_row_values(self):
        """Import parse_row_values."""
        from yahoofinance.utils.display import parse_row_values

        assert parse_row_values is not None
        assert callable(parse_row_values)

    def test_import_print_confidence_details(self):
        """Import print_confidence_details."""
        from yahoofinance.utils.display import print_confidence_details

        assert print_confidence_details is not None
        assert callable(print_confidence_details)

    def test_all_exports_defined(self):
        """__all__ exports are defined."""
        from yahoofinance.utils import display

        assert hasattr(display, '__all__')
        assert len(display.__all__) == 9
