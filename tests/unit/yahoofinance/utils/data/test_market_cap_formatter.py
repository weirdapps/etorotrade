#!/usr/bin/env python3
"""
ITERATION 30: Market Cap Formatter Tests
Target: Test advanced market cap formatting utilities
File: yahoofinance/utils/data/market_cap_formatter.py (56 statements, 18% coverage)
"""

import pytest


class TestFormatMarketCapAdvanced:
    """Test advanced market cap formatting."""

    def test_format_none_returns_none(self):
        """Format None value returns None."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(None)

        assert result is None

    def test_format_trillion_large(self):
        """Format large trillion market cap."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(15_000_000_000_000)

        assert result == "15.0T"

    def test_format_trillion_small(self):
        """Format small trillion market cap."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(2_500_000_000_000)

        assert result == "2.50T"

    def test_format_billion_large(self):
        """Format large billion market cap (>= 100B)."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(250_000_000_000)

        assert result == "250B"

    def test_format_billion_medium(self):
        """Format medium billion market cap (>= 10B, < 100B)."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(50_000_000_000)

        assert result == "50.0B"

    def test_format_billion_small(self):
        """Format small billion market cap (< 10B)."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(5_000_000_000)

        assert result == "5.00B"

    def test_format_million(self):
        """Format million market cap."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(500_000_000)

        assert result == "500.00M"

    def test_format_below_million(self):
        """Format value below million threshold."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(500_000)

        assert result == "500,000"

    def test_format_with_custom_config(self):
        """Format with custom configuration."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        config = {
            "billion_suffix": "Bn",
            "small_billion_precision": 3
        }

        result = format_market_cap_advanced(5_000_000_000, config)

        assert result == "5.000Bn"

    def test_format_with_custom_thresholds(self):
        """Format with custom thresholds."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        config = {
            "billion_threshold": 500_000_000,  # Lower threshold
            "billion_suffix": "B"
        }

        result = format_market_cap_advanced(600_000_000, config)

        # Should use billion scale with lower threshold
        assert "B" in result

    def test_format_invalid_value(self):
        """Format invalid value returns None."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced("invalid")

        assert result is None

    def test_format_zero(self):
        """Format zero value."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(0)

        assert result == "0"

    def test_format_negative_value(self):
        """Format negative value."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(-1_000_000_000)

        # Should still format (absolute value logic not in formatter)
        assert result is not None


class TestGetScaleInfo:
    """Test scale info determination."""

    def test_get_scale_info_trillion(self):
        """Get scale info for trillion value."""
        from yahoofinance.utils.data.market_cap_formatter import _get_scale_info

        scale_info = _get_scale_info(2_000_000_000_000, {})

        assert scale_info["scale"] == "trillion"
        assert scale_info["suffix"] == "T"
        assert scale_info["divisor"] == 1_000_000_000_000

    def test_get_scale_info_large_trillion(self):
        """Get scale info for large trillion (>= 10T)."""
        from yahoofinance.utils.data.market_cap_formatter import _get_scale_info

        scale_info = _get_scale_info(15_000_000_000_000, {})

        assert scale_info["scale"] == "trillion"
        assert scale_info["precision"] == 1

    def test_get_scale_info_small_trillion(self):
        """Get scale info for small trillion (< 10T)."""
        from yahoofinance.utils.data.market_cap_formatter import _get_scale_info

        scale_info = _get_scale_info(2_000_000_000_000, {})

        assert scale_info["scale"] == "trillion"
        assert scale_info["precision"] == 2

    def test_get_scale_info_billion(self):
        """Get scale info for billion value."""
        from yahoofinance.utils.data.market_cap_formatter import _get_scale_info

        scale_info = _get_scale_info(50_000_000_000, {})

        assert scale_info["scale"] == "billion"
        assert scale_info["suffix"] == "B"

    def test_get_scale_info_million(self):
        """Get scale info for million value."""
        from yahoofinance.utils.data.market_cap_formatter import _get_scale_info

        scale_info = _get_scale_info(500_000_000, {})

        assert scale_info["scale"] == "million"
        assert scale_info["suffix"] == "M"

    def test_get_scale_info_raw(self):
        """Get scale info for raw value below million."""
        from yahoofinance.utils.data.market_cap_formatter import _get_scale_info

        scale_info = _get_scale_info(500_000, {})

        assert scale_info["scale"] == "raw"
        assert scale_info["suffix"] == ""
        assert scale_info["divisor"] == 1

    def test_get_scale_info_custom_config(self):
        """Get scale info with custom config."""
        from yahoofinance.utils.data.market_cap_formatter import _get_scale_info

        config = {
            "trillion_suffix": "Tn",
            "small_trillion_precision": 3
        }

        scale_info = _get_scale_info(2_000_000_000_000, config)

        assert scale_info["suffix"] == "Tn"
        assert scale_info["precision"] == 3


class TestFormatWithScale:
    """Test formatting with scale info."""

    def test_format_with_scale_trillion(self):
        """Format with trillion scale."""
        from yahoofinance.utils.data.market_cap_formatter import _format_with_scale

        scale_info = {
            "scale": "trillion",
            "divisor": 1_000_000_000_000,
            "suffix": "T",
            "precision": 2
        }

        result = _format_with_scale(2_500_000_000_000, scale_info)

        assert result == "2.50T"

    def test_format_with_scale_billion(self):
        """Format with billion scale."""
        from yahoofinance.utils.data.market_cap_formatter import _format_with_scale

        scale_info = {
            "scale": "billion",
            "divisor": 1_000_000_000,
            "suffix": "B",
            "precision": 2
        }

        result = _format_with_scale(5_000_000_000, scale_info)

        assert result == "5.00B"

    def test_format_with_scale_raw(self):
        """Format with raw scale (no suffix)."""
        from yahoofinance.utils.data.market_cap_formatter import _format_with_scale

        scale_info = {
            "scale": "raw",
            "divisor": 1,
            "suffix": "",
            "precision": 0
        }

        result = _format_with_scale(500_000, scale_info)

        assert result == "500,000"

    def test_format_with_scale_custom_precision(self):
        """Format with custom precision."""
        from yahoofinance.utils.data.market_cap_formatter import _format_with_scale

        scale_info = {
            "scale": "billion",
            "divisor": 1_000_000_000,
            "suffix": "B",
            "precision": 3
        }

        result = _format_with_scale(5_123_456_789, scale_info)

        assert result == "5.123B"


class TestEdgeCases:
    """Test edge cases."""

    def test_format_very_large_value(self):
        """Format very large value (quadrillion)."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(1_000_000_000_000_000)

        # Should use trillion scale with high value
        assert "1000.0T" in result or "1,000.0T" in result

    def test_format_exact_threshold_trillion(self):
        """Format value exactly at trillion threshold."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(1_000_000_000_000)

        assert "1.00T" in result

    def test_format_exact_threshold_billion(self):
        """Format value exactly at billion threshold."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(1_000_000_000)

        assert "1.00B" in result

    def test_format_just_below_threshold(self):
        """Format value just below billion threshold."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(999_999_999)

        # Should use million scale
        assert "M" in result

    def test_format_with_empty_config(self):
        """Format with empty config dict."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(5_000_000_000, {})

        assert result == "5.00B"

    def test_format_float_value(self):
        """Format float value."""
        from yahoofinance.utils.data.market_cap_formatter import format_market_cap_advanced

        result = format_market_cap_advanced(5.5e9)

        assert "5.50B" in result


