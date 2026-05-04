"""
Tests for Sector-Relative Momentum Provider
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trade_modules.sector_relative_momentum import (
    UNDERPERFORMANCE_THRESHOLD,
    calculate_relative_momentum,
    get_relative_momentum_flags,
    invalidate_cache,
    is_underperforming_sector,
)


class TestSectorRelativeMomentum:
    """Tests for sector-relative momentum calculations."""

    def setup_method(self):
        """Reset cache before each test."""
        invalidate_cache()

    def test_calculate_relative_momentum_outperforming(self):
        """Test calculation for stock outperforming sector."""
        with patch("trade_modules.sector_relative_momentum._fetch_return") as mock_fetch:
            # Stock up 50%, sector up 20% -> relative momentum +30%
            mock_fetch.side_effect = [50.0, 20.0]

            result = calculate_relative_momentum("NVDA", "Technology")

            assert result == pytest.approx(30.0)
            assert mock_fetch.call_count == 2

    def test_calculate_relative_momentum_underperforming(self):
        """Test calculation for stock underperforming sector."""
        with patch("trade_modules.sector_relative_momentum._fetch_return") as mock_fetch:
            # Stock up 5%, sector up 25% -> relative momentum -20%
            mock_fetch.side_effect = [5.0, 25.0]

            result = calculate_relative_momentum("INTC", "Technology")

            assert result == pytest.approx(-20.0)

    def test_calculate_relative_momentum_no_sector(self):
        """Test returns None for missing sector."""
        result = calculate_relative_momentum("AAPL", None)
        assert result is None

    def test_calculate_relative_momentum_unknown_sector(self):
        """Test returns None for unmapped sector."""
        result = calculate_relative_momentum("AAPL", "Unknown Sector")
        assert result is None

    def test_calculate_relative_momentum_fetch_failure(self):
        """Test handles fetch failures gracefully."""
        with patch("trade_modules.sector_relative_momentum._fetch_return") as mock_fetch:
            mock_fetch.return_value = None

            result = calculate_relative_momentum("BADTICKER", "Technology")

            assert result is None

    def test_get_relative_momentum_flags_batch(self):
        """Test batch processing of multiple tickers."""
        with patch("trade_modules.sector_relative_momentum._fetch_return") as mock_fetch:
            # NVDA outperforming (+30%), INTC underperforming (-20%)
            mock_fetch.side_effect = [
                50.0,
                20.0,  # NVDA vs XLK
                5.0,
                25.0,  # INTC vs XLK
            ]

            tickers = [
                ("NVDA", "Technology"),
                ("INTC", "Technology"),
            ]

            results = get_relative_momentum_flags(tickers)

            assert len(results) == 2

            # NVDA outperforming
            assert results["NVDA"]["relative_momentum"] == pytest.approx(30.0)
            assert results["NVDA"]["underperforming"] is False

            # INTC underperforming by >15%
            assert results["INTC"]["relative_momentum"] == pytest.approx(-20.0)
            assert results["INTC"]["underperforming"] is True

    def test_get_relative_momentum_flags_uses_cache(self):
        """Test that batch processing uses cache."""
        with patch("trade_modules.sector_relative_momentum._fetch_return") as mock_fetch:
            mock_fetch.side_effect = [50.0, 20.0]  # Only one set of returns

            tickers = [("NVDA", "Technology")]

            # First call - should fetch
            results1 = get_relative_momentum_flags(tickers)
            assert results1["NVDA"]["relative_momentum"] == pytest.approx(30.0)
            assert mock_fetch.call_count == 2

            # Second call - should use cache
            results2 = get_relative_momentum_flags(tickers)
            assert results2["NVDA"]["relative_momentum"] == pytest.approx(30.0)
            assert mock_fetch.call_count == 2  # No additional calls

    def test_is_underperforming_sector_true(self):
        """Test underperformance detection."""
        with patch("trade_modules.sector_relative_momentum._fetch_return") as mock_fetch:
            # Stock down 10%, sector up 10% -> relative -20% (underperforming)
            mock_fetch.side_effect = [-10.0, 10.0]

            result = is_underperforming_sector("INTC", "Technology")

            assert result is True

    def test_is_underperforming_sector_false(self):
        """Test non-underperformance."""
        with patch("trade_modules.sector_relative_momentum._fetch_return") as mock_fetch:
            # Stock up 20%, sector up 15% -> relative +5% (outperforming)
            mock_fetch.side_effect = [20.0, 15.0]

            result = is_underperforming_sector("NVDA", "Technology")

            assert result is False

    def test_is_underperforming_sector_borderline(self):
        """Test borderline underperformance."""
        with patch("trade_modules.sector_relative_momentum._fetch_return") as mock_fetch:
            # Stock up 5%, sector up 19% -> relative -14% (just below threshold)
            mock_fetch.side_effect = [5.0, 19.0]

            result = is_underperforming_sector("TEST", "Technology", threshold=15.0)

            assert result is False

    def test_fetch_return_valid_data(self):
        """Test _fetch_return with valid historical data."""
        from trade_modules.sector_relative_momentum import _fetch_return

        mock_hist = pd.DataFrame({"Close": [100.0, 110.0, 120.0, 130.0, 150.0]})

        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_hist
            mock_ticker_class.return_value = mock_ticker

            result = _fetch_return("AAPL", period_days=252)

            # (150 - 100) / 100 * 100 = 50%
            assert result == pytest.approx(50.0)

    def test_fetch_return_empty_history(self):
        """Test _fetch_return handles empty history."""
        from trade_modules.sector_relative_momentum import _fetch_return

        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_ticker_class.return_value = mock_ticker

            result = _fetch_return("BADTICKER", period_days=252)

            assert result is None

    def test_fetch_return_exception(self):
        """Test _fetch_return handles exceptions."""
        from trade_modules.sector_relative_momentum import _fetch_return

        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_ticker_class.side_effect = Exception("API error")

            result = _fetch_return("BADTICKER", period_days=252)

            assert result is None

    def test_underperformance_threshold_constant(self):
        """Test the underperformance threshold constant is reasonable."""
        assert UNDERPERFORMANCE_THRESHOLD == pytest.approx(15.0)

    def test_cache_invalidation(self):
        """Test cache invalidation works."""

        with patch("trade_modules.sector_relative_momentum._fetch_return") as mock_fetch:
            mock_fetch.side_effect = [50.0, 20.0, 60.0, 20.0]

            # First call - populate cache
            tickers = [("NVDA", "Technology")]
            results1 = get_relative_momentum_flags(tickers)
            assert results1["NVDA"]["relative_momentum"] == pytest.approx(30.0)

            # Invalidate cache
            invalidate_cache()

            # Second call - should re-fetch
            results2 = get_relative_momentum_flags(tickers)
            assert results2["NVDA"]["relative_momentum"] == pytest.approx(40.0)  # New value
            assert mock_fetch.call_count == 4  # 2 calls each time

    def test_different_period_days(self):
        """Test calculation with different lookback periods."""
        with patch("trade_modules.sector_relative_momentum._fetch_return") as mock_fetch:
            mock_fetch.side_effect = [30.0, 15.0]

            result = calculate_relative_momentum("AAPL", "Technology", period_days=126)

            assert result == pytest.approx(15.0)
            # Verify period_days was passed correctly
            assert mock_fetch.call_args_list[0][0][1] == 126
            assert mock_fetch.call_args_list[1][0][1] == 126
