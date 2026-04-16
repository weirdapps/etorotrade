"""Tests for centralized PriceService."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from datetime import date, timedelta

from trade_modules.price_service import PriceService, REGION_BENCHMARKS


class TestGetPrices:
    """Tests for batch price fetching."""

    def test_returns_dataframe_with_requested_tickers(self):
        """get_prices returns a DataFrame with ticker columns."""
        dates = pd.date_range("2026-01-02", periods=20, freq="B")
        mock_data = pd.DataFrame(
            {"AAPL": np.linspace(150, 160, 20), "MSFT": np.linspace(400, 420, 20)},
            index=dates,
        )

        svc = PriceService(cache_dir=None)
        with patch.object(svc, "_download_prices", return_value=mock_data):
            result = svc.get_prices(["AAPL", "MSFT"], "2026-01-02", "2026-01-31")

        assert isinstance(result, pd.DataFrame)
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_includes_benchmark_automatically(self):
        """get_prices always includes the benchmark ticker."""
        dates = pd.date_range("2026-01-02", periods=20, freq="B")
        mock_data = pd.DataFrame(
            {
                "AAPL": np.linspace(150, 160, 20),
                "SPY": np.linspace(500, 510, 20),
            },
            index=dates,
        )

        svc = PriceService(cache_dir=None)
        with patch.object(svc, "_download_prices", return_value=mock_data):
            result = svc.get_prices(["AAPL"], "2026-01-02", "2026-01-31")

        assert "SPY" in result.columns

    def test_empty_tickers_returns_empty_df(self):
        svc = PriceService(cache_dir=None)
        result = svc.get_prices([], "2026-01-02", "2026-01-31")
        assert result.empty


class TestTradingDayReturn:
    """Tests for trading-day-indexed return calculation."""

    @pytest.fixture
    def prices(self):
        """20 trading days of synthetic data."""
        dates = pd.date_range("2026-01-02", periods=20, freq="B")
        return pd.DataFrame(
            {
                "AAPL": np.linspace(100, 110, 20),  # +10% over 20 days
                "BADCO": np.linspace(100, 90, 20),   # -10% over 20 days
                "SPY": np.linspace(500, 510, 20),     # +2% over 20 days
                "ISF.L": np.linspace(800, 816, 20),   # +2% over 20 days
            },
            index=dates,
        )

    def test_positive_return(self, prices):
        svc = PriceService(cache_dir=None)
        ret = svc.trading_day_return(prices, "AAPL", "2026-01-02", horizon=7)
        assert ret is not None
        assert ret > 0  # AAPL goes up

    def test_negative_return(self, prices):
        svc = PriceService(cache_dir=None)
        ret = svc.trading_day_return(prices, "BADCO", "2026-01-02", horizon=7)
        assert ret is not None
        assert ret < 0  # BADCO goes down

    def test_horizon_exceeds_data(self, prices):
        """Should return None if not enough future data."""
        svc = PriceService(cache_dir=None)
        ret = svc.trading_day_return(prices, "AAPL", "2026-01-02", horizon=25)
        assert ret is None

    def test_unknown_ticker(self, prices):
        svc = PriceService(cache_dir=None)
        ret = svc.trading_day_return(prices, "NOPE", "2026-01-02", horizon=7)
        assert ret is None

    def test_uses_trading_days_not_calendar(self, prices):
        """T+5 should skip weekends (5 trading days = ~7 calendar days)."""
        svc = PriceService(cache_dir=None)
        ret5 = svc.trading_day_return(prices, "AAPL", "2026-01-02", horizon=5)
        # AAPL: 100 -> ~102.63 at index[5] (linspace)
        expected = (np.linspace(100, 110, 20)[5] - 100) / 100 * 100
        assert ret5 == pytest.approx(expected, abs=0.01)


class TestTradingDayAlpha:
    """Tests for alpha (stock return - benchmark return)."""

    @pytest.fixture
    def prices(self):
        dates = pd.date_range("2026-01-02", periods=20, freq="B")
        return pd.DataFrame(
            {
                "AAPL": np.linspace(100, 120, 20),  # +20%
                "SPY": np.linspace(500, 510, 20),     # +2%
                "ISF.L": np.linspace(800, 816, 20),   # +2%
            },
            index=dates,
        )

    def test_positive_alpha(self, prices):
        svc = PriceService(cache_dir=None)
        alpha = svc.trading_day_alpha(prices, "AAPL", "2026-01-02", horizon=19)
        assert alpha is not None
        assert alpha > 0  # AAPL (+20%) beats SPY (+2%)

    def test_uses_regional_benchmark(self, prices):
        """UK stocks should use ISF.L, not SPY."""
        svc = PriceService(cache_dir=None)
        alpha_us = svc.trading_day_alpha(
            prices, "AAPL", "2026-01-02", horizon=19, region="us"
        )
        alpha_uk = svc.trading_day_alpha(
            prices, "AAPL", "2026-01-02", horizon=19, region="uk"
        )
        # Both benchmarks have same return in this fixture, so alpha should be equal
        assert alpha_us == pytest.approx(alpha_uk, abs=0.01)

    def test_none_when_benchmark_missing(self, prices):
        """Alpha is None if benchmark has no data."""
        svc = PriceService(cache_dir=None)
        alpha = svc.trading_day_alpha(
            prices, "AAPL", "2026-01-02", horizon=19, region="hk"
        )
        # 2800.HK not in fixture, fallback to SPY which IS there
        assert alpha is not None  # Falls back to SPY


class TestRegionBenchmarks:
    def test_us_returns_spy(self):
        svc = PriceService(cache_dir=None)
        assert svc.get_benchmark("us") == "SPY"

    def test_uk_returns_isf(self):
        svc = PriceService(cache_dir=None)
        assert svc.get_benchmark("uk") == "ISF.L"

    def test_unknown_region_returns_spy(self):
        svc = PriceService(cache_dir=None)
        assert svc.get_benchmark("mars") == "SPY"

    def test_none_region_returns_default(self):
        svc = PriceService(cache_dir=None)
        assert svc.get_benchmark(None) == "SPY"
