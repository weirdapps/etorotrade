"""Tests for IPO auto-detection using yfinance fallback."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from trade_modules.analysis import signals
from trade_modules.analysis.signals import is_recent_ipo


class FakeYamlConfig:
    """Minimal yaml_config mock for testing."""

    def __init__(self, config):
        self._config = config

    def load_config(self):
        return self._config


def make_config(known_ipos=None, enabled=True, auto_detect=True):
    cfg = {"ipo_grace_period": {"enabled": enabled, "auto_detect": auto_detect}}
    if known_ipos:
        cfg["ipo_grace_period"]["known_ipos"] = known_ipos
    return cfg


class TestIPOConfigOverride:
    """Config-based known_ipos should always take priority."""

    def test_known_ipo_within_grace_period(self):
        recent_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        config = make_config(known_ipos={"TEST": recent_date})
        assert is_recent_ipo("TEST", FakeYamlConfig(config)) is True

    def test_known_ipo_outside_grace_period(self):
        old_date = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")
        config = make_config(known_ipos={"TEST": old_date})
        assert is_recent_ipo("TEST", FakeYamlConfig(config)) is False

    def test_known_ipo_invalid_date(self):
        config = make_config(known_ipos={"TEST": "not-a-date"})
        assert is_recent_ipo("TEST", FakeYamlConfig(config)) is False

    def test_disabled_returns_false(self):
        config = make_config(enabled=False)
        assert is_recent_ipo("AAPL", FakeYamlConfig(config)) is False


class TestIPOAutoDetect:
    """yfinance fallback for tickers not in known_ipos."""

    def setup_method(self):
        # Clear the module-level cache before each test
        signals._ipo_date_cache.clear()

    @patch("yfinance.Ticker")
    def test_recent_ipo_detected_from_yfinance(self, mock_ticker_cls):
        # Simulate a stock that started trading 3 months ago
        first_trade = datetime.now() - timedelta(days=90)
        index = pd.DatetimeIndex([first_trade, first_trade + timedelta(days=1)])
        mock_hist = pd.DataFrame({"Close": [100, 101]}, index=index)
        mock_ticker_cls.return_value.history.return_value = mock_hist

        config = make_config()
        result = is_recent_ipo("NEWSTOCK", FakeYamlConfig(config))
        assert result is True

    @patch("yfinance.Ticker")
    def test_old_stock_not_detected_as_ipo(self, mock_ticker_cls):
        # Simulate a stock that started trading 5 years ago
        first_trade = datetime.now() - timedelta(days=5 * 365)
        index = pd.DatetimeIndex([first_trade, first_trade + timedelta(days=1)])
        mock_hist = pd.DataFrame({"Close": [50, 51]}, index=index)
        mock_ticker_cls.return_value.history.return_value = mock_hist

        config = make_config()
        result = is_recent_ipo("OLDSTOCK", FakeYamlConfig(config))
        assert result is False

    @patch("yfinance.Ticker")
    def test_empty_history_returns_false(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()

        config = make_config()
        assert is_recent_ipo("NODATA", FakeYamlConfig(config)) is False

    @patch("yfinance.Ticker")
    def test_yfinance_exception_returns_false(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.side_effect = Exception("network error")

        config = make_config()
        assert is_recent_ipo("ERRTICKER", FakeYamlConfig(config)) is False

    @patch("yfinance.Ticker")
    def test_cache_prevents_repeated_api_calls(self, mock_ticker_cls):
        first_trade = datetime.now() - timedelta(days=90)
        index = pd.DatetimeIndex([first_trade])
        mock_hist = pd.DataFrame({"Close": [100]}, index=index)
        mock_ticker_cls.return_value.history.return_value = mock_hist

        config = make_config()
        yaml_cfg = FakeYamlConfig(config)

        # First call should query yfinance
        is_recent_ipo("CACHED", yaml_cfg)
        # Second call should use cache
        is_recent_ipo("CACHED", yaml_cfg)

        # history() should only be called once
        assert mock_ticker_cls.return_value.history.call_count == 1

    @patch("yfinance.Ticker")
    def test_cache_stores_none_on_failure(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.side_effect = Exception("fail")

        config = make_config()
        yaml_cfg = FakeYamlConfig(config)

        is_recent_ipo("FAIL1", yaml_cfg)
        assert "FAIL1" in signals._ipo_date_cache
        assert signals._ipo_date_cache["FAIL1"] is None

        # Second call should return False from cache without calling API
        result = is_recent_ipo("FAIL1", yaml_cfg)
        assert result is False
        assert mock_ticker_cls.return_value.history.call_count == 1

    def test_auto_detect_disabled_skips_yfinance(self):
        config = make_config(auto_detect=False)
        # Should return False without even attempting yfinance
        result = is_recent_ipo("UNKNOWN", FakeYamlConfig(config))
        assert result is False

    def test_config_override_takes_priority_over_auto_detect(self):
        # Even with auto_detect enabled, config known_ipos should win
        recent_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        config = make_config(known_ipos={"KNOWN": recent_date}, auto_detect=True)

        # No yfinance mock needed - config should short-circuit
        result = is_recent_ipo("KNOWN", FakeYamlConfig(config))
        assert result is True
