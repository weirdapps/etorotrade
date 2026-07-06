"""
Coverage tests for trade_modules/liquidity_filter.py

Targets the 51 uncovered lines: caching logic, tier lookups, DataFrame
column discovery, transaction-cost arithmetic, and error paths.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trade_modules import liquidity_filter
from trade_modules.liquidity_filter import (
    _ADV_CACHE_TTL_HOURS,
    TIER_MIN_ADV,
    TIER_SPREAD_BPS,
    calculate_cost_adjusted_return,
    check_liquidity,
    estimate_transaction_cost,
    filter_by_liquidity,
    get_adv,
    invalidate_cache,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure the module-level ADV cache is empty before and after each test."""
    invalidate_cache()
    yield
    invalidate_cache()


def _make_hist(close: float = 100.0, volume: int = 1_000_000, rows: int = 20):
    """Build a minimal yfinance-style DataFrame."""
    dates = pd.date_range(end=datetime.now(), periods=rows, freq="B")
    return pd.DataFrame(
        {"Close": [close] * rows, "Volume": [volume] * rows},
        index=dates,
    )


# ===================================================================
# _fetch_adv
# ===================================================================


class TestFetchAdv:
    """Tests for _fetch_adv (private helper)."""

    def test_returns_dollar_volume(self, monkeypatch):
        hist = _make_hist(close=50.0, volume=2_000_000)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        monkeypatch.setattr(liquidity_filter, "yf", mock_yf, raising=False)
        # Reimport not needed -- _fetch_adv does `import yfinance as yf` inside
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = liquidity_filter._fetch_adv("AAPL")
        assert result == pytest.approx(50.0 * 2_000_000, rel=1e-6)

    def test_returns_none_for_empty_hist(self, monkeypatch):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = liquidity_filter._fetch_adv("BAD")
        assert result is None

    def test_returns_none_when_hist_is_none(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = None
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = liquidity_filter._fetch_adv("BAD")
        assert result is None

    def test_returns_none_for_short_hist(self):
        hist = _make_hist(rows=3)  # < 5 rows
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = liquidity_filter._fetch_adv("SHORT")
        assert result is None

    def test_returns_none_for_zero_dollar_volume(self):
        hist = _make_hist(close=100.0, volume=0)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = liquidity_filter._fetch_adv("ZERO")
        assert result is None

    def test_returns_none_on_exception(self):
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = RuntimeError("network")
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = liquidity_filter._fetch_adv("ERR")
        assert result is None


# ===================================================================
# get_adv  (caching layer)
# ===================================================================


class TestGetAdv:
    """Tests for get_adv and the TTL-based cache."""

    def test_cache_hit(self, monkeypatch):
        """A fresh cache entry should be returned without calling _fetch_adv."""
        with liquidity_filter._adv_cache_lock:
            liquidity_filter._adv_cache["CACHED"] = (42.0, datetime.now())
        # Patch _fetch_adv so we can verify it's NOT called
        monkeypatch.setattr(liquidity_filter, "_fetch_adv", lambda t, **kw: 999.0)
        assert get_adv("CACHED") == 42.0

    def test_cache_miss_fetches(self, monkeypatch):
        monkeypatch.setattr(liquidity_filter, "_fetch_adv", lambda t, **kw: 123.0)
        assert get_adv("NEW") == 123.0
        # Verify it was cached
        with liquidity_filter._adv_cache_lock:
            assert "NEW" in liquidity_filter._adv_cache

    def test_stale_cache_refetches(self, monkeypatch):
        """An expired entry should be re-fetched."""
        stale_ts = datetime.now() - timedelta(hours=_ADV_CACHE_TTL_HOURS + 1)
        with liquidity_filter._adv_cache_lock:
            liquidity_filter._adv_cache["STALE"] = (1.0, stale_ts)
        monkeypatch.setattr(liquidity_filter, "_fetch_adv", lambda t, **kw: 999.0)
        assert get_adv("STALE") == 999.0

    def test_cache_not_updated_when_fetch_returns_none(self, monkeypatch):
        monkeypatch.setattr(liquidity_filter, "_fetch_adv", lambda t, **kw: None)
        result = get_adv("MISSING")
        assert result is None
        with liquidity_filter._adv_cache_lock:
            assert "MISSING" not in liquidity_filter._adv_cache


# ===================================================================
# check_liquidity
# ===================================================================


class TestCheckLiquidity:
    """Tests for check_liquidity tier/pass logic."""

    def test_adv_unavailable_mega_passes(self, monkeypatch):
        # MEGA stays lenient on missing ADV (blue-chips are practically always tradable).
        monkeypatch.setattr(liquidity_filter, "get_adv", lambda t: None)
        result = check_liquidity("X", "MEGA")
        assert result["passes"] is True
        assert result["adv"] is None
        assert result["reason"] == "adv_unavailable"

    def test_adv_unavailable_non_mega_fails_closed(self, monkeypatch):
        # FAIL-SAFE: missing ADV must fail the gate for non-MEGA tiers.
        monkeypatch.setattr(liquidity_filter, "get_adv", lambda t: None)
        for tier in ("LARGE", "MID", "SMALL", "MICRO"):
            result = check_liquidity("X", tier)
            assert result["passes"] is False, f"{tier} should fail-closed on missing ADV"
            assert result["adv"] is None
            assert result["reason"] == "adv_unavailable"

    def test_sufficient_liquidity(self, monkeypatch):
        monkeypatch.setattr(liquidity_filter, "get_adv", lambda t: 100_000_000)
        result = check_liquidity("AAPL", "MEGA")
        assert result["passes"] is True
        assert result["adv"] == 100_000_000
        assert result["reason"] == "sufficient"
        assert result["min_adv"] == TIER_MIN_ADV["MEGA"]
        assert result["spread_cost_bps"] == TIER_SPREAD_BPS["MEGA"]

    def test_insufficient_liquidity(self, monkeypatch):
        monkeypatch.setattr(liquidity_filter, "get_adv", lambda t: 1_000_000)
        result = check_liquidity("PENNY", "MEGA")
        assert result["passes"] is False
        assert "below" in result["reason"]

    def test_tier_none_defaults_to_mid(self, monkeypatch):
        monkeypatch.setattr(liquidity_filter, "get_adv", lambda t: 15_000_000)
        result = check_liquidity("X", None)
        assert result["min_adv"] == TIER_MIN_ADV["MID"]
        assert result["spread_cost_bps"] == TIER_SPREAD_BPS["MID"]

    def test_tier_empty_string_defaults_to_mid(self, monkeypatch):
        monkeypatch.setattr(liquidity_filter, "get_adv", lambda t: 15_000_000)
        result = check_liquidity("X", "")
        assert result["min_adv"] == TIER_MIN_ADV["MID"]

    def test_unknown_tier_defaults_to_mid(self, monkeypatch):
        monkeypatch.setattr(liquidity_filter, "get_adv", lambda t: 15_000_000)
        result = check_liquidity("X", "NANO")
        assert result["min_adv"] == TIER_MIN_ADV["MID"]
        assert result["spread_cost_bps"] == TIER_SPREAD_BPS["MID"]

    @pytest.mark.parametrize("tier", ["MEGA", "LARGE", "MID", "SMALL", "MICRO"])
    def test_all_tiers_return_correct_thresholds(self, tier, monkeypatch):
        monkeypatch.setattr(liquidity_filter, "get_adv", lambda t: 1e12)
        result = check_liquidity("X", tier)
        assert result["min_adv"] == TIER_MIN_ADV[tier]
        assert result["spread_cost_bps"] == TIER_SPREAD_BPS[tier]

    def test_case_insensitive_tier(self, monkeypatch):
        monkeypatch.setattr(liquidity_filter, "get_adv", lambda t: 1e12)
        result = check_liquidity("X", "mega")
        assert result["min_adv"] == TIER_MIN_ADV["MEGA"]

    def test_exact_threshold_passes(self, monkeypatch):
        """ADV exactly at the minimum should pass."""
        monkeypatch.setattr(liquidity_filter, "get_adv", lambda t: TIER_MIN_ADV["SMALL"])
        result = check_liquidity("X", "SMALL")
        assert result["passes"] is True


# ===================================================================
# estimate_transaction_cost
# ===================================================================


class TestEstimateTransactionCost:
    """Tests for the transaction cost estimator."""

    def test_basic_cost_structure(self):
        result = estimate_transaction_cost(10_000, "MEGA", holding_period_days=90)
        assert "spread_cost" in result
        assert "financing_cost" in result
        assert "total_cost" in result
        assert "total_cost_pct" in result
        assert result["total_cost"] == pytest.approx(
            result["spread_cost"] + result["financing_cost"], abs=0.01
        )

    def test_spread_cost_calculation(self):
        # MEGA = 2 bps; round trip = 2 * (10000 * 2/10000) = 4
        result = estimate_transaction_cost(10_000, "MEGA", holding_period_days=0)
        expected_spread = 10_000 * (2.0 / 10000) * 2
        assert result["spread_cost"] == pytest.approx(expected_spread, abs=0.01)

    def test_financing_cost_calculation(self):
        result = estimate_transaction_cost(10_000, "MEGA", holding_period_days=365)
        expected_financing = 10_000 * 0.064
        assert result["financing_cost"] == pytest.approx(expected_financing, abs=0.01)

    def test_zero_position_size(self):
        result = estimate_transaction_cost(0, "MEGA", holding_period_days=90)
        assert result["total_cost_pct"] == 0.0
        assert result["spread_cost"] == 0.0
        assert result["financing_cost"] == 0.0

    def test_zero_holding_period(self):
        result = estimate_transaction_cost(10_000, "MID", holding_period_days=0)
        assert result["financing_cost"] == 0.0
        assert result["spread_cost"] > 0

    def test_tier_none_defaults_to_mid(self):
        result = estimate_transaction_cost(10_000, None, holding_period_days=90)
        expected_spread = 10_000 * (TIER_SPREAD_BPS["MID"] / 10000) * 2
        assert result["spread_cost"] == pytest.approx(expected_spread, abs=0.01)

    def test_unknown_tier_defaults_to_mid(self):
        result_unknown = estimate_transaction_cost(10_000, "NANO", holding_period_days=90)
        result_mid = estimate_transaction_cost(10_000, "MID", holding_period_days=90)
        assert result_unknown == result_mid

    @pytest.mark.parametrize("tier", ["MEGA", "LARGE", "MID", "SMALL", "MICRO"])
    def test_spread_increases_with_smaller_tier(self, tier):
        result = estimate_transaction_cost(10_000, tier, holding_period_days=0)
        expected_spread = 10_000 * (TIER_SPREAD_BPS[tier] / 10000) * 2
        assert result["spread_cost"] == pytest.approx(expected_spread, abs=0.01)

    def test_total_cost_pct_accuracy(self):
        result = estimate_transaction_cost(50_000, "LARGE", holding_period_days=180)
        expected_pct = (result["total_cost"] / 50_000) * 100
        assert result["total_cost_pct"] == pytest.approx(expected_pct, abs=0.01)


# ===================================================================
# calculate_cost_adjusted_return
# ===================================================================


class TestCostAdjustedReturn:
    """Tests for calculate_cost_adjusted_return."""

    def test_positive_net_return(self):
        result = calculate_cost_adjusted_return(20.0, 10_000, "MEGA", 90)
        assert result < 20.0
        assert result > 0

    def test_negative_net_return(self):
        """Small expected return eaten by costs."""
        result = calculate_cost_adjusted_return(0.5, 10_000, "MICRO", 365)
        assert result < 0.5  # costs exceed the tiny return

    def test_zero_expected_return(self):
        result = calculate_cost_adjusted_return(0.0, 10_000, "MID", 90)
        assert result < 0

    def test_large_expected_return(self):
        result = calculate_cost_adjusted_return(100.0, 10_000, "MEGA", 30)
        assert result > 90  # costs are small relative to 100%


# ===================================================================
# filter_by_liquidity  (DataFrame logic)
# ===================================================================


class TestFilterByLiquidity:
    """Tests for filter_by_liquidity DataFrame processing."""

    @pytest.fixture
    def _mock_check(self, monkeypatch):
        """Default: everything passes with ADV = 100M."""
        monkeypatch.setattr(
            liquidity_filter,
            "check_liquidity",
            lambda ticker, tier: {
                "passes": True,
                "adv": 100_000_000,
                "min_adv": 50_000_000,
                "spread_cost_bps": 2.0,
                "reason": "sufficient",
            },
        )

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = filter_by_liquidity(df)
        assert result.empty

    def test_standard_columns(self, _mock_check):
        df = pd.DataFrame(
            {"ticker": ["AAPL", "MSFT"], "tier": ["MEGA", "LARGE"], "price": [150, 300]}
        )
        result = filter_by_liquidity(df)
        assert "adv" in result.columns
        assert "liquidity_pass" in result.columns
        assert "spread_cost_bps" in result.columns
        assert result["liquidity_pass"].all()

    def test_ticker_col_TKR(self, _mock_check):
        df = pd.DataFrame({"TKR": ["AAPL"], "tier": ["MEGA"]})
        result = filter_by_liquidity(df, ticker_col="missing_col")
        assert "liquidity_pass" in result.columns

    def test_ticker_col_TICKER(self, _mock_check):
        df = pd.DataFrame({"TICKER": ["AAPL"], "tier": ["MEGA"]})
        result = filter_by_liquidity(df, ticker_col="missing_col")
        assert "liquidity_pass" in result.columns

    def test_ticker_col_symbol(self, _mock_check):
        df = pd.DataFrame({"symbol": ["AAPL"], "tier": ["MEGA"]})
        result = filter_by_liquidity(df, ticker_col="missing_col")
        assert "liquidity_pass" in result.columns

    def test_ticker_from_index(self, _mock_check):
        df = pd.DataFrame({"tier": ["MEGA"]}, index=pd.Index(["AAPL"], name="TKR"))
        result = filter_by_liquidity(df, ticker_col="missing_col")
        assert "liquidity_pass" in result.columns
        # Temp column should be cleaned up
        assert "_ticker" not in result.columns

    def test_ticker_from_index_named_ticker(self, _mock_check):
        df = pd.DataFrame({"tier": ["MEGA"]}, index=pd.Index(["AAPL"], name="ticker"))
        result = filter_by_liquidity(df, ticker_col="missing_col")
        assert "_ticker" not in result.columns

    def test_ticker_from_index_named_TICKER(self, _mock_check):
        df = pd.DataFrame({"tier": ["MEGA"]}, index=pd.Index(["AAPL"], name="TICKER"))
        result = filter_by_liquidity(df, ticker_col="missing_col")
        assert "_ticker" not in result.columns

    def test_no_ticker_column_returns_unmodified(self, _mock_check):
        df = pd.DataFrame({"price": [100], "tier": ["MEGA"]})
        result = filter_by_liquidity(df, ticker_col="missing_col")
        # Should return with no new columns since ticker wasn't found
        assert "liquidity_pass" not in result.columns

    def test_tier_col_TIER(self, monkeypatch):
        captured = {}

        def mock_check(ticker, tier):
            captured["tier"] = tier
            return {
                "passes": True,
                "adv": 100_000_000,
                "min_adv": 50_000_000,
                "spread_cost_bps": 2.0,
                "reason": "sufficient",
            }

        monkeypatch.setattr(liquidity_filter, "check_liquidity", mock_check)
        df = pd.DataFrame({"ticker": ["AAPL"], "TIER": ["LARGE"]})
        filter_by_liquidity(df, tier_col="missing_col")
        assert captured["tier"] == "LARGE"

    def test_tier_col_CAP(self, monkeypatch):
        captured = {}

        def mock_check(ticker, tier):
            captured["tier"] = tier
            return {
                "passes": True,
                "adv": 100_000_000,
                "min_adv": 50_000_000,
                "spread_cost_bps": 2.0,
                "reason": "sufficient",
            }

        monkeypatch.setattr(liquidity_filter, "check_liquidity", mock_check)
        df = pd.DataFrame({"ticker": ["AAPL"], "CAP": ["SMALL"]})
        filter_by_liquidity(df, tier_col="missing_col")
        assert captured["tier"] == "SMALL"

    def test_no_tier_col_defaults_mid(self, monkeypatch):
        captured = {}

        def mock_check(ticker, tier):
            captured["tier"] = tier
            return {
                "passes": True,
                "adv": 100_000_000,
                "min_adv": 50_000_000,
                "spread_cost_bps": 2.0,
                "reason": "sufficient",
            }

        monkeypatch.setattr(liquidity_filter, "check_liquidity", mock_check)
        df = pd.DataFrame({"ticker": ["AAPL"]})
        filter_by_liquidity(df, tier_col="missing_col")
        assert captured["tier"] == "MID"

    def test_failed_stocks_logged(self, monkeypatch, caplog):
        def mock_check(ticker, tier):
            return {
                "passes": False,
                "adv": 1_000_000,
                "min_adv": 50_000_000,
                "spread_cost_bps": 2.0,
                "reason": "adv_1.0M_below_50M",
            }

        monkeypatch.setattr(liquidity_filter, "check_liquidity", mock_check)
        df = pd.DataFrame({"ticker": ["PENNY1", "PENNY2"], "tier": ["MEGA", "MEGA"]})
        import logging

        with caplog.at_level(logging.INFO, logger="trade_modules.liquidity_filter"):
            result = filter_by_liquidity(df)
        assert (~result["liquidity_pass"]).sum() == 2
        assert "2 stocks below ADV threshold" in caplog.text

    def test_adv_none_stored_as_nan(self, monkeypatch):
        def mock_check(ticker, tier):
            return {
                "passes": True,
                "adv": None,
                "min_adv": 10_000_000,
                "spread_cost_bps": 10.0,
                "reason": "adv_unavailable",
            }

        monkeypatch.setattr(liquidity_filter, "check_liquidity", mock_check)
        df = pd.DataFrame({"ticker": ["X"], "tier": ["MID"]})
        result = filter_by_liquidity(df)
        assert np.isnan(result["adv"].iloc[0])


# ===================================================================
# invalidate_cache
# ===================================================================


class TestInvalidateCache:
    """Tests for invalidate_cache."""

    def test_clears_cache(self, monkeypatch):
        with liquidity_filter._adv_cache_lock:
            liquidity_filter._adv_cache["A"] = (1.0, datetime.now())
        invalidate_cache()
        with liquidity_filter._adv_cache_lock:
            assert len(liquidity_filter._adv_cache) == 0
