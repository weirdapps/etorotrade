"""Additional coverage tests for trade_modules/price_service.py

Supplements test_price_service.py by covering:
- _apply_data_fetch_substitutions (module-level helper)
- _download_prices (batching, retries, single/multi-ticker, reverse_map)
- _load_cache / _save_cache (parquet caching)
- Cache integration paths in get_prices
- Edge cases in trading_day_return (all-NaN, zero price)
- Alpha fallback when regional benchmark is missing
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trade_modules.price_service import (
    DEFAULT_CACHE_DIR,
    REGION_BENCHMARKS,
    PriceService,
    _apply_data_fetch_substitutions,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Verify module-level constants are well-formed."""

    def test_region_benchmarks_has_default(self):
        assert "default" in REGION_BENCHMARKS
        assert REGION_BENCHMARKS["default"] == "SPY"

    def test_region_benchmarks_has_standard_regions(self):
        for region in ("us", "eu", "uk", "hk"):
            assert region in REGION_BENCHMARKS

    def test_default_cache_dir_is_path(self):
        from pathlib import Path

        assert isinstance(DEFAULT_CACHE_DIR, Path)
        assert "price_cache" in str(DEFAULT_CACHE_DIR)


# ---------------------------------------------------------------------------
# _apply_data_fetch_substitutions
# ---------------------------------------------------------------------------


class TestApplyDataFetchSubstitutions:
    """Tests for the ticker substitution helper."""

    def test_no_config_returns_original_tickers(self):
        """When config_manager import fails, return tickers unchanged."""
        with patch(
            "trade_modules.price_service.get_config",
            side_effect=ImportError("no config"),
            create=True,
        ):
            # Force the import inside the function to fail
            with patch.dict("sys.modules", {"trade_modules.config_manager": None}):
                result, reverse = _apply_data_fetch_substitutions(["AAPL", "MSFT"])
        assert result == ["AAPL", "MSFT"]
        assert reverse == {}

    def test_empty_substitutions_returns_original(self):
        """When substitutions dict is empty, return tickers unchanged."""
        mock_config = MagicMock()
        mock_config.data_fetch_substitutions = {}
        with patch("trade_modules.config_manager.get_config", return_value=mock_config):
            result, reverse = _apply_data_fetch_substitutions(["AAPL"])
        assert result == ["AAPL"]
        assert reverse == {}

    def test_substitution_applied(self):
        """Known substitution replaces ticker and builds reverse map."""
        mock_config = MagicMock()
        mock_config.data_fetch_substitutions = {"LYXGRE.DE": "GRE.PA"}
        with patch("trade_modules.config_manager.get_config", return_value=mock_config):
            result, reverse = _apply_data_fetch_substitutions(["LYXGRE.DE", "AAPL"])
        assert "GRE.PA" in result
        assert "AAPL" in result
        assert "LYXGRE.DE" not in result
        assert reverse == {"GRE.PA": "LYXGRE.DE"}

    def test_case_insensitive_lookup(self):
        """Tickers are uppercased before lookup."""
        mock_config = MagicMock()
        mock_config.data_fetch_substitutions = {"LYXGRE.DE": "GRE.PA"}
        with patch("trade_modules.config_manager.get_config", return_value=mock_config):
            result, reverse = _apply_data_fetch_substitutions(["lyxgre.de"])
        assert result == ["GRE.PA"]
        assert reverse == {"GRE.PA": "lyxgre.de"}

    def test_none_ticker_skipped(self):
        """None values in tickers list are passed through without crash."""
        mock_config = MagicMock()
        mock_config.data_fetch_substitutions = {"LYXGRE.DE": "GRE.PA"}
        with patch("trade_modules.config_manager.get_config", return_value=mock_config):
            result, reverse = _apply_data_fetch_substitutions([None, "AAPL"])
        assert result == [None, "AAPL"]
        assert reverse == {}

    def test_empty_string_ticker(self):
        """Empty string ticker is passed through (sub = None because falsy)."""
        mock_config = MagicMock()
        mock_config.data_fetch_substitutions = {}
        with patch("trade_modules.config_manager.get_config", return_value=mock_config):
            result, reverse = _apply_data_fetch_substitutions(["", "AAPL"])
        assert result == ["", "AAPL"]

    def test_config_exception_returns_original(self):
        """Any exception from get_config returns tickers unchanged."""
        with patch(
            "trade_modules.config_manager.get_config",
            side_effect=RuntimeError("boom"),
        ):
            result, reverse = _apply_data_fetch_substitutions(["AAPL"])
        assert result == ["AAPL"]
        assert reverse == {}


# ---------------------------------------------------------------------------
# PriceService._download_prices
# ---------------------------------------------------------------------------


class TestDownloadPrices:
    """Tests for the yfinance batch-download method."""

    @pytest.fixture
    def svc(self):
        return PriceService(cache_dir=None)

    def _make_single_ticker_data(self, ticker="AAPL", periods=20):
        dates = pd.date_range("2026-01-02", periods=periods, freq="B")
        return pd.DataFrame(
            {"Close": np.linspace(150, 160, periods)},
            index=dates,
        )

    def _make_multi_ticker_data(self, tickers, periods=20):
        dates = pd.date_range("2026-01-02", periods=periods, freq="B")
        arrays = []
        for t in tickers:
            arrays.append(
                pd.DataFrame(
                    {"Close": np.linspace(100, 110, periods)},
                    index=dates,
                    columns=pd.MultiIndex.from_tuples([(t, "Close")]),
                )
            )
        return pd.concat(arrays, axis=1)

    def test_single_ticker_download(self, svc):
        """Single ticker returns a DataFrame with that ticker as column."""
        mock_data = self._make_single_ticker_data("AAPL")
        with patch("yfinance.download", return_value=mock_data):
            with patch(
                "trade_modules.price_service._apply_data_fetch_substitutions",
                return_value=(["AAPL"], {}),
            ):
                result = svc._download_prices(["AAPL"], "2026-01-02", "2026-01-31")
        assert "AAPL" in result.columns
        assert not result.empty

    def test_multi_ticker_download_multiindex(self, svc):
        """Multi-ticker returns a DataFrame with columns per ticker."""
        tickers = ["AAPL", "MSFT"]
        mock_data = self._make_multi_ticker_data(tickers)
        with patch("yfinance.download", return_value=mock_data):
            with patch(
                "trade_modules.price_service._apply_data_fetch_substitutions",
                return_value=(tickers, {}),
            ):
                result = svc._download_prices(tickers, "2026-01-02", "2026-01-31")
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns

    def test_empty_download_returns_empty_df(self, svc):
        """Empty yfinance response returns empty DataFrame."""
        with patch("yfinance.download", return_value=pd.DataFrame()):
            with patch(
                "trade_modules.price_service._apply_data_fetch_substitutions",
                return_value=(["AAPL"], {}),
            ):
                result = svc._download_prices(["AAPL"], "2026-01-02", "2026-01-31")
        assert result.empty

    def test_retry_on_exception(self, svc):
        """Download retries on exception and succeeds on second attempt."""
        mock_data = self._make_single_ticker_data("AAPL")
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("network error")
            return mock_data

        with patch("yfinance.download", side_effect=side_effect):
            with patch("time.sleep"):  # skip actual wait
                with patch(
                    "trade_modules.price_service._apply_data_fetch_substitutions",
                    return_value=(["AAPL"], {}),
                ):
                    result = svc._download_prices(
                        ["AAPL"], "2026-01-02", "2026-01-31", max_retries=3
                    )
        assert not result.empty
        assert call_count == 2

    def test_all_retries_exhausted_returns_empty(self, svc):
        """When all retries fail, returns empty DataFrame."""
        with patch("yfinance.download", side_effect=ConnectionError("always fails")):
            with patch("time.sleep"):
                with patch(
                    "trade_modules.price_service._apply_data_fetch_substitutions",
                    return_value=(["AAPL"], {}),
                ):
                    result = svc._download_prices(
                        ["AAPL"], "2026-01-02", "2026-01-31", max_retries=2
                    )
        assert result.empty

    def test_batch_splitting(self, svc):
        """Tickers are split into batches of batch_size."""
        tickers = [f"T{i}" for i in range(5)]
        mock_data = self._make_single_ticker_data("T0")
        call_count = 0

        def counting_download(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_data

        with patch("yfinance.download", side_effect=counting_download):
            with patch(
                "trade_modules.price_service._apply_data_fetch_substitutions",
                return_value=(tickers, {}),
            ):
                svc._download_prices(tickers, "2026-01-02", "2026-01-31", batch_size=2)
        # 5 tickers with batch_size=2 -> 3 batches (2+2+1)
        assert call_count == 3

    def test_reverse_map_renames_columns(self, svc):
        """Substituted tickers are renamed back to original symbols."""
        dates = pd.date_range("2026-01-02", periods=5, freq="B")
        mock_data = pd.DataFrame(
            {"Close": np.linspace(100, 110, 5)},
            index=dates,
        )
        with patch("yfinance.download", return_value=mock_data):
            with patch(
                "trade_modules.price_service._apply_data_fetch_substitutions",
                return_value=(["GRE.PA"], {"GRE.PA": "LYXGRE.DE"}),
            ):
                result = svc._download_prices(["LYXGRE.DE"], "2026-01-02", "2026-01-31")
        assert "LYXGRE.DE" in result.columns
        assert "GRE.PA" not in result.columns


# ---------------------------------------------------------------------------
# PriceService._load_cache / _save_cache
# ---------------------------------------------------------------------------


class TestCaching:
    """Tests for parquet cache load/save."""

    def test_load_cache_no_dir(self):
        """No cache_dir means no cache."""
        svc = PriceService(cache_dir=None)
        assert svc._load_cache() is None

    def test_load_cache_file_missing(self, tmp_path):
        """Cache dir exists but no parquet file."""
        svc = PriceService(cache_dir=tmp_path)
        assert svc._load_cache() is None

    def test_save_and_load_cache_roundtrip(self, tmp_path):
        """Save then load produces equivalent DataFrame."""
        svc = PriceService(cache_dir=tmp_path)
        dates = pd.date_range("2026-01-02", periods=5, freq="B")
        prices = pd.DataFrame(
            {"AAPL": [150, 151, 152, 153, 154]},
            index=dates,
        )
        svc._save_cache(prices)

        loaded = svc._load_cache()
        assert loaded is not None
        assert "AAPL" in loaded.columns
        assert len(loaded) == 5
        assert isinstance(loaded.index, pd.DatetimeIndex)

    def test_save_cache_no_dir_is_noop(self):
        """Saving with cache_dir=None does nothing."""
        svc = PriceService(cache_dir=None)
        prices = pd.DataFrame({"AAPL": [150]})
        # Should not raise
        svc._save_cache(prices)

    def test_save_cache_creates_directory(self, tmp_path):
        """Cache dir is created if it doesn't exist."""
        cache_dir = tmp_path / "nested" / "cache"
        svc = PriceService(cache_dir=cache_dir)
        prices = pd.DataFrame({"AAPL": [150]}, index=pd.date_range("2026-01-02", periods=1))
        svc._save_cache(prices)
        assert (cache_dir / "backtest_prices.parquet").exists()

    def test_load_cache_corrupt_file(self, tmp_path):
        """Corrupt parquet file returns None (logs debug)."""
        cache_file = tmp_path / "backtest_prices.parquet"
        cache_file.write_text("not a parquet file")
        svc = PriceService(cache_dir=tmp_path)
        assert svc._load_cache() is None

    def test_save_cache_write_failure(self, tmp_path):
        """Write failure is caught silently (logs debug)."""
        svc = PriceService(cache_dir=tmp_path)
        # Create a directory where the file should be, causing write to fail
        bad_path = tmp_path / "backtest_prices.parquet"
        bad_path.mkdir()
        prices = pd.DataFrame({"AAPL": [150]}, index=pd.date_range("2026-01-02", periods=1))
        # Should not raise
        svc._save_cache(prices)

    def test_load_cache_non_datetime_index(self, tmp_path):
        """Cache with non-DatetimeIndex gets converted."""
        cache_file = tmp_path / "backtest_prices.parquet"
        df = pd.DataFrame(
            {"AAPL": [150, 151, 152]},
            index=["2026-01-02", "2026-01-03", "2026-01-06"],
        )
        df.to_parquet(cache_file)
        svc = PriceService(cache_dir=tmp_path)
        loaded = svc._load_cache()
        assert isinstance(loaded.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# PriceService.get_prices — cache integration
# ---------------------------------------------------------------------------


class TestGetPricesCacheIntegration:
    """Tests for get_prices interaction with the cache layer."""

    def test_uses_cached_data_skips_download(self, tmp_path):
        """When all tickers are cached, no download occurs."""
        svc = PriceService(cache_dir=tmp_path)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        cached = pd.DataFrame(
            {
                "AAPL": np.linspace(150, 160, 10),
                "SPY": np.linspace(500, 510, 10),
                # Include all regional benchmarks to avoid triggering download
                "EXS1.DE": np.linspace(100, 105, 10),
                "ISF.L": np.linspace(800, 810, 10),
                "2800.HK": np.linspace(20, 21, 10),
            },
            index=dates,
        )
        svc._save_cache(cached)

        with patch.object(svc, "_download_prices") as mock_dl:
            result = svc.get_prices(["AAPL"], "2026-01-02", "2026-01-15")
        mock_dl.assert_not_called()
        assert "AAPL" in result.columns

    def test_fetches_only_missing_tickers(self, tmp_path):
        """Only tickers not in cache are downloaded."""
        svc = PriceService(cache_dir=tmp_path)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        cached = pd.DataFrame(
            {
                "AAPL": np.linspace(150, 160, 10),
                "SPY": np.linspace(500, 510, 10),
                "EXS1.DE": np.linspace(100, 105, 10),
                "ISF.L": np.linspace(800, 810, 10),
                "2800.HK": np.linspace(20, 21, 10),
            },
            index=dates,
        )
        svc._save_cache(cached)

        new_data = pd.DataFrame(
            {"MSFT": np.linspace(400, 420, 10)},
            index=dates,
        )

        with patch.object(svc, "_download_prices", return_value=new_data) as mock_dl:
            result = svc.get_prices(["AAPL", "MSFT"], "2026-01-02", "2026-01-15")
        mock_dl.assert_called_once()
        called_tickers = mock_dl.call_args[0][0]
        assert "MSFT" in called_tickers
        assert "AAPL" not in called_tickers
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns

    def test_download_returns_empty_uses_cached(self, tmp_path):
        """When download returns empty, cached data is used."""
        svc = PriceService(cache_dir=tmp_path)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        cached = pd.DataFrame(
            {
                "AAPL": np.linspace(150, 160, 10),
                "SPY": np.linspace(500, 510, 10),
                "EXS1.DE": np.linspace(100, 105, 10),
                "ISF.L": np.linspace(800, 810, 10),
                "2800.HK": np.linspace(20, 21, 10),
            },
            index=dates,
        )
        svc._save_cache(cached)

        with patch.object(svc, "_download_prices", return_value=pd.DataFrame()):
            result = svc.get_prices(["AAPL", "NOPE"], "2026-01-02", "2026-01-15")
        assert "AAPL" in result.columns

    def test_no_cache_no_data_returns_empty(self):
        """No cache and empty download returns empty DataFrame."""
        svc = PriceService(cache_dir=None)
        with patch.object(svc, "_download_prices", return_value=pd.DataFrame()):
            result = svc.get_prices(["AAPL"], "2026-01-02", "2026-01-15")
        assert result.empty

    def test_include_benchmark_false(self):
        """When include_benchmark=False, benchmark is not added."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        mock_data = pd.DataFrame(
            {"AAPL": np.linspace(150, 160, 10)},
            index=dates,
        )

        with patch.object(svc, "_download_prices", return_value=mock_data) as mock_dl:
            svc.get_prices(["AAPL"], "2026-01-02", "2026-01-15", include_benchmark=False)
        called_tickers = mock_dl.call_args[0][0]
        # SPY should still appear because regional benchmarks are always added,
        # but the explicit benchmark add is skipped
        assert "AAPL" in called_tickers

    def test_non_datetime_index_converted(self):
        """get_prices converts non-DatetimeIndex to DatetimeIndex."""
        svc = PriceService(cache_dir=None)
        mock_data = pd.DataFrame(
            {"AAPL": [150, 151, 152]},
            index=["2026-01-02", "2026-01-05", "2026-01-06"],
        )
        with patch.object(svc, "_download_prices", return_value=mock_data):
            result = svc.get_prices(["AAPL"], "2026-01-02", "2026-01-15")
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_price_cache_attribute_set(self):
        """get_prices sets the _price_cache attribute."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=5, freq="B")
        mock_data = pd.DataFrame(
            {"AAPL": np.linspace(150, 155, 5)},
            index=dates,
        )
        with patch.object(svc, "_download_prices", return_value=mock_data):
            svc.get_prices(["AAPL"], "2026-01-02", "2026-01-10")
        assert svc._price_cache is not None

    def test_deduplicates_tickers(self):
        """Duplicate tickers in input are deduplicated."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=5, freq="B")
        mock_data = pd.DataFrame(
            {"AAPL": np.linspace(150, 155, 5)},
            index=dates,
        )
        with patch.object(svc, "_download_prices", return_value=mock_data) as mock_dl:
            svc.get_prices(["AAPL", "AAPL"], "2026-01-02", "2026-01-10")
        called_tickers = mock_dl.call_args[0][0]
        assert called_tickers.count("AAPL") == 1


# ---------------------------------------------------------------------------
# PriceService.trading_day_return — edge cases
# ---------------------------------------------------------------------------


class TestTradingDayReturnEdgeCases:
    """Edge cases not covered by test_price_service.py."""

    def test_all_nan_ticker_returns_none(self):
        """Ticker column with all NaN values returns None."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        prices = pd.DataFrame(
            {"AAPL": [np.nan] * 10},
            index=dates,
        )
        ret = svc.trading_day_return(prices, "AAPL", "2026-01-02", horizon=5)
        assert ret is None

    def test_zero_base_price_returns_none(self):
        """Zero base price returns None to avoid division by zero."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        prices = pd.DataFrame(
            {"AAPL": [0.0] + [100.0] * 9},
            index=dates,
        )
        ret = svc.trading_day_return(prices, "AAPL", "2026-01-02", horizon=5)
        assert ret is None

    def test_negative_base_price_returns_none(self):
        """Negative base price returns None."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        prices = pd.DataFrame(
            {"AAPL": [-1.0] + [100.0] * 9},
            index=dates,
        )
        ret = svc.trading_day_return(prices, "AAPL", "2026-01-02", horizon=5)
        assert ret is None

    def test_signal_date_after_all_data(self):
        """Signal date beyond data range returns None."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        prices = pd.DataFrame(
            {"AAPL": np.linspace(100, 110, 10)},
            index=dates,
        )
        ret = svc.trading_day_return(prices, "AAPL", "2027-01-01", horizon=5)
        assert ret is None

    def test_exact_boundary_horizon(self):
        """Horizon equal to remaining data points returns None."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        prices = pd.DataFrame(
            {"AAPL": np.linspace(100, 110, 10)},
            index=dates,
        )
        # Signal at first date, 10 data points, horizon=10 -> need index[10] but only 0-9 exist
        ret = svc.trading_day_return(prices, "AAPL", "2026-01-02", horizon=10)
        assert ret is None

    def test_horizon_zero(self):
        """Horizon of 0 returns 0% (same-day return)."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        prices = pd.DataFrame(
            {"AAPL": np.linspace(100, 110, 10)},
            index=dates,
        )
        ret = svc.trading_day_return(prices, "AAPL", "2026-01-02", horizon=0)
        assert ret == pytest.approx(0.0)

    def test_partial_nan_before_signal(self):
        """NaN values before signal date are dropped; return still computed."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        vals = [np.nan, np.nan] + list(np.linspace(100, 108, 8))
        prices = pd.DataFrame({"AAPL": vals}, index=dates)
        # Signal at dates[2] (first non-NaN), horizon=3
        ret = svc.trading_day_return(prices, "AAPL", str(dates[2].date()), horizon=3)
        assert ret is not None


# ---------------------------------------------------------------------------
# PriceService.trading_day_alpha — fallback paths
# ---------------------------------------------------------------------------


class TestTradingDayAlphaFallback:
    """Tests for alpha calculation fallback logic."""

    def test_alpha_when_stock_return_is_none(self):
        """Alpha is None when stock return is None."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        prices = pd.DataFrame(
            {"SPY": np.linspace(500, 510, 10)},
            index=dates,
        )
        alpha = svc.trading_day_alpha(prices, "NOPE", "2026-01-02", horizon=5)
        assert alpha is None

    def test_alpha_falls_back_to_default_benchmark(self):
        """When regional benchmark is missing, falls back to default (SPY)."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=20, freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": np.linspace(100, 120, 20),  # +20%
                "SPY": np.linspace(500, 510, 20),  # +2%
                # No ISF.L (UK benchmark) present
            },
            index=dates,
        )
        alpha = svc.trading_day_alpha(prices, "AAPL", "2026-01-02", horizon=19, region="uk")
        # ISF.L missing -> fallback to SPY
        assert alpha is not None
        assert alpha > 0

    def test_alpha_none_when_both_benchmarks_missing(self):
        """Alpha is None when both regional and default benchmarks are missing."""
        svc = PriceService(cache_dir=None, default_benchmark="MISSING_BM")
        dates = pd.date_range("2026-01-02", periods=20, freq="B")
        prices = pd.DataFrame(
            {"AAPL": np.linspace(100, 120, 20)},
            index=dates,
        )
        alpha = svc.trading_day_alpha(prices, "AAPL", "2026-01-02", horizon=19, region="hk")
        assert alpha is None

    def test_alpha_with_none_region_uses_default(self):
        """None region uses default benchmark."""
        svc = PriceService(cache_dir=None)
        dates = pd.date_range("2026-01-02", periods=20, freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": np.linspace(100, 120, 20),
                "SPY": np.linspace(500, 510, 20),
            },
            index=dates,
        )
        alpha = svc.trading_day_alpha(prices, "AAPL", "2026-01-02", horizon=19, region=None)
        assert alpha is not None


# ---------------------------------------------------------------------------
# PriceService.__init__
# ---------------------------------------------------------------------------


class TestPriceServiceInit:
    """Tests for constructor defaults."""

    def test_default_benchmark(self):
        svc = PriceService(cache_dir=None)
        assert svc.default_benchmark == "SPY"

    def test_custom_benchmark(self):
        svc = PriceService(cache_dir=None, default_benchmark="QQQ")
        assert svc.default_benchmark == "QQQ"

    def test_default_cache_dir(self):
        svc = PriceService()
        assert svc.cache_dir == DEFAULT_CACHE_DIR

    def test_none_cache_dir(self):
        svc = PriceService(cache_dir=None)
        assert svc.cache_dir is None

    def test_price_cache_starts_none(self):
        svc = PriceService(cache_dir=None)
        assert svc._price_cache is None
