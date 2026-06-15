"""Coverage tests for sector_pe_provider module."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from trade_modules.sector_pe_provider import (
    DEFAULT_MEDIAN_PE,
    DEFAULT_SECTOR_PE,
    SECTOR_ETF_MAP,
    _is_cache_valid,
    get_all_sector_pe,
    get_dynamic_sector_pe,
    invalidate_cache,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure clean cache state for each test."""
    invalidate_cache()
    yield
    invalidate_cache()


class TestConstants:
    def test_sector_etf_map_has_entries(self):
        assert len(SECTOR_ETF_MAP) > 10
        assert SECTOR_ETF_MAP["Technology"] == "XLK"
        assert SECTOR_ETF_MAP["Financial Services"] == "XLF"

    def test_default_sector_pe_has_entries(self):
        assert len(DEFAULT_SECTOR_PE) > 5
        assert DEFAULT_SECTOR_PE["Technology"] == 28.0

    def test_default_median_pe(self):
        assert DEFAULT_MEDIAN_PE == 20.0

    def test_sector_variants_map_to_same_etf(self):
        assert SECTOR_ETF_MAP["Technology"] == SECTOR_ETF_MAP["Information Technology"]
        assert SECTOR_ETF_MAP["Healthcare"] == SECTOR_ETF_MAP["Health Care"]
        assert SECTOR_ETF_MAP["Consumer Discretionary"] == SECTOR_ETF_MAP["Consumer Cyclical"]


class TestIsCacheValid:
    def test_no_timestamp(self):
        assert _is_cache_valid() is False

    @patch("trade_modules.sector_pe_provider._cache_timestamp", datetime.now())
    def test_valid_cache(self):
        assert _is_cache_valid() is True

    @patch(
        "trade_modules.sector_pe_provider._cache_timestamp",
        datetime.now() - timedelta(hours=5),
    )
    def test_expired_cache(self):
        assert _is_cache_valid() is False


class TestGetDynamicSectorPe:
    @patch("trade_modules.sector_pe_provider._fetch_etf_pe")
    def test_returns_from_cache(self, mock_fetch):
        mock_fetch.return_value = 25.5
        # First call refreshes cache
        result = get_dynamic_sector_pe("Technology")
        assert isinstance(result, float)

    @patch("trade_modules.sector_pe_provider._fetch_etf_pe")
    def test_falls_back_to_defaults(self, mock_fetch):
        mock_fetch.return_value = None  # ETF fetch fails
        result = get_dynamic_sector_pe("Technology")
        assert result == DEFAULT_SECTOR_PE["Technology"]

    @patch("trade_modules.sector_pe_provider._fetch_etf_pe")
    def test_unknown_sector_returns_median(self, mock_fetch):
        mock_fetch.return_value = None
        result = get_dynamic_sector_pe("UnknownSector")
        assert result == DEFAULT_MEDIAN_PE

    @patch("trade_modules.sector_pe_provider._refresh_cache")
    def test_refresh_failure_handled(self, mock_refresh):
        mock_refresh.side_effect = RuntimeError("network error")
        # Should not raise, falls back to defaults
        result = get_dynamic_sector_pe("Technology")
        assert result == DEFAULT_SECTOR_PE["Technology"]

    @patch("trade_modules.sector_pe_provider._fetch_etf_pe")
    def test_cached_value_used(self, mock_fetch):
        mock_fetch.return_value = 30.0
        # First call refreshes
        val1 = get_dynamic_sector_pe("Technology")
        # Second call should use cache
        mock_fetch.return_value = 99.0  # different value
        val2 = get_dynamic_sector_pe("Technology")
        assert val1 == val2  # cache hit, not refreshed


class TestGetAllSectorPe:
    @patch("trade_modules.sector_pe_provider._fetch_etf_pe")
    def test_merges_cache_and_defaults(self, mock_fetch):
        mock_fetch.return_value = 25.0
        result = get_all_sector_pe()
        assert isinstance(result, dict)
        assert len(result) >= len(DEFAULT_SECTOR_PE)

    @patch("trade_modules.sector_pe_provider._fetch_etf_pe")
    def test_cache_overrides_defaults(self, mock_fetch):
        mock_fetch.return_value = 99.0
        result = get_all_sector_pe()
        # XLK maps to Technology, should have cached value
        assert result.get("Technology") == 99.0

    @patch("trade_modules.sector_pe_provider._refresh_cache")
    def test_refresh_failure_returns_defaults(self, mock_refresh):
        mock_refresh.side_effect = RuntimeError("fail")
        result = get_all_sector_pe()
        assert result == DEFAULT_SECTOR_PE


class TestInvalidateCache:
    @patch("trade_modules.sector_pe_provider._fetch_etf_pe")
    def test_invalidate_forces_refresh(self, mock_fetch):
        mock_fetch.return_value = 25.0
        get_dynamic_sector_pe("Technology")
        call_count_1 = mock_fetch.call_count

        # Cache valid, should not call again
        get_dynamic_sector_pe("Technology")
        assert mock_fetch.call_count == call_count_1

        # Invalidate and try again
        invalidate_cache()
        get_dynamic_sector_pe("Technology")
        assert mock_fetch.call_count > call_count_1
