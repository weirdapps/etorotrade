"""Coverage tests for sector_pe_provider module."""

import threading
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from trade_modules import sector_pe_provider as spp
from trade_modules.sector_pe_provider import (
    DEFAULT_MEDIAN_PE,
    DEFAULT_SECTOR_PE,
    SECTOR_ETF_MAP,
    _is_cache_valid,
    get_all_sector_pe,
    get_dynamic_sector_pe,
    invalidate_cache,
)

# One refresh fetches each DISTINCT sector ETF once, not each sector name: the
# map is many-to-one (Technology and Information Technology both mean XLK).
_UNIQUE_ETF_COUNT = len(set(SECTOR_ETF_MAP.values()))


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
        # Asserting the returned VALUE alone cannot distinguish a correctly
        # cached failure from one re-attempted on every call: the fallback is
        # the same 28.0 either way. Count the fetches too.
        assert mock_fetch.call_count == _UNIQUE_ETF_COUNT, (
            f"one failed refresh should cost {_UNIQUE_ETF_COUNT} fetches, "
            f"got {mock_fetch.call_count}"
        )

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


class _CountingFetch:
    """Stand-in for ``_fetch_etf_pe`` that records how often it was called.

    ``results`` are consumed one REFRESH at a time, not one fetch at a time: a
    single refresh loops over every distinct ETF, so a per-fetch queue would
    make "first refresh fails, second succeeds" impossible to express.
    """

    def __init__(self, *per_refresh_results):
        self._results = list(per_refresh_results)
        self.calls = 0
        self.refreshes = 0
        self._seen: set[str] = set()

    def __call__(self, etf_symbol):
        if etf_symbol in self._seen:
            # A new refresh has started; advance to the next scripted result.
            self._seen.clear()
            self._advance()
        if not self._seen:
            self.refreshes += 1
        self._seen.add(etf_symbol)
        self.calls += 1
        return self._current()

    def _current(self):
        return self._results[0] if self._results else None

    def _advance(self):
        if len(self._results) > 1:
            self._results.pop(0)


class TestSectorPeFetchCaching:
    """The refresh must be cached on FAILURE as well as on success.

    Regression guard. ``get_dynamic_sector_pe`` sits on the per-ticker signal
    path (``async_yahoo_finance.calculate_pe_vs_sector`` ->
    ``data_normalizer`` -> here), and one refresh loops over every distinct
    sector ETF. Caching only success meant an unreachable quote source cost a
    whole refresh PER TICKER: measured through the real production caller,
    330 ETF fetches for 30 tickers against 11 on the control.
    """

    def test_a_failed_refresh_is_attempted_only_once(self):
        """The bug: a failing refresh must NOT be repeated on every call."""
        fetch = _CountingFetch(None)

        with patch.object(spp, "_fetch_etf_pe", fetch):
            results = [get_dynamic_sector_pe("Technology") for _ in range(25)]

        assert results == [DEFAULT_SECTOR_PE["Technology"]] * 25
        assert fetch.calls == _UNIQUE_ETF_COUNT, (
            f"expected one refresh ({_UNIQUE_ETF_COUNT} fetches) to be cached "
            f"across 25 calls, but the source was hit {fetch.calls} times"
        )

    def test_POSITIVE_CONTROL_a_working_source_is_still_fetched(self):
        """Sibling to the test above: it must be able to tell 11 from zero.

        Without this, the failure-path count assertion would pass just as
        happily if the fix had disabled fetching altogether.
        """
        fetch = _CountingFetch(27.5)

        with patch.object(spp, "_fetch_etf_pe", fetch):
            results = [get_dynamic_sector_pe("Technology") for _ in range(25)]

        assert results == [27.5] * 25, "a working source must reach the caller"
        assert fetch.calls == _UNIQUE_ETF_COUNT, (
            f"a working source must still be fetched exactly once per distinct "
            f"ETF, got {fetch.calls}"
        )

    def test_a_raising_fetch_is_also_negatively_cached(self):
        """An exception is a failure worth caching, not a reason to retry."""
        calls = []

        def boom(etf_symbol):
            calls.append(etf_symbol)
            raise RuntimeError("quote source unreachable")

        with patch.object(spp, "_fetch_etf_pe", boom):
            results = [get_dynamic_sector_pe("Technology") for _ in range(25)]

        assert results == [DEFAULT_SECTOR_PE["Technology"]] * 25
        assert len(calls) == 1, (
            f"the first raising fetch aborts the refresh, and that failure must "
            f"be cached; expected 1 attempt across 25 calls, got {len(calls)}"
        )

    def test_get_all_sector_pe_is_gated_too(self):
        """The second entry point must back off on the same failure."""
        fetch = _CountingFetch(None)

        with patch.object(spp, "_fetch_etf_pe", fetch):
            for _ in range(25):
                assert get_all_sector_pe() == DEFAULT_SECTOR_PE

        assert fetch.calls == _UNIQUE_ETF_COUNT

    def test_the_two_entry_points_share_one_backoff(self):
        """A failure through one entry point must suppress the other."""
        fetch = _CountingFetch(None)

        with patch.object(spp, "_fetch_etf_pe", fetch):
            get_dynamic_sector_pe("Technology")
            get_all_sector_pe()
            get_dynamic_sector_pe("Healthcare")

        assert fetch.calls == _UNIQUE_ETF_COUNT

    def test_the_negative_ttl_expires_so_an_outage_self_heals(self):
        """A transient outage must not suppress refreshes indefinitely."""
        fetch = _CountingFetch(None, 31.0)

        with patch.object(spp, "_fetch_etf_pe", fetch):
            assert get_dynamic_sector_pe("Technology") == DEFAULT_SECTOR_PE["Technology"]
            assert fetch.calls == _UNIQUE_ETF_COUNT

            # Age the last attempt past the negative TTL.
            spp._last_attempt_timestamp = datetime.now() - timedelta(
                minutes=spp._NEGATIVE_TTL_MINUTES + 1
            )

            assert get_dynamic_sector_pe("Technology") == 31.0

        assert fetch.calls == 2 * _UNIQUE_ETF_COUNT

    def test_the_negative_ttl_is_much_shorter_than_the_success_ttl(self):
        """A 4h backoff on a blip would blind the model for a whole session.

        The negative TTL is also what bounds the behavioural cost of this fix
        on a PARTIAL outage: a ticker that would have succeeded gets the
        cached default for at most this long.
        """
        assert 0 < spp._NEGATIVE_TTL_MINUTES < spp._CACHE_TTL_HOURS * 60

    def test_a_successful_refresh_still_holds_for_the_full_success_ttl(self):
        """Unchanged behaviour: a good refresh is not re-fetched for 4 hours."""
        fetch = _CountingFetch(22.0)

        with patch.object(spp, "_fetch_etf_pe", fetch):
            for _ in range(50):
                assert get_dynamic_sector_pe("Technology") == 22.0

        assert fetch.calls == _UNIQUE_ETF_COUNT

    def test_stale_good_values_are_served_when_a_refresh_fails(self):
        """Unchanged behaviour, pinned: a failed refresh must not clobber data.

        Serving the last good sector PE is strictly better than falling back
        to a hardcoded default that may be years out of date.
        """
        fetch = _CountingFetch(33.0, None)

        with patch.object(spp, "_fetch_etf_pe", fetch):
            assert get_dynamic_sector_pe("Technology") == 33.0

            aged = datetime.now() - timedelta(hours=spp._CACHE_TTL_HOURS + 1)
            spp._cache_timestamp = aged
            spp._last_attempt_timestamp = aged

            # The refresh fails; the stale good value survives it.
            assert get_dynamic_sector_pe("Technology") == 33.0

        assert fetch.calls == 2 * _UNIQUE_ETF_COUNT

    def test_invalidate_cache_clears_the_negative_backoff_too(self):
        """Otherwise the autouse fixture leaks backoff state between tests."""
        fetch = _CountingFetch(None, 24.0)

        with patch.object(spp, "_fetch_etf_pe", fetch):
            assert get_dynamic_sector_pe("Technology") == DEFAULT_SECTOR_PE["Technology"]
            assert fetch.calls == _UNIQUE_ETF_COUNT

            invalidate_cache()

            assert get_dynamic_sector_pe("Technology") == 24.0

        assert fetch.calls == 2 * _UNIQUE_ETF_COUNT

    def test_concurrent_callers_share_a_single_failed_refresh(self):
        """16 threads racing a failing source must trigger one refresh."""
        fetch = _CountingFetch(None)
        results: list[float] = []
        results_lock = threading.Lock()

        def worker():
            value = get_dynamic_sector_pe("Technology")
            with results_lock:
                results.append(value)

        with patch.object(spp, "_fetch_etf_pe", fetch):
            threads = [threading.Thread(target=worker) for _ in range(16)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)

        assert len(results) == 16
        assert results == [DEFAULT_SECTOR_PE["Technology"]] * 16
        assert fetch.calls == _UNIQUE_ETF_COUNT, (
            f"expected 1 refresh across 16 threads, got {fetch.calls} fetches"
        )
