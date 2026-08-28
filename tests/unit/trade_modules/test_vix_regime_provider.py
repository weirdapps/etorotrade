"""Tests for VIX regime provider.

CIO Review v2: Threshold adjustments neutralized. Signal criteria held constant;
risk is managed through position sizing only. All multipliers are 1.0, all offsets are 0.
"""

import threading
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from trade_modules import vix_regime_provider as vix_module
from trade_modules.vix_regime_provider import (
    REGIME_ADJUSTMENTS,
    REGIME_POSITION_MULTIPLIERS,
    VixRegime,
    adjust_buy_criteria,
    adjust_sell_criteria,
    get_adjusted_thresholds,
    get_current_vix,
    get_regime_context,
    get_vix_regime,
    invalidate_cache,
)


class TestRegimeAdjustments:
    """Test that regime adjustments are comprehensive and consistent."""

    def test_all_regimes_have_all_keys(self):
        """Every regime must define the same set of adjustment keys."""
        expected_keys = {
            "min_upside_multiplier",
            "min_buy_pct_multiplier",
            "min_exret_multiplier",
            "min_analysts_offset",
            "max_upside_sell_offset",
            "max_pct_52w_buy_multiplier",
            "max_pe_multiplier",
        }
        for regime in VixRegime:
            assert set(REGIME_ADJUSTMENTS[regime].keys()) == expected_keys, (
                f"{regime.value} missing keys: "
                f"{expected_keys - set(REGIME_ADJUSTMENTS[regime].keys())}"
            )

    def test_normal_regime_is_neutral(self):
        """Normal regime should not change anything."""
        adj = REGIME_ADJUSTMENTS[VixRegime.NORMAL]
        assert adj["min_upside_multiplier"] == pytest.approx(1.0)
        assert adj["min_buy_pct_multiplier"] == pytest.approx(1.0)
        assert adj["min_exret_multiplier"] == pytest.approx(1.0)
        assert adj["min_analysts_offset"] == 0
        assert adj["max_upside_sell_offset"] == pytest.approx(0.0)
        assert adj["max_pct_52w_buy_multiplier"] == pytest.approx(1.0)
        assert adj["max_pe_multiplier"] == pytest.approx(1.0)

    def test_all_regimes_are_neutral(self):
        """CIO v2: All regimes should be neutral (no threshold adjustments)."""
        for regime in VixRegime:
            adj = REGIME_ADJUSTMENTS[regime]
            assert adj["min_upside_multiplier"] == pytest.approx(1.0), (
                f"{regime.value} upside not neutral"
            )
            assert adj["min_buy_pct_multiplier"] == pytest.approx(1.0), (
                f"{regime.value} buy_pct not neutral"
            )
            assert adj["min_exret_multiplier"] == pytest.approx(1.0), (
                f"{regime.value} exret not neutral"
            )
            assert adj["min_analysts_offset"] == 0, f"{regime.value} analysts not neutral"
            assert adj["max_upside_sell_offset"] == pytest.approx(0.0), (
                f"{regime.value} sell offset not neutral"
            )
            assert adj["max_pct_52w_buy_multiplier"] == pytest.approx(1.0), (
                f"{regime.value} 52w not neutral"
            )
            assert adj["max_pe_multiplier"] == pytest.approx(1.0), f"{regime.value} PE not neutral"

    def test_position_sizing_still_varies_by_regime(self):
        """CIO v2: Position sizing multipliers should still vary by regime."""
        assert REGIME_POSITION_MULTIPLIERS[VixRegime.LOW] == pytest.approx(1.00)
        assert REGIME_POSITION_MULTIPLIERS[VixRegime.NORMAL] == pytest.approx(1.00)
        assert REGIME_POSITION_MULTIPLIERS[VixRegime.ELEVATED] == pytest.approx(0.75)
        assert REGIME_POSITION_MULTIPLIERS[VixRegime.HIGH] == pytest.approx(0.50)


class TestAdjustBuyCriteria:
    """Test buy criteria adjustment function."""

    SAMPLE_BUY_CONFIG = {
        "min_upside": 10,
        "min_buy_percentage": 75,
        "min_exret": 6,
        "min_analysts": 8,
        "min_pct_from_52w_high": 45,
        "max_forward_pe": 60,
        "max_trailing_pe": 90,
    }

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_all_regimes_no_change(self, mock_adj):
        """CIO v2: All regimes should produce no threshold changes."""
        for regime in VixRegime:
            mock_adj.return_value = REGIME_ADJUSTMENTS[regime].copy()
            result = adjust_buy_criteria(self.SAMPLE_BUY_CONFIG)
            assert result["min_upside"] == 10
            assert result["min_buy_percentage"] == 75
            assert result["min_exret"] == 6
            assert result["min_analysts"] == 8
            assert result["max_forward_pe"] == 60
            assert result["max_trailing_pe"] == 90

    def test_no_adjustment_when_disabled(self):
        result = adjust_buy_criteria(self.SAMPLE_BUY_CONFIG, apply_adjustments=False)
        assert result == self.SAMPLE_BUY_CONFIG

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_min_analysts_floor(self, mock_adj):
        """min_analysts should never go below 4."""
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.HIGH].copy()
        config = {"min_analysts": 4}
        result = adjust_buy_criteria(config)
        assert result["min_analysts"] >= 4

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_buy_pct_cap_at_95(self, mock_adj):
        """min_buy_percentage should never exceed 95%."""
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.LOW].copy()
        config = {"min_buy_percentage": 93}
        result = adjust_buy_criteria(config)
        assert result["min_buy_percentage"] <= 95.0


class TestAdjustSellCriteria:
    """Test sell criteria adjustment function."""

    SAMPLE_SELL_CONFIG = {
        "max_upside": 0,
        "max_exret": 2,
    }

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_all_regimes_no_change(self, mock_adj):
        """CIO v2: All regimes should produce no sell threshold changes."""
        for regime in VixRegime:
            mock_adj.return_value = REGIME_ADJUSTMENTS[regime].copy()
            result = adjust_sell_criteria(self.SAMPLE_SELL_CONFIG)
            assert result["max_upside"] == 0
            assert result["max_exret"] == 2


class TestGetAdjustedThresholds:
    """Test the unified threshold adjustment interface."""

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_buy_type(self, mock_adj):
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.NORMAL].copy()
        config = {"min_upside": 10}
        result = get_adjusted_thresholds(config, "buy")
        assert result["min_upside"] == 10

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_sell_type(self, mock_adj):
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.NORMAL].copy()
        config = {"max_upside": 0}
        result = get_adjusted_thresholds(config, "sell")
        assert result["max_upside"] == 0


class TestGetRegimeContext:
    """Test regime context generation for committee reports."""

    @patch("trade_modules.vix_regime_provider.get_regime_status")
    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_context_structure(self, mock_adj, mock_status):
        mock_status.return_value = (VixRegime.NORMAL, 18.5, "Normal volatility")
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.NORMAL].copy()

        ctx = get_regime_context()
        assert ctx["regime"] == "normal"
        assert ctx["vix"] == pytest.approx(18.5)
        assert ctx["description"] == "Normal volatility"
        assert "adjustments" in ctx
        assert "implications" in ctx

    @patch("trade_modules.vix_regime_provider.get_regime_status")
    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_high_vix_has_implications(self, mock_adj, mock_status):
        mock_status.return_value = (VixRegime.HIGH, 42.0, "High volatility - defensive mode")
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.HIGH].copy()

        ctx = get_regime_context()
        assert len(ctx["implications"]) > 0
        assert ctx["regime"] == "high"


class TestVixRegimeClassification:
    """Test VIX level to regime mapping."""

    @patch("trade_modules.vix_regime_provider.get_current_vix")
    def test_low_vix(self, mock_vix):
        mock_vix.return_value = 12.0
        assert get_vix_regime() == VixRegime.LOW

    @patch("trade_modules.vix_regime_provider.get_current_vix")
    def test_normal_vix(self, mock_vix):
        mock_vix.return_value = 20.0
        assert get_vix_regime() == VixRegime.NORMAL

    @patch("trade_modules.vix_regime_provider.get_current_vix")
    def test_elevated_vix(self, mock_vix):
        mock_vix.return_value = 30.0
        assert get_vix_regime() == VixRegime.ELEVATED

    @patch("trade_modules.vix_regime_provider.get_current_vix")
    def test_high_vix(self, mock_vix):
        mock_vix.return_value = 40.0
        assert get_vix_regime() == VixRegime.HIGH

    @patch("trade_modules.vix_regime_provider.get_current_vix")
    def test_none_defaults_to_normal(self, mock_vix):
        mock_vix.return_value = None
        assert get_vix_regime() == VixRegime.NORMAL


class _CountingFetch:
    """Stand-in for ``_fetch_vix`` that records how often it was called."""

    def __init__(self, *results):
        self._results = list(results)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if not self._results:
            return None
        if len(self._results) == 1:
            return self._results[0]
        return self._results.pop(0)


@pytest.fixture(autouse=True)
def _reset_vix_cache():
    """Reset the module-level VIX cache around every test in this module.

    The cache is process-global, so without this a real network fetch in one
    test would silently satisfy the next one (and vice versa).
    """
    _clear()
    yield
    _clear()


def _clear():
    vix_module._vix_cache = None
    vix_module._vix_cache_timestamp = None
    vix_module._vix_last_attempt_timestamp = None


class TestVixFetchCaching:
    """The fetch must be cached on FAILURE as well as on success.

    Regression guard: ``adjust_buy_criteria`` / ``adjust_sell_criteria`` are
    called once per ticker inside the signal loop, and both reach
    ``get_current_vix``. Without negative caching, an unreachable or
    rate-limited quote source turns a whole-universe scoring run into one live
    network round-trip per name.
    """

    def test_failed_fetch_is_attempted_only_once(self):
        """A failing fetch must NOT be retried on every call."""
        fetch = _CountingFetch(None)

        with patch.object(vix_module, "_fetch_vix", fetch):
            results = [get_current_vix() for _ in range(25)]

        assert results == [None] * 25, "a failed fetch must not invent a value"
        assert fetch.calls == 1, (
            f"expected the failed fetch to be cached and attempted once, "
            f"but it was attempted {fetch.calls} times in 25 calls"
        )

    def test_raising_fetch_is_attempted_only_once(self):
        """An exception from the fetch is also a failure worth caching."""

        calls = []

        def boom():
            calls.append(1)
            raise RuntimeError("quote source unreachable")

        with patch.object(vix_module, "_fetch_vix", boom):
            results = [get_current_vix() for _ in range(10)]

        assert results == [None] * 10
        assert len(calls) == 1, f"expected 1 attempt, got {len(calls)}"

    def test_negative_cache_expires_and_a_retry_happens(self):
        """The negative TTL must expire so a transient outage self-heals."""
        fetch = _CountingFetch(None, 21.5)

        with patch.object(vix_module, "_fetch_vix", fetch):
            assert get_current_vix() is None
            assert fetch.calls == 1

            # Age the last attempt past the negative TTL.
            vix_module._vix_last_attempt_timestamp = datetime.now() - timedelta(
                minutes=vix_module._VIX_NEGATIVE_TTL_MINUTES + 1
            )

            assert get_current_vix() == pytest.approx(21.5)
            assert fetch.calls == 2

    def test_negative_ttl_is_shorter_than_the_positive_ttl(self):
        """A transient outage must not blind the model for a full success TTL."""
        assert 0 < vix_module._VIX_NEGATIVE_TTL_MINUTES < vix_module._VIX_CACHE_TTL_MINUTES

    def test_success_after_failure_populates_the_cache_normally(self):
        """Once a good value lands it is cached on the normal positive TTL."""
        fetch = _CountingFetch(None, 18.25)

        with patch.object(vix_module, "_fetch_vix", fetch):
            assert get_current_vix() is None
            vix_module._vix_last_attempt_timestamp = datetime.now() - timedelta(
                minutes=vix_module._VIX_NEGATIVE_TTL_MINUTES + 1
            )
            assert get_current_vix() == pytest.approx(18.25)

            # Now cached: no further attempts, no matter how many callers.
            for _ in range(10):
                assert get_current_vix() == pytest.approx(18.25)

        assert fetch.calls == 2

    def test_successful_fetch_is_cached_for_the_positive_ttl(self):
        """Unchanged behaviour: a good value is fetched once per positive TTL."""
        fetch = _CountingFetch(16.0)

        with patch.object(vix_module, "_fetch_vix", fetch):
            for _ in range(50):
                assert get_current_vix() == pytest.approx(16.0)

        assert fetch.calls == 1

    def test_positive_ttl_expiry_triggers_a_refetch(self):
        """Unchanged behaviour: the 30-minute success TTL still expires."""
        fetch = _CountingFetch(16.0, 24.0)

        with patch.object(vix_module, "_fetch_vix", fetch):
            assert get_current_vix() == pytest.approx(16.0)

            aged = datetime.now() - timedelta(minutes=vix_module._VIX_CACHE_TTL_MINUTES + 1)
            vix_module._vix_cache_timestamp = aged
            vix_module._vix_last_attempt_timestamp = aged

            assert get_current_vix() == pytest.approx(24.0)

        assert fetch.calls == 2

    def test_a_stale_good_value_is_served_when_the_refresh_fails(self):
        """Judgement call, pinned: prefer a stale VIX over losing the regime.

        Returning ``None`` here would send ``get_vix_regime`` to its
        ``NORMAL`` default, which is not "no opinion" -- it is an active
        assertion that volatility is normal, and it would quietly restore
        position multipliers from 0.50 to 1.00 in exactly the stressed market
        where the quote source is most likely to be failing. A VIX reading a
        little past its TTL almost always lands in the same 10-point regime
        bucket, so serving it is the strictly safer error.
        """
        fetch = _CountingFetch(38.0, None)

        with patch.object(vix_module, "_fetch_vix", fetch):
            assert get_current_vix() == pytest.approx(38.0)

            aged = datetime.now() - timedelta(minutes=vix_module._VIX_CACHE_TTL_MINUTES + 1)
            vix_module._vix_cache_timestamp = aged
            vix_module._vix_last_attempt_timestamp = aged

            # The refresh fails; the stale value is served rather than None.
            assert get_current_vix() == pytest.approx(38.0)
            assert get_vix_regime() == VixRegime.HIGH

        assert fetch.calls == 2, "the failed refresh must also be negatively cached"

    def test_none_is_returned_when_no_good_value_was_ever_seen(self):
        """No fabricated fallback: an unavailable VIX stays unavailable."""
        with patch.object(vix_module, "_fetch_vix", _CountingFetch(None)):
            assert get_current_vix() is None
            assert get_vix_regime() == VixRegime.NORMAL

    def test_invalidate_cache_clears_the_negative_backoff_too(self):
        """Otherwise a forced invalidation is silently ignored after a failure."""
        fetch = _CountingFetch(None, 19.0)

        with patch.object(vix_module, "_fetch_vix", fetch):
            assert get_current_vix() is None
            assert fetch.calls == 1

            invalidate_cache()

            assert get_current_vix() == pytest.approx(19.0)
            assert fetch.calls == 2

    def test_concurrent_callers_share_a_single_failed_fetch(self):
        """Thread safety: 16 threads racing a failing fetch attempt it once."""
        fetch = _CountingFetch(None)
        results: list[float | None] = []
        results_lock = threading.Lock()

        def worker():
            value = get_current_vix()
            with results_lock:
                results.append(value)

        with patch.object(vix_module, "_fetch_vix", fetch):
            threads = [threading.Thread(target=worker) for _ in range(16)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)

        assert len(results) == 16
        assert results == [None] * 16
        assert fetch.calls == 1, f"expected 1 attempt across 16 threads, got {fetch.calls}"
