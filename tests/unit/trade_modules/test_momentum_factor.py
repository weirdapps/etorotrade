"""Tests for 12-1 month momentum factor (Jegadeesh-Titman skip-month).

Validates the core computation, edge cases, series-based calculation,
and integration with the signal tracker.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trade_modules.analysis.momentum import (
    compute_momentum_12_1m,
    compute_momentum_from_series,
)


class TestComputeMomentum12_1m:
    """Tests for the core skip-month momentum computation."""

    def test_basic_positive_momentum(self):
        result = compute_momentum_12_1m(130, 125, 100)
        assert result == 25.0  # (125/100 - 1) * 100

    def test_basic_negative_momentum(self):
        result = compute_momentum_12_1m(80, 85, 100)
        assert result == -15.0

    def test_none_on_missing_12m(self):
        assert compute_momentum_12_1m(100, 95, None) is None

    def test_none_on_zero_12m(self):
        assert compute_momentum_12_1m(100, 95, 0) is None

    def test_none_on_negative_12m(self):
        assert compute_momentum_12_1m(100, 95, -5) is None

    def test_fallback_to_full_12m_when_1m_missing(self):
        result = compute_momentum_12_1m(130, None, 100)
        assert result == 30.0  # (130/100 - 1) * 100

    def test_skip_month_excludes_recent(self):
        # Stock: was 100, ran to 125 over 11 months, then jumped to 150 in last month
        # Skip-month momentum = (125/100 - 1) = 25%, NOT (150/100 - 1) = 50%
        result = compute_momentum_12_1m(150, 125, 100)
        assert result == 25.0

    def test_zero_1m_price_triggers_fallback(self):
        result = compute_momentum_12_1m(130, 0, 100)
        assert result == 30.0  # Falls back to price_now

    def test_none_1m_and_none_now(self):
        assert compute_momentum_12_1m(None, None, 100) is None

    def test_zero_now_and_none_1m(self):
        assert compute_momentum_12_1m(0, None, 100) is None

    def test_all_none(self):
        assert compute_momentum_12_1m(None, None, None) is None

    def test_rounding_precision(self):
        result = compute_momentum_12_1m(100, 133.33, 100)
        assert result == 33.33  # Should round to 2 decimal places

    def test_flat_momentum(self):
        result = compute_momentum_12_1m(100, 100, 100)
        assert result == 0.0


class TestComputeMomentumFromSeries:
    """Tests for computing momentum from a daily price series."""

    def test_from_series_basic(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(300)) + 100
        result = compute_momentum_from_series(prices)
        assert result is not None

    def test_from_series_too_short(self):
        result = compute_momentum_from_series([100, 101, 102])
        assert result is None

    def test_from_series_exactly_252(self):
        prices = np.linspace(100, 150, 252)
        result = compute_momentum_from_series(prices)
        assert result is not None

    def test_from_series_nan_handling(self):
        prices = np.ones(300) * 100.0
        prices[50:60] = np.nan  # Some NaN in middle
        result = compute_momentum_from_series(prices)
        # After removing NaN, 290 prices remain >= 252, so should work
        assert result is not None

    def test_from_series_too_many_nans(self):
        prices = np.ones(260) * 100.0
        prices[0:20] = np.nan  # 20 NaN values -> 240 remaining < 252
        result = compute_momentum_from_series(prices)
        assert result is None

    def test_from_series_flat_prices(self):
        prices = np.ones(300) * 100.0
        result = compute_momentum_from_series(prices)
        assert result == 0.0

    def test_from_series_uses_correct_offsets(self):
        # Build a controlled series where we know exactly what -21 and -252 are
        prices = np.ones(300) * 100.0
        prices[-252] = 80.0  # 12m ago: 80
        prices[-21] = 120.0  # 1m ago: 120
        prices[-1] = 150.0  # now: 150 (should be skipped in favor of -21)
        result = compute_momentum_from_series(prices)
        # momentum = (120/80 - 1) * 100 = 50.0
        assert result == 50.0


class TestSignalTrackerIntegration:
    """Tests for momentum field in the signal tracker."""

    def test_signal_record_includes_momentum(self):
        from trade_modules.signal_tracker import SignalRecord

        record = SignalRecord(
            ticker="AAPL",
            signal="B",
            momentum_12_1m=23.5,
        )
        d = record.to_dict()
        assert d["momentum_12_1m"] == 23.5

    def test_signal_record_momentum_none_by_default(self):
        from trade_modules.signal_tracker import SignalRecord

        record = SignalRecord(ticker="AAPL", signal="B")
        d = record.to_dict()
        assert d["momentum_12_1m"] is None

    def test_signal_record_roundtrip(self):
        from trade_modules.signal_tracker import SignalRecord

        original = SignalRecord(
            ticker="MSFT",
            signal="H",
            momentum_12_1m=-12.3,
        )
        d = original.to_dict()
        restored = SignalRecord.from_dict(d)
        assert restored.momentum_12_1m == -12.3

    def test_old_record_without_momentum_loads(self):
        """Old JSONL entries missing momentum_12_1m should deserialize fine."""
        from trade_modules.signal_tracker import SignalRecord

        old_data = {
            "ticker": "AAPL",
            "signal": "B",
            "timestamp": "2025-01-01T00:00:00",
        }
        record = SignalRecord.from_dict(old_data)
        assert record.momentum_12_1m is None


class _MomentumSpy:
    """Records every call to fetch_momentum_for_ticker and returns a fixed value.

    fetch_momentum_for_ticker() is the sole entry point to the per-ticker
    yfinance history() call, so "this spy was not called" is exactly
    "no network fetch was attempted".
    """

    def __init__(self, value: float = 42.5):
        self.calls: list[str] = []
        self.value = value

    def __call__(self, ticker: str) -> float:
        self.calls.append(ticker)
        return self.value


@pytest.fixture
def momentum_harness(monkeypatch):
    """Drive log_signal() with every side effect except the momentum fetch stubbed.

    Yields ``run(**env)`` -> ``(spy, record)``. Keys of ``env`` are applied to
    os.environ; a value of None deletes the variable. ``record`` is the
    SignalRecord log_signal() handed to the tracker, i.e. exactly what would
    have been written to signal_log.jsonl.
    """
    import trade_modules.signal_tracker as st

    def run(**env):
        spy = _MomentumSpy()
        for key in ("CI", "GITHUB_ACTIONS"):
            value = env.get(key)
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)

        captured: list = []
        tracker = MagicMock()
        tracker.log_signal.side_effect = lambda record: (captured.append(record), True)[1]

        with (
            patch("trade_modules.analysis.momentum.fetch_momentum_for_ticker", spy),
            patch("trade_modules.vix_regime_provider.get_current_vix", return_value=None),
            patch("trade_modules.regime_detector.get_current_regime", return_value=None),
            patch.object(st, "_get_spy_price", return_value=None),
            patch.object(st, "get_tracker", return_value=tracker),
        ):
            st.log_signal(ticker="AAPL", signal="H", price=100.0)

        return spy, (captured[0] if captured else None)

    return run


class TestMomentumFetchIsOffTheCIPath:
    """The per-ticker momentum fetch is skipped in CI and unchanged locally.

    In CI signal_log.jsonl is destroyed with the runner, so the ~16 minutes of
    per-ticker yfinance history() calls produce records that are thrown away.
    Locally the records are the point, so nothing may change there.
    """

    def test_fetch_IS_attempted_when_not_in_ci(self, momentum_harness):
        """POSITIVE CONTROL: without the CI vars the fetch still runs.

        Without this, the skip assertions below cannot tell "the guard worked"
        from "log_signal never reached the momentum branch at all".
        """
        spy, record = momentum_harness(CI=None, GITHUB_ACTIONS=None)

        assert spy.calls == ["AAPL"], "local runs must still fetch momentum"
        assert record is not None
        assert record.momentum_12_1m == 42.5, "the fetched value must reach the record"

    def test_fetch_is_NOT_attempted_when_ci_is_true(self, momentum_harness):
        spy, record = momentum_harness(CI="true", GITHUB_ACTIONS=None)

        assert spy.calls == [], "CI must not pay for a record it throws away"
        assert record is not None, "the signal itself must still be logged"
        assert record.momentum_12_1m is None

    def test_fetch_is_NOT_attempted_when_github_actions_is_true(self, momentum_harness):
        spy, record = momentum_harness(CI=None, GITHUB_ACTIONS="true")

        assert spy.calls == []
        assert record is not None
        assert record.momentum_12_1m is None

    def test_a_non_true_ci_value_does_NOT_skip(self, momentum_harness):
        """Pins the exact comparison the rest of the suite uses (== "true").

        CI=1 / CI=false are not the convention and must not silently disable a
        local scan that happens to export CI for some other reason.
        """
        spy, _ = momentum_harness(CI="1", GITHUB_ACTIONS="false")

        assert spy.calls == ["AAPL"]

    def test_the_skip_is_ANNOUNCED_not_silent(self, momentum_harness, caplog):
        """A silently absent column is how the next person loses a morning."""
        import trade_modules.signal_tracker as st

        st._MOMENTUM_CI_NOTICE_EMITTED = False
        with caplog.at_level(logging.INFO, logger="trade_modules.signal_tracker"):
            momentum_harness(CI="true", GITHUB_ACTIONS=None)

        notices = [r for r in caplog.records if "momentum_12_1m logging disabled" in r.getMessage()]
        assert len(notices) == 1, "expected exactly one INFO notice naming the skip"
        assert notices[0].levelno == logging.INFO

    def test_the_announcement_is_once_per_run_not_per_ticker(self, momentum_harness, caplog):
        """157k records a night must not become 157k identical log lines."""
        import trade_modules.signal_tracker as st

        st._MOMENTUM_CI_NOTICE_EMITTED = False
        with caplog.at_level(logging.INFO, logger="trade_modules.signal_tracker"):
            momentum_harness(CI="true", GITHUB_ACTIONS=None)
            momentum_harness(CI="true", GITHUB_ACTIONS=None)
            momentum_harness(CI="true", GITHUB_ACTIONS=None)

        notices = [r for r in caplog.records if "momentum_12_1m logging disabled" in r.getMessage()]
        assert len(notices) == 1, "the notice must be emitted once per process"

    def test_the_momentum_module_still_works_in_ci(self):
        """Only the invocation is gated; the code itself is untouched."""
        prices = [100.0] * 231 + [120.0] * 21
        assert compute_momentum_from_series(prices) == 20.0
        assert compute_momentum_12_1m(130, 125, 100) == 25.0
