#!/usr/bin/env python3
"""The signal log recorded a null price for every signal it ever wrote.

Measured on the VPS copy of yahoofinance/output/signal_log.jsonl, 2026-09-14:
6548 records spanning 2026-06-15 to 2026-09-13, and price_at_signal is None in
ALL 6548. target_price likewise. Every other field on the same record, read from
the same DataFrame at the same index in the same call, is populated:

    {"ticker": "MSFT", "signal": "B", "price_at_signal": null,
     "target_price": null, "upside": 43.67, "buy_percentage": 96.67,
     "pct_52w_high": 70.35, "roe": 34.01, ...}

So the frame is fine and the index is fine. What is wrong is the column name the
two price lookups ask for.

calculate_action_vectorized resolves every other field ONCE at function scope
with a fallback chain that ends in the short display name the table renderer
uses -- upside/UPSIDE/UP%, buy_percentage/%BUY/%B, pct_from_52w_high/52W -- and
then takes row_x = x.loc[idx]. Price has such a series too, at signals.py:682,
and its chain is price/PRC/current_price.

The ten log_signal call sites do not use it. Each re-resolves price inline as
df.get("price", df.get("PRICE", ...)) and each omits PRC, which is exactly the
name config.py:495 renames PRICE to for display ("PRICE": "PRC"). The six
target sites omit TGT the same way (config.py:496). The production frame carries
PRC and TGT -- they are the 4th and 5th columns of yahoofinance/output/etoro.csv
-- so all sixteen lookups miss and fall through to their None-filled default.

The cost is not cosmetic. BacktestEngine's calculate_returns rejects a signal
with no price as no_price_data, so the weekly backtest runs on whatever
backfill_signal_prices can rescue from yfinance: on 2026-09-12 that was 2 tickers
out of 3202 unique ticker-date pairs, 32 result rows, all horizon 7, and
buy_count_t7 = 0. ThresholdAnalyzer then returns at backtest_engine.py:924 for
want of a horizon==30 row and never writes backtest_threshold_report.csv. Four
estate data:produced rows trace back here.

The regression these tests pin is the duplication, not the missing key. A chain
spelled once at function scope and re-spelled sixteen times inline is how one
copy drifts, so the fix deletes the inline copies rather than adding PRC to each
of them, and the last test below is what stops them coming back.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from trade_modules.analysis_engine import calculate_action
from trade_modules.signal_tracker import SignalTracker

# signals.py performs a live yfinance earnings lookup per ticker and forces HOLD
# within 7 calendar days of a date, which would make these time-bombs. Same
# mock, same reason, as test_analysis_engine_signals.py.
_CLEAR_EARNINGS = {
    "earnings_date": None,
    "days_until": None,
    "status": "clear",
    "should_hold": False,
    "conviction_boost": False,
    "conviction_adjustment": 0,
}

PRICE = 187.5
TARGET = 215.0

# A frame that clears MEGA-US BUY on every criterion, so the run reaches a
# log_signal call rather than being filtered out before one.
_BASE = {
    "ticker": "AAPL",
    "market_cap": 3_000_000_000_000,
    "region": "US",
    "analyst_count": 20,
    "total_ratings": 15,
    "pe_forward": 25.0,
    "pe_trailing": 28.0,
    "upside": 15.0,
    "buy_percentage": 80.0,
    "EXRET": 12.0,
}


@pytest.fixture(autouse=True)
def _mock_earnings_proximity():
    with patch(
        "trade_modules.earnings_proximity.check_earnings_proximity",
        return_value=_CLEAR_EARNINGS,
    ):
        yield


def _run(price_columns: dict):
    """Run calculate_action over one row and return the SignalRecords logged.

    conftest's autouse fixture already patches SignalTracker.log_signal to a
    plain True. That is patched over here rather than around, because the record
    is built by the module-level log_signal() BEFORE the tracker method is
    reached, so capturing the method's argument sees the finished record.
    """
    captured = []

    def _capture(self, record):
        captured.append(record)
        return True

    df = pd.DataFrame([{**_BASE, **price_columns}])
    with (
        patch.object(SignalTracker, "log_signal", _capture),
        patch.object(SignalTracker, "log_signals_batch", lambda self, *a, **k: 0),
    ):
        calculate_action(df)
    return captured


class TestTheProductionColumnNamesReachTheLog:
    """PRC and TGT are what the frame actually carries in production."""

    def test_prc_is_recorded_as_the_price_at_signal(self):
        records = _run({"PRC": PRICE, "TGT": TARGET})
        assert records, "no signal was logged, so this asserts nothing"
        assert records[0].price_at_signal == pytest.approx(PRICE), (
            "price_at_signal is null for a frame that carries PRC, which is every "
            "record the production log has ever written"
        )

    def test_tgt_is_recorded_as_the_target_price(self):
        """Driven through the catastrophic-drawdown SELL, which is the cheapest
        of the call sites that actually passes `target=`. Most do not, so
        asserting on the BUY path above would pass vacuously against a None."""
        records = _run({"PRC": PRICE, "TGT": TARGET, "pct_from_52w_high": 5.0})
        assert records, "no signal was logged, so this asserts nothing"
        assert records[0].signal == "S", (
            "the drawdown path did not fire, so no call site passing target= was "
            "reached and this test would assert nothing"
        )
        assert records[0].target_price == pytest.approx(TARGET)
        assert records[0].price_at_signal == pytest.approx(PRICE)

    def test_current_price_is_recorded_too(self):
        """The third name in the function's own chain, and it was missing as well."""
        records = _run({"current_price": PRICE})
        assert records, "no signal was logged, so this asserts nothing"
        assert records[0].price_at_signal == pytest.approx(PRICE)


class TestTheNamesThatAlreadyWorkedKeepWorking:
    """The fix must not trade one spelling for another."""

    @pytest.mark.parametrize("column", ["price", "PRICE"])
    def test_the_two_names_the_old_inline_chain_accepted(self, column):
        records = _run({column: PRICE})
        assert records, "no signal was logged, so this asserts nothing"
        assert records[0].price_at_signal == pytest.approx(PRICE)

    def test_a_frame_with_no_price_column_still_logs_a_null_rather_than_raising(self):
        records = _run({})
        assert records, "no signal was logged, so this asserts nothing"
        assert records[0].price_at_signal is None


class TestEveryCallSiteAndNotJustTheEquityOne:
    """Ten call sites carried the defect and two of them were exercised.

    That ratio is the reason it survived: the equity paths get tested, the
    asset-type branches above them do not, and all ten had drifted identically.
    A fix verified on one path is a fix verified on the one path, so these drive
    the branches calculate_action_vectorized picks BEFORE it reaches the equity
    logic, by ticker, which is the only input classify_asset_type reads.
    """

    def test_a_bitcoin_proxy_buy_records_its_price(self):
        """MSTR with momentum and analyst support clears the btc-proxy BUY."""
        records = _run(
            {
                "ticker": "MSTR",
                "PRC": PRICE,
                "TGT": TARGET,
                "pct_from_52w_high": 95.0,
                "two_hundred_day_avg": 100.0,
            }
        )
        assert records, "the bitcoin-proxy branch was not reached"
        assert records[0].price_at_signal == pytest.approx(PRICE)

    def test_a_bitcoin_proxy_sell_records_its_price(self):
        """The same branch's other arm: momentum at or under the sell threshold."""
        records = _run(
            {
                "ticker": "MSTR",
                "PRC": PRICE,
                "TGT": TARGET,
                "pct_from_52w_high": 10.0,
                "two_hundred_day_avg": 100.0,
            }
        )
        assert records, "the bitcoin-proxy branch was not reached"
        assert records[0].signal == "S"
        assert records[0].price_at_signal == pytest.approx(PRICE)

    def test_a_crypto_signal_records_its_price(self):
        """asset_type in (crypto, commodity) has its own log_signal call."""
        records = _run(
            {
                "ticker": "BTC-USD",
                "PRC": PRICE,
                "TGT": TARGET,
                "pct_from_52w_high": 95.0,
                "two_hundred_day_avg": 100.0,
            }
        )
        assert records, "the crypto/commodity branch was not reached"
        assert records[0].price_at_signal == pytest.approx(PRICE)


class TestTheChainIsSpelledOnce:
    """The duplication is the defect; the missing key was only its symptom.

    signals.py resolved price in eleven places and target in seven, and the ten
    inline copies at the log_signal sites had drifted away from the one at :682
    that the scoring uses. Adding PRC to each of the ten would have restored the
    behaviour and left the shape that produced it, so the fix removes them. This
    test is what keeps them removed: it fails the moment somebody re-introduces a
    local price or target resolution inside the function.
    """

    def _source(self):
        from pathlib import Path

        import trade_modules.analysis.signals as mod

        return Path(mod.__file__).read_text()

    def test_price_is_resolved_exactly_once(self):
        src = self._source()
        assert src.count("price_raw_col = df.get(") == 1
        assert "price_raw = df.get(" not in src, (
            "a per-call-site price lookup is back; use the function-scope "
            "price_series, whose chain includes PRC"
        )

    def test_target_is_resolved_exactly_once(self):
        src = self._source()
        assert src.count("target_raw_col = df.get(") == 1
        assert "target_raw = df.get(" not in src, (
            "a per-call-site target lookup is back; use the function-scope "
            "target_series, whose chain includes TGT"
        )

    def test_both_chains_reach_the_short_display_names(self):
        """PRC and TGT are what config.py:495-496 rename PRICE and TARGET_PRICE to."""
        src = self._source()
        assert '"PRC"' in src
        assert '"TGT"' in src
