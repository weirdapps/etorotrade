"""A cached ticker is not a fresh ticker.

Found 2026-09-13. The production cache on the VPS,
~/.weirdapps-trading/price_cache/backtest_prices.parquet, was shape (216, 3556)
with **3129 of its columns holding no valid price after 2026-05-27** and 394
columns entirely NaT. Its mtime was 2026-09-12, the day of the most recent
weekly backtest, so the file was being rewritten every week while its contents
had not moved in three and a half months.

Two defects in get_prices() produced that, and fixing either one alone would
have left it frozen.

1. MEMBERSHIP WAS MISTAKEN FOR FRESHNESS.

       cached_tickers = set(cached.columns)
       missing = [t for t in all_tickers if t not in cached_tickers]

   start_date and end_date are arguments to this method and the cache decision
   ignored both. A ticker present as a column was never refetched, however old
   its last price, so once a column existed it was permanent.

2. THE REFETCH WAS THROWN AWAY ANYWAY.

       prices = pd.concat([cached, new_data], axis=1)
       prices = prices.loc[:, ~prices.columns.duplicated()]

   pandas' duplicated() defaults to keep="first" and `cached` is concatenated
   first, so for any ticker present in both frames the STALE column survived and
   the freshly downloaded one was dropped. A TTL on its own would have queued
   the right downloads and then discarded every one of them.

What it cost: the 2026-09-12 backtest produced backtest_results.csv with 32 rows
covering two tickers out of a 106-ticker universe, all at horizon 7, because
almost no ticker had prices recent enough to evaluate. That file is force-added
and pushed to a public repo on every run.

Nothing under yahoofinance/ touches PriceService, so the daily signal CSVs were
never affected. trade_modules/committee_scorecard.py and
trade_modules/factor_attribution.py do, and both read the same frozen parquet.

Run: cd <repo> && python3 -m pytest tests/unit/trade_modules/test_price_cache_goes_stale.py -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from trade_modules.price_service import PriceService

START = "2026-01-01"
END = "2026-09-12"


def _frame(cols, last_day, start="2026-01-01"):
    """A price frame whose every column stops on `last_day`."""
    idx = pd.bdate_range(start, last_day)
    return pd.DataFrame({c: [100.0 + i for i in range(len(idx))] for c in cols}, index=idx)


@pytest.fixture
def svc(tmp_path):
    return PriceService(cache_dir=tmp_path, default_benchmark="SPY")


def _spy(svc, returns):
    """Record what _download_prices is asked for, and answer with `returns`."""
    asked = []

    def fake(tickers, start_date, end_date):
        asked.append(list(tickers))
        return returns(tickers)

    svc._download_prices = fake  # noqa: SLF001
    return asked


class TestStaleColumnsAreRefetched:
    def test_a_ticker_frozen_months_ago_is_refetched(self, svc):
        """The production case: the column exists, so it was never asked for."""
        svc._save_cache(_frame(["AAPL", "SPY"], "2026-05-27"))  # noqa: SLF001
        asked = _spy(svc, lambda t: _frame(t, END))

        svc.get_prices(["AAPL"], START, END)

        assert asked, "nothing was downloaded; the stale column was accepted"
        assert "AAPL" in asked[0], f"AAPL not refetched, asked for {asked[0]}"

    def test_a_column_that_is_entirely_empty_is_refetched(self, svc):
        """394 of the 3556 production columns were all-NaT. last_valid_index()
        returns None there, which must read as stale and not as 'covered'."""
        idx = pd.bdate_range(START, END)
        svc._save_cache(pd.DataFrame({"DEAD": [None] * len(idx)}, index=idx))  # noqa: SLF001
        asked = _spy(svc, lambda t: _frame(t, END))

        svc.get_prices(["DEAD"], START, END)

        assert asked and "DEAD" in asked[0]

    def test_a_fresh_ticker_is_not_refetched(self, svc):
        """The cache must still be a cache. Over-correcting into 'always
        download' would put a 106-ticker yfinance pull on every call."""
        svc._save_cache(_frame(["AAPL", "SPY", "EXS1.DE", "ISF.L", "2800.HK"], END))  # noqa: SLF001
        asked = _spy(svc, lambda t: _frame(t, END))

        svc.get_prices(["AAPL"], START, END)

        assert not asked, f"refetched despite fresh coverage: {asked}"

    def test_a_weekend_end_date_does_not_invalidate_everything(self, svc):
        """end_date is routinely a Saturday, and every venue here keeps its own
        calendar. Demanding coverage to the exact day would refetch the whole
        universe every run, which is the opposite failure."""
        svc._save_cache(_frame(["AAPL", "SPY", "EXS1.DE", "ISF.L", "2800.HK"], "2026-09-11"))  # noqa: SLF001
        asked = _spy(svc, lambda t: _frame(t, "2026-09-13"))

        svc.get_prices(["AAPL"], START, "2026-09-13")  # a Sunday

        assert not asked, f"a two-day gap over a weekend triggered a refetch: {asked}"


class TestTheRefetchActuallyWins:
    def test_fresh_data_replaces_the_stale_column(self, svc):
        """keep='first' kept the cached column, so a refetch changed nothing.
        This is the defect that would have survived a TTL-only fix."""
        svc._save_cache(_frame(["AAPL", "SPY"], "2026-05-27"))  # noqa: SLF001
        _spy(svc, lambda t: _frame(t, END))

        prices = svc.get_prices(["AAPL"], START, END)

        last = prices["AAPL"].last_valid_index()
        assert last is not None
        assert last >= pd.Timestamp("2026-09-10"), (
            f"AAPL still ends {last}: the stale column won the merge"
        )

    def test_the_cache_on_disk_is_updated_too(self, svc):
        """The production parquet re-persisted its own staleness every week, so
        the freeze was self-perpetuating. The write-back must carry the new data."""
        svc._save_cache(_frame(["AAPL", "SPY"], "2026-05-27"))  # noqa: SLF001
        _spy(svc, lambda t: _frame(t, END))

        svc.get_prices(["AAPL"], START, END)

        reloaded = svc._load_cache()  # noqa: SLF001
        assert reloaded is not None
        assert reloaded["AAPL"].last_valid_index() >= pd.Timestamp("2026-09-10")

    def test_an_untouched_ticker_survives_the_merge(self, svc):
        """Scope check: refetching AAPL must not drop a healthy sibling."""
        svc._save_cache(_frame(["AAPL", "MSFT", "SPY"], "2026-05-27"))  # noqa: SLF001
        _spy(svc, lambda t: _frame(t, END))

        prices = svc.get_prices(["AAPL"], START, END)

        assert "MSFT" in prices.columns


class TestDegradedDownloads:
    def test_a_failed_download_keeps_the_stale_cache_rather_than_nothing(self, svc):
        """yfinance returning empty must not erase what we had. Old prices beat
        no prices; the caller's own emptiness gates decide what to do next."""
        svc._save_cache(_frame(["AAPL", "SPY"], "2026-05-27"))  # noqa: SLF001
        _spy(svc, lambda t: pd.DataFrame())

        prices = svc.get_prices(["AAPL"], START, END)

        assert "AAPL" in prices.columns
