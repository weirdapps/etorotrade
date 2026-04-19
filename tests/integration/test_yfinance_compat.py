"""
yfinance API compatibility smoke test.

Covers the 11 yfinance surface points the codebase actually depends on.
Run before/after any yfinance version bump:

    pytest tests/integration/test_yfinance_compat.py -v --no-header

Each test asserts shape/type only — never specific market values.
Tests skip (not fail) on network errors so a flaky Yahoo doesn't break CI.
"""
from __future__ import annotations

import socket

import pandas as pd
import pytest
import yfinance as yf

CANONICAL_TICKER = "AAPL"
BULK_TICKERS = ["AAPL", "MSFT", "SPY"]


def _skip_if_no_network():
    try:
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=5).close()
    except OSError as e:
        pytest.skip(f"No network to Yahoo Finance: {e}")


@pytest.fixture(scope="module")
def ticker():
    _skip_if_no_network()
    return yf.Ticker(CANONICAL_TICKER)


def test_yf_version_recorded():
    assert hasattr(yf, "__version__")
    print(f"\nyfinance version under test: {yf.__version__}")


def test_ticker_constructor(ticker):
    assert ticker.ticker == CANONICAL_TICKER


def test_history_period(ticker):
    df = ticker.history(period="1mo")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert col in df.columns, f"missing column {col!r}"


def test_history_start_end(ticker):
    df = ticker.history(start="2025-01-02", end="2025-01-31")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_info_dict(ticker):
    info = ticker.info
    assert isinstance(info, dict)
    assert info, "info dict is empty"
    for key in ("symbol", "sector", "industry"):
        assert key in info, f"missing info key {key!r}"


def test_fast_info(ticker):
    fi = ticker.fast_info
    last = fi["lastPrice"] if "lastPrice" in fi else getattr(fi, "last_price", None)
    assert last is not None and last > 0


def test_calendar(ticker):
    cal = ticker.calendar
    assert cal is None or isinstance(cal, (dict, pd.DataFrame))


def test_earnings_dates(ticker):
    dates = ticker.earnings_dates
    assert dates is None or isinstance(dates, pd.DataFrame)


def test_news(ticker):
    news = ticker.news
    assert isinstance(news, list)


def test_institutional_holders(ticker):
    inst = ticker.institutional_holders
    assert inst is None or isinstance(inst, pd.DataFrame)


def test_insider_transactions(ticker):
    insiders = ticker.insider_transactions
    assert insiders is None or isinstance(insiders, pd.DataFrame)


def test_recommendations(ticker):
    recs = ticker.recommendations
    assert recs is None or isinstance(recs, pd.DataFrame)


def test_upgrades_downgrades(ticker):
    ud = ticker.upgrades_downgrades
    if ud is not None and isinstance(ud, pd.DataFrame) and not ud.empty:
        assert "ToGrade" in ud.columns, "ToGrade column missing — committee_scorecard depends on it"


def test_download_bulk():
    _skip_if_no_network()
    data = yf.download(
        BULK_TICKERS, period="5d", progress=False, auto_adjust=True
    )
    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_download_single_with_period():
    _skip_if_no_network()
    spy = yf.download("SPY", period="30d", progress=False, auto_adjust=True)
    assert isinstance(spy, pd.DataFrame)
    assert not spy.empty
    assert "Close" in spy.columns or ("Close", "SPY") in spy.columns


def test_tickers_multi():
    _skip_if_no_network()
    multi = yf.Tickers(" ".join(BULK_TICKERS))
    assert hasattr(multi, "tickers")
    assert set(multi.tickers.keys()) >= set(BULK_TICKERS)
