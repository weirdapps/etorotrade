"""A lost Yahoo session is rebuilt in process and the ticker fetched again.

The real yfinance code runs end to end; only ``YfData._make_request`` (the one network choke
point, already blocked by tests/unit/conftest.py) is replaced with a fake Yahoo that refuses the
session with HTTP 401 "Invalid Crumb" until the test's reset function restores it.
"""

import json
import threading

import pytest
import yfinance as yf
import yfinance._http as yf_http
from yfinance.data import YfData

from tests.unit.conftest import LiveYahooBlockedError
from yahoofinance.utils.network import yahoo_session
from yahoofinance.utils.network.yahoo_session import (
    YahooSessionRefresher,
    fetch_info_with_session_refresh,
)

INVALID_CRUMB = {
    "finance": {"result": None, "error": {"code": "Unauthorized", "description": "Invalid Crumb"}}
}


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)
        self.url = "https://query2.finance.yahoo.com/"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise yf_http.HTTPError(f"HTTP Error {self.status_code}", response=self)

    def json(self):
        return self._body


class _FakeYahoo:
    """Answers the three requests ``Ticker.info`` makes; refuses them while the session is lost."""

    def __init__(self, session_ok=True, timeseries_refused=False, barrier=None):
        self.session_ok = session_ok
        self.timeseries_refused = timeseries_refused
        self.barrier = barrier
        self.requests = 0
        self._lock = threading.Lock()

    def __call__(self, url, *args, **kwargs):  # set on the class, so not bound to YfData
        modules = (kwargs.get("params") or {}).get("modules", "")
        if "quoteSummary" in url and "financialData" not in modules:
            # Any read other than Ticker.info's three is blocked, as tests/unit/conftest.py does.
            raise LiveYahooBlockedError(url)
        with self._lock:
            self.requests += 1
        if "timeseries" in url:
            if self.timeseries_refused and not self.session_ok:
                return _Response(401, INVALID_CRUMB)
            return _Response(200, {"timeseries": {"result": [], "error": None}})
        if not self.session_ok:
            if self.barrier is not None and "quoteSummary" in url:
                self.barrier.wait(timeout=5)
            return _Response(401, INVALID_CRUMB)
        if "quoteSummary" in url:
            return _Response(
                200,
                {"quoteSummary": {"result": [{"financialData": {"currentPrice": 123.0}}]}},
            )
        return _Response(
            200, {"quoteResponse": {"result": [{"symbol": "AAPL", "regularMarketPrice": 123.0}]}}
        )

    def restore(self):
        self.session_ok = True


@pytest.fixture
def fake_yahoo(monkeypatch):
    def install(**kwargs):
        fake = _FakeYahoo(**kwargs)
        monkeypatch.setattr(YfData, "_make_request", fake)
        # Undo the detector's class patch after the test.
        monkeypatch.setattr(YfData, "get_raw_json", YfData.get_raw_json)
        YfData.cache_get.cache_clear()
        return fake

    yield install
    YfData.cache_get.cache_clear()


def _refresher(fake, **kwargs):
    sleeps = []
    resets = []

    def reset():
        resets.append(1)
        fake.restore()
        YfData.cache_get.cache_clear()  # as reset_yfinance_session does

    refresher = YahooSessionRefresher(reset=reset, sleep=sleeps.append, **kwargs)
    return refresher, sleeps, resets


def _fetch(symbol, refresher):
    return fetch_info_with_session_refresh(
        yf.Ticker(symbol), lambda: yf.Ticker(symbol), refresher=refresher
    )


def test_lost_session_is_rebuilt_and_the_ticker_priced(fake_yahoo):
    fake = fake_yahoo(session_ok=False)
    refresher, sleeps, resets = _refresher(fake, backoff_seconds=5.0)

    _, info = _fetch("AAPL", refresher)

    assert info["regularMarketPrice"] == 123.0
    assert resets == [1]
    assert sleeps == [5.0]
    assert refresher.refreshes == 1


def test_refused_read_that_raises_is_also_rebuilt(fake_yahoo):
    # With every endpoint refused, yfinance raises from the timeseries read instead of
    # returning an empty info.
    fake = fake_yahoo(session_ok=False, timeseries_refused=True)
    refresher, _, resets = _refresher(fake)

    _, info = _fetch("AAPL", refresher)

    assert info["regularMarketPrice"] == 123.0
    assert resets == [1]


def test_hard_block_stops_at_the_bound_and_leaves_the_ticker_unpriced(fake_yahoo):
    fake = fake_yahoo(session_ok=False)
    resets = []
    sleeps = []
    refresher = YahooSessionRefresher(
        max_refreshes=2, backoff_seconds=5.0, reset=lambda: resets.append(1), sleep=sleeps.append
    )

    _, info = _fetch("AAPL", refresher)

    assert "regularMarketPrice" not in info
    assert refresher.refreshes == 2
    assert sleeps == [5.0, 10.0]  # backed off, doubling

    # Once the bound is spent, a later ticker is read once and not refreshed again.
    before = fake.requests
    _, info = _fetch("MSFT", refresher)
    assert "regularMarketPrice" not in info
    assert len(resets) == 2
    assert fake.requests - before == 3  # quoteSummary, v7 quote, timeseries: one read


def test_hard_block_that_raises_reraises_after_the_bound(fake_yahoo):
    fake = fake_yahoo(session_ok=False, timeseries_refused=True)
    refresher = YahooSessionRefresher(max_refreshes=1, reset=lambda: None, sleep=lambda s: None)

    with pytest.raises(Exception, match="Invalid Crumb"):
        _fetch("AAPL", refresher)
    assert refresher.refreshes == 1
    assert fake.session_ok is False


def test_concurrent_failures_cost_one_refresh(fake_yahoo):
    workers = 4
    fake = fake_yahoo(session_ok=False, barrier=threading.Barrier(workers))
    refresher, _, resets = _refresher(fake)
    prices = {}
    errors = []

    def work(symbol):
        try:
            _, info = _fetch(symbol, refresher)
            prices[symbol] = info.get("regularMarketPrice")
        except Exception as e:  # surfaced by the assertion below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(f"T{i}",)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert resets == [1]
    assert refresher.refreshes == 1
    assert prices == {f"T{i}": 123.0 for i in range(workers)}


def test_negative_control_healthy_session_is_never_refreshed(fake_yahoo):
    fake = fake_yahoo(session_ok=True)
    refresher, sleeps, resets = _refresher(fake)

    plain = yf.Ticker("AAPL").info
    plain_requests = fake.requests
    YfData.cache_get.cache_clear()

    _, info = _fetch("AAPL", refresher)

    assert info == plain
    assert fake.requests - plain_requests == plain_requests  # no extra request
    assert resets == []
    assert sleeps == []
    assert refresher.refreshes == 0


def test_detector_installs_once(fake_yahoo):
    fake_yahoo()
    yahoo_session.install_auth_failure_detector()
    wrapped = YfData.get_raw_json
    yahoo_session.install_auth_failure_detector()
    assert YfData.get_raw_json is wrapped


class _StubTicker:
    """Stands in for yf.Ticker in the provider: .info is refused while the session is lost."""

    session_ok = False
    recommendations = None
    upgrades_downgrades = None
    calendar = None
    earnings_dates = None

    def __init__(self, symbol):
        self.ticker = symbol

    @property
    def info(self):
        if not _StubTicker.session_ok:
            yahoo_session._call_state.auth_failed = True  # what the detector records on a 401
            return {}
        return {"regularMarketPrice": 123.0, "quoteType": "EQUITY"}


@pytest.mark.asyncio
async def test_provider_prices_a_ticker_after_the_session_is_rebuilt(monkeypatch):
    from yahoofinance.api.providers.async_yahoo_finance import AsyncYahooFinanceProvider

    monkeypatch.setattr(yf, "Ticker", _StubTicker)
    monkeypatch.setattr(_StubTicker, "session_ok", False)
    resets = []

    def reset():
        resets.append(1)
        _StubTicker.session_ok = True

    refresher = YahooSessionRefresher(reset=reset, sleep=lambda s: None)
    monkeypatch.setattr(yahoo_session, "_default_refresher", refresher)

    info = await AsyncYahooFinanceProvider().get_ticker_info("AAPL")

    assert info["price"] == 123.0
    assert resets == [1]


@pytest.mark.asyncio
async def test_provider_negative_control_makes_no_refresh(monkeypatch):
    from yahoofinance.api.providers.async_yahoo_finance import AsyncYahooFinanceProvider

    monkeypatch.setattr(yf, "Ticker", _StubTicker)
    monkeypatch.setattr(_StubTicker, "session_ok", True)
    resets = []
    refresher = YahooSessionRefresher(reset=lambda: resets.append(1), sleep=lambda s: None)
    monkeypatch.setattr(yahoo_session, "_default_refresher", refresher)

    info = await AsyncYahooFinanceProvider().get_ticker_info("AAPL")

    assert info["price"] == 123.0
    assert resets == []
    assert refresher.refreshes == 0


def test_reset_discards_session_cookie_crumb_and_persisted_cookie(monkeypatch):
    from yfinance import cache

    data = YfData()
    stored = []
    monkeypatch.setattr(
        cache,
        "get_cookie_cache",
        lambda: type("C", (), {"store": lambda s, k, v: stored.append((k, v))})(),
    )
    old_session = data._session
    monkeypatch.setattr(data, "_cookie", True)
    monkeypatch.setattr(data, "_crumb", "stale")

    yahoo_session.reset_yfinance_session()

    assert data._session is not old_session
    assert data._cookie is None
    assert data._crumb is None
    assert stored == [("curlCffi", None)]


# 2026-11-17: Yahoo refused shard 3's session for 17+ minutes at a time. Three rebuilds backed off
# 5s, 10s and 20s were spent in the first 35 seconds, and every later name was written blank.
# The refresher now waits out a block of that length, so the scan pauses instead of blanking.


class _Block:
    """A session refused until ``lifts_after`` rebuilds have happened."""

    def __init__(self, fake, lifts_after):
        self.fake = fake
        self.lifts_after = lifts_after
        self.resets = 0

    def reset(self):
        self.resets += 1
        if self.resets >= self.lifts_after:
            self.fake.restore()
        YfData.cache_get.cache_clear()


def test_backoff_is_capped(fake_yahoo):
    fake = fake_yahoo(session_ok=False)
    sleeps = []
    refresher = YahooSessionRefresher(
        max_refreshes=4,
        backoff_seconds=5.0,
        max_backoff_seconds=20.0,
        max_wait_seconds=1000.0,
        reset=lambda: None,
        sleep=sleeps.append,
    )

    _fetch("AAPL", refresher)

    assert sleeps == [5.0, 10.0, 20.0, 20.0]


def test_total_wait_is_bounded(fake_yahoo):
    fake = fake_yahoo(session_ok=False)
    sleeps = []
    refresher = YahooSessionRefresher(
        max_refreshes=10,
        backoff_seconds=5.0,
        max_wait_seconds=12.0,
        reset=lambda: None,
        sleep=sleeps.append,
    )

    _, info = _fetch("AAPL", refresher)

    assert "regularMarketPrice" not in info
    assert sleeps == [5.0]  # a 10s wait would take the total past 12s


def test_a_priced_read_restores_the_budget_for_the_next_block(fake_yahoo):
    fake = fake_yahoo(session_ok=False)
    block = _Block(fake, lifts_after=2)
    sleeps = []
    refresher = YahooSessionRefresher(
        max_refreshes=2,
        backoff_seconds=5.0,
        max_wait_seconds=1000.0,
        reset=block.reset,
        sleep=sleeps.append,
    )

    _, info = _fetch("AAPL", refresher)
    assert info["regularMarketPrice"] == 123.0

    # A second block later in the scan gets the full per-block bound again, from the first step.
    fake.session_ok = False
    block.resets = 0
    _, info = _fetch("MSFT", refresher)

    assert info["regularMarketPrice"] == 123.0
    assert sleeps == [5.0, 10.0, 5.0, 10.0]
    assert refresher.refreshes == 4


def test_defaults_wait_out_a_thirty_minute_block():
    r = YahooSessionRefresher(reset=lambda: None, sleep=lambda s: None)
    waits = [r.delay_for(step) for step in range(r.max_refreshes)]
    assert 30 * 60 <= sum(waits) <= r.max_wait_seconds <= 60 * 60
