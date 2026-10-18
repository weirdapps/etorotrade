"""Unit-test fixtures: no live Yahoo Finance traffic."""

import pytest
from yfinance.data import YfData


class LiveYahooBlockedError(RuntimeError):
    """Raised in place of a live yfinance request during a unit test."""


@pytest.fixture(autouse=True)
def _no_live_yahoo(monkeypatch):
    """Fail every yfinance request fast, at its one choke point.

    Signal code reaches Yahoo on several paths no test asks for (the earnings
    calendar, the IPO first-trade probe, log_signal's momentum history), and each
    caller catches the error and falls back. Live, those calls passed on CI
    runners and hung locally on Yahoo's crumb fetch. A test that mocks
    yf.Ticker or the provider never reaches this; live checks belong in
    tests/integration.
    """

    def _blocked(self, url, *args, **kwargs):
        raise LiveYahooBlockedError(f"live Yahoo call in a unit test: {url}")

    monkeypatch.setattr(YfData, "_make_request", _blocked)
    yield
