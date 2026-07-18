"""TDD — BUILD ⑥c (2026-07-18, D25): EUR-denominated S&P 500 benchmark.

The owner's home currency is EUR, so the benchmark is the EUR return of holding SPY
= SPY's USD return combined with the EUR/USD move. Pure conversion + period-return
helpers (tested here) plus a thin fetch wrapper (tested with an injected fetcher).
"""

from __future__ import annotations

import pandas as pd
import pytest

from trade_modules.v3.benchmark import (
    fetch_spy_eur_return_pct,
    period_return_pct,
    spy_eur_return_pct,
    to_eur,
)


def test_to_eur_divides_by_usd_per_eur():
    assert to_eur(110.0, 1.10) == pytest.approx(100.0)  # $110 at 1.10 USD/EUR = 100 EUR


def test_period_return_pct():
    assert period_return_pct([100.0, 110.0]) == pytest.approx(10.0)
    assert pd.isna(period_return_pct([100.0]))  # need >= 2 points
    assert pd.isna(period_return_pct([0.0, 100.0]))  # zero base


def test_spy_eur_return_pure_usd_move():
    spy = pd.Series([100.0, 120.0])  # +20% USD
    fx = pd.Series([1.0, 1.0])  # no FX move
    assert spy_eur_return_pct(spy, fx) == pytest.approx(20.0)


def test_spy_eur_return_pure_fx_move():
    spy = pd.Series([100.0, 100.0])  # flat USD
    fx = pd.Series([1.0, 1.25])  # EUR strengthens 25% -> SPY worth less in EUR
    # EUR value: 100/1.0=100 -> 100/1.25=80 -> -20%
    assert spy_eur_return_pct(spy, fx) == pytest.approx(-20.0)


def test_spy_eur_return_aligns_on_common_dates():
    idx1 = pd.date_range("2026-01-01", periods=3, freq="D")
    spy = pd.Series([100.0, 110.0, 120.0], index=idx1)
    fx = pd.Series([1.0, 1.0], index=idx1[:2])  # missing the 3rd date
    # aligned window is the first two dates: 100 -> 110 = +10%
    assert spy_eur_return_pct(spy, fx) == pytest.approx(10.0)


def test_fetch_spy_eur_return_with_injected_fetcher():
    def fake_fetch(tickers, period="1y", **_kw):
        return pd.DataFrame({"SPY": [100.0, 120.0], "EURUSD=X": [1.0, 1.0]})

    assert fetch_spy_eur_return_pct("1y", fetch=fake_fetch) == pytest.approx(20.0)


def test_fetch_spy_eur_return_nan_on_empty_or_error():
    assert pd.isna(fetch_spy_eur_return_pct(fetch=lambda *a, **k: pd.DataFrame()))

    def boom(*a, **k):
        raise RuntimeError("network")

    assert pd.isna(fetch_spy_eur_return_pct(fetch=boom))
