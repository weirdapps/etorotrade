"""Regression tests for calculate_analyst_momentum row-ordering (2026-07-30).

Bug: yfinance returns `.recommendations` NEWEST-FIRST (row 0 = current month '0m',
last row = oldest '-3m'). The old code used `sort_index().iloc[-1]` as "latest",
which is actually the OLDEST row — so it compared -3m vs -1m: sign-inverted AND the
wrong window, never reading the true current month. Verified systematic across 10
tickers (AAPL stored +7.1 vs true -5.0, Toyota -3.1 vs +8.4, etc.).

Fix: select the '0m' and '-3m' rows by their `period` label so row order cannot
invert the comparison.
"""

import pandas as pd
import pytest

from yahoofinance.api.providers.async_modules.recommendations_parser import (
    calculate_analyst_momentum,
)


class _FakeTicker:
    def __init__(self, rec, ticker="TEST"):
        self.recommendations = rec
        self.ticker = ticker


def _rec(rows):
    """Build a yfinance-style recommendations frame (a RangeIndex + 'period' column)."""
    return pd.DataFrame(rows)


# The real INVE-B.ST shape: buy% 20.0 (0m) vs 33.3 (-3m) -> analysts turned MORE bearish.
_INVE_ROWS = [
    {"period": "0m", "strongBuy": 0, "buy": 1, "hold": 3, "sell": 1, "strongSell": 0},  # 20.0%
    {"period": "-1m", "strongBuy": 0, "buy": 1, "hold": 4, "sell": 1, "strongSell": 0},  # 16.7%
    {"period": "-2m", "strongBuy": 0, "buy": 2, "hold": 3, "sell": 1, "strongSell": 0},  # 33.3%
    {"period": "-3m", "strongBuy": 0, "buy": 2, "hold": 3, "sell": 1, "strongSell": 0},  # 33.3%
]


def test_momentum_is_current_minus_three_months_ago():
    res = calculate_analyst_momentum(_FakeTicker(_rec(_INVE_ROWS)))
    # 0m 20.0% - (-3m) 33.3% = -13.3 (bearish). The OLD bug returned +16.7 (bullish).
    assert res["analyst_momentum"] == pytest.approx(-13.3, abs=0.1)
    assert res["analyst_momentum"] < 0  # sign must reflect the real (bearish) trend


def test_momentum_bullish_when_buy_share_rises():
    rows = [
        {"period": "0m", "strongBuy": 3, "buy": 3, "hold": 2, "sell": 0, "strongSell": 0},  # 75%
        {"period": "-1m", "strongBuy": 2, "buy": 2, "hold": 4, "sell": 0, "strongSell": 0},
        {"period": "-2m", "strongBuy": 1, "buy": 2, "hold": 5, "sell": 0, "strongSell": 0},
        {"period": "-3m", "strongBuy": 0, "buy": 2, "hold": 6, "sell": 0, "strongSell": 0},  # 25%
    ]
    res = calculate_analyst_momentum(_FakeTicker(_rec(rows)))
    assert res["analyst_momentum"] == pytest.approx(50.0, abs=0.1)  # 75 - 25


def test_momentum_is_row_order_invariant():
    """Selecting by 'period' label must give the same answer no matter the row order."""
    forward = calculate_analyst_momentum(_FakeTicker(_rec(_INVE_ROWS)))
    shuffled = _rec(list(reversed(_INVE_ROWS)))  # oldest-first
    reverse = calculate_analyst_momentum(_FakeTicker(shuffled))
    assert (
        forward["analyst_momentum"] == reverse["analyst_momentum"] == pytest.approx(-13.3, abs=0.1)
    )


def test_momentum_falls_back_to_oldest_when_3m_missing():
    rows = [
        {"period": "0m", "strongBuy": 0, "buy": 1, "hold": 3, "sell": 0, "strongSell": 0},  # 25%
        {"period": "-1m", "strongBuy": 0, "buy": 2, "hold": 2, "sell": 0, "strongSell": 0},
        {
            "period": "-2m",
            "strongBuy": 0,
            "buy": 3,
            "hold": 1,
            "sell": 0,
            "strongSell": 0,
        },  # 75% (oldest)
    ]
    res = calculate_analyst_momentum(_FakeTicker(_rec(rows)))
    assert res["analyst_momentum"] == pytest.approx(-50.0, abs=0.1)  # 25 - 75 (oldest present)


def test_momentum_none_with_insufficient_data():
    res = calculate_analyst_momentum(_FakeTicker(_rec(_INVE_ROWS[:1])))
    assert res["analyst_momentum"] is None
