"""Tests for riskfirst.prices — the price-history factor primitives that replace
the snapshot proxies: true 12-1 (skip-month) momentum, realized volatility, and a
shrunk (Ledoit-Wolf-style) covariance from returns.
"""

import numpy as np
import pandas as pd
import pytest

from trade_modules.riskfirst.prices import (
    daily_returns,
    momentum_12_1,
    realized_vol,
    shrunk_cov,
)

# ---- 12-1 skip-month momentum ----


def test_momentum_skips_the_last_month():
    # Flat for ~12 months, then a spike ONLY in the last month. 12-1 momentum must
    # ignore the last month -> ~0 (a naive last/12m would read +50%).
    prices = pd.Series([100.0] * 279 + list(np.linspace(100, 150, 21)))
    assert momentum_12_1(prices) == pytest.approx(0.0, abs=1e-6)


def test_momentum_positive_for_steady_riser():
    prices = pd.Series(np.linspace(100, 200, 300))
    assert momentum_12_1(prices) > 0.3


def test_momentum_nan_when_too_short():
    assert np.isnan(momentum_12_1(pd.Series(np.linspace(100, 110, 50))))


# ---- realized volatility ----


def test_realized_vol_near_zero_for_smooth_growth():
    prices = pd.Series(100 * (1.0005 ** np.arange(260)))
    assert realized_vol(prices) < 0.01


def test_realized_vol_matches_alternating_series():
    # alternating +1%/-1% daily -> daily std ~0.01 -> annualised ~0.01*sqrt(252)
    p = [100.0]
    for i in range(260):
        p.append(p[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
    assert realized_vol(pd.Series(p)) == pytest.approx(0.01 * np.sqrt(252), abs=0.03)


# ---- shrunk covariance ----


def test_shrunk_cov_shrinks_offdiagonal_preserves_diagonal():
    r = pd.DataFrame({"A": [0.01, -0.01, 0.01, -0.01], "B": [0.01, -0.01, 0.01, -0.01]})
    cov = shrunk_cov(r, shrink=0.5, annualize=1)
    assert cov == pytest.approx(cov.T)  # symmetric
    assert cov[0, 1] == pytest.approx(0.5 * cov[0, 0])  # off-diag halved, diag intact


def test_daily_returns_basic():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    r = daily_returns(prices)
    assert r["A"].iloc[0] == pytest.approx(0.10)
    assert r["A"].iloc[1] == pytest.approx(-0.10)
