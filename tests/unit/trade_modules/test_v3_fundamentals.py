"""TDD — PIT fundamental factor derivations (the data-blocked durable premia)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trade_modules.v3.fundamentals import (
    accruals,
    asset_growth,
    book_to_price,
    factor_panel,
    gross_profit_to_assets,
    srw_sue,
)


def test_book_to_price():
    assert book_to_price(40.0, 100.0) == pytest.approx(0.40)  # cheap on book
    assert math.isnan(book_to_price(-5.0, 100.0))  # negative book -> meaningless
    assert math.isnan(book_to_price(40.0, 0.0))  # no market cap


def test_asset_growth_is_yoy():
    assert asset_growth(110.0, 100.0) == pytest.approx(0.10)  # +10% (CMA: low is good)
    assert asset_growth(90.0, 100.0) == pytest.approx(-0.10)  # conservative investment
    assert math.isnan(asset_growth(110.0, 0.0))


def test_gross_profit_to_assets():
    assert gross_profit_to_assets(30.0, 120.0) == pytest.approx(0.25)  # Novy-Marx quality
    assert math.isnan(gross_profit_to_assets(30.0, 0.0))


def test_accruals():
    # (net income - operating cash flow) / assets; high accruals = low earnings quality.
    assert accruals(20.0, 12.0, 200.0) == pytest.approx(0.04)
    assert accruals(10.0, 25.0, 300.0) == pytest.approx(-0.05)  # cash-backed earnings
    assert math.isnan(accruals(20.0, 12.0, 0.0))


def test_srw_sue_positive_on_accelerating_eps():
    # Seasonal-random-walk SUE = (eps_t - eps_{t-4}) / std(seasonal diffs). Needs the
    # EPS history; the last jump (1.6 vs 1.2 = +0.4) is large vs prior +0.2 steps.
    eps = pd.Series([1.0, 1.1, 1.2, 1.3, 1.2, 1.3, 1.4, 1.6])
    assert srw_sue(eps) > 0


def test_srw_sue_needs_enough_history():
    assert math.isnan(srw_sue(pd.Series([1.0, 1.1, 1.2])))  # < 5 quarters -> NaN


def test_factor_panel_matches_scalar_derivations():
    fasof = pd.DataFrame(
        {
            "equity": [40.0, -5.0],
            "assets": [200.0, 100.0],
            "gp": [30.0, 20.0],
            "netinc": [20.0, 5.0],
            "ncfo": [12.0, 8.0],
            "sharesbas": [10.0, 10.0],
        },
        index=["A", "B"],
    )
    fprior = pd.DataFrame({"assets": [180.0, 120.0]}, index=["A", "B"])
    price = pd.Series({"A": 10.0, "B": 5.0})
    fp = factor_panel(fasof, fprior, price)
    assert fp.loc["A", "book_to_price"] == pytest.approx(0.40)  # 40 / (10*10)
    assert fp.loc["A", "asset_growth"] == pytest.approx(200 / 180 - 1)
    assert fp.loc["A", "gp_assets"] == pytest.approx(0.15)
    assert fp.loc["A", "accruals"] == pytest.approx(0.04)
    assert math.isnan(fp.loc["B", "book_to_price"])  # negative equity -> NaN
