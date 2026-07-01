"""Tests for riskfirst.engine — composite, selection, construction, recommend.

Factors are dependency-injected (callables df->Series) so the engine is testable
independently of the individual factor modules.
"""

import numpy as np
import pandas as pd
import pytest

from trade_modules.riskfirst.engine import composite_score, recommend, select_and_construct
from trade_modules.riskfirst.stats import zscore


def _f_a(df):
    return zscore(df["a"], winsor=False)


def _f_b(df):
    return zscore(df["b"], winsor=False)


def _universe():
    return pd.DataFrame(
        {
            "a": [5.0, 4.0, 3.0, 2.0, 1.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "B": [1.0, 1.1, 0.9, 1.2, 0.8],  # beta
        },
        index=["AAPL", "MSFT", "SAP.DE", "0700.HK", "NVDA"],
    )


def test_composite_is_equal_weight_mean_of_factor_zscores():
    df = _universe()
    comp = composite_score(df, [_f_a, _f_b])
    # a and b identical -> composite equals each factor's z-score
    assert comp.loc["AAPL"] == pytest.approx(zscore(df["a"], winsor=False).loc["AAPL"])
    assert comp.idxmax() == "AAPL"  # highest a/b


def test_construct_respects_caps_and_selection():
    df = _universe()
    res = select_and_construct(
        df,
        [_f_a, _f_b],
        top_n=3,
        market_vol=0.18,
        idio_vol=0.25,
        target_vol=0.10,
        name_cap=0.45,
        usd_bloc_cap=0.60,
    )
    w = res["weights"]
    nonzero = w[w > 1e-9]
    assert len(nonzero) == 3  # only top-3 selected
    assert set(nonzero.index) == {"AAPL", "MSFT", "SAP.DE"}
    assert nonzero.max() <= 0.45 + 1e-6  # single-name cap
    assert w.sum() <= 1.0 + 1e-6  # no leverage
    # USD-bloc (AAPL only among the 3; SAP.DE=EUR) within cap
    usd = w.loc["AAPL"]
    assert usd <= 0.60 + 1e-6


def test_construct_vol_target_holds_cash_when_too_hot():
    df = _universe()
    res = select_and_construct(
        df,
        [_f_a, _f_b],
        top_n=3,
        market_vol=0.40,
        idio_vol=0.50,
        target_vol=0.05,
        name_cap=1.0,
        usd_bloc_cap=1.0,
    )
    # very hot names + low target -> scaled down, gross well below 1 (cash held)
    assert res["weights"].sum() < 0.9
    assert res["cash"] > 0.1


def test_sector_cap_binds_when_sector_column_present():
    df = pd.DataFrame(
        {
            "a": [5.0, 4.0, 3.0, 2.0, 1.0],
            "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "B": [1.0, 1.0, 1.0, 1.0, 1.0],
            "SECTOR": ["tech", "tech", "tech", "energy", "health"],
        },
        index=["AAPL", "MSFT", "NVDA", "XOM", "JNJ"],
    )
    res = select_and_construct(
        df,
        [_f_a, _f_b],
        top_n=4,
        market_vol=0.05,
        idio_vol=0.05,
        target_vol=0.50,
        name_cap=1.0,
        usd_bloc_cap=1.0,
        sector_cap=0.5,
    )
    w = res["weights"]
    tech = w.loc[["AAPL", "MSFT", "NVDA"]].sum()
    assert tech <= 0.5 + 1e-6  # 3 tech names capped from ~75% to 50%


def test_cov_fn_hook_drives_weights_from_empirical_cov():
    df = _universe()
    seen = {}

    def cov_fn(selected):
        seen["selected"] = list(selected)
        # diagonal with increasing variance -> ERC gives decreasing (inverse-vol) weights
        return np.diag([0.01, 0.04, 0.09])

    res = select_and_construct(
        df,
        [_f_a, _f_b],
        top_n=3,
        target_vol=0.50,
        name_cap=1.0,
        usd_bloc_cap=1.0,
        cov_fn=cov_fn,
    )
    assert seen["selected"] == ["AAPL", "MSFT", "SAP.DE"]  # cov_fn got the selection
    w = res["weights"].loc[seen["selected"]]
    assert w.iloc[0] > w.iloc[1] > w.iloc[2]  # inverse-vol ordering


def test_recommend_classifies_deltas():
    target = pd.Series({"AAPL": 0.30, "MSFT": 0.20, "NVDA": 0.0})
    current = pd.Series({"AAPL": 0.10, "MSFT": 0.20, "NVDA": 0.15})
    recs = recommend(target, current)
    actions = dict(zip(recs["ticker"], recs["action"]))
    assert actions["AAPL"] == "ADD"  # 0.10 -> 0.30
    assert actions["MSFT"] == "HOLD"  # unchanged
    assert actions["NVDA"] == "SELL"  # 0.15 -> 0.0


def _toy_df():
    # 4 eligible-looking names with betas for the fallback covariance
    return pd.DataFrame(
        {"B": [1.0, 1.0, 1.0, 1.0], "CAP": ["100B", "100B", "100B", "100B"]},
        index=["AAA", "BBB", "CCC", "DDD"],
    )


def _one_factor():
    return [lambda df: pd.Series([1.0, 0.5, 0.0, -0.5], index=df.index)]


def test_regime_multiplier_scales_gross():
    df, fns = _toy_df(), _one_factor()
    base = select_and_construct(df, fns, top_n=4, target_vol=0.50)
    scaled = select_and_construct(df, fns, top_n=4, target_vol=0.50, regime_multiplier=0.5)
    assert scaled["gross"] == pytest.approx(base["gross"] * 0.5, rel=1e-9)
    assert scaled["cash"] == pytest.approx(1.0 - base["gross"] * 0.5, rel=1e-9)
    np.testing.assert_allclose(scaled["weights"].values, base["weights"].values * 0.5, rtol=1e-9)


def test_regime_multiplier_default_is_noop():
    df, fns = _toy_df(), _one_factor()
    a = select_and_construct(df, fns, top_n=4, target_vol=0.50)
    b = select_and_construct(df, fns, top_n=4, target_vol=0.50, regime_multiplier=1.0)
    np.testing.assert_allclose(a["weights"].values, b["weights"].values)
    assert a["gross"] == pytest.approx(b["gross"])
    assert a["cash"] == pytest.approx(b["cash"])
