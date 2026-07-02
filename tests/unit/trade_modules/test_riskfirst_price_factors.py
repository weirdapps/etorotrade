"""Tests for the price-history factor closures (true momentum + realized-vol low-vol)
that replace the snapshot proxies once price history is available."""

import numpy as np
import pandas as pd

from trade_modules.riskfirst.prices import price_lowvol_factor, price_momentum_factor


def _prices():
    n = 300
    return pd.DataFrame(
        {
            "RISE": np.linspace(100, 200, n),
            "FALL": np.linspace(200, 100, n),
            "CALM": 100 * (1.0003 ** np.arange(n)),
            "WILD": [100 * (1.03 if i % 2 == 0 else 1 / 1.03) ** 1 for i in range(n)],
        }
    )


def test_price_momentum_ranks_riser_above_faller():
    compute = price_momentum_factor(_prices())
    z = compute(pd.DataFrame(index=["RISE", "FALL"]))
    assert z.loc["RISE"] > z.loc["FALL"]


def test_price_lowvol_ranks_calm_above_wild():
    compute = price_lowvol_factor(_prices())
    z = compute(pd.DataFrame(index=["CALM", "WILD"]))
    assert z.loc["CALM"] > z.loc["WILD"]


def test_missing_ticker_is_nan():
    compute = price_momentum_factor(_prices())
    z = compute(pd.DataFrame(index=["RISE", "NOPRICE"]))
    assert np.isnan(z.loc["NOPRICE"])
    assert list(z.index) == ["RISE", "NOPRICE"]
