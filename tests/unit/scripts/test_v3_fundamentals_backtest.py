"""_preferred_price: the fundamentals backtest must use a TRUE price return where
available, not the marketcap/shares proxy whose share-count term mechanically couples
the forward return to the net_issuance / asset_growth factors.
"""

import numpy as np
import pandas as pd

from scripts.v3_fundamentals_backtest import _preferred_price


def _idx(n):
    return pd.date_range("2020-01-31", periods=n, freq="ME")


def test_preferred_price_uses_sep_where_covered_else_proxy():
    idx = _idx(30)
    proxy = pd.DataFrame({"AAA": np.arange(1.0, 31.0), "BBB": np.arange(1.0, 31.0)}, index=idx)
    sep = pd.DataFrame({"AAA": np.arange(100.0, 130.0)}, index=idx)  # SEP covers AAA only
    price = _preferred_price(proxy, sep, min_months=24)
    assert (price["AAA"] == sep["AAA"]).all()  # TRUE SEP price used where covered
    assert (price["BBB"] == proxy["BBB"]).all()  # proxy fallback where SEP absent


def test_preferred_price_ignores_thin_sep_coverage():
    idx = _idx(30)
    proxy = pd.DataFrame({"AAA": np.arange(1.0, 31.0)}, index=idx)
    sep = pd.DataFrame({"AAA": [np.nan] * 28 + [5.0, 6.0]}, index=idx)  # only 2 months
    price = _preferred_price(proxy, sep, min_months=24)
    assert (price["AAA"] == proxy["AAA"]).all()  # too thin -> keep proxy (no boundary corruption)


def test_preferred_price_none_sep_is_identity():
    idx = _idx(5)
    proxy = pd.DataFrame({"AAA": np.arange(1.0, 6.0)}, index=idx)
    assert _preferred_price(proxy, None).equals(proxy)
    assert _preferred_price(proxy, pd.DataFrame()).equals(proxy)
