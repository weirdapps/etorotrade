import numpy as np
import pandas as pd

from trade_modules.v3.labels import demean_by_date, forward_returns


def _close():
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {"A": [100, 101, 102, 103, 104, 105], "B": [100, 100, 100, 100, 100, 100]},
        index=idx,
        dtype=float,
    )


def test_forward_returns_are_strictly_forward():
    c = _close()
    asof = c.index[0]
    fwd = forward_returns(c, [asof], [2])
    a = fwd[(fwd.ticker == "A") & (fwd.horizon == 2)]["fwd_ret"].iloc[0]
    assert np.isclose(a, 102 / 100 - 1)
    # no row uses data at/before as_of for the future leg
    assert (fwd["horizon"] == 2).all()


def test_forward_returns_skips_when_no_future_bar():
    c = _close()
    fwd = forward_returns(c, [c.index[5]], [2])  # last bar, no +2
    assert fwd.empty


def test_demean_by_date():
    c = _close()
    asof = c.index[0]
    fwd = demean_by_date(forward_returns(c, [asof], [2]))
    grp = fwd[fwd.horizon == 2]
    assert np.isclose(grp["net_alpha"].sum(), 0.0, atol=1e-9)
