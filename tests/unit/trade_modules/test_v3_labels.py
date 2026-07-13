import numpy as np
import pandas as pd

from trade_modules.v3.labels import cross_sectional_ic, demean_by_date, forward_returns, ic_summary


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


def test_cross_sectional_ic_perfect_rank():
    asof = pd.Timestamp("2026-01-01")
    scores = pd.DataFrame(
        {"as_of": [asof] * 3, "ticker": ["A", "B", "C"], "score": [1.0, 2.0, 3.0]}
    )
    fwd = pd.DataFrame(
        {
            "as_of": [asof] * 3,
            "ticker": ["A", "B", "C"],
            "horizon": [5, 5, 5],
            "fwd_ret": [0.01, 0.02, 0.03],
        }
    )
    ic = cross_sectional_ic(scores, fwd, 5)
    assert np.isclose(ic.loc[asof], 1.0)


def test_ic_summary_stats():
    ic = pd.Series([0.1, 0.1, 0.1, 0.1])
    s = ic_summary(ic)
    assert s["n"] == 4 and np.isclose(s["mean_ic"], 0.1) and s["hit_rate"] == 1.0
