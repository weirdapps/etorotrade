import numpy as np
import pandas as pd

from trade_modules.v3.spine import low_vol, momentum_12_1, spine_scores


def _trend_close(n=300):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    up = 100 * (1.001 ** np.arange(n))  # steady uptrend
    flat = 100 * np.ones(n)
    down = 100 * (0.999 ** np.arange(n))
    return pd.DataFrame({"UP": up, "FLAT": flat, "DOWN": down}, index=idx)


def test_momentum_ranks_uptrend_highest():
    c = _trend_close()
    asof = c.index[-1]
    m = momentum_12_1(c, asof)
    assert m["UP"] > m["FLAT"] > m["DOWN"]


def test_momentum_empty_without_warmup():
    c = _trend_close()
    asof = c.index[10]  # < lookback
    assert momentum_12_1(c, asof).empty


def test_low_vol_prefers_stable():
    idx = pd.date_range("2025-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    calm = 100 + np.cumsum(rng.normal(0, 0.05, 300))
    wild = 100 + np.cumsum(rng.normal(0, 2.0, 300))
    c = pd.DataFrame({"CALM": calm, "WILD": wild}, index=idx)
    lv = low_vol(c, c.index[-1])
    assert lv["CALM"] > lv["WILD"]


def test_spine_scores_shape():
    c = _trend_close()
    dates = [c.index[-1]]
    s = spine_scores(c, dates)
    assert set(s.columns) == {"as_of", "ticker", "score"}
    assert len(s) == 3
