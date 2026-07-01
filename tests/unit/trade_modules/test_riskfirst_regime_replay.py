import numpy as np
import pytest

from scripts.regime_overlay_replay import build_daily_data, simulate


def test_simulate_full_exposure_matches_buy_and_hold():
    r = np.array([0.01, -0.02, 0.03, 0.00])
    m = np.ones(4)
    s = simulate(r, m)
    # cumulative return equals product of (1+r)-1
    expected = np.prod(1 + r) - 1
    assert s["total_return"] == pytest.approx(expected, rel=1e-9)
    assert s["pct_derisked"] == 0.0


def test_simulate_half_exposure_reduces_drawdown():
    r = np.array([-0.10, -0.10, 0.05])
    full = simulate(r, np.ones(3))
    half = simulate(r, np.full(3, 0.5))
    assert abs(half["max_drawdown"]) < abs(full["max_drawdown"])
    assert half["pct_derisked"] == 1.0


def test_build_daily_data_windows():
    vix = np.linspace(15, 25, 600)
    vix3m = np.linspace(16, 24, 600)
    spy = np.linspace(100, 130, 600)
    d = build_daily_data(vix, vix3m, spy, i=599, lookback=504)
    assert d["vix_current"] == vix[599]
    assert d["spy_current"] == spy[599]
    assert d["spy_52w_high"] == spy[599]  # rising series -> today is the high
    assert len(d["vix_history"]) <= 505
