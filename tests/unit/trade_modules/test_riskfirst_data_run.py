"""Tests for run_data regime wiring (network calls mocked)."""

import numpy as np
import pandas as pd

import trade_modules.riskfirst.data_run as dr


def _fake_prices(tickers, period="2y"):
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    data = {}
    for i, t in enumerate(list(tickers)):
        rets = 0.001 + 0.01 * np.sin(np.arange(300) / 7.0 + i)
        data[t] = 100.0 * np.cumprod(1 + rets)
    return pd.DataFrame(data, index=idx)


def _fake_sectors(tickers):
    labels = ["Tech", "Energy", "Health"]
    return {t: labels[i % 3] for i, t in enumerate(list(tickers))}


def test_run_data_overlay_disabled_stub(monkeypatch):
    monkeypatch.setattr(dr, "fetch_prices", _fake_prices)
    monkeypatch.setattr(dr, "fetch_sectors", _fake_sectors)
    res = dr.run_data(regime_overlay_enabled=False)
    assert res["regime"]["applied_multiplier"] == 1.0


def test_run_data_overlay_scales_gross(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "fetch_prices", _fake_prices)
    monkeypatch.setattr(dr, "fetch_sectors", _fake_sectors)
    sp = str(tmp_path / "state.json")
    # config.yaml is now AGGRESSIVE: crisis=0.20
    res = dr.run_data(
        regime_overlay_enabled=True,
        regime_fn=lambda: "crisis",
        regime_state_path=sp,
        persistence_days=1,
    )
    assert res["regime"]["confirmed_regime"] == "crisis"
    assert res["regime"]["applied_multiplier"] == 0.20
