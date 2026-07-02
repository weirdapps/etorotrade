"""Tests: config.yaml is authoritative for regime overlay and event gate runners.

Covers:
  - news_gate.load_config: defaults on missing file; values from a temp config
  - Runner honours config: crisis -> 0.20 (aggressive profile) via resolve_regime_multiplier
  - Kwarg override: regime_overlay_enabled=False suppresses overlay even when config enables it
  - Existing shadow/data_run/news tests must stay green (they pass explicit kwargs)
"""

import numpy as np
import pandas as pd
import pytest

import trade_modules.riskfirst.data_run as dr
import trade_modules.riskfirst.shadow_run as sr
from trade_modules.riskfirst.news_gate import load_config as load_event_cfg

# ── news_gate.load_config ──────────────────────────────────────────────────────


def test_news_gate_load_config_defaults_on_missing_file(tmp_path):
    cfg = load_event_cfg(str(tmp_path / "nope.yaml"))
    assert cfg["enabled"] is True
    assert cfg["earnings_blackout_days"] == 7
    assert "event_risk" in cfg["event_risk_path"]


def test_news_gate_load_config_reads_values(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "event_gate:\n  enabled: false\n  earnings_blackout_days: 3\n"
        "  event_risk_path: /tmp/risk.json\n"
    )
    cfg = load_event_cfg(str(p))
    assert cfg["enabled"] is False
    assert cfg["earnings_blackout_days"] == 3
    assert cfg["event_risk_path"] == "/tmp/risk.json"


def test_news_gate_load_config_partial_override(tmp_path):
    """Only earnings_blackout_days set; enabled and path fall back to defaults."""
    p = tmp_path / "cfg.yaml"
    p.write_text("event_gate:\n  earnings_blackout_days: 5\n")
    cfg = load_event_cfg(str(p))
    assert cfg["enabled"] is True
    assert cfg["earnings_blackout_days"] == 5


# ── Runner honours config (aggressive profile crisis=0.20) ────────────────────

_HEADER = "TKR,CAP,PET,PEF,P/S,FCF,ROE,DE,EG,%B,AM,B,52W"
_ROWS = [
    "AAA,100B,15,14,3,6,25,40,10,80,2,1.0,90",
    "BBB,50B,20,18,4,4,18,60,8,70,1,1.1,85",
    "CCC,30B,12,11,2,8,22,30,12,75,3,0.9,95",
    "DDD,10B,25,22,5,3,15,80,5,65,0,1.2,80",
]


def _write_universe(tmp_path):
    p = tmp_path / "etoro.csv"
    p.write_text(_HEADER + "\n" + "\n".join(_ROWS) + "\n")
    return str(p)


def test_shadow_runner_crisis_uses_aggressive_table(tmp_path):
    """repo config.yaml (aggressive) crisis=0.20 drives the multiplier when runner
    reads config; persistence_days=1 via kwarg to confirm on first run."""
    sp = str(tmp_path / "state.json")
    res = sr.run(
        universe_path=_write_universe(tmp_path),
        portfolio_path="/nonexistent",
        top_n=3,
        regime_fn=lambda: "crisis",
        regime_state_path=sp,
        persistence_days=1,  # kwarg override: confirm on first run
    )
    assert res["regime"]["confirmed_regime"] == "crisis"
    assert res["regime"]["applied_multiplier"] == pytest.approx(0.20)


# ── Kwarg override: regime_overlay_enabled=False suppresses overlay ────────────


def test_shadow_kwarg_override_disables_overlay(tmp_path):
    """regime_overlay_enabled=False must produce multiplier=1.0 even when
    config.yaml has enabled: true."""
    res = sr.run(
        universe_path=_write_universe(tmp_path),
        portfolio_path="/nonexistent",
        top_n=3,
        regime_overlay_enabled=False,
    )
    assert res["regime"]["applied_multiplier"] == pytest.approx(1.0)


# ── data_run: kwarg override and crisis table ─────────────────────────────────


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


def test_data_run_kwarg_override_disables_overlay(monkeypatch):
    monkeypatch.setattr(dr, "fetch_prices", _fake_prices)
    monkeypatch.setattr(dr, "fetch_sectors", _fake_sectors)
    res = dr.run_data(regime_overlay_enabled=False)
    assert res["regime"]["applied_multiplier"] == pytest.approx(1.0)


def test_data_run_crisis_uses_aggressive_table(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "fetch_prices", _fake_prices)
    monkeypatch.setattr(dr, "fetch_sectors", _fake_sectors)
    sp = str(tmp_path / "state.json")
    res = dr.run_data(
        regime_overlay_enabled=True,
        regime_fn=lambda: "crisis",
        regime_state_path=sp,
        persistence_days=1,
    )
    assert res["regime"]["confirmed_regime"] == "crisis"
    assert res["regime"]["applied_multiplier"] == pytest.approx(0.20)
