"""Tests for news event gate wiring into shadow_run and data_run."""

import datetime
import json

import numpy as np
import pandas as pd
import pytest

import trade_modules.riskfirst.data_run as dr
import trade_modules.riskfirst.shadow_run as sr

# ── helpers shared with test_riskfirst_shadow.py ──────────────────────────────

_HEADER = "TKR,CAP,PET,PEF,P/S,FCF,ROE,DE,EG,%B,AM,B,52W"
_ROWS = [
    "AAA,100B,15,14,3,6,25,40,10,80,2,1.0,90",
    "BBB,50B,20,18,4,4,18,60,8,70,1,1.1,85",
    "CCC,30B,12,11,2,8,22,30,12,75,3,0.9,95",
    "DDD,10B,25,22,5,3,15,80,5,65,0,1.2,80",
    "MICRO,50M,,,,,,,,,,,",  # filtered by investability gate
]


def _write_universe(tmp_path):
    p = tmp_path / "etoro.csv"
    p.write_text(_HEADER + "\n" + "\n".join(_ROWS) + "\n")
    return str(p)


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


# ── shadow_run wiring ─────────────────────────────────────────────────────────


def test_shadow_event_gate_excludes_ticker(tmp_path):
    """A ticker in the event_risk file is absent from target_weights; event_excluded >= 1."""
    univ = _write_universe(tmp_path)
    risk_file = tmp_path / "event_risk.json"
    risk_file.write_text(json.dumps(["AAA"]))  # AAA is in the universe

    res = sr.run(
        universe_path=univ,
        portfolio_path="/nonexistent",
        top_n=3,
        event_gate_enabled=True,
        event_risk_path=str(risk_file),
    )
    assert "AAA" not in res["target_weights"].index
    assert res["event_excluded"] >= 1


def test_shadow_event_gate_disabled_no_exclusion(tmp_path):
    """With gate disabled, even a matching risk file has no effect; event_excluded == 0."""
    univ = _write_universe(tmp_path)
    risk_file = tmp_path / "event_risk.json"
    risk_file.write_text(json.dumps(["AAA", "BBB", "CCC", "DDD"]))

    res = sr.run(
        universe_path=univ,
        portfolio_path="/nonexistent",
        top_n=3,
        event_gate_enabled=False,
        event_risk_path=str(risk_file),
    )
    assert res["event_excluded"] == 0


def test_shadow_event_gate_missing_file_no_exclusion(tmp_path):
    """Missing event_risk file -> gate is a no-op; event_excluded == 0."""
    univ = _write_universe(tmp_path)

    res = sr.run(
        universe_path=univ,
        portfolio_path="/nonexistent",
        top_n=3,
        event_gate_enabled=True,
        event_risk_path=str(tmp_path / "nonexistent.json"),
    )
    assert res["event_excluded"] == 0


# ── data_run wiring ───────────────────────────────────────────────────────────


def test_data_run_earnings_blackout_excludes(tmp_path, monkeypatch):
    """A ticker whose next earnings date is within blackout_days is excluded."""
    monkeypatch.setattr(dr, "fetch_prices", _fake_prices)
    monkeypatch.setattr(dr, "fetch_sectors", _fake_sectors)

    # Pick a candidate that will survive pre-screening — any from the top of the real
    # universe (mocked prices give all tickers plausible data). We use an injected
    # earnings_fn to flag one candidate name deterministically.
    # We can't know which tickers survive the pre-screen without running it, so we
    # run once to discover a candidate, then run again with it flagged.
    probe = dr.run_data(
        event_gate_enabled=False,
        earnings_fn=lambda tks: {},
    )
    # pick any ticker from target_weights
    if not len(probe["target_weights"]):
        pytest.skip("pre-screen returned empty book on this machine")

    victim = probe["target_weights"].index[0]
    today = datetime.date.today()
    blackout_date = (today + datetime.timedelta(days=3)).isoformat()

    res = dr.run_data(
        event_gate_enabled=True,
        earnings_blackout_days=7,
        earnings_fn=lambda tks: {victim: blackout_date} if victim in tks else {},
    )
    assert victim not in res["target_weights"].index
    assert res["event_excluded"] >= 1


def test_data_run_event_gate_disabled(tmp_path, monkeypatch):
    """With gate disabled, event_excluded == 0 even with an earnings_fn that flags everything."""
    monkeypatch.setattr(dr, "fetch_prices", _fake_prices)
    monkeypatch.setattr(dr, "fetch_sectors", _fake_sectors)

    today = datetime.date.today()
    flood = (today + datetime.timedelta(days=2)).isoformat()

    res = dr.run_data(
        event_gate_enabled=False,
        earnings_fn=lambda tks: dict.fromkeys(tks, flood),
    )
    assert res["event_excluded"] == 0
