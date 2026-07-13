# tests/unit/trade_modules/test_v3_validate_spine.py
import numpy as np
import pandas as pd

from trade_modules.v3.validate_spine import build_rows, classify_regimes, run_gate


def _strong_signal(n_dates=40, n_names=30, horizon=5, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="W")
    srows, frows = [], []
    for d in dates:
        s = rng.normal(0, 1, n_names)
        for k in range(n_names):
            tkr = f"T{k}"
            srows.append({"as_of": d, "ticker": tkr, "score": float(s[k])})
            # forward return strongly increasing in score + noise
            frows.append(
                {
                    "as_of": d,
                    "ticker": tkr,
                    "horizon": horizon,
                    "fwd_ret": float(0.02 * s[k] + rng.normal(0, 0.01)),
                }
            )
    return pd.DataFrame(srows), pd.DataFrame(frows)


def test_build_rows_shape_and_keys():
    scores, fwd = _strong_signal()
    rows = build_rows(scores, fwd, [5], top_q=0.2)
    assert rows and {"ticker", "signal_date", "horizon", "net_alpha", "signal"}.issubset(rows[0])
    assert all(r["signal"] == "spine" for r in rows)


def test_run_gate_passes_ic_on_strong_signal():
    scores, fwd = _strong_signal()
    v = run_gate(scores, fwd, [5], n_trials=2, min_obs=5)
    assert v["primary_ic_pass"] is True
    assert v["ic"][5]["mean_ic"] > 0.5


def test_build_rows_attaches_regime_label():
    scores, fwd = _strong_signal()
    dates = scores["as_of"].unique()
    regime = dict.fromkeys(dates, "RISK_ON")
    rows = build_rows(scores, fwd, [5], regime=regime)
    assert all(r["regime"] == "RISK_ON" for r in rows)


def test_build_rows_no_regime_key_when_none():
    scores, fwd = _strong_signal()
    rows = build_rows(scores, fwd, [5])
    assert all("regime" not in r for r in rows)


def test_classify_regimes_uptrend_mostly_risk_on():
    """A steady uptrend with low vol should classify as RISK_ON after warmup."""
    idx = pd.date_range("2023-01-01", periods=400, freq="B")
    s = pd.Series(np.linspace(100.0, 200.0, 400), index=idx)
    result = classify_regimes(s)
    # Should have entries after the 200-bar warmup period
    assert len(result) > 0
    labels = list(result.values())
    # Majority should be RISK_ON (steady uptrend = above MA, low vol)
    risk_on_count = labels.count("RISK_ON")
    assert risk_on_count > len(labels) * 0.5, f"Expected mostly RISK_ON, got {labels}"


def test_classify_regimes_high_vol_downtrend_risk_off():
    """A noisy downtrend should classify mostly RISK_OFF after warmup."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # Downward drift + high noise
    prices = np.cumprod(1 + rng.normal(-0.003, 0.025, n)) * 100
    s = pd.Series(prices, index=idx)
    result = classify_regimes(s)
    assert len(result) > 0
    labels = list(result.values())
    risk_off_count = labels.count("RISK_OFF")
    # At least a material fraction should be RISK_OFF
    assert risk_off_count > len(labels) * 0.2, f"Expected substantial RISK_OFF, got {labels}"


def test_classify_regimes_short_series_returns_empty():
    """Series shorter than 200 bars → no entries (insufficient history)."""
    s = pd.Series(
        np.linspace(100, 110, 100), index=pd.date_range("2025-01-01", periods=100, freq="B")
    )
    result = classify_regimes(s)
    assert result == {}


def test_classify_regimes_values_are_valid_labels():
    """All output values must be one of the three valid regime labels."""
    idx = pd.date_range("2022-01-01", periods=300, freq="B")
    s = pd.Series(np.linspace(100.0, 150.0, 300), index=idx)
    result = classify_regimes(s)
    valid = {"RISK_ON", "NEUTRAL", "RISK_OFF"}
    assert all(v in valid for v in result.values())


def test_run_gate_rejects_noise():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2025-01-01", periods=40, freq="W")
    srows, frows = [], []
    for d in dates:
        for k in range(30):
            srows.append({"as_of": d, "ticker": f"T{k}", "score": float(rng.normal())})
            frows.append(
                {
                    "as_of": d,
                    "ticker": f"T{k}",
                    "horizon": 5,
                    "fwd_ret": float(rng.normal(0, 0.01)),
                }
            )
    v = run_gate(pd.DataFrame(srows), pd.DataFrame(frows), [5], n_trials=2, min_obs=5)
    assert v["primary_ic_pass"] is False
