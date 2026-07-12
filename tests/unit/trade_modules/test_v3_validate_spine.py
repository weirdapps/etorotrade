# tests/unit/trade_modules/test_v3_validate_spine.py
import numpy as np
import pandas as pd

from trade_modules.v3.validate_spine import build_rows, run_gate


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
