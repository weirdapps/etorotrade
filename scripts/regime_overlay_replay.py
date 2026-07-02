"""Historical replay to calibrate the regime-overlay exposure profile.

Reuses the (already pure) RegimeDetector.compute_features/classify on a rolling
window of historical VIX / VIX3M / SPY, applies confirm_regime hysteresis, and
compares the dialled strategy against always-invested. SPY is a PROXY for the
book (the risk-first book has no own history) — this measures the dial mechanism,
not the full 20-name book. Output: report + chart to ~/Downloads.
"""

from __future__ import annotations

import datetime
import os

import numpy as np

from trade_modules.regime_detector import RegimeDetector
from trade_modules.riskfirst.regime_overlay import confirm_regime, exposure_for_regime

PROFILES = {
    "mild": {"risk_on": 1.00, "neutral": 1.00, "risk_off": 0.75, "crisis": 0.50},
    "moderate": {"risk_on": 1.00, "neutral": 0.90, "risk_off": 0.65, "crisis": 0.40},
    "aggressive": {"risk_on": 1.00, "neutral": 0.80, "risk_off": 0.50, "crisis": 0.20},
}


def build_daily_data(vix, vix3m, spy, i, lookback=504):
    lo = max(0, i - lookback)
    vh = np.asarray(vix[lo : i + 1], dtype=float)
    sh = np.asarray(spy[lo : i + 1], dtype=float)
    d = {
        "vix_current": float(vix[i]),
        "vix_history": vh,
        "vix_5d_ago": float(vix[i - 5]) if i >= 5 else float(vix[i]),
        "vix3m_current": (None if (vix3m is None or np.isnan(vix3m[i])) else float(vix3m[i])),
        "spy_current": float(spy[i]),
        "spy_history": sh,
        "spy_52w_high": float(np.max(spy[max(0, i - 252) : i + 1])),
    }
    if len(sh) >= 504:
        d["spy_2yr_return"] = float((sh[-1] - sh[-504]) / sh[-504] * 100)
    elif len(sh) >= 252:
        d["spy_2yr_return"] = float((sh[-1] - sh[-252]) / sh[-252] * 100)
    else:
        d["spy_2yr_return"] = None
    return d


def regime_series(vix, vix3m, spy, persistence_days, table=None, warmup=252):
    det = RegimeDetector()
    raw = []
    for i in range(len(spy)):
        if i < warmup:
            raw.append("risk_on")
            continue
        data = build_daily_data(vix, vix3m, spy, i)
        feats = det.compute_features(data)
        raw.append(det.classify(feats, data)["regime"])
    confirmed = [confirm_regime(raw[: i + 1], persistence_days) for i in range(len(raw))]
    return confirmed


def simulate(returns, multipliers):
    returns = np.asarray(returns, dtype=float)
    mult = np.clip(np.asarray(multipliers, dtype=float), 0.0, 1.0)
    strat = mult * returns
    equity = np.cumprod(1 + strat)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    n = len(returns)
    ann = 252.0 / n if n else 0.0
    total = float(equity[-1] - 1) if n else 0.0
    vol = float(np.std(strat, ddof=1) * np.sqrt(252)) if n > 1 else 0.0
    return {
        "total_return": total,
        "cagr": float((equity[-1]) ** ann - 1) if n else 0.0,
        "vol": vol,
        "sharpe": float(np.mean(strat) / np.std(strat, ddof=1) * np.sqrt(252))
        if n > 1 and np.std(strat, ddof=1) > 0
        else 0.0,
        "max_drawdown": float(dd.min()) if n else 0.0,
        # NOTE: day 0's multiplier is forced to 1.0 by the caller's lag, so
        # pct_derisked is understated by at most 1/N (immaterial).
        "pct_derisked": float(np.mean(mult < 1.0)) if n else 0.0,
        "switches": int(np.sum(mult[1:] != mult[:-1])) if n > 1 else 0,
    }


def _fetch(symbol, start):  # pragma: no cover - network
    import yfinance as yf

    h = yf.Ticker(symbol).history(start=start)["Close"]
    return h


def main(argv=None):  # pragma: no cover - network integration entry point
    import sys

    args = argv if argv is not None else sys.argv[1:]
    start = args[0] if args else "2020-01-01"

    vix = _fetch("^VIX", start)
    vix3m = _fetch("^VIX3M", start)
    spy = _fetch("SPY", start)
    # Normalize to date-only to avoid timezone-mismatch (VIX=Chicago, SPY=NY)
    vix.index = vix.index.normalize().tz_localize(None)
    vix3m.index = vix3m.index.normalize().tz_localize(None)
    spy.index = spy.index.normalize().tz_localize(None)
    # Build working index from SPY ∩ VIX only (both go back to the 1990s).
    # VIX3M only starts ~2011; reindex it onto the wider index so pre-2011
    # dates get NaN rather than silently dropping all pre-2011 rows.
    idx = spy.index.intersection(vix.index)
    spy, vix = spy.reindex(idx), vix.reindex(idx)
    vix3m = vix3m.reindex(idx)  # NaN where VIX3M has no history
    spy_ret = spy.pct_change().fillna(0).to_numpy()

    lines = [
        f"# Regime overlay replay (SPY proxy, {start} to present)",
        "",
        "| Profile | Total | CAGR | Vol | Sharpe | MaxDD | %derisked | switches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = simulate(spy_ret, np.ones(len(spy_ret)))
    lines.append(
        f"| always-100% | {base['total_return']:.1%} | {base['cagr']:.1%} | "
        f"{base['vol']:.1%} | {base['sharpe']:.2f} | {base['max_drawdown']:.1%} | 0% | 0 |"
    )
    for name, table in PROFILES.items():
        conf = regime_series(vix.to_numpy(), vix3m.to_numpy(), spy.to_numpy(), 2, table)
        mult = np.array([exposure_for_regime(c, table) for c in conf])
        mult = np.concatenate([[1.0], mult[:-1]])  # lag: yesterday's dial on today's return
        s = simulate(spy_ret, mult)
        lines.append(
            f"| {name} | {s['total_return']:.1%} | {s['cagr']:.1%} | {s['vol']:.1%} | "
            f"{s['sharpe']:.2f} | {s['max_drawdown']:.1%} | {s['pct_derisked']:.0%} | {s['switches']} |"
        )

    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{ts}_regime_overlay_replay.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
