"""
regime_join.py — Attach contemporaneous market-regime labels to historical signal rows.

Part of the S0 validation harness.  Stratifies signal performance by regime
so IC-decay, perf-matrix, and turnover primitives can be sliced by:
    risk_on | neutral | risk_off | crisis

Public API
----------
label_dates(dates, vix, vix3m, spy, date_index) -> dict[str, str]
attach_regime(rows, date_to_regime, date_key="signal_date") -> list[dict]
fetch_regime_inputs(start, end) -> tuple  # pragma: no cover
"""

import numpy as np

from trade_modules.regime_detector import RegimeDetector

# ---------------------------------------------------------------------------
# Private helper — copied verbatim from scripts/regime_overlay_replay.py
# (regime-overlay branch) to avoid a cross-branch import.
# ---------------------------------------------------------------------------


def _build_daily_data(vix, vix3m, spy, i, lookback=504):
    """Copied verbatim from scripts/regime_overlay_replay.py (regime-overlay branch)."""
    lo = max(0, i - lookback)
    vh = np.asarray(vix[lo : i + 1], dtype=float)
    sh = np.asarray(spy[lo : i + 1], dtype=float)
    d = {
        "vix_current": float(vix[i]),
        "vix_history": vh,
        "vix_5d_ago": float(vix[i - 5]) if i >= 5 else float(vix[i]),
        "vix3m_current": float(vix3m[i]) if vix3m is not None else None,
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


# ---------------------------------------------------------------------------
# Module-level detector (stateless, no network, reused across calls)
# ---------------------------------------------------------------------------
_detector = RegimeDetector()


# ---------------------------------------------------------------------------
# Public pure functions
# ---------------------------------------------------------------------------


def label_dates(dates, vix, vix3m, spy, date_index):
    """PURE.  Given aligned historical arrays (vix, vix3m, spy) and their
    ``date_index`` (list of ISO-date strings matching the arrays), return
    ``{iso_date: regime_label}`` for each requested ``dates`` (list of ISO
    strings).

    For each date:
    - Find its position in date_index.
    - Positions < 30 → ``'neutral'`` (not enough history).
    - Dates not found in date_index → ``'neutral'``.
    - Otherwise build the RegimeDetector data dict via _build_daily_data,
      call compute_features → classify, and return the label string.

    Never raises regardless of input.
    """
    # Build a position lookup once
    date_pos = {d: i for i, d in enumerate(date_index)}

    result: dict[str, str] = {}
    for date in dates:
        try:
            pos = date_pos.get(date)
            if pos is None or pos < 30:
                result[date] = "neutral"
                continue

            data = _build_daily_data(vix, vix3m, spy, pos)
            features = _detector.compute_features(data)
            classification = _detector.classify(features, data)
            result[date] = classification["regime"]
        except Exception:
            result[date] = "neutral"

    return result


def attach_regime(rows, date_to_regime, date_key="signal_date"):
    """PURE.  Return a new list of new dicts — each original row plus a
    ``'regime'`` key sourced from ``date_to_regime[row[date_key]]``.
    Falls back to ``'neutral'`` when the date is absent.

    Input rows are never mutated.
    """
    result = []
    for row in rows:
        new_row = dict(row)
        new_row["regime"] = date_to_regime.get(row.get(date_key, ""), "neutral")
        result.append(new_row)
    return result


def fetch_regime_inputs(start, end):  # pragma: no cover
    """Fetch ^VIX, ^VIX3M, SPY closes over [start, end] via yfinance;
    align on SPY ∩ VIX index (VIX3M reindexed, NaN pre-2011 handled);
    return (date_index: list[iso], vix, vix3m, spy) as numpy arrays.

    Args:
        start: ISO date string e.g. ``"2018-01-01"``.
        end:   ISO date string e.g. ``"2024-12-31"``.
    """
    import yfinance as yf

    raw = yf.download(
        ["^VIX", "^VIX3M", "SPY"],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Drop rows where VIX or SPY is NaN; reindex VIX3M onto that index
    aligned = raw[["^VIX", "SPY"]].dropna()
    vix3m_aligned = raw["^VIX3M"].reindex(aligned.index)  # NaN where unavailable

    date_index = [d.strftime("%Y-%m-%d") for d in aligned.index]
    vix = aligned["^VIX"].to_numpy(dtype=float)
    spy = aligned["SPY"].to_numpy(dtype=float)
    vix3m = vix3m_aligned.to_numpy(dtype=float)

    return date_index, vix, vix3m, spy
