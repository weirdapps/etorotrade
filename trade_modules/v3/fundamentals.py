"""PIT fundamental factor derivations — the data-blocked durable premia we cannot
build from the current panel (need Sharadar SF1 balance-sheet / cash-flow history).

Pure, unit-tested; no I/O. Each takes already-point-in-time values (from
``fundamentals_store.read_asof`` / ``read_history``, i.e. datekey <= as_of) and
returns the raw factor value. Directional signs are applied later in the combiner
(book_to_price + / asset_growth − / gp_assets + / accruals − / sue +).

References: Fama-French (book-to-market; CMA investment), Hou-Xue-Zhang (q-factor
investment), Novy-Marx (gross-profitability/assets), Sloan (accruals), Foster-
Olsen-Shevlin / Bernard-Thomas (PEAD, seasonal-random-walk SUE).
"""

from __future__ import annotations

import math

import pandas as pd


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def book_to_price(equity, market_cap) -> float:
    """Book-to-price = shareholders' equity / market cap (high = cheap on book).

    The Fama-French value anchor and the right valuation metric for financials.
    NaN when book is non-positive (meaningless — e.g. buyback-heavy / intangible
    firms) or market cap is missing/zero.
    """
    e, m = _num(equity), _num(market_cap)
    if math.isnan(e) or math.isnan(m) or e <= 0 or m <= 0:
        return math.nan
    return e / m


def asset_growth(assets_now, assets_year_ago) -> float:
    """Year-over-year total-asset growth (the CMA / investment factor; low is good)."""
    a, a0 = _num(assets_now), _num(assets_year_ago)
    if math.isnan(a) or math.isnan(a0) or a0 == 0:
        return math.nan
    return a / a0 - 1.0


def gross_profit_to_assets(gross_profit, assets) -> float:
    """Gross profitability / total assets (Novy-Marx quality; high is good)."""
    g, a = _num(gross_profit), _num(assets)
    if math.isnan(g) or math.isnan(a) or a == 0:
        return math.nan
    return g / a


def accruals(net_income, operating_cash_flow, assets) -> float:
    """Balance-sheet accruals = (net income − operating cash flow) / assets.

    High accruals = earnings not backed by cash = lower earnings quality (Sloan);
    scored negative in the combiner.
    """
    ni, cf, a = _num(net_income), _num(operating_cash_flow), _num(assets)
    if math.isnan(ni) or math.isnan(cf) or math.isnan(a) or a == 0:
        return math.nan
    return (ni - cf) / a


def srw_sue(eps_history) -> float:
    """Seasonal-random-walk standardized unexpected earnings (PEAD signal).

    ``eps_history`` is a ticker's chronological quarterly EPS (oldest→newest). SUE =
    the latest year-over-year EPS change divided by the volatility of those seasonal
    changes: ``(eps_t − eps_{t-4}) / std(Δ4 eps)``. Needs ≥5 quarters (≥2 seasonal
    diffs); NaN otherwise or on zero dispersion. Uses only actuals — no consensus feed.
    """
    e = pd.to_numeric(pd.Series(eps_history), errors="coerce").dropna()
    diffs = e.diff(4).dropna()
    if len(diffs) < 2:
        return math.nan
    sd = float(diffs.std(ddof=1))
    if not sd or math.isnan(sd) or sd == 0:
        return math.nan
    return float(diffs.iloc[-1] / sd)
