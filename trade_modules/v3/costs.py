"""BUILD ⑥b (2026-07-18): transaction-cost model for v3.

Round-trip cost = entry+exit spread (by cap tier) + eToro overnight financing over
the holding horizon. Reuses ``liquidity_filter``'s ``TIER_SPREAD_BPS`` and
``ETORO_OVERNIGHT_ANNUAL_RATE`` so the cost constants have a single home.

Principled use: a NET-of-cost IC in validation (trial-register ``net_ir_min``) —
grading net alpha. Applying cost at the ACTION level (suppressing trades whose
expected gain does not clear the round trip) would change LIVE trading and is a
separate, sign-off-required increment.
"""

from __future__ import annotations

import pandas as pd

from trade_modules.liquidity_filter import ETORO_OVERNIGHT_ANNUAL_RATE, TIER_SPREAD_BPS
from trade_modules.v3.liquidity import cap_tier
from trade_modules.v3.universe import parse_cap

# V3_SIGNAL_HORIZON is 21 trading days ~ 30 calendar days; eToro financing accrues
# on calendar days, so the cost horizon is expressed in calendar days.
DEFAULT_HOLDING_DAYS = 30


def roundtrip_cost_pct(cap_usd, *, holding_days: int = DEFAULT_HOLDING_DAYS) -> float:
    """Round-trip cost as a % of position for a name of this cap tier: entry+exit
    spread (2x the tier's bps) + eToro overnight financing over the holding period."""
    tier = cap_tier(cap_usd)
    spread_pct = TIER_SPREAD_BPS[tier] / 10000.0 * 2.0 * 100.0
    financing_pct = ETORO_OVERNIGHT_ANNUAL_RATE * (holding_days / 365.0) * 100.0
    return spread_pct + financing_pct


def net_of_cost_return(
    gross_return_pct, cap_usd, *, holding_days: int = DEFAULT_HOLDING_DAYS
) -> float:
    """Gross return % minus this name's round-trip cost %."""
    return float(gross_return_pct) - roundtrip_cost_pct(cap_usd, holding_days=holding_days)


def cost_map_from_panel(etoro_csv_path, *, holding_days: int = DEFAULT_HOLDING_DAYS) -> dict:
    """``{TICKER_UPPER: roundtrip_cost_pct}`` built from the panel's CAP column.

    Cap is slow-changing, so a current-panel lookup is an adequate per-name cost for
    grading the (short) IC log net of cost. An unreadable panel returns ``{}``.
    """
    try:
        df = pd.read_csv(etoro_csv_path, na_values=["--"])
    except (OSError, ValueError):
        return {}
    if "TKR" not in df.columns or "CAP" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for tkr, cap in zip(df["TKR"], df["CAP"], strict=False):
        t = str(tkr).strip().upper()
        if not t or t == "NAN":
            continue
        out[t] = roundtrip_cost_pct(parse_cap(cap), holding_days=holding_days)
    return out
