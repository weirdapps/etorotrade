"""v3 Phase 5C — PM construction: suggested actions vs live portfolio.

Turns the risk-gated target book (from build_portfolio) plus the live portfolio
weights into a per-ticker action list for DECISION-SUPPORT.  The user reviews
the list and decides whether to execute.  This module does NOT trade.

Pure function: no I/O, no network, no yahoofinance.core.config import.
"""

from __future__ import annotations

import pandas as pd

# A ticker with weight below this threshold is considered "not held / not targeted".
_EPSILON: float = 1e-6

_BUY_ADD_ACTIONS: frozenset[str] = frozenset({"BUY", "ADD"})
_TRIM_SELL_ACTIONS: frozenset[str] = frozenset({"TRIM", "SELL"})


def _safe_get(row, col: str):
    """Extract a column value from a scored row, returning None on NaN / missing."""
    if row is None:
        return None
    val = row.get(col) if hasattr(row, "get") else getattr(row, col, None)
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def build_actions(
    target_weights: pd.Series,
    current_weights: pd.Series,
    scored: pd.DataFrame,
    *,
    nav: float | None = None,
    add_trim_threshold: float = 0.005,
) -> list[dict]:
    """Build per-ticker action list: BUY / ADD / TRIM / SELL / HOLD.

    For every ticker in the UNION of ``target_weights`` and ``current_weights``,
    emit one action dict.  Names where both target and current are zero (below
    ``_EPSILON``) are silently skipped.

    Action rules (epsilon = 1e-6):
    - **BUY**:  target > epsilon AND current < epsilon.
    - **SELL**: current > epsilon AND target < epsilon.
    - **ADD**:  both > epsilon AND delta_pct > +add_trim_threshold.
    - **TRIM**: both > epsilon AND delta_pct < -add_trim_threshold.
    - **HOLD**: both > epsilon AND |delta_pct| <= add_trim_threshold.

    Args:
        target_weights: Risk-gated target allocation (fractions summing <= 1).
            Indexed by ticker.  Zero-weight names may be absent.
        current_weights: Current live allocation (fractions summing <= 1).
            Indexed by ticker.  May be empty (e.g. portfolio.csv has no weight
            column; the caller should pass equal weights over listed holdings as a
            best-effort approximation in that case).
        scored: Output of compute_scores — carries ``name``, ``sector``,
            ``conviction``, ``price``, ``stop_loss``, ``take_profit`` columns
            (indexed by ticker).  SELL names absent from scored get None for those
            enriched fields.
        nav: Total portfolio NAV in USD.  When provided,
            ``delta_usd = delta_pct * nav``.  When ``None`` (portfolio.csv has no
            per-position USD value column), ``delta_usd`` is ``None``.
        add_trim_threshold: Minimum |delta_pct| required to escalate from HOLD to
            ADD / TRIM.  Default 0.005 (0.5 pp).

    Returns:
        List of action dicts sorted:
        **BUY + ADD** (by target_pct desc) → **TRIM + SELL** (by current_pct desc)
        → **HOLD** (by target_pct desc).

        Each dict has keys::

            {ticker, name, sector, conviction, action, current_pct, target_pct,
             delta_pct, delta_usd, price, stop_loss, take_profit}
    """
    all_tickers: set[str] = set(target_weights.index) | set(current_weights.index)

    actions: list[dict] = []
    for ticker in all_tickers:
        target_pct = float(target_weights.get(ticker, 0.0))
        current_pct = float(current_weights.get(ticker, 0.0))
        delta_pct = target_pct - current_pct
        delta_usd = float(delta_pct) * nav if nav is not None else None

        in_target = target_pct > _EPSILON
        in_current = current_pct > _EPSILON

        if in_target and not in_current:
            action = "BUY"
        elif in_current and not in_target:
            action = "SELL"
        elif in_target and in_current:
            if delta_pct > add_trim_threshold:
                action = "ADD"
            elif delta_pct < -add_trim_threshold:
                action = "TRIM"
            else:
                action = "HOLD"
        else:
            # Both zero / sub-epsilon — ghost ticker, skip.
            continue

        row = scored.loc[ticker] if (scored is not None and ticker in scored.index) else None

        actions.append(
            {
                "ticker": ticker,
                "name": _safe_get(row, "name"),
                "sector": _safe_get(row, "sector"),
                "conviction": _safe_get(row, "conviction"),
                "action": action,
                "current_pct": current_pct,
                "target_pct": target_pct,
                "delta_pct": delta_pct,
                "delta_usd": delta_usd,
                "price": _safe_get(row, "price"),
                "stop_loss": _safe_get(row, "stop_loss"),
                "take_profit": _safe_get(row, "take_profit"),
            }
        )

    # Sort: BUY+ADD (target_pct desc) → TRIM+SELL (current_pct desc) → HOLD
    def _sort_key(item: dict) -> tuple:
        act = item["action"]
        if act in _BUY_ADD_ACTIONS:
            return (0, -item["target_pct"])
        if act in _TRIM_SELL_ACTIONS:
            return (1, -item["current_pct"])
        return (2, -item["target_pct"])  # HOLD

    actions.sort(key=_sort_key)
    return actions
