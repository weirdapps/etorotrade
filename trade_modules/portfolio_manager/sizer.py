"""ERC-based action-plan sizer.

Takes the S3 final universe (action + conviction per ticker) and sizes each
position using Equal Risk Contribution, subject to name cap, sector cap, and a
minimum position floor. Long-only; core positions are never touched by ERC.

Pure function: no I/O, no mutation of inputs.
"""

from __future__ import annotations

from trade_modules.riskfirst.construct import erc_weights
from trade_modules.riskfirst.covariance import single_factor_cov

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

_DEFAULT_CFG: dict = {
    "name_cap": 0.12,
    "sector_cap": 0.35,
    "trim_to": 0.80,
    "min_position": 0.01,
    "max_gross": 1.0,  # hard no-leverage ceiling on total post-sizing gross
}

_DEPLOY_ACTIONS = {"BUY", "ADD"}
_SORT_ORDER = {"BUY": 0, "ADD": 1, "TRIM": 2, "SELL": 3, "HOLD": 4}


def _merge_cfg(user_cfg: dict | None) -> dict:
    cfg = dict(_DEFAULT_CFG)
    if user_cfg:
        cfg.update(user_cfg)
    return cfg


def _erc_additions(deploy_rows: list[dict], budget_frac: float) -> dict[str, float]:
    """Return {ticker: raw_add} using ERC weights scaled by budget_frac."""
    if not deploy_rows or budget_frac <= 0:
        return {r["ticker"]: 0.0 for r in deploy_rows}

    if len(deploy_rows) == 1:
        return {deploy_rows[0]["ticker"]: budget_frac}

    betas = [float(r.get("beta") or 1.0) for r in deploy_rows]
    cov = single_factor_cov(betas, market_vol=0.18, idio_vol=0.15)
    w = erc_weights(cov)  # sums to 1
    return {r["ticker"]: float(w[i]) * budget_frac for i, r in enumerate(deploy_rows)}


def size_book(
    final_universe: list[dict],
    current_weights: dict[str, float],
    budget_frac: float,
    cfg: dict | None = None,
    nav: float | None = None,
) -> list[dict]:
    """Size the S3 final universe into a concrete action plan.

    Args:
        final_universe: List of {ticker, action, conviction, beta?, sector?}.
            actions ∈ {BUY, ADD, HOLD, TRIM, SELL}.
        current_weights: {ticker: weight_frac} real book (fractions, long-only).
        budget_frac: Deployable fraction of NAV from deployable_budget().
        cfg: Optional overrides — name_cap, sector_cap, trim_to, min_position.
        nav: Portfolio NAV in USD. When provided, delta_usd is computed as
            delta_pct * nav. When None, delta_usd is None.

    Returns:
        List of {ticker, action, current_pct, target_pct, delta_pct, delta_usd,
        conviction} sorted BUY/ADD by target descending, then TRIM/SELL, HOLD.
        Long-only: target_pct >= 0 always.

    Note:
        Budget may be partially undeployed when name or sector caps bind.
        Freed budget is not redistributed — this is intentional behaviour to
        keep the sizer simple and single-pass.
    """
    # Sizing is risk-parity (ERC); conviction is passed through for reference, not used for weighting.
    # Pure: never mutate inputs
    universe = [dict(r) for r in final_universe]
    weights = dict(current_weights)

    c = _merge_cfg(cfg)
    name_cap: float = c["name_cap"]
    sector_cap: float = c["sector_cap"]
    trim_to: float = c["trim_to"]
    min_position: float = c["min_position"]
    max_gross: float = c["max_gross"]

    # Step 0 — hard no-leverage guard (F5): the deployed additions must never
    # push total post-sizing gross above max_gross (default 1.0 = fully invested,
    # no leverage).  Existing holds are fixed in this single pass, so we clamp the
    # deployable budget to the gross headroom that REMAINS after accounting for
    # every non-addition target (HOLD→current, TRIM→current*trim_to, SELL→0, and
    # the retained core of ADD names→current).  This bounds the manual --cash-pct
    # override, which can otherwise request more budget than free cash exists.
    committed_gross = 0.0
    for row in universe:
        action = row.get("action", "HOLD")
        current = float(weights.get(row["ticker"], 0.0))
        if action == "TRIM":
            committed_gross += current * trim_to
        elif action == "SELL":
            committed_gross += 0.0
        elif action == "BUY":
            committed_gross += 0.0  # fresh addition, counted separately
        elif action == "ADD":
            committed_gross += current  # retained core; top-up is an addition
        else:  # HOLD / unknown → held at current
            committed_gross += current
    gross_headroom = max(0.0, max_gross - committed_gross)
    budget_frac = min(budget_frac, gross_headroom)

    # Step 1 — compute raw ERC additions for deploy set
    deploy_rows = [r for r in universe if r.get("action") in _DEPLOY_ACTIONS]
    raw_add = _erc_additions(deploy_rows, budget_frac)

    # Step 2 & 3 — raw targets + name cap
    targets: dict[str, float] = {}
    for row in universe:
        ticker = row["ticker"]
        action = row.get("action", "HOLD")
        current = float(weights.get(ticker, 0.0))

        if action == "BUY":
            raw = raw_add.get(ticker, 0.0)
            # Name cap: BUY capped at name_cap
            target = min(raw, name_cap)
        elif action == "ADD":
            raw = raw_add.get(ticker, 0.0)
            # ADD: existing + top-up ≤ name_cap → top-up = min(raw, name_cap - current)
            headroom = max(0.0, name_cap - current)
            top_up = min(raw, headroom)
            target = current + top_up
        elif action == "HOLD":
            target = current
        elif action == "TRIM":
            target = current * trim_to
        elif action == "SELL":
            target = 0.0
        else:
            target = current  # unknown action → HOLD semantics

        targets[ticker] = max(0.0, target)

    # Step 4 — sector cap: scale down BUY/ADD additions for over-concentrated sectors
    # Group by sector; compute hold totals per sector; scale BUY/ADD additions.
    sector_hold_total: dict[str, float] = {}  # sum of HOLD/TRIM/SELL targets in sector
    sector_buy_add: dict[str, list[str]] = {}  # tickers with BUY/ADD per sector

    for row in universe:
        ticker = row["ticker"]
        action = row.get("action", "HOLD")
        sector = row.get("sector")
        if not sector:
            continue  # no sector → skip sector cap

        if action in _DEPLOY_ACTIONS:
            sector_buy_add.setdefault(sector, []).append(ticker)
        else:
            sector_hold_total[sector] = sector_hold_total.get(sector, 0.0) + targets[ticker]

    # Build sector→action lookup for the current universe
    action_lookup: dict[str, str] = {r["ticker"]: r.get("action", "HOLD") for r in universe}

    for sector, tickers_in_deploy in sector_buy_add.items():
        hold_total = sector_hold_total.get(sector, 0.0)
        # "additions" contributed by BUY/ADD names
        additions = {
            t: (
                targets[t] if action_lookup[t] == "BUY" else targets[t] - float(weights.get(t, 0.0))
            )
            for t in tickers_in_deploy
        }
        total_additions = sum(additions.values())
        projected_total = hold_total + total_additions

        if projected_total > sector_cap + 1e-12 and total_additions > 0:
            # Scale additions proportionally so hold_total + scaled_additions = sector_cap
            available = max(0.0, sector_cap - hold_total)
            scale = available / total_additions if total_additions > 0 else 0.0
            for t in tickers_in_deploy:
                add = additions[t]
                scaled_add = add * scale
                if action_lookup[t] == "BUY":
                    targets[t] = max(0.0, scaled_add)
                else:
                    # ADD: keep existing core, only scale the top-up
                    core = float(weights.get(t, 0.0))
                    targets[t] = max(core, core + scaled_add)

    # Step 5 — drop BUY names with final target < min_position
    # ADD names are NOT dropped (preserving existing core)
    dropped: set[str] = set()
    for row in universe:
        ticker = row["ticker"]
        if action_lookup[ticker] == "BUY" and targets[ticker] < min_position:
            dropped.add(ticker)

    # Step 6 — build output rows
    output: list[dict] = []
    for row in universe:
        ticker = row["ticker"]
        if ticker in dropped:
            continue
        action = row.get("action", "HOLD")
        current = float(weights.get(ticker, 0.0))
        target = targets[ticker]
        delta = target - current

        out = {
            "ticker": ticker,
            "action": action,
            "current_pct": current,
            "target_pct": target,
            "delta_pct": delta,
            "delta_usd": delta * nav if nav is not None else None,
            "conviction": row.get("conviction"),
        }
        # Propagate optional fields that may be useful to callers
        if "sector" in row:
            out["sector"] = row["sector"]
        output.append(out)

    # Step 7 — sort: BUY/ADD by target_pct desc, then TRIM/SELL, then HOLD
    def sort_key(r: dict) -> tuple:
        action = r["action"]
        group = _SORT_ORDER.get(action, 99)
        # Within BUY/ADD group, sort by target_pct descending
        if action in _DEPLOY_ACTIONS:
            return (group, -r["target_pct"])
        return (group, 0)

    output.sort(key=sort_key)
    return output
