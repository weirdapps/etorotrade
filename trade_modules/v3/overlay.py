"""v3 OVERLAY construction — keep the live book, sell only the genuinely weak.

The from-scratch top-N constructor (:func:`trade_modules.v3.construct.build_portfolio`)
rebuilds the book from zero and therefore recommends selling most of a 44-name
live book purely as a 44->20 structural artifact, not because the tilt says so.
This module instead treats the CURRENT book as the anchor and applies a minimal,
conviction-driven overlay:

  * **SELL** only holdings that are genuinely weak — ineligible / dataless
    (conviction NaN) OR whose conviction sits at/below the ``sell_pctile``
    percentile of the eligible universe. Their weight is freed.
  * **KEEP** every other holding at its current weight (the anchor).
  * **BUY** up to ``max_new`` of the highest-conviction NON-held names whose
    conviction clears the ``buy_pctile`` percentile, funded from the freed
    weight plus cash toward ``gross_target``.
  * then run the combined book (keeps + buys) through the SAME Phase 5A hard risk
    gate (:func:`trade_modules.v3.risk_gate.apply_risk_gate`) that
    :func:`build_portfolio` uses — vol ceiling, concentration caps, redistribution
    to convergence. The returned book is the GATED book.

Turnover (``sum(|Δw|)/2``) is REPORTED, never capped — the owner sees the churn
the overlay implies and decides. Pure orchestration: no network access and no
``yahoofinance.core.config`` import (module-level or otherwise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.construct import portfolio_vol
from trade_modules.riskfirst.covariance import single_factor_cov
from trade_modules.riskfirst.fx import currency_of
from trade_modules.riskfirst.prices import daily_returns, shrunk_cov
from trade_modules.v3.construct import (
    _norm_name,
    _root_of,
    _tiered_name_caps,
    dedup_dual_listings,
    region_of,
)
from trade_modules.v3.risk_gate import apply_risk_gate

# A holding/target weight at or below this is treated as flat (not held).
_EPS = 1e-9
# Minimum overlapping return observations before the empirical cov is trusted
# (mirrors construct._MIN_COV_OBS); below it, fall back to single-factor beta cov.
_MIN_COV_OBS = 60
# Single-factor fallback assumptions (mirror construct._MARKET_VOL / _IDIO_VOL).
_MARKET_VOL = 0.18
_IDIO_VOL = 0.25


def _as_weight_series(weights) -> pd.Series:
    """Coerce a current-weights input (Series / dict) to a clean float Series.

    Non-numeric entries are dropped, the index is stringified, and duplicate
    tickers are summed so downstream membership tests are unambiguous.
    """
    if weights is None:
        return pd.Series(dtype=float)
    s = weights if isinstance(weights, pd.Series) else pd.Series(weights)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s):
        s.index = s.index.map(str)
        if not s.index.is_unique:
            s = s.groupby(level=0).sum()
    return s.astype(float)


def _beta_array(names: list[str], scored: pd.DataFrame, betas) -> np.ndarray:
    """Per-name betas aligned to ``names`` (explicit ``betas`` -> scored -> 1.0)."""
    bser = None
    if betas is not None:
        bser = betas if isinstance(betas, pd.Series) else pd.Series(betas)
        bser = pd.to_numeric(bser, errors="coerce")
        bser.index = bser.index.map(str)
    have_col = scored is not None and "beta" in scored.columns
    out: list[float] = []
    for t in names:
        v = np.nan
        if bser is not None and t in bser.index:
            v = float(bser.loc[t]) if pd.notna(bser.loc[t]) else np.nan
        if not np.isfinite(v) and have_col and t in scored.index:
            v = pd.to_numeric(scored.loc[t, "beta"], errors="coerce")
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = np.nan
        out.append(fv if np.isfinite(fv) else 1.0)
    return np.asarray(out, dtype=float)


def _cov_matrix(prices: pd.DataFrame, names: list[str], betas_arr: np.ndarray) -> np.ndarray:
    """Shrunk empirical cov from ``prices`` when usable, else single-factor beta cov."""
    if not names:
        return np.zeros((0, 0), dtype=float)
    have = prices is not None and not prices.empty and all(t in prices.columns for t in names)
    if have:
        rets = daily_returns(prices[names]).dropna(how="any")
        if len(rets) >= _MIN_COV_OBS:
            return np.asarray(shrunk_cov(rets), dtype=float)
    return np.asarray(single_factor_cov(betas_arr, _MARKET_VOL, _IDIO_VOL), dtype=float)


def _company_key(scored: pd.DataFrame, ticker: str) -> str:
    """Company identity for buy-side dedup: normalized ``name`` else the root.

    The normalized name collapses cross-root dual listings (e.g. ``AMC.AX`` /
    ``AMCR``, the ASX + NYSE listings of Amcor) that :func:`dedup_dual_listings`
    — which groups by the suffix-stripped root — cannot merge on its own. Falls
    back to the suffix-stripped root when no usable name is present.
    """
    if scored is not None and "name" in scored.columns and ticker in scored.index:
        nm = _norm_name(scored.loc[ticker, "name"])
        if nm:
            return nm
    return _root_of(ticker)


def _dedup_buy_candidates(cand_list: list[str], held: list[str], scored: pd.DataFrame) -> list[str]:
    """Filter buy candidates so we never buy a company twice or a held dual line.

    Order-preserving (candidates arrive conviction-descending):

      1. collapse same-root dual listings among the candidates via the shared
         :func:`dedup_dual_listings` util (keeps the mother-market listing);
      2. drop any candidate whose company is ALREADY HELD (a current holding);
      3. collapse any remaining cross-root same-company candidates, keeping the
         first (highest-conviction) listing per company.
    """
    if scored is not None:
        present = [t for t in cand_list if t in scored.index]
        if present:
            survivors = set(dedup_dual_listings(scored.loc[present]).index)
            cand_list = [t for t in cand_list if t not in scored.index or t in survivors]

    held_keys = {_company_key(scored, t) for t in held}
    seen: set[str] = set()
    out: list[str] = []
    for t in cand_list:
        key = _company_key(scored, t)
        if key in held_keys or key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _screen_buy_candidates(
    cand_list: list[str],
    scored: pd.DataFrame,
    *,
    max_short_interest: float | None,
    min_current_ratio: float | None,
    max_de: float | None,
    min_analyst_mom: float | None = None,
    min_analyst_n: int = 4,
    min_earn_trajectory: float | None = None,
) -> tuple[list[str], list[str]]:
    """FIX-NOW 2026-07-18 (D4/D8): drop squeeze-risky / distressed names from the BUY
    pool. Applied to buys ONLY — never force-sells a holding.

    A candidate is screened out when ANY active criterion trips:
      * short_interest (% of float) > ``max_short_interest``  — squeeze/borrow risk;
      * current_ratio < ``min_current_ratio``                 — distress (illiquid);
      * de > ``max_de``                                       — distress (over-levered);
      * analyst_mom <= ``min_analyst_mom`` AND n_analysts >= ``min_analyst_n`` — the
        Street is actively downgrading the name with meaningful coverage (owner
        2026-07-22). Asymmetric: we reject active downgrades but never *require*
        positive coverage (that would anti-select momentum winners);
      * earn_trajectory (= trailing/forward P/E) < ``min_earn_trajectory`` — forward
        P/E materially above trailing, i.e. earnings expected to FALL (value-trap tilt;
        owner 2026-07-22). Use ``1/1.05`` to block a > 1.05x forward/trailing rise.
    A criterion whose param is ``None`` is off; an absent column or NaN value never
    excludes (missing data is not treated as a red flag). Returns (kept, screened_out).
    """
    if scored is None or not cand_list:
        return cand_list, []

    def _val(t: str, col: str) -> float:
        if col not in scored.columns or t not in scored.index:
            return float("nan")
        try:
            return float(scored.at[t, col])
        except (TypeError, ValueError):
            return float("nan")

    kept: list[str] = []
    out: list[str] = []
    for t in cand_list:
        si, cr, de = _val(t, "short_interest"), _val(t, "current_ratio"), _val(t, "de")
        am, na, traj = _val(t, "analyst_mom"), _val(t, "n_analysts"), _val(t, "earn_trajectory")
        squeeze = max_short_interest is not None and pd.notna(si) and si > max_short_interest
        illiquid = min_current_ratio is not None and pd.notna(cr) and cr < min_current_ratio
        levered = max_de is not None and pd.notna(de) and de > max_de
        downgraded = (
            min_analyst_mom is not None
            and pd.notna(am)
            and am <= min_analyst_mom
            and pd.notna(na)
            and na >= min_analyst_n
        )
        value_trap = (
            min_earn_trajectory is not None and pd.notna(traj) and traj < min_earn_trajectory
        )
        excluded = squeeze or illiquid or levered or downgraded or value_trap
        (out if excluded else kept).append(t)
    return kept, out


# Portfolio-aware buy selection (owner 2026-07-22): tilt the SHOWN buys toward what the
# book needs, not just standalone score. Crowding = current-book group weight / group cap;
# the tilt strengths are in conviction units, kept small so conviction stays primary.
_DIV_TILT_SECTOR = 0.5
_DIV_TILT_REGION = 0.3
_SELECT_REGION_CAP = 0.65


def _portfolio_aware_order(
    cand_list: list[str],
    scored: pd.DataFrame,
    current: pd.Series,
    conv: pd.Series,
    *,
    sector_cap: float,
    region_cap: float,
) -> tuple[list[str], list[str]]:
    """Re-order buy candidates by portfolio FIT, not standalone conviction.

    Using the current book's sector + region exposure (as a fraction of the invested
    book): (1) SKIP a candidate whose sector or region is already at its cap — the risk
    gate would zero it at sizing anyway, so surfacing it is a wasted recommendation;
    (2) re-rank the rest by conviction minus a mild crowding penalty
    (``_DIV_TILT_* * group_weight / group_cap``) so buys tilt toward UNDER-weight
    sectors/markets. Conviction stays primary — the penalty only re-orders near-ties,
    never buys a weak name to diversify. Returns (reordered_kept, skipped).
    """
    if scored is None or not cand_list:
        return cand_list, []
    held = [t for t in current.index if float(current.get(t, 0.0)) > _EPS]
    invested = float(sum(float(current[t]) for t in held)) or 1.0

    def _sector_of(t: str) -> str:
        if "sector" in scored.columns and t in scored.index and pd.notna(scored.at[t, "sector"]):
            return str(scored.at[t, "sector"]).upper()
        return "UNKNOWN"

    def _grp_weights(labeler) -> dict:
        w: dict = {}
        for t in held:
            g = labeler(t)
            w[g] = w.get(g, 0.0) + float(current[t]) / invested
        return w

    sec_w = _grp_weights(_sector_of)
    reg_w = _grp_weights(region_of)

    kept: list[str] = []
    skipped: list[str] = []
    adj: dict[str, float] = {}
    for t in cand_list:
        sc = sec_w.get(_sector_of(t), 0.0)
        rc = reg_w.get(region_of(t), 0.0)
        # Hard skip only on SECTOR (the gate-enforced cap; sectors are diverse so this
        # rarely binds). Region is NOT hard-capped at sizing — a US-heavy book is ~100%
        # one region, so a region skip would drop every US buy; region stays a soft tilt.
        if sc >= sector_cap:
            skipped.append(t)
            continue
        penalty = _DIV_TILT_SECTOR * min(1.0, sc / sector_cap) + _DIV_TILT_REGION * min(
            1.0, rc / region_cap
        )
        adj[t] = float(conv.get(t, 0.0)) - penalty
        kept.append(t)
    kept.sort(key=lambda t: -adj[t])
    return kept, skipped


def _sector_labels(names: list[str], scored: pd.DataFrame) -> list[str]:
    """Uppercased sector label per name ("UNKNOWN" when missing)."""
    if scored is None or "sector" not in scored.columns:
        return ["UNKNOWN"] * len(names)
    ser = scored.reindex(names)["sector"].astype("string").str.upper().fillna("UNKNOWN")
    return [s if s else "UNKNOWN" for s in ser.tolist()]


def build_overlay(
    scored: pd.DataFrame,
    current_weights,
    prices: pd.DataFrame,
    *,
    sell_pctile: float = 0.15,
    buy_pctile: float = 0.85,
    max_new: int = 8,
    gross_target: float = 0.95,
    name_cap: float = 0.08,
    sector_cap: float = 0.25,
    usd_bloc_cap: float = 0.60,
    vol_ceiling: float = 0.18,
    betas=None,
    core_list=None,
    cap_mode: str | None = None,
    managed_vol_ceiling: float = 0.18,
    sell_negative_noncore: bool = False,
    noncore_sell_floor: float = 0.0,
    protect_core: bool = False,
    core_floor: pd.Series | None = None,
    max_short_interest: float | None = 20.0,
    min_current_ratio: float | None = 1.0,
    max_de: float | None = None,
    min_analyst_mom: float | None = -3.0,  # owner 2026-07-22: veto buys the Street is downgrading
    min_analyst_n: int = 4,  # ...only when coverage is meaningful (>=4 analysts)
    min_earn_trajectory: float | None = 1.0 / 1.05,  # veto buys w/ fwd P/E > 1.05x trailing
    min_conviction: float | None = None,  # absolute conviction floor for a new buy (owner)
    portfolio_aware: bool = False,  # tilt buy selection toward under-weight sectors/markets
    tier_name_caps: bool = False,
) -> dict:
    """Build a minimal keep/sell/buy overlay on the live book, then risk-gate it.

    Args:
        scored: Output of :func:`trade_modules.v3.combine.compute_scores` — must
            carry ``conviction`` and (ideally) ``eligible`` / ``sector`` / ``beta``
            columns, indexed by ticker. Ineligible names carry a NaN conviction.
        current_weights: Live book weights (Series or dict, fractions of NAV)
            indexed by ticker — the anchor the overlay preserves.
        prices: Daily closes (dates x tickers) for the covariance estimate; pass an
            empty frame to force the single-factor (beta) covariance fallback.
        sell_pctile: A held name is SOLD when its conviction is at or below this
            percentile of the eligible-universe convictions (default 0.15 = the
            weakest ~15%), in addition to always selling ineligible/dataless names.
        buy_pctile: A non-held name is BUY-eligible only when its conviction is at
            or above this percentile (default 0.85 = the strongest ~15%).
        max_new: Cap on the number of new names bought (strongest first).
        gross_target: Fraction of capital the book is funded toward — buys draw
            from the freed weight + cash to bring gross up to this target.
        name_cap / sector_cap / usd_bloc_cap / vol_ceiling: Passed to the Phase 5A
            risk gate applied to the combined keeps+buys book.
        betas: Optional per-name betas (Series/dict keyed by ticker). ``None`` ->
            read ``scored["beta"]`` (missing -> 1.0). Report-only in the gate (net
            beta) and used for the single-factor covariance fallback.
        core_list: Optional list of "core" tickers to report kept/sold status for
            (e.g. the mega-cap anchors), surfaced under ``core_retention``.
        cap_mode: Optional cap-scaling vol mode passed to the risk gate.
            ``None`` / ``'uniform'`` → standard behavior (unchanged). Options:
            ``'cap_budget'`` (RC allowance ∝ log-cap, no hard vol ceiling),
            ``'cap_exempt'`` (sigmoid-exempt split, mega-caps ~fully held),
            ``'cap_ordered'`` (lever-1 trims smallest-cap first). Market caps
            are read from ``scored["cap"]``; names with missing cap default to
            1.0 (treated as the smallest cap in the allowance calculation).
        managed_vol_ceiling: Vol ceiling for the MANAGED sleeve in
            ``cap_exempt`` mode (default 0.18). Ignored for other modes.

    Returns:
        ``{"weights", "diagnostics"}``:

        * ``weights``: the GATED target book as a ``pd.Series`` over keeps+buys
          (sold names are absent, i.e. weight 0).
        * ``diagnostics``: ``n_sell`` / ``n_buy`` / ``n_keep``, ``turnover``
          (= ``sum(|Δw|)/2`` vs the current book, REPORTED not capped),
          ``freed_weight``, ``keep_weight``, ``buy_weight``, ``sell_threshold``,
          ``buy_threshold``, ``gross_target``, the ``sold`` / ``kept`` / ``bought``
          ticker lists, ``core_retention`` (when ``core_list`` is given: per-name
          kept/sold/not_held + ``n_kept`` / ``n_sold``), and ``gate`` (the
          apply_risk_gate diagnostics on the final book).
    """
    cur = _as_weight_series(current_weights)
    held = [t for t in cur.index if cur[t] > _EPS]
    held_set = set(held)

    # Conviction lookup + eligibility (ineligible names carry a NaN conviction).
    if scored is not None and "conviction" in scored.columns:
        conv_all = pd.to_numeric(scored["conviction"], errors="coerce")
    else:
        conv_all = pd.Series(dtype=float)
    if scored is not None and "eligible" in scored.columns:
        elig_mask = scored["eligible"].fillna(False).astype(bool)
    elif scored is not None:
        elig_mask = pd.Series(True, index=scored.index)
    else:
        elig_mask = pd.Series(dtype=bool)

    elig_conv = conv_all[elig_mask].dropna() if len(conv_all) else pd.Series(dtype=float)
    if len(elig_conv):
        sell_threshold = float(elig_conv.quantile(sell_pctile))
        buy_threshold = float(elig_conv.quantile(buy_pctile))
    else:  # no eligible universe -> nothing to keep against / buy
        sell_threshold = float("nan")
        buy_threshold = float("nan")

    core_set = set(core_list or [])

    def _is_weak(t: str) -> bool:
        if protect_core and t in core_set:
            return False  # AI core is the thesis sleeve; never sold when protected
        if t not in conv_all.index:  # dataless (absent from scored)
            return True
        c = conv_all.loc[t]
        if pd.isna(c):  # ineligible names have a NaN conviction
            return True
        if t in elig_mask.index and not bool(elig_mask.loc[t]):
            return True
        if sell_negative_noncore and t not in core_set and float(c) < noncore_sell_floor:
            return True  # owner rule: drop clearly-weak non-core names (deadband)
        if not np.isnan(sell_threshold) and float(c) <= sell_threshold:
            return True  # genuinely weak by percentile
        return False

    sold = [t for t in held if _is_weak(t)]
    sold_set = set(sold)
    kept = [t for t in held if t not in sold_set]
    freed_weight = float(sum(cur[t] for t in sold))
    keep_sum = float(sum(cur[t] for t in kept))

    # BUY: strongest non-held eligible names clearing the buy percentile.
    bought: list[str] = []
    screened_out: list[str] = []
    if not np.isnan(buy_threshold) and max_new > 0:
        cand = elig_conv[[t for t in elig_conv.index if t not in held_set]]
        # Absolute conviction floor (owner 2026-07-22): a buy must clear BOTH the
        # percentile AND the floor, so a weak universe yields fewer buys, not 8 mediocre.
        buy_floor = buy_threshold if min_conviction is None else max(buy_threshold, min_conviction)
        cand = cand[cand >= buy_floor].sort_values(ascending=False)
        # FIX A: never buy the same company twice, nor a dual listing of a name
        # already in the book (dedup the candidate pool + drop held companies).
        cand_list = _dedup_buy_candidates(list(cand.index), held, scored)
        # FIX-NOW (D4/D8): drop squeeze-risky / distressed names from the buy pool.
        cand_list, screened_out = _screen_buy_candidates(
            cand_list,
            scored,
            max_short_interest=max_short_interest,
            min_current_ratio=min_current_ratio,
            max_de=max_de,
            min_analyst_mom=min_analyst_mom,
            min_analyst_n=min_analyst_n,
            min_earn_trajectory=min_earn_trajectory,
        )
        # Owner 2026-07-22: tilt selection toward what the book NEEDS — skip at-cap
        # sectors/markets and re-rank by conviction minus a mild crowding penalty.
        if portfolio_aware:
            cand_list, _div_skipped = _portfolio_aware_order(
                cand_list,
                scored,
                cur,
                conv_all,
                sector_cap=sector_cap,
                region_cap=_SELECT_REGION_CAP,
            )
            screened_out = screened_out + _div_skipped
        bought = cand_list[: int(max_new)]

    # Pre-gate target: keeps anchored at current weight; buys funded from the freed
    # weight + cash toward gross_target (conviction-proportional across buys).
    target: dict[str, float] = {t: float(cur[t]) for t in kept}
    buy_budget = max(0.0, float(gross_target) - keep_sum)
    if bought and buy_budget > 0.0:
        bconv = conv_all.reindex(bought).astype(float).clip(lower=0.0).to_numpy()
        tot = float(bconv.sum())
        frac = bconv / tot if tot > 0 else np.full(len(bought), 1.0 / len(bought))
        for t, f in zip(bought, frac, strict=True):
            target[t] = buy_budget * float(f)

    target_names = list(target.keys())
    core_floor_applied: dict[str, float] = {}

    # Phase 5A hard risk gate on the combined book (keeps + buys).
    if target_names:
        betas_arr = _beta_array(target_names, scored, betas)
        cov = _cov_matrix(prices, target_names, betas_arr)
        tw = pd.Series([target[t] for t in target_names], index=target_names, dtype=float)
        # Extract market caps from scored["cap"] for cap-scaling modes.
        if cap_mode is not None and cap_mode != "uniform" and scored is not None:
            _cap_col = (
                pd.to_numeric(scored["cap"], errors="coerce")
                if "cap" in scored.columns
                else pd.Series(dtype=float)
            )
            caps_ser: pd.Series | None = _cap_col.reindex(target_names)
        else:
            caps_ser = None
        final, gate_diag = apply_risk_gate(
            tw,
            cov,
            sectors=_sector_labels(target_names, scored),
            currencies=[currency_of(t) for t in target_names],
            betas=betas_arr,
            conviction=conv_all.reindex(target_names),
            vol_ceiling=vol_ceiling,
            name_cap=name_cap,
            sector_cap=sector_cap,
            usd_bloc_cap=usd_bloc_cap,
            cap_mode=cap_mode,
            caps=caps_ser,
            managed_vol_ceiling=managed_vol_ceiling,
            # Owner sizing: cap each name at its market-cap tier (mega 10% .. micro 0.25%)
            # so small-caps can't be sized like mega-caps regardless of conviction.
            name_caps=(
                pd.Series(_tiered_name_caps(scored, target_names), index=target_names)
                if tier_name_caps and scored is not None
                else None
            ),
        )

        # Deliberate thesis overlay: FLOOR core names at a chosen minimum (their
        # current weights, pre-capped by name_cap), scaling non-core proportionally
        # so gross holds. The conviction gate trims middling-conviction core; this
        # re-expresses the owner's high-conviction AI view as a VISIBLE post-gate floor.
        if core_floor is not None and len(final):
            floors = pd.to_numeric(core_floor, errors="coerce").dropna()
            floors = floors[floors > 0.0]
            raised = 0.0
            for t in [t for t in final.index if t in floors.index]:
                fv = float(floors[t])
                if float(final[t]) < fv:
                    core_floor_applied[t] = fv - float(final[t])
                    raised += fv - float(final[t])
                    final[t] = fv
            if raised > 1e-9:
                noncore = [t for t in final.index if t not in floors.index]
                nc_sum = float(final.reindex(noncore).sum())
                if nc_sum > raised:  # scale non-core down to fund the floor
                    scale = (nc_sum - raised) / nc_sum
                    for t in noncore:
                        final[t] = float(final[t]) * scale
                else:  # non-core cannot fully fund the floor -> haircut the lifts to
                    # what it frees, then zero it, so gross is PRESERVED (never inflated).
                    haircut = nc_sum / raised  # raised > 1e-9 above, so safe
                    for t in list(core_floor_applied):
                        applied = core_floor_applied[t] * haircut
                        final[t] = float(final[t]) - core_floor_applied[t] + applied
                        core_floor_applied[t] = applied
                    for t in noncore:
                        final[t] = 0.0
                final = final[final > 1e-12]
                pos = {t: i for i, t in enumerate(target_names)}
                names_f = [t for t in final.index if t in pos]
                if names_f:
                    ix = [pos[t] for t in names_f]
                    vol_f = float(
                        portfolio_vol(final.reindex(names_f).to_numpy(), cov[np.ix_(ix, ix)])
                    )
                    gate_diag = {
                        **gate_diag,
                        "vol_after_prefloor": gate_diag.get("vol_after"),
                        "vol_after": vol_f,
                    }
    else:  # everything sold / nothing to hold -> all cash
        final = pd.Series(dtype=float)
        gate_diag = {}

    # Turnover = sum(|Δw|)/2 over the union of tickers (reported, not capped).
    idx = final.index.union(cur.index)
    turnover = 0.5 * float(
        (final.reindex(idx).fillna(0.0) - cur.reindex(idx).fillna(0.0)).abs().sum()
    )

    # Region exposure monitor (informational). The from-scratch constructor
    # (build_portfolio) HARD-caps region; the overlay only REPORTS it, so a region
    # breach never force-sells a held/core name against the owner rules.
    region_exposures: dict[str, float] = {}
    _final_gross = float(final.sum()) if len(final) else 0.0
    for _t, _v in final.items():
        if float(_v) > 1e-12:
            _rg = region_of(_t)
            region_exposures[_rg] = region_exposures.get(_rg, 0.0) + float(_v)
    max_region = (
        max(region_exposures.values(), default=0.0) / _final_gross if _final_gross > 0 else 0.0
    )

    diagnostics: dict = {
        "n_sell": len(sold),
        "n_buy": len(bought),
        "n_keep": len(kept),
        "turnover": turnover,
        "region_exposures": region_exposures,
        "max_region": max_region,
        "freed_weight": freed_weight,
        "keep_weight": keep_sum,
        "buy_weight": float(sum(target[t] for t in bought)) if bought else 0.0,
        "sell_threshold": sell_threshold,
        "buy_threshold": buy_threshold,
        "gross_target": float(gross_target),
        "sold": sold,
        "kept": kept,
        "bought": bought,
        "screened_out": screened_out,
        "gate": gate_diag,
        "core_floor_applied": core_floor_applied,
    }

    if core_list is not None:
        per: dict[str, str] = {}
        for c in core_list:
            cc = str(c)
            per[cc] = "kept" if cc in set(kept) else ("sold" if cc in sold_set else "not_held")
        diagnostics["core_retention"] = {
            "per_name": per,
            "n_kept": sum(1 for v in per.values() if v == "kept"),
            "n_sold": sum(1 for v in per.values() if v == "sold"),
        }

    return {"weights": final, "diagnostics": diagnostics}
