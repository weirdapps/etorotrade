"""S5 Pipeline Orchestrator — PURE given loaded inputs.

Wires S1→S2→S3→S4 into a single deterministic run.  No I/O, no clock,
no mutation.  All impure work (loading files, calling clock, writing output)
lives in scripts/trading_pipeline_v2.py.

SHADOW ONLY — this pipeline NEVER places orders.  It emits a decision-support
action plan for manual review and execution by the user.
"""

from __future__ import annotations

import pandas as pd

from trade_modules.committee_v2.cio import synthesize
from trade_modules.portfolio_manager.budget import deployable_budget
from trade_modules.portfolio_manager.sizer import size_book
from trade_modules.signals_v2.composite import factor_composite, long_only_signal
from trade_modules.universe.filter import filter_universe

from .adapters import etoro_row_to_candidate, portfolio_to_weights, universe_df_for_filter

# ---------------------------------------------------------------------------
# Caveats emitted in every action plan
# ---------------------------------------------------------------------------

_FIXED_CAVEATS = [
    "SHADOW / decision-support only — this pipeline NEVER places orders.",
    "Fundamentals are forward-gated (not point-in-time); use for direction only.",
    "S2 runs on a single regime snapshot (no DSR PASS walk-forward yet).",
    "Price sleeve skipped — no OHLCV price-history matrix available in etoro.csv.",
    "User executes manually; go-live is user-triggered.",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile_rank(series: pd.Series, value: float | None) -> float | None:
    """Return the percentile rank of ``value`` within ``series`` [0, 1].

    None if value is None or series is empty.
    """
    if value is None:
        return None
    valid = series.dropna()
    if len(valid) == 0:
        return None
    return float((valid <= value).mean())


def _to_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    universe_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    regime_mult: float,
    cash_pct: float,
    generated_at: str,
    config: dict | None = None,
) -> dict:
    """Run the full S1→S2→S3→S4 pipeline.

    PURE given loaded inputs.  Deterministic (no clock — generated_at passed in).

    Args:
        universe_df:   Raw etoro.csv DataFrame (one row per instrument).
        portfolio_df:  Portfolio positions DataFrame (columns: symbol,
                       totalInvestmentPct, isBuy, ...).
        regime_mult:   Float [0, 1] from resolve_regime_multiplier.
        cash_pct:      Current cash as a fraction of NAV (e.g. 0.20 = 20%).
        generated_at:  ISO-8601 timestamp string (Athens tz); passed in, not computed.
        config:        Optional overrides for S3 (cio.synthesize) and S4
                       (size_book) thresholds.

    Returns:
        action_plan dict:
            generated_at, regime_mult, budget_frac,
            actions: [sized rows with action/conviction/persona_consensus/
                      persona_dissent/delta_pct/target_pct/current_pct/rationale],
            holds:   [held rows with HOLD action + sizing],
            resulting_gross, resulting_cash,
            caveats: list of caveat strings.
    """
    cfg = config or {}
    caveats: list[str] = list(_FIXED_CAVEATS)

    # ------------------------------------------------------------------
    # S1: Universe routing + quality filter
    # ------------------------------------------------------------------
    adapted_df = universe_df_for_filter(universe_df)

    if adapted_df.empty:
        return {
            "generated_at": generated_at,
            "regime_mult": regime_mult,
            "budget_frac": 0.0,
            "actions": [],
            "holds": [],
            "resulting_gross": 0.0,
            "resulting_cash": cash_pct,
            "caveats": caveats + ["Empty universe — no instruments to process."],
        }

    s1_result = filter_universe(adapted_df, config=cfg.get("filter"))
    eligible_df: pd.DataFrame = s1_result["eligible"]
    excluded: dict = s1_result.get("excluded", {})
    price_only_tickers: set[str] = {str(t).strip() for t in s1_result.get("price_only", [])}

    # ------------------------------------------------------------------
    # S2: Factor composite + long_only_signal on fundamental sleeve
    #
    # factor_composite expects df indexed by ticker.
    # etoro.csv uses integer index; re-index by TKR.
    # ------------------------------------------------------------------
    if not eligible_df.empty:
        # Set TKR as index for factor functions (they expect ticker index)
        if "TKR" in eligible_df.columns:
            factor_df = eligible_df.set_index("TKR")
        else:
            factor_df = eligible_df.copy()

        # Coerce numeric columns that factors rely on
        for col in ("PET", "PEF", "FCF", "ROE", "DE", "EG", "%B", "B", "52W", "AM", "P/S"):
            if col in factor_df.columns:
                factor_df[col] = pd.to_numeric(factor_df[col], errors="coerce")

        composite_scores: pd.Series = factor_composite(factor_df)
        signals: pd.Series = long_only_signal(composite_scores)

        # Build composite percentile ranks (used for conviction scoring)
        composite_pct_series: pd.Series = composite_scores.rank(pct=True)

    else:
        composite_scores = pd.Series(dtype=float)
        signals = pd.Series(dtype=str)
        composite_pct_series = pd.Series(dtype=float)
        caveats.append("No eligible fundamentals passed S1 quality gates.")

    # Price sleeve: skipped — no OHLCV history available in etoro.csv.
    # (already noted in _FIXED_CAVEATS)

    # ------------------------------------------------------------------
    # Portfolio: current weights + held tickers
    # ------------------------------------------------------------------
    current_weights, held_tickers = portfolio_to_weights(portfolio_df)

    # ------------------------------------------------------------------
    # S3 Candidate assembly  (asymmetric held vs non-held — see F1/F2)
    #
    # ALL S1-eligible fundamental names become candidates (including EXIT-signal
    # names), each carrying composite_pct for conviction scoring and in_universe
    # implicitly True.  cio.synthesize then GRADES them by conviction:
    #   - non-held → BUY only if conviction clears the BUY threshold, else NONE
    #   - held     → ADD / HOLD / TRIM / SELL-at-floor  (NEVER force-SELL'd for
    #                merely landing in the bottom relative-signal quantile)
    #
    # A held name is force-SELL'd ONLY if it truly dropped from the investable
    # universe (failed S1 or excluded) — passed to synthesize as
    # held_failed_eligibility.
    #
    # Held PRICE-ONLY names (ETFs / crypto) are unscoreable because the price
    # sleeve is skipped; they are emitted as HOLD directly (untouched), never a
    # fabricated SELL.
    # ------------------------------------------------------------------
    candidates: list[dict] = []

    eligible_tickers: set[str] = set()
    for ticker in signals.index:
        # Build candidate dict from the eligible row
        # Recover the original row from eligible_df (pre-index-set)
        if "TKR" in eligible_df.columns:
            row_matches = eligible_df[eligible_df["TKR"] == ticker]
        else:
            row_matches = pd.DataFrame()

        if row_matches.empty:
            # Fallback: build a minimal row
            raw_row = {"TKR": ticker}
        else:
            raw_row = row_matches.iloc[0].to_dict()

        candidate = etoro_row_to_candidate(raw_row)

        # Attach composite_pct [0, 1] for conviction scoring
        pct_rank = composite_pct_series.get(ticker) if not composite_pct_series.empty else None
        candidate["composite_pct"] = _to_float(pct_rank)

        candidates.append(candidate)
        eligible_tickers.add(str(candidate.get("ticker")).strip())

    # Held names that TRULY dropped from the investable universe: held, and
    # neither S1-eligible nor price-only (i.e. failed a quality gate or were
    # excluded for no usable price).  These — and only these — become SELL.
    held_failed_eligibility: set[str] = {
        t for t in held_tickers if t not in eligible_tickers and t not in price_only_tickers
    }

    # ------------------------------------------------------------------
    # S3: CIO synthesize (grades candidates; SELLs only failed-eligibility held)
    # ------------------------------------------------------------------
    s3_result: list[dict] = synthesize(
        candidates=candidates,
        held_tickers=held_tickers,
        held_failed_eligibility=held_failed_eligibility,
        cfg=cfg.get("cio"),
    )

    # ------------------------------------------------------------------
    # Held PRICE-ONLY names → HOLD (untouched; price sleeve unavailable)
    # ------------------------------------------------------------------
    held_price_only: set[str] = {t for t in held_tickers if t in price_only_tickers}
    s3_tickers = {r.get("ticker") for r in s3_result}
    for ticker in sorted(held_price_only):
        if ticker in s3_tickers:
            continue  # already represented (shouldn't happen — price-only ≠ candidate)
        s3_result.append(
            {
                "ticker": ticker,
                "action": "HOLD",
                "conviction": None,
                "persona_consensus": "neutral",
                "persona_dissent": [],
                "rationale": (
                    f"{ticker}: HOLD | price-only (ETF/crypto) held position — "
                    f"unscored (price sleeve skipped); left untouched"
                ),
            }
        )

    # ------------------------------------------------------------------
    # S4: Budget + size_book
    # ------------------------------------------------------------------
    budget_frac = deployable_budget(
        cash_pct=cash_pct,
        target_cash_pct=cfg.get("target_cash_pct", 0.07),
        regime_mult=regime_mult,
    )

    # Propagate sector to size_book rows (None is fine — sizer handles it)
    for row in s3_result:
        if "sector" not in row:
            row["sector"] = None

    sized: list[dict] = size_book(
        final_universe=s3_result,
        current_weights=current_weights,
        budget_frac=budget_frac,
        cfg=cfg.get("sizer"),
    )

    # ------------------------------------------------------------------
    # Assemble action plan
    # ------------------------------------------------------------------
    actions: list[dict] = []
    holds: list[dict] = []

    for row in sized:
        action = row.get("action", "HOLD")
        # Enrich with S3 fields (rationale, persona_consensus, persona_dissent)
        s3_meta = next((r for r in s3_result if r.get("ticker") == row.get("ticker")), {})
        enriched = {
            "ticker": row.get("ticker"),
            "action": action,
            "conviction": row.get("conviction"),
            "current_pct": row.get("current_pct"),
            "target_pct": row.get("target_pct"),
            "delta_pct": row.get("delta_pct"),
            "persona_consensus": s3_meta.get("persona_consensus"),
            "persona_dissent": s3_meta.get("persona_dissent"),
            "rationale": s3_meta.get("rationale"),
        }
        if action == "HOLD":
            holds.append(enriched)
        else:
            actions.append(enriched)

    # Compute resulting gross exposure and cash
    resulting_gross = sum(r.get("target_pct") or 0.0 for r in sized)
    resulting_cash = max(0.0, 1.0 - resulting_gross)

    return {
        "generated_at": generated_at,
        "regime_mult": regime_mult,
        "budget_frac": budget_frac,
        "actions": actions,
        "holds": holds,
        "resulting_gross": resulting_gross,
        "resulting_cash": resulting_cash,
        "caveats": caveats,
    }
