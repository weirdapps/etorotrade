"""
Waterfall Modifier Categorisation (CIO v17 L1)

The conviction waterfall has ~53 distinct modifier keys, which is too
many to scan in an HTML report. This module groups them into 6
human-readable categories (Quality, Momentum, Consensus, Macro, Risk,
Regime) so the report can show aggregated category contributions with
the per-key detail collapsed.

The categories mirror the implicit taxonomy in factor_attribution.py
(category="analyst|valuation|risk|fundamental|technical|modifier") but
collapsed for display. A modifier may belong to exactly one category.
Unrecognised keys fall through to the "Other" bucket so nothing is lost.
"""

from typing import Any

# Modifier key → human category. Keep this list in sync with the keys
# emitted by committee_synthesis.compute_adjustments + synthesize_stock.
MODIFIER_CATEGORY: dict[str, str] = {
    # ── Quality (fundamentals, balance sheet, earnings quality) ──────
    "piotroski_quality": "Quality",
    "piotroski_weak": "Quality",
    "revenue_growth": "Quality",
    "eps_revisions_up": "Quality",
    "eps_revisions_down": "Quality",
    "fcf_quality_strong": "Quality",
    "fcf_quality_concern": "Quality",
    "fund_composite_cap": "Quality",
    "debt_high_risk": "Quality",
    "dividend_yield_trap": "Quality",
    "quality_trap": "Quality",
    # ── Momentum (technical indicators, price action, trend) ────────
    "rsi_overbought": "Momentum",
    "rsi_oversold": "Momentum",
    "rsi_divergence_bullish": "Momentum",
    "rsi_divergence_bearish": "Momentum",
    "tech_momentum_neg": "Momentum",
    "tech_disagree": "Momentum",
    "rel_strength_spy": "Momentum",
    "adx_trend_confirm": "Momentum",
    "adx_downtrend": "Momentum",
    "adx_ranging": "Momentum",
    "confluence": "Momentum",
    "confluence_conflict": "Momentum",
    "volume_confirm": "Momentum",
    "volume_confirm_sell": "Momentum",
    "iv_low_entry": "Momentum",
    "iv_x_earnings": "Momentum",
    "signal_velocity": "Momentum",
    # ── Consensus (analyst, target, census) ──────────────────────────
    "consensus_crowded": "Consensus",
    "consensus_crowded_tier_waived": "Consensus",
    "stale_targets": "Consensus",
    "target_consensus": "Consensus",
    "target_dispersion_wide": "Consensus",
    "low_dir_confidence": "Consensus",
    "agent_consensus": "Consensus",
    "census_alignment": "Consensus",
    "census_distribution": "Consensus",
    "census_accumulation": "Consensus",
    # ── Macro (regime, sector, FX, currency) ─────────────────────────
    "macro_sector": "Macro",
    "macro_regime_risk_off": "Macro",
    "sector_rotation": "Macro",
    "sector_concentration": "Macro",
    "currency_risk_USD": "Macro",
    "currency_risk_GBP": "Macro",
    "currency_risk_HKD": "Macro",
    "currency_risk_JPY": "Macro",
    "currency_risk_CHF": "Macro",
    "regional_EU": "Macro",
    "regional_HK": "Macro",
    "earnings_proximity": "Macro",
    "news_catalyst_pos": "Macro",
    "news_catalyst_neg": "Macro",
    "earnings_surprise": "Macro",
    # ── Risk (kill thesis, risk warnings, sizing/floor caps) ─────────
    "kill_thesis": "Risk",
    "high_beta": "Risk",
    "short_squeeze": "Risk",
    "short_interest_weakness": "Risk",
    "contradiction": "Risk",
    "sell_tech_disagree": "Risk",
    "sell_macro_disagree": "Risk",
    "sell_census_disagree": "Risk",
    "risk_off_cap": "Risk",
    # ── Regime / Cap-management (debate, structural overrides) ──────
    "debate_strengthen_bull": "Regime",
    "debate_weaken_bull": "Regime",
    "debate_strengthen_bear": "Regime",
    "debate_weaken_bear": "Regime",
    "trim_regime_gate": "Regime",
    "proportionality_cap": "Regime",
    "floor_applied": "Regime",
    "base_bonus_cap": "Regime",
    "base_penalty_cap": "Regime",
    "signal_quality_cap": "Regime",
}

# Category display order for the HTML report.
CATEGORY_ORDER: tuple[str, ...] = (
    "Quality",
    "Momentum",
    "Consensus",
    "Macro",
    "Risk",
    "Regime",
    "Other",
)


def categorize_waterfall(
    waterfall: dict[str, int],
) -> dict[str, dict[str, int]]:
    """
    Group a waterfall dict {modifier: int} into category buckets.

    Returns {category: {"total": int, "modifiers": {key: int, ...}}}
    in CATEGORY_ORDER. Empty categories are omitted.
    """
    out: dict[str, dict[str, Any]] = {}
    for key, val in (waterfall or {}).items():
        if key.startswith("_") or key.startswith("~"):
            continue
        cat = MODIFIER_CATEGORY.get(key, "Other")
        bucket = out.setdefault(cat, {"total": 0, "modifiers": {}})
        try:
            v_int = int(val)
        except (TypeError, ValueError):
            continue
        bucket["total"] += v_int
        bucket["modifiers"][key] = v_int
    # Re-emit in the canonical order.
    ordered: dict[str, dict[str, int]] = {}
    for cat in CATEGORY_ORDER:
        if cat in out and out[cat]["modifiers"]:
            ordered[cat] = out[cat]
    return ordered


def render_category_summary(
    waterfall: dict[str, int],
    *,
    text_only: bool = False,
) -> str:
    """
    Render the categorised waterfall as either a plain-text or HTML summary.

    text_only=True returns "Quality +5 · Momentum -8 · Consensus +5"
    suitable for log lines or unit tests.
    Otherwise returns inline-styled HTML for the action-card report.
    """
    cats = categorize_waterfall(waterfall)
    if not cats:
        return ""

    if text_only:
        return " · ".join(
            f"{cat} {('+' if v['total'] > 0 else '')}{v['total']}" for cat, v in cats.items()
        )

    parts: list[str] = []
    for cat, v in cats.items():
        total = v["total"]
        color = "#2e7d32" if total > 0 else "#c62828" if total < 0 else "#666"
        sign = "+" if total > 0 else ""
        parts.append(f'<span style="color:{color};font-weight:600;">' f"{cat} {sign}{total}</span>")
    return (
        '<div style="font-size:10px;color:#666;margin-top:4px;'
        'font-family:Menlo,Consolas,monospace;">' + " · ".join(parts) + "</div>"
    )
