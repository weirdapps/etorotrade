# CIO v36 — Empirical Refoundation (Operational Reference)

**Date shipped**: 2026-05-04
**Predecessor**: v35.0 (literature-validated 22-modifier set)
**Driving evidence**: parameter_study_results.json (n=32,589 obs, T+30) +
modifier_t30_calibration.json (n=5,801 obs)

The v17 review documented Spearman ρ(conviction, α30) ≈ −0.002 — zero
predictive power on the user's 30-day horizon. The v17 ops sprint shipped
modules to address it but they remained instrumented-only (n=0 in
production). The v36 refoundation:

1. Removes 16 modifiers proven to have no T+30 rank-predictive power
2. Adds 2 PREDICTIVE modifiers found by the calibrator
3. Stops conviction-weighting noise into Kelly sizing
4. Stops shipping fabricated/erroneous data through the pipeline
5. Adds FX-aware sizing for the EUR-home account
6. Adds a 4-week kill-thesis cooldown
7. Adds an empirically-grounded BUY gate (M4) — opt-in until validated

## Module map (15 changes)

| ID  | Title | File | Status | Test file |
|-----|-------|------|--------|-----------|
| M1  | Reject placeholder/error news | `committee_synthesis.validate_news_report` | shipped | `test_news_placeholder_rejection.py` |
| M2  | Conviction-multiplier clamp to 1.0 | `conviction_sizer.CONVICTION_CLAMP_TO_UNITY` | shipped (default ON) | `test_conviction_clamp.py` |
| M3  | Dynamic base position % of NAV | `enrich_with_position_sizes(base_position_pct)` + `config.yaml:base_position_pct` | shipped | `test_dynamic_base_position.py` |
| M4  | EmpiricalFactorScore BUY gate | `trade_modules/empirical_factor.py` | shipped (opt-in via `apply_empirical_gate()`) | `test_empirical_factor.py` |
| M5  | Drop NaN-ρ modifiers, add PREDICTIVE ones | `committee_synthesis.ACTIVE_MODIFIERS` (v36 set: 6 modifiers) | shipped | `test_active_modifiers_v36.py` |
| M8  | FX-aware sizing in EUR | `trade_modules/fx_sizing.py` + `enrich_with_position_sizes(fx_aware)` | shipped (opt-in) | `test_fx_aware_sizing.py` |
| M10 | Kill-thesis 4-week cooldown | `trade_modules/kill_thesis_cooldown.py` | shipped | `test_kill_thesis_cooldown.py` |
| M11 | Per-modifier T+30 calibrator | `scripts/calibrate_modifiers_t30.py` | shipped | `test_calibrate_modifiers_t30.py` |
| M15 | Hard-fail on broken agent JSON | `committee_synthesis.{validate_agent_report,load_agent_report,BrokenAgentReportError}` + `committee_html.generate_report_from_files` | shipped | `test_agent_report_validator.py` |

Modules M6/M7/M9/M12/M13/M14 from the design doc are not yet shipped — see "Backlog" below.

## ACTIVE_MODIFIERS — v35 → v36 diff

**v35.0 set (22 modifiers, literature-validated)** → all in
`committee_synthesis.py:138-161` before the change.

**v36.0 set (6 modifiers, empirically-validated)**:

```python
ACTIVE_MODIFIERS = {
    "sector_concentration",   # PREDICTIVE  ρ=−0.32 n=202
    "consensus_crowded",      # PREDICTIVE  ρ=+0.18 n=226 (NEW)
    "tech_disagree",          # PREDICTIVE  ρ=+0.26 n= 66 (NEW)
    "revenue_growth",         # MARGINAL    ρ=+0.13 n=180
    "earnings_surprise",      # PEAD precedent; n<30 — retain
    "signal_velocity",        # SHADOW      n= 33 too small to drop
}
```

**REMOVED (16 modifiers — all NaN/SHADOW per calibrator)**:

- `census_alignment` (n=759, ρ=−0.003)
- `eps_revisions_up`, `eps_revisions_down` (n=184/42)
- `iv_low_entry`, `iv_x_earnings` (n=50/15)
- `news_catalyst_pos`, `news_catalyst_neg` (n=41/87)
- `target_consensus` (n=318)
- `piotroski_quality` (n=136, ρ=−0.23 INVERTED — pending sign-flip review)
- `fcf_quality_strong` (n=52)
- `currency_risk_USD/HKD/JPY/GBP` (constant value per currency)
- `sector_rotation` (n=20)
- `dividend_yield_trap` (n=20)
- `short_interest_weakness` (n=15)
- `volume_confirm` (n=4)

The removed modifiers are tracked in the waterfall as `~name` (shadow
prefix) so historical concordance comparisons remain possible.

## Sizing pipeline changes

```text
Pre-v36                       v36
=======                       ===
conviction × 0.35–1.0         × 1.0 (clamped) — no edge to size on
base = $2500 (static)         base = NAV × 0.5% (dynamic)
no FX adjustment              fx_aware=True → σ_stock/σ_total haircut
no kill-thesis cooldown       28-day cooldown forces size=0 per ticker
```

Operator activation:

```python
enrich_with_position_sizes(
    concordance,
    portfolio_value=current_nav_eur,
    base_position_pct=0.005,
    fx_aware=True,
    ref_currency="EUR",
)
```

The conviction clamp can be reverted to legacy curve only after M11
calibrator shows per-cell ρ > 0.05 in a 12-week sample. Until then,
sizing is conviction-invariant (= equal weight at tier baseline).

## EmpiricalFactorScore (M4) usage

Multi-factor BUY gate using the 4 empirically-validated parameters:

```python
from trade_modules.empirical_factor import apply_empirical_gate

# Right after build_concordance:
demotions = apply_empirical_gate(concordance, threshold=0.0)
# Each row gets: empirical_score, empirical_components, empirical_verdict
# BUY/ADD with score ≤ 0 → action="HOLD", original_action preserved
# returns the count of demotions for audit
```

Validated on today's 12 BUY signals (2026-05-03):

- 6 KEPT (score > 0): BAC, TSM, AMZN, SCHW, NVDA, V — all near 52w highs
  with low SI
- 6 DEMOTED (score ≤ 0): 0700.HK, META, MSFT, 2899.HK, DTE.DE, LLY — high
  upside but farther from 52w highs (analyst-darling-fade profile)

## Quarterly maintenance loop

```bash
# 1. Refresh signal_log forward returns
# (signal_log.jsonl already has 4-month rolling history)

# 2. Re-run parameter study to validate factor signs
python scripts/signal_parameter_study.py
# → output: ~/.weirdapps-trading/parameter_study_results.json

# 3. Re-run modifier calibration
python scripts/calibrate_modifiers_t30.py
# → output: ~/.weirdapps-trading/committee/modifier_t30_calibration.json

# 4. If a factor sign flips or a modifier verdict changes, update:
#    - empirical_factor.FACTOR_WEIGHTS (if sign flip)
#    - committee_synthesis.ACTIVE_MODIFIERS (if verdict shift)
```

## Backlog (not shipped in v36 first wave)

- **M6** (continuous z-scored agent outputs) — agents emit categorical bins
  destroying rank info. Larger refactor across `~/.weirdapps-trading/committee/scripts/`.
- **M7** (single-call macro lookup) — 9 unique reasoning strings × 33 stocks
  = 3.7× duplication. Replaces per-stock LLM call with sector-regime matrix.
- **M9** (vol-targeting at portfolio level) — 60d realized vol > 12% → scale
  all sizing × 12/realized_vol. Catches slow grinding losses VaR misses.
- **M12** (agent_sign_calibrator field-name fix) — calibrator looks for
  `fund_view`/`tech_signal`/`macro_fit`, concordance writes different keys
  → 0 evidence collected.
- **M13** (sell-trigger logging) — sell triggers not in signal_log per row
  → parameter study can't measure trigger predictive power.
- **M14** (census z-score thresholds + EMA) — current absolute pp/pct
  thresholds fire on 1-investor changes (noise).

## Acceptance criteria for promoting to "validated" status

After 8-12 weeks of v36 in production:

1. Backtest re-run shows ρ(conviction, α30) > 0.05 on signal_log (was ≈0)
2. Hit rate at T+30 on BUY signals > 50% (was 44.6%)
3. Mean alpha > 0 (was −0.34%)
4. EmpiricalFactorScore demotions show ≥0.5pp T+30 alpha advantage vs
   non-demoted BUYs

Until those are met, sizing stays clamped to baseline and the gate stays
opt-in.

## Cross-references

- Design doc: `docs/superpowers/specs/2026-05-04-cio-empirical-refoundation-design.md`
- Predecessor: `docs/CIO_V17_OPS.md`
- Empirical evidence: `~/.weirdapps-trading/parameter_study_results.json`,
  `~/.weirdapps-trading/committee/modifier_t30_calibration.json`
