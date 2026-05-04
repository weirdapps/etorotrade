# CIO v36 — Empirical Refoundation

**Date**: 2026-05-04 (Athens 01:01)
**Author**: Senior IO critical review (Ralph loop, opus-1m)
**Status**: Draft → execute on approval

---

## Executive Summary

After 35 review cycles and 14 v17 self-improvement modules, the Investment Committee still has **zero predictive power on the user's actual investment horizon** (30+ days):

- **Spearman ρ(conviction, α30) ≈ −0.002** — explicitly documented in `docs/CIO_V17_OPS.md`
- **Mean alpha = −0.34%** at T+30, **hit rate 44.6%** (n=32,589) per `parameter_study_results.json`
- **Primary BUY criteria are inverted on T+30**: upside ρ=−0.087, exret ρ=−0.079, buy_pct ρ=−0.015 — the system buys analyst-darling stocks that **underperform**
- **The strongest predictive signals are unused**: pct_52w_high ρ=+0.103, short_interest ρ=−0.111
- **News pipeline ships hardcoded fake headlines** to "breaking_news" every committee run (`fetch_news_events.py:142-181`)
- **Macro agent is template copy-paste**: 9 unique reasoning strings × 33 stocks
- **Position sizing applies Kelly to noise**: μ from conviction buckets where ρ(conv, α30)≈0, then concentrates in 4.74%-sized bets
- **EUR-home account holds USD positions with zero FX hedging** — ~560bps unhedged FX risk on the USD sleeve
- **27 of 36 modifiers fire identical Δ**: no rank discrimination at all
- **calibration_report.json shows `modifiers_evaluated: 0`** — the 22-modifier "active" set was literature-validated, not data-validated

The previous Ralph loop concluded "production-ready, no changes required" after smoke tests. This was wrong. Smoke tests verify the system *runs*; backtests verify it *works*. It runs. It does not work for the user's horizon.

---

## Three Deep Errors Behind The Surface Symptoms

### Error 1: The system optimizes for analyst alignment, not realized return

The BUY classifier scores stocks by analyst upside, %buy, exret. These metrics measure **how much the sell-side likes the stock**, not whether the stock outperforms. Empirically (32,589 obs), **highly-loved stocks underperform the market by 95-102 bps over 30 days** while **least-loved bottom-quintile stocks outperform by 193-217 bps**. The classifier is reliably picking the wrong end of a real signal. This is **the value/contrarian effect** that AQR/Cliff Asness wrote 30 years of papers about, applied to analyst sentiment as a proxy for crowdedness.

The fix is not to invert blindly — high-upside-AND-positive-momentum stocks do work (Jegadeesh/Titman 1993). The fix is to **add a momentum gate**: require pct_52w_high to be high (≥75%) AND upside to be moderate-not-extreme. Today's system does the opposite: it uses upside as the engine and ignores momentum.

### Error 2: The conviction waterfall calibrates on zero data

The 22 active modifiers in `committee_synthesis.py:138-161` are commented "Reviewed all 63 modifiers against T+30 forward returns... on 2026-05-02 with portfolio manager." But `calibration_report.json` shows `modifiers_evaluated: 0`. The 2026-05-02 review was **manual** against academic literature, not against this implementation's actual data. Ours has known issues (binned outputs, hardcoded fake news, inverted primary signals) — citing Novy-Marx 2013 doesn't mean *our* revenue_growth modifier behaves like Novy-Marx's.

Worse, the v17 self-improvement modules (`conviction_cells`, `bayesian_conviction`, `debate_scorecard`) all show n=0 in the latest backtest. They were shipped but not connected to data — they wait for evidence that never arrives because the calibration loop wasn't wired to the right fields (`agent_sign_calibrator` looks for `fund_view`/`tech_signal`/`macro_fit` but concordance rows carry different keys).

### Error 3: Sizing concentrates noise into 4.74% bets

`conviction_sizer.py:90-153` maps conviction ≥70 → 4.74% Kelly position. Quarter-Kelly fraction is correct (Carver/Thorp standard). The disease is the input: μ comes from a conviction-bucket lookup, and ρ(conviction, α30)≈0 means **μ is noise**. Kelly-on-noise is `0.25 × noise/σ²`. We are taking concentrated 4.74% positions on stocks where our edge measurement has no out-of-sample validity. Adding FX-blind USD-denominated caps on a EUR book (560bps unhedged) and a static $2,500 base position (which inflates relative risk during drawdowns) compounds the problem.

---

## The Actual Empirical Evidence (T+30, n=32,589)

| Parameter | Spearman ρ | p | Q5−Q1 spread | Current use | Correct use |
|---|---:|---:|---:|---|---|
| short_interest | **−0.111** | <0.001 | −2.09% | sell trigger only | LONG SCREEN: prefer low SI |
| peg | **+0.111** | <0.001 | +2.81% | not used | LONG SCREEN: high PEG = momentum proxy |
| **pct_52w_high** | **+0.103** | <0.001 | +3.02% | sell trigger only | **PRIMARY BUY: momentum factor** |
| upside | −0.087 | <0.001 | **−3.12%** | PRIMARY BUY | DEMOTE: confirmation only |
| exret | −0.079 | <0.001 | −2.95% | PRIMARY BUY | DEMOTE: confirmation only |
| roe | +0.035 | <0.001 | +1.04% | screen | KEEP: quality factor |
| pe_forward | +0.024 | <0.001 | +0.88% | inverted screen | DEMOTE: weak |
| vix_level | −0.020 | <0.001 | −0.31% | regime | KEEP: regime adjuster |
| buy_percentage | −0.015 | 0.008 | −0.87% | PRIMARY BUY | DEMOTE: confirmation only |
| pe_trailing | −0.005 | 0.43 | n/a | screen | DROP: not significant |
| debt_equity | −0.002 | 0.70 | n/a | screen | DROP: not significant |

**Reading**: the system's three highest-weighted BUY criteria (upside, exret, %buy) **all have negative T+30 correlation**. The two strongest positive predictors (pct_52w_high, short_interest) are used only as sell triggers, not as buy criteria. Mean alpha is negative because we are systematically choosing stocks from the **wrong end** of real signals.

---

## Implementation Plan — Ranked by (Impact × Tractability)

### Phase 1 — Stop the bleeding (1 day, must ship first)

**M1. Disable hardcoded fake news** (`fetch_news_events.py:142-181`)

- Today's `news.json` is also a 404 OAuth blob — the fake headlines are silently the only news input. Either wire to news-reader MCP or remove the breaking_news/regulatory_risks/economic_events arrays. Until wired, return `data_status: "INSUFFICIENT_DATA"` and abort the news_catalyst modifiers.
- **Acceptance**: `news_catalyst_pos`, `news_catalyst_neg` modifiers fire `0` when news data is missing instead of using fabricated headlines. Test: assert `breaking_news` is empty when source is `placeholder`.

**M2. Clamp conviction-multiplier to 1.0 in sizing**

- In `conviction_sizer.py:get_conviction_multiplier`, return `1.0` (= equal-weight at tier baseline) until per-cell ρ proves >0.05 out-of-sample.
- This removes the largest avoidable source of risk: concentrated bets sized on noise.
- **Acceptance**: All BUY positions get the tier-baseline size, no conviction lift. `cell_confidence_multiplier` returns 1.0 for n<200 cells.

**M3. Make base position dynamic % of NAV**

- `config.yaml:656` `base_position_size: 2500` → make it `base_position_pct: 0.005` and read current `portfolio_value_eur` at sizing time.
- **Acceptance**: After a 10% drawdown, base position is 0.5% × new NAV (not 0.55% of new NAV).

### Phase 2 — Fix the inputs (2 days)

**M4. EmpiricalFactorScore — new BUY/SELL signal using actual predictive parameters**

- New function `empirical_factor_score(stock)` returns a continuous z-score in [-3, +3].
- Coefficients from rolling 6-month regression on signal_log (out-of-sample): `+1.00 × momentum_z − 0.80 × short_interest_z + 0.30 × roe_z − 0.40 × upside_z`
- Negative coefficient on `upside` reflects empirical inversion. Coefficient signs validated quarterly via parameter study.
- This becomes a **gate**: stocks need EmpiricalFactorScore > 0 to be eligible for BUY, regardless of conviction.
- **Acceptance**: Gate reduces BUY count by ~50%; the remaining BUYs have higher ρ vs T+30 alpha than current set (validated on holdout signal_log).

**M5. Drop or shadow modifiers with NaN/zero discrimination**

- 27 of 36 modifiers fire identical Δ across all stocks → no rank discrimination → pure noise.
- Move `tech_momentum_neg`, `confluence_conflict`, `news_catalyst_*`, `target_consensus`, `iv_low_entry`, `volume_confirm` (all NaN ρ) to shadow (`~`-prefix) until they show ρ ≠ NaN with n ≥ 100.
- **Acceptance**: `len(ACTIVE_MODIFIERS)` drops from 22 → ≤10. Backtest shows no degradation in conviction ρ (since they were noise anyway).

### Phase 3 — Fix the agents (2 days)

**M6. Make agents emit continuous z-scored outputs**

- Fundamental, Technical, Macro agents bin outputs into 3-5 categories → rank info destroyed.
- Change agent prompts to return continuous score [0, 100] *and* provide cross-sectional rank within the run. Synthesis weights by z-score not bin-vote.
- **Acceptance**: Per-agent score has ≥10 unique values across 30+ stocks (vs current 3-5).

**M7. Replace per-stock Macro agent with sector-regime lookup**

- 9 unique macro reasoning strings × 33 stocks = 3.7× duplication. Sector-regime is the only real signal.
- Single Sonnet call returns regime + per-sector score map. Synthesis applies sector→stock lookup. Removes 33 Opus/Sonnet calls per run with zero predictive loss.
- **Acceptance**: Macro section in concordance has 1 reasoning string per sector, not per stock.

### Phase 4 — Risk overlay (2 days)

**M8. FX-aware sizing in EUR**

- All caps and VaR convert to EUR. Add `fx_vol_multiplier(currency)` that down-sizes positions in non-EUR by `1 / (1 + σ_FX/σ_stock)`.
- USD position with σ_stock=20%, σ_EURUSD=8% → 28.6% size reduction. Or add explicit EURUSD/EURHKD hedge sleeves.
- **Acceptance**: Reading portfolio in EUR, max single position ≤ 5% of EUR NAV. New positions in non-EUR currencies are sized down per vol formula.

**M9. Vol-targeted portfolio with 60d realized vol**

- `realized_portfolio_vol_60d > 12% annualized` → multiply all sizing × `12 / realized_vol`.
- Catches slow grinding losses that VaR misses.
- **Acceptance**: When portfolio realized vol exceeds 12%, new BUY sizes proportionally scale down.

**M10. Kill-thesis 4-week cooldown per ticker**

- After any kill-thesis trigger on TKR, set `recent_kill_thesis: True` → conviction-multiplier × 0 for 28 days.
- Stops the "size penalty 0.85→0.75 then re-enter next week" leak.
- **Acceptance**: Ticker that triggered kill thesis on day D returns size 0 for days D+1..D+28.

### Phase 5 — Validation infrastructure (3 days)

**M11. Per-modifier T+30 calibrator (the missing scorecard)**

- New `scripts/calibrate_modifiers_t30.py`. Reads signal_log + concordance history + price_cache. For each modifier in ACTIVE_MODIFIERS, computes Spearman ρ(modifier_value, α30) on out-of-sample window. Auto-shadows any modifier with |ρ| < 0.02 or NaN.
- Replaces the literature-review pruning with data-driven pruning.
- **Acceptance**: Output JSON at `~/.weirdapps-trading/committee/modifier_t30_calibration.json` with per-modifier Spearman ρ, n, p-value. Each modifier has a `verdict: "predictive"|"shadow"|"drop"` field.

**M12. Fix agent_sign_calibrator field-name mismatch**

- Calibrator looks for `fund_view`/`tech_signal`/`macro_fit` but concordance rows have `fundamental_signal`/`technical_signal`/etc. Either fix calibrator OR make concordance write the expected fields.
- **Acceptance**: After 4 weeks of runs, `agent_sign_calibration.json` shows `evidence_total > 0` for all agents.

**M13. Sell trigger logging**

- Currently `parameter_study_results.json` shows ALL sell triggers with `n_triggered=0`. The trigger fields aren't logged in signal_log. Add to per-row write so we can measure which sell triggers actually catch drawdowns.
- **Acceptance**: After 30 days, parameter study shows non-zero n for all 22+ sell triggers.

### Phase 6 — Cleanup (1 day)

**M14. Census z-score thresholds + EMA smoothing**

- Replace absolute pp/pct thresholds with z-score vs ticker's 90d holder volatility. Use 7d EMA over snapshots, not first/last.
- **Acceptance**: census_alignment doesn't fire on 1-day blips; requires |z| > 2 sustained for 3+ snapshots.

**M15. Hard-fail on broken agent outputs**

- If `news.json` or `census.json` is an HTTP error blob (has `error`/`error_description` keys), abort committee run with non-zero exit code.
- Today the system silently feeds empty `breaking_news=[]` and `divergences={}` into synthesis as if all-clear.
- **Acceptance**: Test that committee_synthesis raises `RuntimeError` when an agent output has the OAuth error envelope.

---

## Acceptance for the Whole Refoundation

After all 15 modules ship:

1. **Backtest re-run shows ρ(conv, α30) > 0.05** on the 32k+ obs in signal_log (currently ≈0).
2. **Mean alpha > 0** at T+30 (currently −0.34%).
3. **Hit rate > 50%** at T+30 (currently 44.6%).
4. **No fake/hardcoded data** in production agent outputs (audited in test).
5. **EUR-denominated risk caps**, FX-vol-adjusted sizing.
6. **Modifier calibration evidence-driven**, not literature-driven (json artifact + test).
7. **All 818 existing trade_modules tests still pass**, plus new tests for each module.
8. **Documentation updated** — TECHNICAL.md gets a new "Empirical Validation" section; CIO_V17_OPS.md becomes CIO_V36_OPS.md or is appended.

The bar to mark done: **a 30-day forward backtest on the new signal set must show empirical improvement in alpha, not just better code organization**. If the backtest doesn't move, ship reverts.

---

## Out of Scope (intentional)

- Replacing eToro as broker (sizing math is broker-agnostic).
- Changing the 5-tier market-cap system (orthogonal to predictive validity).
- Adding crypto-specific models (separate workstream).
- Replacing the 7-agent committee architecture (the agents are the cheap part; the integration is broken).
- Real-time execution loop (we're running daily/weekly cadence, not intraday).
- News-reader MCP improvements (we depend on it but don't own it).

---

## Order of Execution (this Ralph loop)

1. **M1** (disable fake news) — 5 min, integrity issue
2. **M11** (modifier calibrator) — needed to validate everything else
3. **M2** (clamp conviction) — sizing fix, immediate risk reduction
4. **M3** (dynamic base) — sizing fix
5. **M4** (EmpiricalFactorScore) — flagship signal change
6. **M5** (drop noise modifiers) — depends on M11
7. **M8** (FX-aware sizing) — risk overlay
8. **M10** (kill-thesis cooldown) — risk overlay
9. **M15** (hard-fail on broken agents) — integrity
10. **M9** (vol targeting) — risk overlay
11. **M14** (census z-score) — quality fix
12. **M12** (agent_sign_calibrator fix) — instrumentation
13. **M13** (sell trigger logging) — instrumentation
14. **M6** (continuous agent outputs) — agent refactor
15. **M7** (single-call macro) — agent refactor

Each module ships with: failing test → implementation → passing test → backtest delta. Loop completion requires the backtest delta on signal_log to be positive at T+30.
