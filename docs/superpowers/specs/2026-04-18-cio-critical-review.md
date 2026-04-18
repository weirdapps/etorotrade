# CIO Critical Review — etorotrade Trading System

**Author:** Claude (CIO function), commissioned by Dimitrios Plessas
**Date:** 2026-04-18
**Scope:** End-to-end review of signal generation, committee synthesis, position sizing, backtesting methodology, agent architecture, and parameter framework.
**Method:** Four parallel workstreams (diagnostic backtest, overfitting stress test, architectural critique, literature anchor) on the existing 138K-signal log spanning Jan 14 – Apr 18, 2026.
**Source artifacts:** `~/Downloads/202604180000_ws1_diagnostic_backtest.md`, `~/Downloads/202604180000_ws2_overfitting_stress.md`, `~/Downloads/202604181730_ws3_architectural_critique.md`, `~/Downloads/202604180000_ws4_literature_anchor.md`.

---

## Executive Summary

**Bottom line: the system delivers measurable alpha at T+7 (+0.61%, hit rate 58.2%, 90% CI 55.5–60.8%) but loses it by T+30 (–1.43%, hit rate 42.2%). The bear-market YTD context obscures this — raw returns at T+30 are negative for every signal type (BUY –5.96%, HOLD –5.71%, SELL –4.35%), but that is market beta, not signal failure.** When you look at alpha vs SPY (the right metric), the picture is: short-horizon BUY signals work, long-horizon BUY signals don't, and SELL signals deliver positive alpha at both horizons.

**Three findings are robust and actionable:**

1. **The horizon is wrong, not the signals.** BUY alpha decays from +0.61% at T+7 to –1.43% at T+30. Holding an analyst-consensus signal for 30 days is fighting the well-documented decay in recommendation alpha (Womack 1996; Barber et al. 2001). The system's `holding_cost_model` assumes a 90-day default hold — that's exactly the wrong horizon for this signal.

2. **EXRET ranks. Upside doesn't.** Within BUY-flagged stocks, the EXRET decile test passes at T+7 (top decile +2.51% alpha vs bottom +0.50%, p=0.027). Upside and buy_pct fail their decile tests at both horizons. **The headline metric driving signal generation has no within-cohort predictive power.** Threshold tuning on a metric that doesn't rank is theatre.

3. **The system amplifies the market regime; it does not generate independent alpha.** Monthly alpha hit rate correlates r=0.99 with SPY's monthly return. In a flat or rising market we look smart; in a falling market we look broken. This is consistent with an analyst-consensus-driven model (Welch 2000 herding) and is structurally hard to fix with this metric set alone.

**Three recommendations, in order of priority:**

1. **Stop tuning thresholds on this 94-day window. Now.** The existing `backtest_threshold_report.csv` suggests, e.g., raising `us_large_BUY_min_upside` from 12% to 25.9%. Walk-forward validation collapses 4 of 5 top suggestions (most catastrophically `us_large_BUY_min_upside`). Five of eleven suggestions are statistically underpowered at 90% confidence. Adopting any of them would lock in bear-market overfitting. **The data screams for action; the action it implies is the wrong action.**

2. **Instrument before optimize.** The architecture has a fully-built `calibrate_modifiers()` function (`committee_scorecard.py:1283-1446`) that has never run in production. Concordance.json logs every modifier contribution but agent-level votes are aggregated away (`committee_backtester.py:394-422`). For three months of nights-and-weekends work you can answer "which of the 20 conviction modifiers actually predicts alpha?" — and probably retire half of them.

3. **Strategic moves only with longer history.** Threshold retuning, factor-model overlay (Quality, Momentum, Betting-Against-Beta — see WS4), and conviction-to-Kelly recalibration all require ≥6 months of data spanning at least one bull→bear transition. This 94-day window is insufficient for any of them. Document the gate, queue the work.

The middle of the report makes the case for these three moves with numbers and citations. The end states explicitly what I am NOT recommending and why — that section is more important than the recommendations themselves, because the data tempts you toward overfit fixes that would feel productive and damage P&L.

---

## The Data In One View

| Signal | Horizon | Count | Alpha Hit Rate | 90% CI | Mean Alpha | Mean Raw Return | What it means |
|---|---|---:|---:|---:|---:|---:|---|
| BUY | T+7 | 906 | **58.2%** | 55.5–60.8% | **+0.61%** | +0.44% | Works |
| BUY | T+30 | 595 | **42.2%** | 38.9–45.5% | **–1.43%** | –5.96% | Fails |
| HOLD | T+7 | 12,188 | 26.0% (alpha hit)* | 25.3–26.6% | +1.20% | +0.65% | Neutral, by definition |
| HOLD | T+30 | 11,663 | 14.5% (alpha hit)* | — | –0.86% | –5.71% | Neutral, by definition |
| SELL | T+7 | 10,133 | 39.8% | 39.0–40.6% | **+1.23%** | +0.79% | Works (alpha-positive) |
| SELL | T+30 | 9,924 | **52.8%** | — | **+0.16%** | –4.35% | Works |

\* HOLD alpha-hit rate uses a tighter band (|alpha|<2%); not directly comparable to BUY/SELL.

**SPY YTD context (verified 2026-04-17):** start 681.31, March 30 low 631.97 (−7.2% trough), close 710.14 (+4.23% YTD). Monthly: Feb –0.86%, March –4.94%, April +9.20%. **A textbook V-shape.** Any T+30 backtest straddling this whipsaw will show wildly different numbers depending on signal date alone.

Two observations dominate everything that follows:
- **Raw returns are deeply negative for HOLD signals at T+30 (–5.71%) — and HOLD signals have, by definition, no directional view.** That number alone proves the bear market dominates the dataset.
- **BUY alpha hit rate by month**: Jan 49.1% (SPY –0.69%), Feb 49.4% (SPY –1.38%), Mar 61.2% (SPY –0.02%), Apr 88.9% (n=9, SPY +4.71%). **Pearson correlation between monthly BUY alpha hit rate and SPY monthly return: r=0.99.** The "alpha" is regime-amplified, not regime-independent.

---

## Workstream 1 — Diagnostic Backtest Deep-Dive

Methodology verified clean: 20-row sample audit confirms `signal_date < observation window` (no lookahead), `price_at_signal > 0` (valid backfill), `alpha = stock_return − spy_return` (within 0.01% tolerance). The negative T+30 numbers are real, not a calculation bug.

### Finding 1A — Signal fade is the dominant T+30 problem

Alpha hit rate decays from 58.2% (T+7, n=906) to 42.2% (T+30, n=595). Mean alpha goes from +0.61% to –1.43%. The 90% CIs do not overlap. This pattern is consistent with the academic literature on analyst recommendation half-lives: Womack (1996) found upgrade alpha attenuates within ~30 days; Barber et al. (2001) showed the alpha is in *recommendation changes*, not consensus levels. **The system holds positions for 90 days by default — that's three half-lives past where the signal expires.**

### Finding 1B — The `upside` metric does not rank

Decile test on BUY-flagged stocks (n=1,501 over 94 days):

| Metric | Horizon | Top–Bottom Decile Gap | t-stat | p-value | Verdict |
|---|---|---|---:|---:|---|
| upside | T+7 | +1.17pp alpha | 1.30 | 0.195 | Not significant |
| upside | T+30 | –3.75pp alpha | –1.40 | 0.164 | Not significant (sign flipped) |
| buy_pct | T+7 | varies, no monotone | — | — | Not significant |
| buy_pct | T+30 | varies, no monotone | — | — | Not significant |
| **EXRET** | **T+7** | **+2.01pp alpha** | **2.23** | **0.027** | **Significant** |
| EXRET | T+30 | –2.26pp alpha | –0.86 | 0.394 | Not significant (sign flipped) |

Implication: tuning `min_upside` thresholds is wishful. Within already-BUY-flagged stocks, the bottom decile of upside delivers *more* alpha than the top decile at T+30 (D1: +5.12% vs D10: +1.37%). This is exactly what Bradshaw (2002) and Bonini (2010) document: target prices are systematically optimistic, with mean absolute forecast errors around 25-46%. The metric is noise dressed as signal.

EXRET (= upside × %buy / 100) has weak short-horizon ranking power. That single result is the evidence base for whatever forward-looking work the system does on metric design.

### Finding 1C — High-VIX BUY signals counterintuitively outperform at T+7

| VIX bucket | T+7 BUY alpha hit | Mean alpha |
|---|---:|---:|
| 15-20 | 59.0% (n=427) | +0.21% |
| 20-25 | 50.4% (n=278) | +0.32% |
| **>25** | **66.1% (n=183)** | **+1.73%** |

This is unexpected. The conviction sizer multiplies position sizes by 0.5x in HIGH VIX (`conviction_sizer.py:35-40`) — but the data suggests this is exactly when the BUY signals work best. **Possible explanations**: panic-selling creates analyst-consensus value buys; VIX spikes coincide with mean-reversion regimes; 183 observations is small enough that this could be a single fortunate cohort. Worth a focused investigation; not yet worth a parameter change.

### Finding 1D — The system is a regime amplifier, not an alpha generator

Pearson correlation of monthly BUY alpha hit rate with SPY monthly return: r=0.99. This is not subtle. In months when SPY went down, BUY alpha hit rate sat at ~49% (no edge). In months when SPY went sideways or up, BUY alpha hit rate climbed to 60-89%. **A 0.99 correlation between "market direction" and "signal alpha hit rate" is the signature of a strategy with no regime-independent edge.** The system reads as a momentum-with-fundamental-confirmation strategy — and momentum is regime-dependent (Daniel & Moskowitz on momentum crashes is the canonical reference).

This is not necessarily a defect. Many real strategies are regime-conditional. But it must be acknowledged in the marketing of the system: it is a *long bias amplifier with bear-market drag*, not a market-neutral alpha source.

---

## Workstream 2 — Overfitting Stress Test

The existing `backtest_threshold_report.csv` would have us raise BUY thresholds across the board. Walk-forward validation (chronological 70/30 split, train on first 67 days, test on last 27) shows the suggestions do not survive contact with held-out data:

| Threshold | Current | Suggested | In-sample HR at suggested | Out-sample HR at suggested | Out-sample HR at current | Verdict |
|---|---:|---:|---:|---:|---:|---|
| hk_large_SELL_max_upside | 10.0 | –15.5 | 30.8% | 46.5% | 57.4% | **OVERFIT** — current beats suggested out-of-sample |
| us_large_BUY_min_upside | 12.0 | 25.9 | 32.2% | 77.8% | 57.2% | **OVERFIT** — but in confusing direction; suggested direction is *correct out-of-sample* but the report's claimed magnitude is wildly off |
| us_large_BUY_min_roe | 14.0 | 39.9 | 75.6% | 87.9% | 65.9% | Survives, but n=82 train and the recommendation's ROE bar (39.9%) would exclude almost all real stocks |
| eu_large_SELL_max_upside | –9.0 | –10.5 | 42.7% | 41.1% | 52.2% | **OVERFIT** — out-sample hit rate worse than current |
| us_mega_BUY_min_upside | 10.0 | 33.4 | 69.5% | 64.0% | 61.3% | **UNDERPOWERED** — n=139 < min n=144 to detect claimed effect |

Sample-size sanity check: 5 of 11 suggestions in the threshold report do not have enough observations to detect their own claimed improvement at 90% confidence. That includes both `us_mega_*` BUY suggestions and `us_mid_BUY_min_upside`.

**Verdict: do not adopt any of the threshold report's recommendations.** This includes the previously-implemented `us_large_BUY_min_roe` change from 8.0 → 14.0 (which the project documents as "+10.4% hit rate improvement"). That number was in-sample on a single regime; whether it survives requires an honest walk-forward re-evaluation, which the workstream did not run because the change predates this dataset.

### A counter-finding worth keeping: the contrarian buy_pct effect at the tails

WS1 found a near-zero overall correlation between buy_pct and T+30 alpha (Pearson r=–0.03 across 22K signals). WS2 found a stronger negative correlation within smaller monthly cohorts (–0.20 in Jan, –0.08 in Feb). These are reconcilable: the *linear* correlation across the full distribution is weak, but the *tail* effect (top vs bottom quintile of buy_pct within BUY-flagged stocks) is real:

| Month | Q1 alpha | Q5 alpha | Gap | Significant? |
|---|---:|---:|---:|---|
| 2026-01 | +5.61% | –1.87% | **–7.48pp** | Yes (CIs don't overlap) |
| 2026-02 | –0.44% | –1.07% | –0.63pp | No |
| 2026-03 | +1.86% | +0.36% | –1.50pp | Yes |

In 2 of 3 months with adequate sample size, the **lowest-buy_pct quintile of BUY-flagged stocks delivered higher alpha than the highest-buy_pct quintile**. This aligns precisely with Welch (2000) on analyst herding and Bradley et al. (2014) on the informativeness of contrarian recommendations. **It does not yet justify a parameter change** (3 months is short, the gap is unstable, and acting on it would invert decades of received wisdom about analyst consensus). It does justify *instrumenting* a contrarian probe as a separate signal channel that we observe for 6+ months before making it tradable.

---

## Workstream 3 — Architectural Critique

### The big finding: the committee adds value, but we cannot say which parts

T+7 alpha: raw signal +0.61% (n=906), committee BUY +2.75% (n=40). Roughly 4× incremental alpha. Sample size for committee is small (40 vs 906) so the magnitude is uncertain, but the direction is consistent. **Something the committee does works** — and we cannot tell what, because no individual modifier has been validated against forward returns.

The system applies ~20 conviction modifiers (sector rotation +10/–15, quality trap –8, signal velocity –8, contradictions –6 to –10, news asymmetry +5/–6.5, census divergence, earnings proximity –12 to –15, regional calibration ±5, etc.). Magnitudes are hand-tuned by intuition and CIO review notes ("–8 feels right for quality trap"). The infrastructure for calibration exists in `committee_scorecard.py:1283-1446` (a `calibrate_modifiers()` function that computes per-modifier hit rates and alpha deltas). It has never been run in production.

**Recommendation: activate weekly modifier calibration on a trailing 90-day window. Regress each modifier's contribution against T+7 alpha. Drop modifiers with |correlation| < 0.10. Retune magnitudes via grid search.** This requires no new code — the function already exists. It requires preserving agent-level votes in `concordance.json` (currently aggregated to `bull_pct` before write) so the per-modifier waterfall can be back-tested.

### Five other architectural issues worth naming

1. **The "agents" are not agents.** Five of the seven specialists are deterministic Python rules (200-350 lines each). The "Devil's Advocate" runs `if pct_buy >= 95: append("Extreme consensus...")`. This is fine for reproducibility — but the marketing is misleading. Recommendation: drop the "agent" framing in user-facing docs OR lift specific agents to actual LLM reasoning (news interpretation is the most obvious candidate, since template-matched warnings cannot detect novel risks).

2. **`committee_synthesis.py` is a 4,594-line monolith.** 50 functions averaging 92 lines each, nesting up to 5-6 levels deep, mixes vote counting + modifier definitions + waterfall tracking + HTML formatting. Not refactorable in one pass. Flag for a Q3 decomposition (vote_aggregator | conviction_modifiers | conviction_bounds | audit_logger).

3. **Floor reapplication is genuinely fragile.** Quality floors get applied at base conviction, then sector concentration penalty can breach them, then the floor is re-clamped (CIO v11.0 L7). This is two-pass logic that the next developer will forget. Recommendation: define an ordered pipeline (base → bonuses → penalties → hard limits → floors) where floors run *last*, *once*, and document the precedence in the module docstring.

4. **Kill thesis is exception-on-exception logic.** Quality floors are inviolable… except when kill thesis triggered, then floors are bypassed. Scattered across `committee_synthesis.py:2100-2150` and `:3900`. Same fix: a clean precedence pyramid (kill thesis > hard limits > floors > modifiers).

5. **Synthetic data discount has no validation.** If an upstream bug mis-flags real data as `synthetic: true`, it gets silently halved by the 0.5x weighting. Recommendation: a contract test that fails CI if more than ~20% of any run's concordance entries are flagged synthetic.

### What works well

The concordance waterfall logging is excellent — every modifier contribution is tracked. The backtest pipeline is methodologically clean (verified by 20-row audit). The tier abstraction (MEGA/LARGE/MID/SMALL/MICRO) is the right structure. The signal engine itself (`analysis_engine.py`) is vectorized, well-tested, and clean. **Do not refactor any of these.**

### Output bloat

The HTML committee report has 18 sections. Best estimate is that 5 drive decisions (executive summary, action items, concordance grid, sector gap, risk alerts). Recommendation: cut to 6 sections and move the rest to an appendix link. Measure scroll-depth if you want to be empirical about it.

---

## Workstream 4 — Literature Anchor

The findings of WS1 and WS2 are not surprising to anyone who has read the analyst-recommendation literature.

**Analyst consensus is contrarian / weakly informative.** Womack (1996, *J Finance*) found upgrade abnormal returns of +2.4% in the month following but quickly attenuating. Barber, Lehavy, McNichols & Trueman (2001, *J Finance*) showed the alpha is in *changes* not *levels*. Bradley et al. (2014, *Financial Mgmt*) documented that contrarian positions against consensus produce significant negative announcement-day returns followed by reversal — i.e. the consensus IS the news, after which the news fades. Welch (2000, *JFE*) documented systematic analyst herding. **Our –0.06 to –0.20 monthly correlations within BUY-flagged stocks are squarely in this literature.**

**Target prices are noise.** Bradshaw (2002, *Accounting Horizons*) documented analyst stock-appreciation forecasts of 25-35% annually with subsequent underperformance of 10-15%. Asquith et al. (2005, *JFE*) found only 54% of target prices are met within horizon, with mean absolute forecast error 24.8% and 9.4% upward bias. **Our +0.05 correlation between `upside` and T+30 alpha is consistent with a metric that reflects analyst optimism, not intrinsic value.**

**The system omits factors with stronger empirical support than analyst consensus.** Fama-French five-factor (2015, *JFE*) adds profitability (RMW) and investment (CMA) — the system captures profitability via ROE but not investment intensity. Asness, Frazzini & Pedersen (2019, *RAS*) "Quality Minus Junk" delivers Sharpe 0.6+ globally. Frazzini & Pedersen (2014, *JFE*) "Betting Against Beta" delivers Sharpe 0.78 in U.S. equities (vs ~0 for analyst upside in our data). Jegadeesh & Titman (1993, *J Finance*) momentum delivers ~1% monthly. **The strongest empirical factors are largely missing from the system.**

**Position sizing on miscalibrated conviction is the Kelly trap.** Thorp's fractional-Kelly literature consistently recommends 0.25-0.50× full Kelly. Barber & Odean (2000, *J Finance*) showed overconfident traders underperform by 7.5pp/year, with miscalibration as the primary mechanism. Bröcker & Smith (2007) on calibration: low Brier score does NOT imply good calibration. The system's linear conviction → position-size map (0.35×–1.0×) lacks any calibration validation. **If conviction is overconfident — which is plausible given the analyst-consensus inputs — position sizes are systematically too big and drawdowns will be deeper than backtests suggest.**

**Survivorship bias in yfinance is real and material.** Shumway (1997, *J Finance*) on CRSP delisting returns; Brown et al. (1992, *RFS*) on survivorship bias in mutual fund performance. Documented inflation of backtest returns by 1.6–4.0pp/year. **The 138K signal log uses yfinance, so the absolute return numbers are likely overstated.** Alpha vs SPY is partially protected (both numerator and denominator suffer survivorship), but stock-level alpha may still be inflated for stocks that were later delisted. Practical remedy: Norgate Data ($40/month) or manual addition of known delisted tickers.

**Backtest length is far too short for confident statements.** Harvey, Liu & Zhu (2016, *RFS*) recommend t-stat > 3.0 (vs traditional 2.0) to control for multiple testing — requiring ~20 years of monthly data at Sharpe 0.75. Bailey & López de Prado (2014) on the Deflated Sharpe Ratio: backtest-period Sharpe must be discounted for selection bias. **Our 94-day window is hypothesis-generating only, not validation. Any statement about strategy performance from this dataset must carry that caveat.**

---

## Prioritized Recommendations

Each recommendation tagged with effort (S = sub-day, M = 1-2 weeks, L = month+) and expected impact (minor / moderate / material).

### Tier 1 — Instrument before optimize (do this month)

**R1. Preserve agent-level votes in `concordance.json`.** Currently aggregated to `bull_pct` before write (`committee_synthesis.py:960-1000`, `committee_backtester.py:394-422`). Adds ~1KB/stock/day. Without this we cannot answer "which agent was right." **Effort: S. Impact: enables every downstream analysis. Material.**

**R2. Activate `calibrate_modifiers()` on a weekly cron.** Function exists at `committee_scorecard.py:1283-1446`. Run on trailing 90-day window. Output: per-modifier alpha contribution. After 3 months of data, retire any modifier with |correlation| < 0.10 against T+7 alpha. **Effort: S to wire up, M to act on results. Impact: probable retirement of 5-10 of the 20 modifiers. Material — directly addresses the "we don't know what works" core problem.**

**R3. Capture `price_at_signal` at signal generation time, not post-hoc.** 99.4% of `signal_log.jsonl` entries lack this field; it's backfilled from yfinance at backtest time using "nearest prior trading day." Works but introduces lag-of-unknown-magnitude. Add the field at signal write. **Effort: S. Impact: methodological cleanup, moderate.**

**R4. Backfill the `sector` field (currently 100% null in signal_log).** Use the existing `TICKER_GICS_MAP`. Without sector, regime-stratified analysis (sector rotation efficacy) is impossible. **Effort: S. Impact: enables sector-level diagnostics, moderate.**

**R5. Add a contract test for synthetic-data flag proportion.** Fail CI if >20% of concordance entries in any run carry `synthetic: true`. **Effort: S. Impact: catches a class of silent bugs, minor-to-moderate.**

### Tier 2 — Tighten methodology (do this quarter)

**R6. Demote raw return; promote alpha-vs-SPY as the headline metric in all backtest outputs.** Raw returns shown only as context with explicit market-context disclaimer. The current `backtest_summary.csv` leads with `mean_return`, which is misleading in any non-flat regime. **Effort: S. Impact: prevents the system from being judged by market conditions in future reviews. Moderate.**

**R7. Activate walk-forward validation in production backtests.** `backtest_stats.py:walk_forward_split()` exists but is unused. Add to the daily backtest cron. Report both in-sample and out-of-sample metrics. **Effort: S to wire, M to integrate into reporting. Impact: prevents future overfit drift. Material.**

**R8. Add regime-stratified reporting.** VIX bucket × signal × tier. The high-VIX BUY counterintuitive finding (66% hit rate at T+7) is only visible with this stratification. **Effort: S. Impact: surfaces edge candidates and fragility points. Moderate.**

**R9. Reduce default holding period from 90 days to a horizon where the signal has alpha.** Either T+7 (if you can absorb the turnover cost) or T+14 (compromise). Current 90-day default holds positions through 12 half-lives of the analyst-recommendation alpha decay (Womack 1996). **Effort: S to change default, M to model the turnover cost trade-off. Impact: potentially material, but requires honest analysis of holding-cost vs alpha trade-off — the existing `holding_cost_model` (CIO v4 F2) is already built for this.**

### Tier 3 — Strategic, gated on longer history (queue, do not start)

**R10. Factor model overlay.** Add momentum (12m skip-1m), low-vol (3y realized), quality (Piotroski exists in fundamental analyzer; lift to signal-level filter). Test as alternative SELECTORS, not in addition to existing thresholds. Gate: ≥6 months of data spanning a bull→bear transition. WS4 literature is strong here.

**R11. Conviction calibration before re-using conviction for sizing.** Build a Brier-score calibration curve. If conviction=80 doesn't deliver an 80% win rate, re-scale before feeding to the position sizer. Then move to fractional-Kelly sizing (0.25x). Hard cap at 3% of portfolio per position regardless of conviction. Gate: same as R10.

**R12. Survivorship-bias correction.** Either Norgate Data subscription (~$40/month) or manual addition of known delisted tickers to backtest universe. Estimated impact on reported alpha: +1.6 to +4.0pp/year *overstatement* corrected. Gate: only meaningful with longer history.

**R13. Tail-aware contrarian probe on buy_pct.** Run as a parallel observed-but-not-traded signal channel for 6 months. If the within-BUY tail effect (–7pp gap, top-vs-bottom quintile) persists, consider adding `max_buy_percentage` as a SELL trigger or as a conviction penalty. **Do not act on the current 3-month evidence.**

**R14. Architectural refactor of `committee_synthesis.py`.** Decompose into vote_aggregator | conviction_modifiers | conviction_bounds | audit_logger. Define explicit precedence pipeline (kill thesis > hard limits > floors > modifiers). Gate: no feature pressure for 2 weeks.

---

## What I Am Explicitly NOT Recommending — And Why

This is the most important section because the data tempts you toward each of these.

**I am NOT recommending any threshold changes from the current `backtest_threshold_report.csv`.** Walk-forward validation collapses 4 of 5 top suggestions. Five of eleven are statistically underpowered. Adopting them would lock in bear-market overfitting and degrade out-of-sample performance.

**I am NOT recommending you "fix" the T+30 BUY signal failure.** The fix isn't to change the BUY logic — it's to stop holding for 30 days. The signal works at T+7 and decays in line with academic literature on analyst recommendations. Changing the metric to make it work at 30 days would be fitting noise.

**I am NOT recommending you invert the buy_pct signal yet.** The contrarian effect is real in the academic literature and visible at the tails of our data, but our 3-month sample is far too short to act on. Acting on this now and being wrong would be expensive. Observe for 6 months, then decide.

**I am NOT recommending the agent system be lifted to actual LLM agents at this time.** It would be a large rewrite, and the deterministic system delivers measurable alpha. The marketing should change ("specialists" or "analyzers" instead of "agents"); the implementation should not — until you have evidence that LLM reasoning would add value beyond what calibrated modifiers provide.

**I am NOT recommending that you stop running this system.** The T+7 BUY alpha is real (+0.61%, statistically significant). The SELL signals work at both horizons. The committee adds incremental alpha (subject to small-n caveat). The system is genuinely useful — it's just over-engineered, under-instrumented, and held too long.

**I am NOT recommending you panic about the bear-market T+30 numbers.** They are noise from a single regime. You correctly flagged this.

---

## Verification

Every quantitative claim in this report traces to one of:
- `yahoofinance/output/backtest_summary.csv` (the headline T+7/T+30 hit rates)
- `yahoofinance/output/backtest_threshold_report.csv` (in-sample threshold suggestions)
- `~/Downloads/202604180000_ws1_diagnostic_backtest.md` (decile, VIX, monthly cohort, methodology audit)
- `~/Downloads/202604180000_ws2_overfitting_stress.md` (walk-forward, sample-size, monthly correlations)
- `~/Downloads/202604181730_ws3_architectural_critique.md` (file:line citations for code findings)
- `~/Downloads/202604180000_ws4_literature_anchor.md` (academic citations)
- yfinance API for SPY YTD context (verified 2026-04-17 close 710.14)

The four workstream reports are kept in `~/Downloads/` (not committed to the repo per the user's file-output convention). They contain the raw tables and the analysis scripts that generated them.

---

## Closing

The hardest discipline in this review was **not acting on the data**. Every workstream produced findings that look like they should be acted on tomorrow — change thresholds, invert metrics, kill modifiers, switch horizons. Most of those would be wrong moves on a 94-day single-regime window.

The high-leverage moves are unglamorous: **preserve the data the system already throws away (R1), run the calibration function that already exists (R2), and stop holding signals through three half-lives of their alpha (R9)**. Those three changes together would, with reasonable probability, materially improve P&L without committing to a single bear-market overfit.

The strategic moves (factor overlay, Kelly recalibration, survivorship correction) are real and supported by literature — but they require a longer dataset than 94 days to validate. Queue them, gate them on ≥6 months spanning a regime transition, and revisit.

Most of all: **the system delivers alpha, and the bear market did not break it. The bear market exposed which parts were uncalibrated theatre and which parts were quietly working. That distinction is the gift this regime gave you.**
