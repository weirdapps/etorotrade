# CIO v17 Operational Modules — Reference

The v17 review surfaced an uncomfortable fact: **Spearman ρ(conviction,
α30) within BUY-signal stocks ≈ −0.002**. The conviction score, after 16
review cycles and 53 modifiers, has zero rank-order predictive power on
T+30 alpha among the stocks the upstream signal already classified as
BUY. The system *does* add value at the upstream classifier (BUY +0.92%
α at T+30), but the modifier waterfall doesn't differentiate within it.

The v17 sprint shipped 14 implementations to address this finding (see
[the v17 review](.claude/cio-review-v17.md)). The ops sprint then added
**7 modules + 4 weekly-backtest phases** to close the self-improvement
loop so future calibration is evidence-driven, not opinion-driven.

This document is the reference for those 7 modules.

## Quick map

| Module | Purpose | Used by |
|--------|---------|---------|
| `price_cache` | Parquet 1y cache → eliminates yfinance SPOF | All downstream consumers |
| `kill_thesis_auditor` | Classify triggered theses TRUE/FALSE positive | Weekly backtest Phase 10 |
| `conviction_cells` | Per-cell Spearman ρ(conv, α30) | Synthesis confidence multiplier |
| `debate_scorecard` | Adversarial-debate effectiveness | Weekly backtest Phase 7 |
| `post_mortem` | Auto-detect -10%/30d drawdowns | Lessons-learned log → next /committee |
| `bayesian_conviction` | Posterior P(α30 > 0) shadow scoring | Future replacement for waterfall |
| `agent_sign_calibrator` (v17 H1) | Detect inverted agents (shadow mode) | Synthesis vote-weight sign |

The weekly backtest now runs **10 phases** (was 3 before v17):

```
Phase 1  — Signal-level backtest (BacktestEngine)
Phase 2  — Committee conviction backtest (CommitteeBacktester)
Phase 3  — Scorecard + modifier calibration
Phase 4  — Rolling-percentile thresholds       [v17 H4.b]
Phase 5  — Agent sign calibrator (SHADOW)      [v17 H1]
Phase 6  — Per-cell conviction calibration     [v17 op #4]   ← NEW
Phase 7  — Adversarial debate scorecard        [v17 op #5]   ← NEW
Phase 8  — Bayesian conviction likelihoods     [v17 op #7]   ← NEW
Phase 9  — Post-mortem detection               [v17 op #6]   ← NEW
Phase 10 — Kill thesis audit                   [v17 op #2]   ← NEW
```

---

## `trade_modules/price_cache.py` — Op #8

Parquet cache at `~/.weirdapps-trading/price_cache/{ticker}_1y.parquet`.
Every consumer hits this first; only falls back to yfinance when the
cache is missing or stale.

### Public API
```python
from trade_modules.price_cache import (
    freshness_status,    # → "missing" | "fresh" | "stale" | "very_stale"
    load_prices,         # batch load DataFrames from cache
    fetch_and_cache,     # refresh tickers from yfinance
    refresh_if_stale,    # smart refresh based on freshness_status
    cache_stats,         # health snapshot
    write_health_report, # persist health JSON
)
```

### Refresh policy
| Status | Last bar age | Behaviour |
|--------|--------------|-----------|
| `fresh` | ≤2 trading days | Use cache, no refresh |
| `stale` | 2-7 days | Use cache, log warning |
| `very_stale` | >7 days | Force refresh from yfinance |
| `missing` | n/a | Fetch from yfinance |

### Daily cron
```bash
python scripts/refresh_price_cache.py [--force]
```
Walks the union of (portfolio.csv ∪ buy.csv ∪ last 5 concordance archives ∪ {SPY}) and refreshes any missing/very_stale entries. `--force` also refreshes `stale`.

---

## `trade_modules/kill_thesis_auditor.py` — Op #2

When `check_kill_theses()` reports many triggered theses, this module
distinguishes:
- **TRUE_POSITIVE** — position dropped ≥8% within 30d → trigger justified
- **FALSE_POSITIVE** — position rose ≥8% → trigger fired prematurely
- **INCONCLUSIVE** — position moved within ±8% → noise band
- **UNVERIFIED** — <7 days since trigger or no price data

### Public API
```python
from trade_modules.kill_thesis_auditor import audit_triggered_theses

result = audit_triggered_theses()
# → {
#     "total_audited": 61,
#     "true_positives": [...], "false_positives": [...],
#     "summary": {"true_positive_count": 8, "false_positive_count": 3, ...},
#     "by_pattern": {...},  # grouped by thesis fingerprint (first 5 words)
# }
```

### Live findings (2026-04-20)
First run on real data immediately surfaced **8 FALSE positives** among
61 triggered theses:
- ABI.BR rose +9.5% in 28d after "VIX>40" trigger
- GOOG rose +13.5% in 28d after "macro deterioration" trigger
- TMO rose +10.1% in 28d after same generic pattern

**Action implication**: the generic "macro deterioration / VIX>40" kill
thesis is over-triggering. Future committees should generate
stock-specific kill theses with concrete numeric triggers.

---

## `trade_modules/conviction_cells.py` — Op #4

Slices the historical concordance × forward-returns sample into
(signal × tier × regime × consensus_band) cells and computes Spearman ρ
per cell. Direct attack on the v17 headline finding that aggregate ρ
≈ −0.002 — the cell-level signal may be much stronger than the
aggregate.

### Public API
```python
from trade_modules.conviction_cells import (
    compute_cells,                 # build the cell table
    cell_confidence_multiplier,    # consume in synthesis
    persist_cells, load_cells,
)

# In synthesis: get cell-aware confidence multiplier
mult = cell_confidence_multiplier(
    signal="B", market_cap_billions=600, regime="RISK_ON",
    buy_pct=80, cells_data=loaded_cells,
)
# mult ∈ {0.7, 1.0, 1.2}:
#   high-IC cell (|ρ|≥0.3 with n≥5) → 1.2
#   low-IC cell  (|ρ|<0.10 with n≥5) → 0.7
#   otherwise → 1.0
```

### Cell axes
- **Signal**: B / H / S / I
- **Tier**: MEGA (≥$500B) / LARGE (≥$100B) / MID (≥$10B) / SMALL (≥$2B) / MICRO (<$2B)
- **Regime**: RISK_ON / CAUTIOUS / RISK_OFF
- **Consensus**: EXTREME (≥90%) / HIGH (≥75%) / MODERATE (≥60%) / LOW (<60%)

Up to 4×5×3×4 = 240 cells; in practice 30-60 are populated.

---

## `trade_modules/debate_scorecard.py` — Op #5

Tracks adversarial-debate effectiveness. For each `debate_signal`
(STRENGTHEN_BULL, WEAKEN_BULL, STRENGTHEN_BEAR, WEAKEN_BEAR), measures
realized α(T+30) vs same-conviction control stocks **without** debate.

If the verdict is `NO_EDGE` (|excess α| < 0.5pp) for many runs, we
should consider deprecating the adversarial-debate path — it's the
most expensive component (Round 1 + Round 2 each consume Opus tokens
per contentious stock).

### Public API
```python
from trade_modules.debate_scorecard import (
    compute_debate_scorecard,
    persist_scorecard, load_scorecard,
)

sc = compute_debate_scorecard(history, forward_returns)
# → {
#     "control_n": 50,
#     "control_mean_alpha": 1.2,
#     "signals": {
#         "STRENGTHEN_BULL": {
#             "count": 8, "mean_conviction_delta": 4.2,
#             "mean_alpha": 3.1, "hit_rate": 0.625,
#             "excess_alpha_vs_control": 1.9,
#         },
#         ...
#     },
#     "verdict": "POSITIVE_EDGE — weighted excess α = +1.4pp",
# }
```

---

## `trade_modules/post_mortem.py` — Op #6

When a recommended ADD/BUY drops ≥10% within 30 days, auto-generates a
structured post-mortem and appends it to a lessons-learned log. The
log is read by the next /committee run as part of agent_memory.

### Public API
```python
from trade_modules.post_mortem import (
    detect_post_mortems,        # scan history for drawdown breaches
    append_lessons,             # dedupe + append to JSONL log
    load_recent_lessons,        # last N lessons
    summarise_for_committee,    # one-page markdown for next /committee
)

pms = detect_post_mortems()  # uses price_cache by default
n = append_lessons(pms)      # appends new ones, dedups by (ticker, date)
```

### PostMortem fields
- `ticker`, `recommendation_date`, `action`, `conviction`, `entry_price`
- `drawdown_date`, `drawdown_pct`, `days_to_drawdown`
- `endorsing_agents` — agents that voted bullish before the drop
- `dissenting_agents` — agents that warned but were overruled
- `waterfall_top` — top-5 modifier contributions by abs(value)
- `kill_thesis_text` — what we said would invalidate the trade
- `lesson` — 1-2 sentence summary

### Lessons log format
JSONL at `~/.weirdapps-trading/committee/lessons_learned.jsonl`. One
line per audited failure. The next committee reads the most recent N
lines and includes them in the agent_memory section of each agent's
prompt.

---

## `trade_modules/bayesian_conviction.py` — Op #7

Replaces the additive bonus/penalty waterfall with a Bayesian update.
Runs in **shadow mode** for the first 8 weeks alongside the existing
waterfall scorer; user actions still come from the waterfall until the
Bayesian engine has 8+ weeks of comparable output.

### Math
```
prior(α30 > 0)  = sigmoid_from_conviction(conviction)
for each agent view v:
    posterior ∝ prior × P(α30 > 0 | view = v)
recalibrated_conviction = sigmoid⁻¹(posterior) × 100
```

Per-agent likelihoods come from rolling history with **Beta(2,2)
shrinkage** — adds 4 pseudo observations split between hits and
misses, so a 0/0 view starts at p=0.5 not undefined.

### Public API
```python
from trade_modules.bayesian_conviction import (
    compute_likelihoods,           # build per-view likelihood table
    bayesian_posterior,            # update one stock's conviction
    shadow_score_concordance,      # batch shadow-score
    persist_likelihoods, load_likelihoods,
)

# In weekly backtest Phase 8:
lik = compute_likelihoods(history, forward_returns, horizon="T+30")
persist_likelihoods(lik)

# In shadow scoring (alongside the waterfall):
shadow = shadow_score_concordance(concordance, lik)
# → {"rows": [{ticker, conviction_prior, conviction_posterior, delta, ...}],
#    "summary": {"mean_delta": 1.5, "n_upgrade_10pt": 3, ...}}
```

### Promotion criteria
After 8 weeks of shadow output:
1. Compare shadow vs waterfall conviction at T+30
2. If shadow ρ(conv, α30) > waterfall ρ by ≥0.05 — promote to AUTO
3. Otherwise stay in shadow until evidence accumulates

---

## `scripts/run_weekly_backtest.py` — orchestrator

Runs all 10 phases in order. Each phase is wrapped in try/except so
one failure doesn't take down the rest.

### Output
Consolidated `yahoofinance/output/backtest_report.json` with these
top-level keys:
```
{
  "headline":            {buy_count_t7, sell_count_t7, signal_backtest_status, ...},
  "signal_backtest":     {status, metrics},
  "committee_backtest":  {status, performance_7d, performance_30d},
  "scorecard":           {buy_total, ..., conviction_predictive},
  "calibration":         {sufficient_data, modifiers_evaluated},
  "rolling_thresholds":          {B, H, S, I, _meta},          # Phase 4
  "agent_sign_calibration":      {status, agents: {...}},      # Phase 5
  "conviction_cells":            {total_observations, ...},    # Phase 6
  "debate_scorecard_summary":    {control_n, verdict},         # Phase 7
  "bayesian_likelihoods_summary":{evidence_total, n_agents},   # Phase 8
  "post_mortems_appended":       <int>,                        # Phase 9
  "kill_thesis_audit":           {summary, status}             # Phase 10
}
```

### CIO v17 op #1 — soft-fail on Phase 1 data drought
`build_report()` always emits `headline.buy_count_t7` (int 0 not None)
and `headline.signal_backtest_status` ("ok" or "no_data"). The
weekly-backtest workflow validation now checks **schema presence**, not
value, so the Sunday cron stops failing on environmental issues.

---

## Dependencies between modules

```
                    ┌──────────────────────┐
                    │ price_cache (op #8)  │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌──────────────────┐    ┌──────────────┐
│ post_mortem   │    │ kill_thesis_     │    │ committee_   │
│   (op #6)     │    │   auditor (op #2)│    │ backtester   │
└───────┬───────┘    └──────────────────┘    └──────┬───────┘
        │                                           │
        ▼                                           ▼
┌─────────────────┐         ┌─────────────────────────────────┐
│ lessons_learned │         │ history → forward_returns       │
│ .jsonl          │         └────────────────┬────────────────┘
└────────┬────────┘                          │
         │                  ┌────────────────┼─────────────────┐
         │                  │                │                 │
         ▼                  ▼                ▼                 ▼
┌─────────────────┐  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ next /committee │  │ conviction_  │ │ debate_      │ │ bayesian_    │
│ agent_memory    │  │  cells (#4)  │ │ scorecard(#5)│ │ conviction(#7│
└─────────────────┘  └──────┬───────┘ └──────────────┘ └──────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │ committee_synthesis │
                  │ confidence_mult     │
                  └─────────────────────┘
```

---

## Test coverage

`tests/unit/trade_modules/test_v17_ops.py` — 35 cases covering each
module's public API, edge cases, and the Bayesian sigmoid roundtrip.

Combined with the v17 implementations test (`test_v17_improvements.py`,
62 cases) and the broader committee suite, the **focused
trade_modules suite has 818 passing tests**.
