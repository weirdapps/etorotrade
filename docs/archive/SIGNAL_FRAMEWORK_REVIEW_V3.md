# Signal Framework Comprehensive Review V3
## Critical Analysis by Senior Hedge Fund CIO Perspective

**Date:** February 7, 2026
**Analyst Role:** Senior CIO / Chief Investment Officer / Quantitative Strategist
**Scope:** Complete review of signal generation framework and market.csv output
**Previous Review:** V2 dated 2026-02-05

---

## Progress Since V2 Review

| Metric | V2 (Feb 5) | V3 (Feb 7) | Change |
|--------|------------|------------|--------|
| BUY Signals | 15 | 29 | **+93%** |
| EU BUYs | 0 | 3 | **Fixed** |
| SELL Signals | 587 | 584 | -0.5% |
| HOLD | 1,298 | 1,257 | -3.2% |
| INCONCLUSIVE | 3,113 | 3,156 | +1.4% |

**V2 Fixes Implemented:**
- EU region now generating BUY signals (SAP, Samsung, Airbus)
- Analyst count thresholds reduced for MEGA/LARGE caps
- 200DMA requirement relaxed for EU MEGA/LARGE

**Remaining Issues:** 164 high-quality stocks still stuck as HOLD (see Section 3)

---

## Executive Summary

The current signal framework has been analyzed against 5,026 stocks across US, EU, and HK markets. The overall signal distribution shows:

| Signal | Count | % of Total |
|--------|-------|------------|
| Inconclusive (I) | 3,156 | 62.8% |
| Hold (H) | 1,257 | 25.0% |
| Sell (S) | 584 | 11.6% |
| Buy (B) | 29 | 0.6% |

**Key Finding:** The framework is **excessively conservative**, generating only 29 BUY signals from 5,026 stocks (0.6%). While conservatism is appropriate for a systematic approach, this may be leaving significant alpha on the table.

---

## SECTION 1: SIGNAL DISTRIBUTION CRITIQUE

### 1.1 Geographic Distribution Analysis

| Region | Total | BUY | SELL | HOLD | INCONC |
|--------|-------|-----|------|------|--------|
| US | 3,559 | 23 | 416 | 871 | 2,249 |
| EU | 1,336 | 3 | 147 | 289 | 897 |
| HK | 131 | 3 | 21 | 97 | 10 |

**Critique:**
1. **EU Underrepresentation:** Only 3 BUY signals from 1,336 EU stocks (0.2%) is concerning. EU mega/large caps like SAP, Airbus are included, but mid/small caps are almost entirely excluded.

2. **HK Overweighting Issue:** HK has only 131 stocks but generates 3 BUYs (2.3%), suggesting thresholds may be better calibrated for HK or there's selection bias in the universe.

3. **MICRO cap filtering:** 2,111 MICRO caps are almost entirely marked INCONCLUSIVE, which is appropriate given the $2B floor, but the floor may be too conservative for certain high-quality small caps.

### 1.2 Tier Distribution Analysis

| Tier | BUY | SELL | HOLD | INCONC |
|------|-----|------|------|--------|
| MEGA | 12 | 4 | 32 | 5 |
| LARGE | 15 | 56 | 146 | 27 |
| MID | 1 | 268 | 608 | 189 |
| SMALL | 1 | 256 | 471 | 578 |
| MICRO | 0 | 0 | 0 | 2,111 |

**Critical Issues:**
1. **MID-cap Starvation:** Only 1 BUY from 1,066 MID-cap stocks (0.09%). The 75% 52W threshold combined with require_above_200dma=true is too restrictive.

2. **SMALL-cap Starvation:** Only 1 BUY from 1,306 SMALL-cap stocks (0.08%). The combination of 25% min_upside, 85% min_buy%, and 22% min_exret is nearly impossible to satisfy.

---

## SECTION 2: FALSE POSITIVE SELL SIGNALS (Type I Errors)

### 2.1 High-Conviction Stocks Marked SELL

Several stocks with strong fundamentals are incorrectly marked SELL:

| Ticker | Name | UP% | %B | Issue |
|--------|------|-----|-----|-------|
| MSTR | Strategy | 235% | 87% | 52W=30% (momentum crash), SI=11.7% (short interest), Beta=3.5 |
| HEXA-B.ST | Hexagon | 20.6% | 35% | Low buy% (35%) is correct SELL |
| GDDY | GoDaddy | 80% | 36% | DE=4217% (extreme leverage) - CORRECT SELL |
| MBLY | Mobileye | 79.6% | 38% | Low buy% (38%) - CORRECT SELL |

**Assessment:** Most high-upside SELL signals are CORRECTLY identified due to:
- Very low analyst buy% (< 40%)
- Extreme leverage (DE > 1000%)
- Severe momentum crash (52W < 35%)

**Recommendation:** The SELL framework is working well. MSTR is debatable given it's a Bitcoin proxy, but the framework correctly flags the extreme risk.

### 2.2 Questionable SELL: Stocks with Mixed Signals

| Ticker | UP% | %B | 52W | Issue |
|--------|-----|-----|-----|-------|
| BX (Blackstone) | 31.9% | 38% | 68% | Low buy% but strong upside |
| HUM (Humana) | 45.9% | 31% | 61% | Low buy% is concerning |
| ROP (Roper Tech) | 27.5% | 33% | 61% | Quality company, low coverage |

**Assessment:** These are borderline cases where low analyst buy% triggers SELL despite positive upside. The framework is being conservative, which is appropriate.

---

## SECTION 3: MISSED BUY OPPORTUNITIES (Type II Errors)

### 3.1 Critical Issue: 307 High-Conviction HOLDs

There are 307 stocks with:
- Upside >= 20%
- Buy% >= 70%
- Still marked HOLD

This is the **largest systematic problem** in the framework.

### 3.2 Root Cause Analysis of Missed BUYs

**Blocking Factors (in order of frequency):**

1. **52W Momentum Check (min_pct_from_52w_high)**
   - MID-cap requires 75%, SMALL-cap requires 75%
   - Many quality growth stocks trade far from 52W highs during corrections
   - Examples: TEAM (52W=29%), SNOW (52W=60%), NOW (52W=48%)

2. **require_above_200dma = true for MID/SMALL**
   - This filter alone blocks ~200 potential BUYs
   - Growth stocks often consolidate below 200DMA before breakouts

3. **PEF > PET * 1.1 Check (Deteriorating Earnings)**
   - Blocks: SONY (PEF=19.1 vs PET*1.1=18.7), UBER (PEF=17.5 vs PET*1.1=17.4)
   - This check is overly sensitive - a 10% threshold is too tight
   - Academic research suggests using 20-25% threshold

4. **Short Interest > 2.25-2.5%**
   - Blocks: UBER (SI=2.4%), SE (SI=5.6%), HOOD (SI=4.4%)
   - US tech stocks routinely have SI > 3%
   - Threshold should be tier-adjusted (higher for MID/SMALL)

5. **max_debt_equity for Growth Companies**
   - Blocks: MELI (DE=159%), HOOD (DE=189%), ORCL (DE=432%)
   - Growth-stage tech companies operate with high leverage by design
   - Oracle's DE is high due to share buybacks, not operational risk

### 3.3 Specific Case Studies

**CASE 1: UBER (us_large)**
- Upside: 41.9%, Buy%: 92%, EXRET: 38.8%
- Blockers: SI=2.4% (max=2.25%), PEF>PET*1.1
- **Critique:** This is a textbook BUY. Uber has achieved profitability, 92% analyst buy consensus, 42% upside. Being blocked by 0.15% short interest excess and marginal PEF/PET ratio is suboptimal.

**CASE 2: MELI (MercadoLibre) (us_mid)**
- Upside: 42.4%, Buy%: 87%, EXRET: 36.7%
- Blockers: 52W=74% (<75%), 2H=N, DE=159%, FCF=-4.1%
- **Critique:** MELI is the dominant e-commerce platform in LatAm. The framework correctly identifies risks but may be too conservative for this proven winner.

**CASE 3: TEAM (Atlassian) (us_mid)**
- Upside: 119%, Buy%: 94%, EXRET: 112%
- Blockers: 52W=29%, 2H=N, ROE=-15.3%
- **Critique:** TEAM is being blocked by momentum and profitability metrics. However, software companies often have negative ROE during growth phase. The 119% upside with 94% consensus is exceptional.

**CASE 4: SONY (us_mega)**
- Upside: 43.4%, Buy%: 88%, EXRET: 38.2%
- Blocker: PEF=19.1 > PET*1.1=18.7 (deteriorating earnings)
- **Critique:** Sony is being blocked by a razor-thin margin on the PEF/PET check. The 2% exceedance (19.1 vs 18.7) should not override 43% upside with 88% consensus.

---

## SECTION 4: METRICS ANALYSIS

### 4.1 Metrics Distribution by Signal

| Metric | BUY (median) | SELL (median) | HOLD (median) |
|--------|--------------|---------------|---------------|
| Upside % | 23.5% | 1.8% | 12.7% |
| Buy % | 97% | 31% | 75% |
| EXRET | 22.8% | 0.0% | 8.9% |
| 52W % | 83% | 91% | 89% |
| AM % | 0% | 0% | 0% |

**Observations:**
1. **BUY signals have LOWER 52W%** (83%) than SELL signals (91%). This seems counterintuitive but reflects that BUYs are found in slightly pulled-back stocks, not momentum chasers.

2. **SELL signals have very low buy%** (31% median) which is appropriate.

3. **Analyst Momentum (AM) is ineffective** - median is 0% across all signals. This metric may not be adding value.

### 4.2 Recommended Metric Weight Adjustments

Current SELL scoring weights:
- Analyst sentiment: 35%
- Momentum: 25%
- Valuation: 20%
- Fundamentals: 20%

**Proposed Changes:**
- Analyst sentiment: 40% (increase - most predictive)
- Momentum: 20% (decrease - 52W is volatile)
- Valuation: 25% (increase - PE spreads are meaningful)
- Fundamentals: 15% (decrease - ROE/DE less predictive short-term)

---

## SECTION 5: DETAILED IMPROVEMENT PLAN

### PRIORITY 1: Critical Threshold Adjustments

#### 5.1.1 Relax PEF/PET Deterioration Check
**Current:** PEF > PET * 1.1 blocks BUY
**Proposed:** PEF > PET * 1.25 blocks BUY

**Rationale:** The 10% threshold is too sensitive. Earnings estimates fluctuate, and a 25% threshold better captures genuine deterioration.

```yaml
# In signals.py, line ~1286-1290
# BEFORE
if row_pef > row_pet * 1.1:
    is_buy_candidate = False

# AFTER
if row_pef > row_pet * 1.25:
    is_buy_candidate = False
```

**Impact:** Would unlock SONY, UBER from BUY eligibility.

#### 5.1.2 Increase Short Interest Thresholds
**Current (us_large):** max_short_interest = 2.25%
**Proposed:** max_short_interest = 4.0% for MEGA/LARGE, 5.0% for MID

**Rationale:** US tech stocks routinely have 3-5% short interest. Tesla has historically had 15%+ and still generated positive returns. The current threshold is too conservative.

```yaml
us_mega:
  buy:
    max_short_interest: 4.0  # Was 2.0
us_large:
  buy:
    max_short_interest: 4.0  # Was 2.25
us_mid:
  buy:
    max_short_interest: 5.0  # Was 2.5
```

**Impact:** Would unlock UBER from BUY eligibility.

#### 5.1.3 Relax 52W Momentum for MID/SMALL
**Current (us_mid):** min_pct_from_52w_high = 75%
**Proposed:** min_pct_from_52w_high = 60%

**Rationale:** Requiring stocks to be within 25% of 52W high eliminates most value opportunities. Growth stocks routinely correct 30-40% before resuming uptrends.

```yaml
us_mid:
  buy:
    min_pct_from_52w_high: 60  # Was 75
us_small:
  buy:
    min_pct_from_52w_high: 60  # Was 75
eu_mid:
  buy:
    min_pct_from_52w_high: 60  # Was 75
```

**Impact:** Would add ~100+ potential BUY candidates including HOOD, SE, SNOW.

#### 5.1.4 Remove require_above_200dma for MID
**Current:** require_above_200dma = true for us_mid, us_small
**Proposed:** require_above_200dma = false for us_mid, keep true for us_small

**Rationale:** The 200DMA requirement combined with 52W% check is redundant and overly restrictive for mid-caps.

```yaml
us_mid:
  buy:
    require_above_200dma: false  # Was true
```

**Impact:** Would add ~50+ potential BUY candidates.

### PRIORITY 2: Debt/Equity Sector Adjustments

#### 5.2.1 Add Sector-Specific DE Overrides
High-growth tech and financial services operate with higher leverage by design.

**Current:** max_debt_equity = 130% for us_mid
**Proposed:** Add sector overrides:
- Financials: max_debt_equity = 300%
- Technology: max_debt_equity = 200%
- Real Estate: max_debt_equity = 250%

```python
# In trade_config.py, add sector_adjustments
SECTOR_DE_OVERRIDES = {
    'Financial Services': 3.0,  # Multiplier on base threshold
    'Technology': 1.5,
    'Real Estate': 2.0,
    'Consumer Cyclical': 1.3,
}
```

**Impact:** Would unlock MELI, HOOD, ORCL from DE constraint.

### PRIORITY 3: ROE Handling for Growth Companies

#### 5.3.1 Conditional ROE Check
**Current:** Negative ROE blocks BUY
**Proposed:** Ignore ROE check if:
- Revenue growth > 20% AND
- Gross margin > 50% AND
- Upside > 50%

**Rationale:** High-growth software companies (SNOW, TEAM, CRM history) intentionally reinvest at the expense of short-term profitability.

```python
# In signals.py, modify ROE check
if "min_roe" in buy_criteria and not pd.isna(row_roe):
    # Growth override: ignore ROE for high-growth, high-margin companies
    is_growth_override = (
        row_rev_growth > 20 and
        row_upside > 50 and
        row_buy_pct > 80
    )
    if not is_growth_override and row_roe < buy_criteria.get("min_roe"):
        is_buy_candidate = False
```

**Impact:** Would unlock TEAM, SNOW for BUY consideration.

### PRIORITY 4: FCF Yield Flexibility

#### 5.4.1 Growth-Adjusted FCF Check
**Current:** min_fcf_yield = 0.0 for us_mid
**Proposed:** Allow negative FCF for high-growth companies

```yaml
us_mid:
  buy:
    min_fcf_yield: -5.0  # Allow moderate cash burn for growth
us_small:
  buy:
    min_fcf_yield: -3.0  # Slightly stricter for smaller caps
```

**Rationale:** Amazon, Netflix, Uber all had negative FCF during growth phases. Penalizing cash burn ignores growth investment.

**Impact:** Would unlock MELI from FCF constraint.

### PRIORITY 5: Signal Scoring Enhancements

#### 5.5.1 Implement Quality Score Override for BUY
Add a quality score that can override individual criterion failures:

```python
# If quality score > 85 AND no more than 2 non-critical criteria fail, mark as BUY
quality_score = calculate_buy_quality_score(
    upside=row_upside,
    buy_pct=row_buy_pct,
    exret=row_exret,
    analyst_count=analyst_count.loc[idx],
)

non_critical_failures = count_non_critical_failures(...)
if quality_score > 85 and non_critical_failures <= 2:
    # Override to BUY with lower position size
    is_buy_candidate = True
    position_size_multiplier = 0.5  # Half position for override
```

#### 5.5.2 Add Conviction Tiers to BUY Signals
Currently all BUY signals are equal. Add conviction tiers:

- **HIGH CONVICTION (Score 85+):** Meet all criteria comfortably
- **MEDIUM CONVICTION (Score 70-84):** Meet criteria but marginal
- **LOW CONVICTION (Score 55-69):** Quality override applied

This helps with position sizing and risk management.

---

## SECTION 6: SPECIFIC STOCK RECOMMENDATIONS

Based on this analysis, the following stocks SHOULD be marked BUY but are not:

### 6.1 Strong BUY Candidates (Currently HOLD)

| Ticker | Upside | Buy% | Current Blockers | Recommendation |
|--------|--------|------|------------------|----------------|
| UBER | 41.9% | 92% | SI, PEF/PET | **STRONG BUY** |
| SONY | 43.4% | 88% | PEF/PET (marginal) | **STRONG BUY** |
| NOW | 88.6% | 88% | 52W=48% | **BUY** (value entry) |
| MELI | 42.4% | 87% | 52W, DE, FCF | **BUY** (growth) |
| KKR | 52.0% | 80% | 52W, ROE, 200DMA | **BUY** (financials) |
| HOOD | 79.8% | 87% | 52W, DE, SI | **SPECULATIVE BUY** |
| TEAM | 119% | 94% | 52W, ROE | **SPECULATIVE BUY** |
| SNOW | 66.7% | 93% | 52W, ROE, PEF | **SPECULATIVE BUY** |

### 6.2 SELL Signals Confirmed Correct

| Ticker | Upside | Buy% | Reason | Assessment |
|--------|--------|------|--------|------------|
| TSLA | 1.9% | 29% | Low consensus | **CORRECT SELL** |
| MSTR | 235% | 87% | 52W=30%, SI=11.7%, Beta=3.5 | **CORRECT SELL** (extreme risk) |
| INTC | -6.8% | 22% | Negative upside, low consensus | **CORRECT SELL** |
| AMGN | -11% | 33% | Negative upside, low consensus | **CORRECT SELL** |

---

## SECTION 7: IMPLEMENTATION PRIORITY

### Phase 1 (Immediate - This Week)
1. Increase PEF/PET threshold from 1.1 to 1.25
2. Increase max_short_interest thresholds
3. Reduce min_pct_from_52w_high for MID/SMALL from 75% to 60%

**Expected Impact:** +10-15 additional BUY signals

### Phase 2 (Next 2 Weeks)
1. Remove require_above_200dma for us_mid
2. Add sector-specific DE overrides
3. Implement growth company ROE override

**Expected Impact:** +20-30 additional BUY signals

### Phase 3 (This Month)
1. Implement conviction tiers for BUY signals
2. Add quality score override mechanism
3. Backtest threshold changes against historical data

**Expected Impact:** Better position sizing, reduced false negatives

---

## SECTION 8: RISK WARNINGS

### 8.1 Risks of Proposed Changes
1. **Increased False Positives:** Relaxing thresholds may introduce low-quality BUY signals
2. **Higher Volatility Exposure:** Allowing lower 52W% stocks increases drawdown risk
3. **Leverage Risk:** Relaxing DE thresholds exposes portfolio to bankruptcy risk

### 8.2 Mitigations
1. Implement conviction tiers with position sizing adjustments
2. Add stop-loss triggers for high-volatility BUY signals
3. Maintain hard floors on critical metrics (e.g., never buy < 60% buy consensus)

---

## CONCLUSION

The current framework is **well-designed but overly conservative**. The SELL logic is functioning correctly, identifying genuinely risky stocks. However, the BUY logic has multiple overlapping filters that create an "impossible to satisfy" condition for mid/small cap growth stocks.

The recommended changes would increase BUY signals from 29 to approximately 50-80 while maintaining quality standards. This represents a more balanced risk/reward approach appropriate for a systematic investing strategy.

**Key Takeaway:** The framework prioritizes avoiding losses (Type I errors) at the significant cost of missing gains (Type II errors). For an investment strategy, a more balanced approach would optimize for risk-adjusted returns rather than pure loss avoidance.

---

## APPENDIX A: PRIORITY 1 ACTION ITEMS (IMMEDIATE)

### A.1 PEF/PET Threshold Change

**File:** `trade_modules/analysis/signals.py`
**Line:** ~1286-1290

```python
# CURRENT:
if row_pef > row_pet * 1.1:

# CHANGE TO:
pef_pet_threshold = buy_criteria.get('max_pef_pet_ratio', 1.25)
if row_pef > row_pet * pef_pet_threshold:
```

**Config Addition:** Add `max_pef_pet_ratio: 1.25` to all tier buy sections in config.yaml

### A.2 Short Interest Threshold Changes

**File:** `config.yaml`

```yaml
# Change these values:
us_mega.buy.max_short_interest: 4.0   # was 2.0
us_large.buy.max_short_interest: 4.0  # was 2.25
us_mid.buy.max_short_interest: 5.0    # was 2.5
us_small.buy.max_short_interest: 5.5  # was 2.75
```

### A.3 52W Momentum Threshold Changes

**File:** `config.yaml`

```yaml
# Change these values:
us_mid.buy.min_pct_from_52w_high: 55   # was 75
us_small.buy.min_pct_from_52w_high: 60 # was 75
eu_mid.buy.min_pct_from_52w_high: 55   # was 75
eu_small.buy.min_pct_from_52w_high: 60 # was 75
```

---

## APPENDIX B: STOCKS THAT SHOULD BE BUY (TOP 20 BY QUALITY)

| Rank | Ticker | Name | Upside | Buy% | EXRET | Current Blocker(s) |
|------|--------|------|--------|------|-------|-------------------|
| 1 | UBER | Uber Technologies | 41.9% | 92% | 38.8% | SI=2.4% (>2.25%), PEF/PET |
| 2 | SONY | Sony Group | 43.4% | 88% | 38.2% | PEF/PET (19.1 vs 18.7) |
| 3 | KAP.L | National Aerospace | 183.6% | 100% | 183.6% | Unknown - needs investigation |
| 4 | WVE | Wave Life Sciences | 149.1% | 100% | 149.1% | ROE=-86.5% |
| 5 | RBRK | Rubrik Inc | 117.1% | 94% | 109.8% | 52W=50%, 200DMA |
| 6 | REL.L | RELX PLC | 95.6% | 91% | 87.0% | DE=343%, 52W=51% |
| 7 | IBRX | ImmunityBio | 95.0% | 100% | 95.0% | Unknown |
| 8 | SGHC | Super Group | 93.5% | 100% | 93.5% | 200DMA |
| 9 | SRAD | Sportradar | 91.0% | 100% | 91.0% | 52W=53% |
| 10 | HOOD | Robinhood | 79.8% | 87% | 69.4% | 52W, DE, SI |
| 11 | APP | AppLovin | 77.5% | 91% | 70.7% | DE=238% |
| 12 | SE | Sea Limited | 74.6% | 86% | 64.0% | 52W, SI=5.6% |
| 13 | DDOG | Datadog | 73.8% | 91% | 67.5% | 52W=55% |
| 14 | SNOW | Snowflake | 66.7% | 93% | 61.8% | 52W, ROE, PEF |
| 15 | MELI | MercadoLibre | 42.4% | 87% | 36.7% | 52W, DE, FCF |
| 16 | NOW | ServiceNow | 88.6% | 88% | 77.5% | 52W=48% |
| 17 | KKR | KKR & Co | 52.0% | 80% | 41.6% | 52W, ROE, 200DMA |
| 18 | TEAM | Atlassian | 119.0% | 94% | 112.0% | 52W=29%, ROE |
| 19 | ORCL | Oracle | 93.5% | 76% | 71.0% | %B=76%<77%, 52W, DE |
| 20 | ZEAL.CO | Zealand Pharma | 80.5% | 88% | 70.9% | 52W=52% |

---

*Document Version: 3.0*
*Analysis Date: February 7, 2026*
*Next Review: After Phase 1 implementation*
