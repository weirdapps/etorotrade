# Signal Generation Framework - Critical Review

**Date:** 2026-02-04
**Reviewed By:** Claude (Senior Hedge Fund Manager / CIO Perspective)
**Data Source:** market.csv with 5,535 stocks

---

## Executive Summary

The current signal framework is **too restrictive**, generating only **13 BUY signals (0.2%)** out of 5,535 stocks. This is far below the 5-10% BUY rate typical of institutional frameworks. The issues fall into three categories:

1. **Asset Type Misclassification** - Causing incorrect signal paths
2. **Momentum Requirements Too Strict** - 200DMA blocking quality stocks
3. **Fundamental Thresholds Too Tight** - FCF, debt limits blocking valid opportunities

---

## Signal Distribution Analysis

| Signal | Count | Percentage | Expected Range |
|--------|-------|------------|----------------|
| BUY | 13 | 0.2% | 5-10% |
| SELL | 583 | 10.5% | 10-20% |
| HOLD | 1,320 | 23.8% | 30-50% |
| INCONCLUSIVE | 3,619 | 65.4% | 20-30% |

**Key Finding:** Too many stocks are INCONCLUSIVE (65.4%) due to strict analyst coverage requirements.

---

## Critical Issue #1: Asset Type Misclassification

### Bug: "TRUST" Pattern Matching ETFs Incorrectly

**Problem:** The word "TRUST" in company names triggers ETF classification, but many financial services companies have "TRUST" in their names.

**Affected Stocks:**
- NTRS (Northern Trust Corp.) → Classified as "etf" → BUY signal with only 33% buy rating!
- STB.L (Secure Trust Bank PLC) → Classified as "etf" → BUY signal with only 1 analyst!

**Root Cause:** Line 167 in `asset_type_utils.py`:
```python
etf_patterns = ['ETF', 'INDEX', 'TRUST']
```

**Impact:** These stocks bypass analyst-based signal logic and go through momentum-based logic, resulting in incorrect BUY signals.

### Bug: Ticker Collision with Cryptocurrency

**Problem:** VET (Vermilion Energy Inc.) is classified as "crypto" due to collision with VET (Vechain cryptocurrency).

**Root Cause:** Line 127 in `asset_type_utils.py`:
```python
known_crypto = {
    'BTC', 'ETH', 'XRP', ..., 'VET', ...  # VET is Vechain crypto
}
```

**Impact:** Vermilion Energy (a Canadian oil company) is treated as cryptocurrency and gets incorrect signals.

### Recommended Fixes:

1. **Remove "TRUST" from ETF patterns** or add exclusions for financial services companies:
   - Add to known_non_etfs: 'NTRS', 'BNY', 'STB.L'
   - Or check for patterns like "BANK", "FINANCIAL", "CORP" to exclude

2. **Add company name check for crypto tickers:**
   ```python
   # In _is_crypto_asset:
   if ticker in known_crypto:
       # Check company name doesn't contain non-crypto indicators
       if company_name and any(x in company_name.upper() for x in ['ENERGY', 'OIL', 'GAS', 'MINING', 'INC', 'CORP']):
           return False
       return True
   ```

---

## Critical Issue #2: 200DMA Requirement Too Restrictive

### Problem

The `require_above_200dma: true` setting is blocking **212 high-quality stocks** that would otherwise qualify for BUY.

**Notable Stocks Blocked:**
| Ticker | Name | Buy% | Upside | ROE | Why Blocked |
|--------|------|------|--------|-----|-------------|
| MSFT | Microsoft | 100% | 46.7% | 34.4% | Below 200DMA |
| V | Visa | 100% | 20.9% | 54.0% | Below 200DMA |
| MA | Mastercard | 100% | 20.0% | 209.9% | Below 200DMA |
| SONY | Sony | 88% | 52.2% | 15.4% | Below 200DMA |
| NFLX | Netflix | 75% | 39.9% | 42.8% | Below 200DMA |
| UBER | Uber | 96% | 40.3% | 73.0% | Below 200DMA |

### Academic Context

**200DMA as Momentum Indicator:**
- Jegadeesh & Titman (1993): Momentum is a proven factor
- However, binary above/below is crude vs. continuous momentum scoring
- Market-wide corrections can push many quality stocks below 200DMA temporarily

**Research on Buying Below Moving Averages:**
- Seyhun (1998): Insiders buy more when stocks are below moving averages
- Lakonishok et al. (1994): Value stocks often trade below technical levels before recovering

### Recommended Fix:

**Option A: Relax the Requirement for MEGA/LARGE Caps**
```yaml
us_mega:
  buy:
    require_above_200dma: false  # Remove for large, liquid names

us_large:
  buy:
    require_above_200dma: false  # Remove for large caps
```

**Option B: Make it a Soft Factor (Score Reduction)**
Instead of blocking, reduce the conviction score when below 200DMA:
- Above 200DMA: +10 points
- Below 200DMA: -10 points (but don't block)

**Option C: Add Proximity Check**
```yaml
require_above_200dma: true
dma_proximity_override: 0.95  # Allow if within 5% of 200DMA
```

---

## Critical Issue #3: FCF Yield Requirement Too Strict

### Problem

The `min_fcf_yield: 0.5` setting is blocking **66 high-quality stocks** including BABA.

**Notable Stocks Blocked:**
| Ticker | Name | Buy% | Upside | FCF Yield | Why Blocked |
|--------|------|------|--------|-----------|-------------|
| BABA | Alibaba | 97% | 20.3% | -1.6% | Negative FCF |
| INSM | Insmed | 96% | 36.9% | -1.3% | Negative FCF |
| NBIS | Nebius Group | 83% | 63.1% | -9.1% | Negative FCF |
| AIR.PA | Airbus | 75% | 18.8% | -0.3% | Negative FCF |

### Academic Context

**FCF as Quality Factor:**
- Sloan (1996): Accruals anomaly - cash flow predicts returns better than earnings
- Lakonishok et al. (1994): High FCF yield stocks outperform

**However:**
- Growth companies often have negative FCF due to reinvestment
- Tech/biotech sectors commonly have negative FCF during expansion phases
- Alibaba's negative FCF is due to heavy cloud infrastructure investment

### Recommended Fix:

**Option A: Relax Threshold**
```yaml
min_fcf_yield: -5.0  # Allow moderately negative FCF
```

**Option B: Sector-Specific Thresholds**
```yaml
fcf_yield_thresholds:
  technology: -10.0  # Growth sectors can have negative FCF
  healthcare: -15.0  # Biotech R&D requires investment
  financials: 0.0    # Traditional requirement
  consumer: 0.5      # Traditional requirement
```

**Option C: Score Penalty Instead of Block**
Instead of blocking, reduce conviction score:
- FCF yield > 5%: +15 points (cash cow)
- FCF yield 0-5%: +5 points
- FCF yield -5 to 0%: 0 points
- FCF yield < -5%: -10 points

---

## Critical Issue #4: Debt/Equity Threshold

### Problem

AVGO (Broadcom) with 96% buy rating and 43.2% upside is HOLD because DE is 166% vs max of 150%.

**Academic Context:**
- Technology companies often carry higher debt post-acquisitions
- Broadcom's debt is from strategic acquisitions (VMware, Symantec)
- Interest coverage ratio is more important than absolute debt level

### Recommended Fix:

```yaml
us_mega:
  buy:
    max_debt_equity: 200.0  # Increase from 150 to 200 for MEGA caps
```

---

## Critical Issue #5: Too Many INCONCLUSIVE Signals

### Problem

3,619 stocks (65.4%) are INCONCLUSIVE, primarily due to:
- min_analyst_count: 4 requirement
- min_price_targets: 4 requirement

### Analysis

Many valid stocks have fewer than 4 analysts:
- Small/mid-cap companies with 2-3 analysts covering
- International stocks with limited US analyst coverage
- Recently IPO'd companies

### Recommended Fix:

**Tiered Analyst Requirements:**
```yaml
analyst_requirements:
  mega_cap:    # $500B+
    min_analyst_count: 8
    min_price_targets: 6
  large_cap:   # $100-500B
    min_analyst_count: 6
    min_price_targets: 4
  mid_cap:     # $10-100B
    min_analyst_count: 4
    min_price_targets: 3
  small_cap:   # $2-10B
    min_analyst_count: 3
    min_price_targets: 2
```

---

## Summary: Improvement Plan

### Priority 1 (Critical Bugs)
1. **Fix "TRUST" ETF classification** - Add financial services companies to exclusion list
2. **Fix VET ticker collision** - Check company name before classifying as crypto

### Priority 2 (Signal Quality)
3. **Relax 200DMA for MEGA/LARGE caps** - Set `require_above_200dma: false`
4. **Relax FCF yield threshold** - Change to `min_fcf_yield: -5.0`
5. **Increase DE threshold for MEGA** - Change to `max_debt_equity: 200.0`

### Priority 3 (Coverage)
6. **Reduce analyst requirements for smaller caps** - Tiered approach
7. **Consider score-based approach** - Replace binary blocks with score adjustments

### Expected Outcome

After implementing Priority 1 and 2 fixes:
- BUY signals: ~50-100 (1-2% of universe)
- SELL signals: ~500-600 (10%)
- HOLD signals: ~2000-2500 (40-50%)
- INCONCLUSIVE: ~2500-3000 (45-55%)

---

## Appendix: Questionable BUY Signals

These stocks should NOT have BUY signals:

| Ticker | Name | Buy% | Upside | Issue |
|--------|------|------|--------|-------|
| NTRS | Northern Trust | 33% | 3.2% | Misclassified as ETF, bypassed criteria |
| STB.L | Secure Trust Bank | 100% | 26.1% | Only 1 analyst, misclassified as ETF |
| VET | Vermilion Energy | 36% | -- | Misclassified as crypto |

---

*This review was conducted from the perspective of a senior hedge fund manager with focus on practical applicability, signal quality, and academic research support.*
