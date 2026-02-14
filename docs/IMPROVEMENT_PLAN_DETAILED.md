# Detailed Implementation Plan for Signal Framework Improvements

## Summary of Changes

This document contains specific, actionable changes to address the issues identified in the Signal Framework Review V3.

---

## CHANGE 1: Relax PEF/PET Deterioration Threshold

### Location
`trade_modules/analysis/signals.py`, lines 1286-1290

### Current Code
```python
# PEF < PET requirement: Forward PE should be lower than Trailing PE (improving earnings)
# Only apply this check when both values are meaningful (> 10) and PEF is significantly higher
if not pd.isna(row_pef) and not pd.isna(row_pet) and row_pet > 10 and row_pef > 10:
    # PEF should not be more than 10% higher than PET
    if row_pef > row_pet * 1.1:
        is_buy_candidate = False
        logger.debug(f"Ticker {ticker}: Failed PEF<PET check - PEF:{row_pef:.1f} > PET*1.1:{row_pet * 1.1:.1f}")
```

### Proposed Code
```python
# PEF < PET requirement: Forward PE should be lower than Trailing PE (improving earnings)
# Only apply this check when both values are meaningful (> 10) and PEF is significantly higher
# Research shows 25% threshold better captures genuine deterioration vs. estimate noise
if not pd.isna(row_pef) and not pd.isna(row_pet) and row_pet > 10 and row_pef > 10:
    # PEF should not be more than 25% higher than PET (relaxed from 10%)
    pef_pet_threshold = buy_criteria.get('max_pef_pet_ratio', 1.25)
    if row_pef > row_pet * pef_pet_threshold:
        is_buy_candidate = False
        logger.debug(f"Ticker {ticker}: Failed PEF<PET check - PEF:{row_pef:.1f} > PET*{pef_pet_threshold}:{row_pet * pef_pet_threshold:.1f}")
```

### Config Addition (config.yaml)
```yaml
# Add to each tier's buy criteria
us_mega:
  buy:
    max_pef_pet_ratio: 1.25  # Allow 25% PEF > PET (was hardcoded 10%)
us_large:
  buy:
    max_pef_pet_ratio: 1.25
us_mid:
  buy:
    max_pef_pet_ratio: 1.20  # Slightly stricter for mid-caps
```

### Rationale
- SONY blocked by PEF=19.1 vs threshold of 18.7 (2% over)
- UBER blocked by PEF=17.5 vs threshold of 17.4 (0.6% over)
- Academic research on earnings momentum uses 20-25% thresholds
- Current 10% threshold is overly sensitive to estimate noise

### Impact
Stocks unlocked: SONY, UBER, ~5-10 others

---

## CHANGE 2: Increase Short Interest Thresholds

### Location
`config.yaml`, all tier definitions

### Current Values
```yaml
us_mega:
  buy:
    max_short_interest: 2.0
us_large:
  buy:
    max_short_interest: 2.25
us_mid:
  buy:
    max_short_interest: 2.5
us_small:
  buy:
    max_short_interest: 2.75
```

### Proposed Values
```yaml
us_mega:
  buy:
    max_short_interest: 4.0   # Mega caps can handle higher SI
us_large:
  buy:
    max_short_interest: 4.0   # Large caps well-covered
us_mid:
  buy:
    max_short_interest: 5.0   # Mid-caps often shorted more
us_small:
  buy:
    max_short_interest: 5.5   # Small caps naturally have higher SI
```

### Rationale
- UBER blocked at SI=2.4% (threshold 2.25%)
- US tech sector median SI is ~3%
- Tesla has historically had 15-20% SI and still outperformed
- Short interest is a weak negative predictor when paired with strong fundamentals
- High SI + high analyst buy% often indicates "misunderstood" opportunity

### Research Support
- Asquith, Pathak & Ritter (2005): High SI with high institutional ownership shows positive returns
- Boehmer, Jones & Zhang (2008): SI only predictive when combined with other signals

### Impact
Stocks unlocked: UBER, and ~10-15 others with SI between 2.5-5%

---

## CHANGE 3: Reduce 52W Momentum Threshold for MID/SMALL

### Location
`config.yaml`, us_mid and us_small sections

### Current Values
```yaml
us_mid:
  buy:
    min_pct_from_52w_high: 75
us_small:
  buy:
    min_pct_from_52w_high: 75
eu_mid:
  buy:
    min_pct_from_52w_high: 75
eu_small:
  buy:
    min_pct_from_52w_high: 75
```

### Proposed Values
```yaml
us_mid:
  buy:
    min_pct_from_52w_high: 55   # Allow 45% drawdown from high
us_small:
  buy:
    min_pct_from_52w_high: 60   # Allow 40% drawdown from high
eu_mid:
  buy:
    min_pct_from_52w_high: 55
eu_small:
  buy:
    min_pct_from_52w_high: 60
```

### Rationale
- 121 stocks have UP% >= 40%, Buy% >= 80% but are HOLD
- Most are blocked by 52W% threshold
- Examples:
  - TEAM: 52W=29%, UP%=119%, %B=94%
  - SNOW: 52W=60%, UP%=66.7%, %B=93%
  - NOW: 52W=48%, UP%=88.6%, %B=88%
- Growth stocks routinely correct 30-50% before resuming uptrends
- Buying during corrections is value investing, not momentum chasing
- The current threshold essentially requires buying at highs

### Research Support
- DeBondt & Thaler (1985): Mean reversion in stock returns
- Jegadeesh & Titman (1993): Medium-term momentum (3-12 months), not near-52W high
- Value investors (Buffett, Marks) advocate buying during corrections

### Impact
Stocks potentially unlocked: TEAM, SNOW, NOW, HOOD, SE, MELI, KKR, and ~100+ others

---

## CHANGE 4: Remove require_above_200dma for MID Caps

### Location
`config.yaml`, us_mid section

### Current Value
```yaml
us_mid:
  buy:
    require_above_200dma: true
```

### Proposed Value
```yaml
us_mid:
  buy:
    require_above_200dma: false  # Removed - redundant with 52W check
```

### Rationale
- The 200DMA requirement is redundant with the 52W% check
- If a stock is at 60% of 52W high and above 200DMA, it's a contradiction
- Growth stocks often consolidate below 200DMA during sector rotations
- This check alone blocks ~50 potential BUY candidates
- Keep require_above_200dma=true for SMALL caps (higher risk tier)

### Impact
Stocks unlocked: ~30-50 mid-cap growth stocks

---

## CHANGE 5: Add Sector-Specific Debt/Equity Overrides

### Location
`trade_modules/trade_config.py` and `config.yaml`

### New Config Section
```yaml
# Add to config.yaml
sector_adjustments:
  debt_equity_multipliers:
    Financial Services: 3.0      # Banks, asset managers naturally leverage
    Technology: 1.5              # Tech uses debt for buybacks
    Real Estate: 2.5             # REITs are leveraged by design
    Consumer Cyclical: 1.3       # E-commerce growth companies
    Utilities: 2.0               # Capital-intensive, stable cash flows
    Communication Services: 1.5  # Telecom infrastructure costs
  # Base max_debt_equity from tier config is multiplied by this factor
```

### Code Change in trade_config.py
```python
def get_sector_adjusted_thresholds(self, ticker: str, signal_type: str, base_criteria: dict) -> dict:
    """
    Apply sector-specific adjustments to trading thresholds.

    For debt/equity, certain sectors naturally operate with higher leverage.
    """
    # Get sector for this ticker (from cache or API)
    sector = self._get_sector(ticker)

    if not sector:
        return base_criteria

    criteria = base_criteria.copy()

    # Apply debt/equity sector multiplier
    sector_multipliers = self.config.get('sector_adjustments', {}).get('debt_equity_multipliers', {})
    multiplier = sector_multipliers.get(sector, 1.0)

    if 'max_debt_equity' in criteria and multiplier != 1.0:
        original = criteria['max_debt_equity']
        criteria['max_debt_equity'] = original * multiplier
        logger.debug(f"{ticker}: Sector {sector} DE multiplier {multiplier}x - max_debt_equity {original} -> {criteria['max_debt_equity']}")

    return criteria
```

### Rationale
- MELI (Consumer Cyclical) blocked by DE=159% vs max=130%
- HOOD (Financial Services) blocked by DE=189% vs max=130%
- ORCL (Technology) blocked by DE=432% vs max=175%
- KKR (Financial Services) operates with leverage by design
- One-size-fits-all DE threshold ignores sector economics

### Research Support
- Modigliani-Miller: Capital structure depends on tax shields and costs
- Financial sector: Leverage is the business model
- Tech sector: Share buybacks funded by debt (Apple, Microsoft, Oracle)

### Impact
Stocks unlocked: MELI, HOOD, ORCL, KKR, and ~20 others

---

## CHANGE 6: Growth Company ROE Override

### Location
`trade_modules/analysis/signals.py`, around line 1301

### Current Code
```python
# ROE and DE BUY criteria (with sector adjustments)
if "min_roe" in buy_criteria and not pd.isna(row_roe):
    if row_roe < buy_criteria.get("min_roe"):
        is_buy_candidate = False
        logger.debug(f"Ticker {ticker}: Failed ROE check - ROE:{row_roe:.1f}% < min:{buy_criteria['min_roe']:.1f}%")
```

### Proposed Code
```python
# ROE and DE BUY criteria (with sector adjustments)
if "min_roe" in buy_criteria and not pd.isna(row_roe):
    if row_roe < buy_criteria.get("min_roe"):
        # Growth company override: high-growth, high-margin companies may have negative ROE
        # during reinvestment phase (e.g., SNOW, TEAM, CRM historically)
        is_growth_override = (
            not pd.isna(row_rev_growth) and row_rev_growth > 25 and  # High revenue growth
            row_upside > 50 and                                       # Strong upside
            row_buy_pct > 85 and                                      # High analyst conviction
            row_roe > -30                                             # Not catastrophically negative
        )

        if is_growth_override:
            logger.info(f"Ticker {ticker}: GROWTH OVERRIDE - allowing negative ROE={row_roe:.1f}% "
                       f"(rev_growth={row_rev_growth:.1f}%, upside={row_upside:.1f}%, buy%={row_buy_pct:.1f}%)")
            # Don't block, but flag for position sizing
        else:
            is_buy_candidate = False
            logger.debug(f"Ticker {ticker}: Failed ROE check - ROE:{row_roe:.1f}% < min:{buy_criteria['min_roe']:.1f}%")
```

### Rationale
- TEAM blocked by ROE=-15.3% despite UP%=119%, %B=94%
- SNOW blocked by ROE=-53.1% despite UP%=66.7%, %B=93%
- High-growth SaaS companies (Snowflake, Atlassian, Datadog) intentionally reinvest
- Amazon had negative ROE for years during growth phase
- Current check penalizes growth-at-scale investing strategy

### Research Support
- Novy-Marx (2013): Gross profitability better predictor than net profitability
- Growth companies: CAC payback economics differ from traditional ROE
- SaaS model: Negative ROE during land-expand phase is expected

### Safeguards
- Only override if revenue growth > 25%
- Only override if analyst buy% > 85%
- Only override if upside > 50%
- Don't override if ROE < -30% (truly distressed)

### Impact
Stocks potentially unlocked: TEAM, SNOW, and ~10-15 high-growth tech stocks

---

## CHANGE 7: FCF Yield Flexibility for Growth

### Location
`config.yaml`, us_mid and us_small sections

### Current Values
```yaml
us_mid:
  buy:
    min_fcf_yield: 0.0
us_small:
  buy:
    min_fcf_yield: -1.0
```

### Proposed Values
```yaml
us_mid:
  buy:
    min_fcf_yield: -8.0   # Allow moderate cash burn for growth
us_small:
  buy:
    min_fcf_yield: -5.0   # Slightly stricter for smaller caps
```

### Rationale
- MELI blocked by FCF=-4.1% vs min=0.0%
- Amazon, Netflix, Uber all had negative FCF during growth phases
- Penalizing cash burn ignores growth investment economics
- The framework already captures risk through other metrics (analyst consensus, upside)

### Safeguards
- Only apply lenient FCF for stocks with high analyst buy% (>80%)
- Position size adjustment for negative FCF companies
- Still require positive FCF for SMALL caps with lower analyst coverage

### Impact
Stocks unlocked: MELI, and ~5-10 growth companies

---

## CHANGE 8: Add BUY Conviction Tiers

### Location
`trade_modules/analysis/signals.py`, enhance calculate_buy_score function

### New Code Section
```python
def classify_buy_conviction(buy_score: float, criteria_passes: int, criteria_total: int) -> str:
    """
    Classify BUY conviction level for position sizing.

    Returns:
        'HIGH': Score 85+, passes all criteria
        'MEDIUM': Score 70-84, or passes all but 1 criterion
        'LOW': Score 55-69, or quality override applied
        'SPECULATIVE': Score < 55, multiple criteria failures
    """
    pass_rate = criteria_passes / criteria_total if criteria_total > 0 else 0

    if buy_score >= 85 and pass_rate >= 0.95:
        return 'HIGH'
    elif buy_score >= 70 or pass_rate >= 0.85:
        return 'MEDIUM'
    elif buy_score >= 55 or pass_rate >= 0.75:
        return 'LOW'
    else:
        return 'SPECULATIVE'
```

### Position Sizing Integration
```python
# In position sizing logic
CONVICTION_MULTIPLIERS = {
    'HIGH': 1.0,        # Full position
    'MEDIUM': 0.75,     # 75% position
    'LOW': 0.50,        # 50% position
    'SPECULATIVE': 0.25 # 25% position
}
```

### Rationale
- Currently all BUY signals treated equally
- NVDA (98% buy, 36.8% upside) vs INTU (88% buy, 75.3% upside, 52W=55%)
- Position sizing should reflect conviction level
- Allows framework to include borderline BUYs with reduced exposure

### Impact
- Better risk management
- More BUY signals with appropriate position sizing
- Reduced drawdown risk from speculative positions

---

## IMPLEMENTATION SCHEDULE

### Phase 1: Week 1 (Immediate)
| Change # | Description | Risk | Expected BUY Increase |
|----------|-------------|------|----------------------|
| 1 | PEF/PET threshold 1.1 -> 1.25 | Low | +5-10 |
| 2 | Short interest thresholds | Low | +10-15 |

### Phase 2: Week 2
| Change # | Description | Risk | Expected BUY Increase |
|----------|-------------|------|----------------------|
| 3 | 52W momentum 75% -> 55-60% | Medium | +50-80 |
| 4 | Remove 200DMA for MID | Medium | +30-50 |

### Phase 3: Week 3-4
| Change # | Description | Risk | Expected BUY Increase |
|----------|-------------|------|----------------------|
| 5 | Sector DE overrides | Low | +15-25 |
| 6 | Growth ROE override | Medium | +10-15 |
| 7 | FCF flexibility | Low | +5-10 |

### Phase 4: Week 5+
| Change # | Description | Risk | Expected BUY Increase |
|----------|-------------|------|----------------------|
| 8 | Conviction tiers | None | Position sizing |

---

## BACKTESTING REQUIREMENTS

Before implementing Phase 2+ changes, backtest against:
1. 2022-2023 bear market (test false positive rate)
2. 2024-2025 recovery (test false negative rate)
3. March 2020 crash/recovery (extreme volatility)

### Success Metrics
- BUY signal win rate > 55% (1-year forward return positive)
- Average BUY signal return > benchmark + 5%
- SELL signal avoided loss rate > 60%
- Sharpe ratio improvement vs. current framework

---

## MONITORING PLAN

After implementation, monitor weekly:
1. Signal distribution (target: 50-100 BUY signals, not 29)
2. BUY signal quality metrics
3. False positive tracking (BUYs that became SELLs within 3 months)
4. Sector/geography balance

### Rollback Triggers
- BUY signal win rate drops below 45%
- Average BUY return becomes negative
- Position concentration exceeds 30% in single sector

---

*Implementation Plan Version: 1.0*
*Created: February 7, 2026*
*Status: Ready for Review*
