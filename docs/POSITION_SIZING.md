# Position Sizing Methodology

## Overview

The etorotrade system employs a sophisticated, multi-factor position sizing approach that balances market capitalization tiers, expected returns, and risk management. The methodology has been modernized in 2025 to provide more intuitive scaling while maintaining rigorous risk controls.

## 2025 Current Framework

### Portfolio Configuration
- **Portfolio Value**: $450,000 (updated from previous $500K)
- **Position Limits**: 
  - Minimum: $1,000 (0.22% of portfolio)
  - Maximum: $40,000 (8.89% of portfolio)
- **Base Position**: 0.5% of portfolio = $2,250 for standard opportunities

### Step 1: Base Position Calculation

The foundation starts with a base position that is then scaled by multiple factors:

**Base Position**: 0.5% of portfolio = $2,250

This represents the standard allocation for a typical investment opportunity with moderate expected returns.

### Step 2: EXRET-Based Scaling (Primary Driver)

The EXRET (Expected Return) multiplier provides the primary position sizing signal based on opportunity quality:

| EXRET Range | Multiplier | Position Type | Example Amount |
|-------------|------------|---------------|----------------|
| ≥40% | 5.0x | Exceptional opportunity | $11,250 |
| ≥30% | 4.0x | High opportunity | $9,000 |
| ≥25% | 3.0x | Good opportunity | $6,750 |
| ≥20% | 2.0x | Standard opportunity | $4,500 |
| ≥15% | 1.5x | Lower opportunity | $3,375 |
| ≥10% | 1.0x | Base position | $2,250 |
| <10% | 0.5x | Conservative | $1,125 |

**Academic Basis**: 
- Kelly Criterion principles: Position size proportional to expected edge
- Simplified from complex stepped functions for cleaner scaling
- Reduced maximum multiplier from 10x to 5x to limit concentration risk

### Step 3: Market Cap Tier Scaling (Secondary Driver)

Market capitalization tier provides risk-adjusted scaling based on company stability using the new 5-tier system:

| Tier | Market Cap Range | Multiplier | Rationale |
|------|------------------|------------|-----------|
| **MEGA** | ≥$500B | 3.0x | Mega-cap champions premium |
| **LARGE** | $100B-$500B | 2.5x | Large-cap stability premium |
| **MID** | $10B-$100B | 1.5x | Mid-cap balanced allocation |
| **SMALL** | $2B-$10B | 0.75x | Small-cap opportunity/risk balance |
| **MICRO** | <$2B | 0.5x | Micro-cap risk management |

**Academic Basis**:
- Fama-French size factor research
- Lower volatility and higher Sharpe ratios in large-cap stocks
- Institutional investor preferences for liquid, stable positions

### Step 4: Geographic Risk Adjustment

| Market | Multiplier | Rationale |
|--------|------------|-----------|
| Hong Kong (.HK) | 0.75x | Concentration risk management |
| All Others | 1.0x | No adjustment |

### Final Calculation

```
Position Size = Base Position × EXRET Multiplier × Market Cap Tier × Geographic Risk
```

**Post-Processing**:
1. Apply minimum ($1,000) and maximum ($40,000) limits
2. Round to nearest dollar for display (no forced rounding)

## Implementation Examples

### MEGA Tier: Mega-Cap Technology (AAPL-like)
- **Market Cap**: $3T (MEGA tier)
- **EXRET**: 15%
- **Geographic**: US

**Calculation**:
```
Base: $2,250 (0.5% of $450K)
EXRET: 1.5x (15% opportunity)
Market Cap: 3.0x (MEGA tier)
Geographic: 1.0x (US stock)

Position = $2,250 × 1.5 × 3.0 × 1.0 = $10,125
```

### MID Tier: Mid-Cap Growth
- **Market Cap**: $25B (MID tier)
- **EXRET**: 22%
- **Geographic**: US

**Calculation**:
```
Base: $2,250 (0.5% of $450K)
EXRET: 2.0x (22% opportunity)
Market Cap: 1.5x (MID tier)
Geographic: 1.0x (US stock)

Position = $2,250 × 2.0 × 1.5 × 1.0 = $6,750
```

### MICRO Tier: Micro-Cap with Hong Kong Risk
- **Market Cap**: $1.5B (MICRO tier)
- **EXRET**: 28%
- **Geographic**: Hong Kong

**Calculation**:
```
Base: $2,250 (0.5% of $450K)
EXRET: 3.0x (28% opportunity)
Market Cap: 0.5x (MICRO tier)
Geographic: 0.75x (Hong Kong risk)

Position = $2,250 × 3.0 × 0.5 × 0.75 = $2,531
```

## Academic References & Rationale

### Modern Portfolio Theory Integration
- **Markowitz (1952)**: Mean-variance optimization principles
- **Fama & French (1992)**: Size and value factors
- **Black & Litterman (1990)**: Bayesian approach to expected returns

### Risk Management Principles
- **Kelly (1956)**: Optimal betting strategies and position sizing
- **Bridgewater Risk Parity**: Equal risk contribution methodology
- **Institutional Research**: Vanguard, BlackRock position sizing frameworks

### Key Advantages

1. **Reduces Estimation Error**: Limited reliance on expected return forecasts
2. **Systematic Risk Management**: Beta-based volatility adjustment
3. **Market Cap Focus**: Aligns with institutional investment preferences
4. **Mathematical Consistency**: Smooth linear formulas replace stepped tiers
5. **Concentration Control**: Geographic and size-based diversification

### Evolution from Previous Approach

| Aspect | Previous (2024) | Current (2025) | Improvement |
|--------|----------|---------|-------------|
| Portfolio Size | $500K | $450K | Realistic capacity |
| Base Position | Tier-based | 0.5% uniform | Simplified foundation |
| Primary Driver | Market cap tiers | EXRET scaling | Opportunity-focused |
| EXRET Range | 0.0167 coefficient | 0.5x-5.0x steps | Intuitive scaling |
| Beta Adjustment | Linear formula | Removed | Simplified system |
| Tier System | 3 tiers | 5 tiers | Granular classification |
| MEGA Multiplier | N/A | 3.0x tier | New mega-cap category |
| LARGE Multiplier | 2.0x base | 2.5x tier | Enhanced large-cap |
| MID Multiplier | 1.0x base | 1.5x tier | Balanced approach |
| SMALL Multiplier | N/A | 0.75x tier | New small-cap balance |
| MICRO Multiplier | 0.2x base | 0.5x tier | Better micro-cap |

## Implementation Location

The position sizing logic is implemented in:
- **Primary Function**: `yahoofinance/utils/data/format_utils.py::calculate_position_size()`
- **Constants**: `trade_modules/constants.py` - All thresholds and multipliers
- **Configuration**: `yahoofinance/core/config.py::PORTFOLIO_CONFIG`
- **Integration**: Trade processing pipeline automatically applies position sizing

## Configuration Constants

Key constants controlling position sizing behavior:

```python
# From trade_modules/trade_config.py
PORTFOLIO_VALUE = 450_000            # Total portfolio value
BASE_POSITION_PERCENTAGE = 0.5      # 0.5% base position
MIN_POSITION_SIZE = 1_000           # $1K minimum
MAX_POSITION_SIZE = 40_000          # $40K maximum

# Market cap tier thresholds (5-tier system)
TIER_THRESHOLDS = {
    "mega_tier_min": 500_000_000_000,    # $500B+ = MEGA
    "large_tier_min": 100_000_000_000,   # $100B-500B = LARGE
    "mid_tier_min": 10_000_000_000,      # $10B-100B = MID
    "small_tier_min": 2_000_000_000,     # $2B-10B = SMALL
    # Below $2B = MICRO
}
```

## Testing & Validation

Comprehensive unit tests validate the methodology:
- **Test File**: `tests/trade_modules/test_analysis_engine.py`
- **Coverage**: All tier combinations, EXRET ranges, and geographic adjustments
- **Verification**: Mathematical formulas and edge case handling
- **Integration**: End-to-end position sizing within trade recommendations

## Benefits of Current Approach

1. **Intuitive Scaling**: Clear relationship between opportunity quality and position size
2. **Risk Management**: Tier-based and geographic risk adjustments
3. **Simplified Logic**: Removed complex beta calculations for maintainability  
4. **Opportunity Focus**: EXRET-driven sizing aligns with investment merit
5. **Realistic Limits**: $1K-$40K range appropriate for $450K portfolio

---
*Last Updated: January 2025*
*Framework Version: 2025.1 (Modernized)*