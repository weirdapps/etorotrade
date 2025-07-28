# Position Sizing Methodology

## Overview

The etorotrade system employs a sophisticated, academically-backed position sizing approach that prioritizes market capitalization tiers while incorporating risk adjustments and expected return tilts. This methodology was designed to optimize risk-adjusted returns while minimizing estimation errors common in traditional expected return-based approaches.

## 2024 Updated Framework

### Portfolio Configuration
- **Portfolio Value**: $500,000
- **Position Limits**: 
  - Minimum: $1,000 (0.2% of portfolio)
  - Maximum: $40,000 (8.0% of portfolio)
- **Rounding**: All positions rounded to nearest $500 for cleaner execution

### Step 1: Tier-Based Base Allocation (Primary Driver)

The foundation of our approach is market capitalization-based tiering, reflecting the empirical size effect documented in academic literature:

| Tier | Market Cap Range | Base Allocation | Dollar Amount | Rationale |
|------|------------------|-----------------|---------------|-----------|
| **VALUE** | ≥$100B | 2.0% | $10,000 | Large-cap stability premium |
| **GROWTH** | $5B-$100B | 1.0% | $5,000 | Standard mid-cap allocation |
| **BETS** | <$5B | 0.2% | $1,000 | Small-cap risk management |

**Academic Basis**: 
- Fama-French size factor research
- Lower volatility and higher Sharpe ratios in large-cap stocks
- Institutional investor preferences for liquid, stable positions

### Step 2: Linear Beta Risk Adjustment (Secondary Driver)

**Formula**: `risk_multiplier = 1.4 - (beta × 0.4)`

**Range**: 1.2x (low risk) to 0.8x (high risk)

| Beta Range | Multiplier | Risk Profile | Example |
|------------|------------|--------------|---------|
| ≤0.5 | 1.2x | Low volatility premium | Utilities, Consumer staples |
| 1.0 | 1.0x | Market risk (baseline) | Broad market ETFs |
| 1.5 | 0.8x | High risk penalty | Growth tech, Biotech |
| ≥2.5 | 0.8x | Maximum penalty (floor) | Speculative stocks |

**Academic Basis**:
- Kelly Criterion: Optimal position sizing inversely related to variance
- Risk Parity principles: Equal risk contribution across positions
- CAPM systematic risk adjustment

### Step 3: Linear EXRET Tilt (Tertiary Driver)

**Formula**: `exret_multiplier = 1.0 + (exret × 0.0167)`

**Range**: 1.0x (0% EXRET) to 1.5x (30%+ EXRET)

| EXRET Level | Multiplier | Interpretation |
|-------------|------------|----------------|
| 0% | 1.0x | No expected return advantage |
| 10% | 1.167x | Modest opportunity |
| 20% | 1.33x | Strong opportunity |
| 30%+ | 1.5x | Exceptional opportunity (capped) |

**Academic Basis**:
- Reduced from previous 10x range to minimize estimation error
- Black-Litterman model: Modest tilts toward expected outperformance
- Practitioner research on forecast uncertainty

### Step 4: Geographic Risk Adjustment

| Market | Multiplier | Rationale |
|--------|------------|-----------|
| Hong Kong (.HK) | 0.75x | Concentration risk management |
| All Others | 1.0x | No adjustment |

### Final Calculation

```
Position Size = Base Allocation × Beta Risk × EXRET Tilt × Geographic Risk
```

**Post-Processing**:
1. Apply minimum ($1,000) and maximum ($40,000) limits
2. Round to nearest $500 for execution efficiency

## Implementation Examples

### VALUE Tier: Large-Cap Technology (AAPL-like)
- **Market Cap**: $3T (VALUE tier)
- **Beta**: 1.2 (moderate risk)
- **EXRET**: 10%

**Calculation**:
```
Base: $10,000 (2.0% of $500K)
Beta: 1.4 - (1.2 × 0.4) = 1.0
EXRET: 1.0 + (10 × 0.0167) = 1.167
Geographic: 1.0 (US stock)

Position = $10,000 × 1.0 × 1.167 × 1.0 = $11,670 → $11,500
```

### GROWTH Tier: Mid-Cap Growth
- **Market Cap**: $25B (GROWTH tier)
- **Beta**: 1.0 (market risk)
- **EXRET**: 15%

**Calculation**:
```
Base: $5,000 (1.0% of $500K)
Beta: 1.4 - (1.0 × 0.4) = 1.0
EXRET: 1.0 + (15 × 0.0167) = 1.25
Geographic: 1.0

Position = $5,000 × 1.0 × 1.25 × 1.0 = $6,250 → $6,000
```

### BETS Tier: Small-Cap Speculation
- **Market Cap**: $2B (BETS tier)
- **Beta**: 1.8 (high risk)
- **EXRET**: 25%

**Calculation**:
```
Base: $1,000 (0.2% of $500K)
Beta: 1.4 - (1.8 × 0.4) = 0.68 → capped at 0.8
EXRET: 1.0 + (25 × 0.0167) = 1.417
Geographic: 1.0

Position = $1,000 × 0.8 × 1.417 × 1.0 = $1,134 → $1,000 (minimum)
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

| Aspect | Previous | Current | Improvement |
|--------|----------|---------|-------------|
| Portfolio Size | $450K | $500K | Increased capacity |
| VALUE Base | 1.5% | 2.0% | Enhanced large-cap focus |
| GROWTH Base | 1.0% | 1.0% | Maintained balance |
| BETS Base | 0.5% | 0.2% | Better risk management |
| Beta Range | 0.75x-1.25x | 0.8x-1.2x | Tighter risk bounds |
| EXRET Range | 1.75x | 1.5x | Reduced estimation error |
| Formula Type | Stepped tiers | Linear scaling | Mathematical consistency |

## Implementation Location

The position sizing logic is implemented in:
- **Primary Function**: `yahoofinance/utils/data/format_utils.py::calculate_position_size()`
- **Configuration**: `yahoofinance/core/trade_criteria_config.py::TradingCriteria`
- **Integration**: `yahoofinance/presentation/console.py::_add_position_size_column()`

## Testing & Validation

Comprehensive unit tests validate the methodology:
- **Test File**: `tests/yahoofinance/utils/data/test_format_utils.py`
- **Coverage**: All tier combinations, edge cases, and boundary conditions
- **Verification**: Mathematical formulas and rounding behavior

---
*Last Updated: July 2024*
*Framework Version: 2024.1*